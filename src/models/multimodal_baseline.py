from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, WhisperFeatureExtractor, WhisperModel


class AudioEncoder(nn.Module):
    def __init__(self, sample_rate: int, n_mels: int, conv_channels: List[int], dropout: float):
        super().__init__()
        self.register_buffer("_log_clamp_min", torch.tensor(1e-4), persistent=False)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self._mel_transform = None

        c1, c2, c3 = conv_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(2, c1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.GELU(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c3),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
        )
        self.out_dim = c3

    def _ensure_mel(self, device: torch.device):
        if self._mel_transform is None:
            import torchaudio
            self._mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate, n_mels=self.n_mels,
                n_fft=1024, hop_length=320, win_length=1024,
            )
        self._mel_transform = self._mel_transform.to(device)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        self._ensure_mel(wave.device)
        bsz, chans, _ = wave.shape
        mel_list = []
        for c in range(chans):
            with torch.amp.autocast("cuda", enabled=False):
                m = self._mel_transform(wave[:, c, :].float())
                m = torch.clamp(m, min=float(self._log_clamp_min.item()))
                m = torch.log(m)
            mel_list.append(m)
        mel = torch.stack(mel_list, dim=1)
        return self.encoder(mel)


# ---------------------------------------------------------------------------
# Learnable attention pooling: attend to a subset of time steps
# ---------------------------------------------------------------------------
class AttentionPooling(nn.Module):
    """Single-head attention pooling over a sequence dimension."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.scale = hidden_dim ** -0.5

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, D]
        scores = (self.query * x).sum(dim=-1) * self.scale  # [B, T]
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.squeeze(-1)
            mask = mask.to(dtype=torch.bool, device=x.device)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)  # [B, T]
        if mask is not None:
            weights = weights * mask.to(dtype=weights.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        weights = weights.unsqueeze(-1)  # [B, T, 1]
        return (x * weights).sum(dim=1)  # [B, D]


class LowRankTensorFusion(nn.Module):
    """Low-rank tensor fusion for compact high-order modality interactions."""
    def __init__(self, input_dims: List[int], output_dim: int, rank: int):
        super().__init__()
        self.input_dims = list(input_dims)
        self.output_dim = output_dim
        self.rank = rank
        self.factors = nn.ParameterList([
            nn.Parameter(torch.empty(rank, input_dim + 1, output_dim))
            for input_dim in self.input_dims
        ])
        self.fusion_weights = nn.Parameter(torch.ones(rank, output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        for factor in self.factors:
            nn.init.xavier_normal_(factor)
        nn.init.constant_(self.fusion_weights, 1.0 / max(1, self.rank))
        nn.init.zeros_(self.bias)

    def forward(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        if len(modalities) != len(self.factors):
            raise ValueError(f"Expected {len(self.factors)} modalities, got {len(modalities)}")

        fused = None
        for x, factor in zip(modalities, self.factors):
            ones = x.new_ones(x.shape[0], 1)
            augmented = torch.cat([ones, x], dim=-1)
            projected = torch.einsum("bd,rdo->bro", augmented, factor)
            fused = projected if fused is None else fused * projected

        return (fused * self.fusion_weights.unsqueeze(0)).sum(dim=1) + self.bias


class WhisperAudioEncoder(nn.Module):
    def __init__(
        self, model_name: str, sample_rate: int, proj_dim: int,
        freeze: bool = True, tail_ratio: float = 0.2,
        unfreeze_layers: int = 0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.freeze = freeze
        self.tail_ratio = tail_ratio
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.encoder = WhisperModel.from_pretrained(model_name).encoder
        if self.freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
        if unfreeze_layers > 0 and self.freeze:
            # Unfreeze the last N encoder layers for task adaptation
            total_layers = len(self.encoder.layers)
            for layer_idx in range(max(0, total_layers - unfreeze_layers), total_layers):
                for p in self.encoder.layers[layer_idx].parameters():
                    p.requires_grad = True
        self.encoder_has_trainable_layers = any(p.requires_grad for p in self.encoder.parameters())
        hidden_size = int(self.encoder.config.d_model)
        self.attn_pool = AttentionPooling(hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.out_dim = proj_dim

    def _build_input_features(self, wave: torch.Tensor) -> torch.Tensor:
        mono = wave.mean(dim=1)
        mono_np = mono.detach().float().cpu().numpy()
        inputs = self.feature_extractor(
            [x for x in mono_np],
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        return inputs["input_features"]

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            input_features = self._build_input_features(wave).to(wave.device)

        if self.freeze and not self.encoder_has_trainable_layers:
            with torch.no_grad():
                hidden = self.encoder(input_features=input_features).last_hidden_state
        else:
            hidden = self.encoder(input_features=input_features).last_hidden_state

        # Only attend to the tail portion of the time axis
        T = hidden.shape[1]
        tail_start = max(0, T - int(T * self.tail_ratio))
        tail_hidden = hidden[:, tail_start:, :]  # [B, tail_T, D]
        pooled = self.attn_pool(tail_hidden)
        return self.proj(pooled)


class ContextLabelEncoder(nn.Module):
    """Encode context label sequence with strong tail-awareness."""
    def __init__(self, vocab_size: int, embed_dim: int, channels: List[int],
                 tail_k: int = 50):
        super().__init__()
        c1, c2 = channels
        self.tail_k = tail_k
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Tail branch: only last K chunks → richer conv + flatten (no global pool)
        self.tail_conv = nn.Sequential(
            nn.Conv1d(embed_dim, c1, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(c1, c2, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.tail_proj = nn.Linear(c2 * tail_k, c2)

        # Full branch: whole sequence → conv + attention pool
        self.full_conv = nn.Sequential(
            nn.Conv1d(embed_dim, c1, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.full_attn_pool = AttentionPooling(c2)

        self.out_dim = c2 * 2  # tail + full concatenated

    def forward(self, context_labels: torch.Tensor) -> torch.Tensor:
        x = self.embedding(context_labels).transpose(1, 2)  # [B, E, L]

        # Tail branch
        tail_x = x[:, :, -self.tail_k:]  # [B, E, K]
        tail_feat = self.tail_conv(tail_x)  # [B, c2, K]
        tail_feat = self.tail_proj(tail_feat.flatten(1))  # [B, c2]

        # Full branch with attention pooling
        full_feat = self.full_conv(x)  # [B, c2, L]
        full_feat = self.full_attn_pool(full_feat.transpose(1, 2))  # [B, c2]

        return torch.cat([tail_feat, full_feat], dim=-1)  # [B, c2*2]


class HandcraftedFeatures(nn.Module):
    """Compute hand-crafted statistics from context labels."""
    def __init__(self, num_labels: int = 5, context_chunks: int = 375):
        super().__init__()
        self.num_labels = num_labels
        self.context_chunks = context_chunks
        self.out_dim = num_labels * 3 + 4  # 3 windows * 5 ratios + 4 extra

    def forward(self, context_labels: torch.Tensor) -> torch.Tensor:
        B, L = context_labels.shape
        device = context_labels.device
        one_hot = F.one_hot(context_labels.long(), self.num_labels).float()  # [B, L, 5]

        tail25 = one_hot[:, -25:, :].mean(dim=1)    # [B, 5]
        tail50 = one_hot[:, -50:, :].mean(dim=1)    # [B, 5]
        tail100 = one_hot[:, -100:, :].mean(dim=1)  # [B, 5]

        # Distance to last event (T=1, BC=2, I=3)
        event_mask = (context_labels == 1) | (context_labels == 2) | (context_labels == 3)
        indices = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        event_positions = torch.where(event_mask, indices, torch.zeros_like(indices))
        last_event_pos = event_positions.max(dim=1).values  # [B]
        has_event = event_mask.any(dim=1).float()
        dist_to_last = ((L - 1 - last_event_pos).float() / L) * has_event + (1.0 - has_event)

        # Last 3 raw labels normalized
        last1 = context_labels[:, -1].float() / (self.num_labels - 1)
        last2 = context_labels[:, -2].float() / (self.num_labels - 1) if L > 1 else torch.zeros(B, device=device)
        last3 = context_labels[:, -3].float() / (self.num_labels - 1) if L > 2 else torch.zeros(B, device=device)

        return torch.cat([
            tail25, tail50, tail100,
            dist_to_last.unsqueeze(1),
            last1.unsqueeze(1), last2.unsqueeze(1), last3.unsqueeze(1),
        ], dim=-1)


class TextEncoder(nn.Module):
    def __init__(self, model_name: str, freeze_backbone: bool = True,
                 tail_ratio: float = 0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.out_dim = int(self.backbone.config.hidden_size)
        self.tail_ratio = tail_ratio
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.attn_pool = AttentionPooling(self.out_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # [B, L, H]

        # Focus on the tail portion of the sequence (later utterances)
        L = hidden.shape[1]
        tail_start = max(0, L - int(L * self.tail_ratio))
        tail_hidden = hidden[:, tail_start:, :]
        tail_mask = attention_mask[:, tail_start:].to(dtype=torch.bool, device=hidden.device)
        pooled = self.attn_pool(tail_hidden, mask=tail_mask)
        return pooled


class MultimodalFusion(nn.Module):
    """Lightweight cross-modal fusion with low-rank multimodal interaction + adaptive gating.

    Key ideas:
    - Low-rank tensor fusion captures high-order interactions across all modalities
      without the full tensor product cost.
    - Adaptive gates let the model decide how much to trust each modality per sample.
    """

    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        context_dim: int,
        hand_dim: int,
        hidden_dim: int,
        bilinear_rank: int = 48,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Low-rank interaction over audio, text, context, and hand-crafted signals.
        self.low_rank_fusion = LowRankTensorFusion(
            input_dims=[audio_dim, text_dim, context_dim, hand_dim],
            output_dim=hidden_dim,
            rank=bilinear_rank,
        )
        self.low_rank_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Per-modality projections to hidden_dim
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.hand_proj = nn.Sequential(
            nn.Linear(hand_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )

        # Adaptive modality gates
        gate_in = audio_dim + text_dim + context_dim + hand_dim
        self.gate_net = nn.Sequential(
            nn.Linear(gate_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )

        # Final fusion projection
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_dim = hidden_dim

    def forward(
        self, audio: torch.Tensor, text: torch.Tensor,
        context: torch.Tensor, hand: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Low-rank tensor interaction: joint space across all modalities
        interaction_feat = self.low_rank_proj(
            self.low_rank_fusion([audio, text, context, hand])
        )  # [B, H]

        # 2. Per-modality projections
        a = self.audio_proj(audio)
        t = self.text_proj(text)
        c = self.context_proj(context)
        h = self.hand_proj(hand)

        # 3. Adaptive gates: learn when to trust each modality
        all_raw = torch.cat([audio, text, context, hand], dim=-1)
        gates = self.gate_net(all_raw)  # [B, 4]
        a_g, t_g, c_g, h_g = gates[:, 0:1], gates[:, 1:2], gates[:, 2:3], gates[:, 3:4]

        # 4. Concatenate all features (interaction + 4 gated modalities)
        fused = torch.cat([interaction_feat, a * a_g, t * t_g, c * c_g, h * h_g], dim=-1)
        return self.out_proj(fused)


class MultimodalTurnTakingModel(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        audio_type = str(cfg["audio_encoder"].get("type", "cnn")).lower()
        if audio_type == "whisper":
            self.audio_encoder = WhisperAudioEncoder(
                model_name=cfg["audio_encoder"]["model_name"],
                sample_rate=cfg["sample_rate"],
                proj_dim=int(cfg["audio_encoder"]["proj_dim"]),
                freeze=bool(cfg["audio_encoder"].get("freeze", True)),
                tail_ratio=float(cfg["audio_encoder"].get("tail_ratio", 0.2)),
                unfreeze_layers=int(cfg["audio_encoder"].get("unfreeze_layers", 0)),
            )
        else:
            self.audio_encoder = AudioEncoder(
                sample_rate=cfg["sample_rate"],
                n_mels=cfg["audio_encoder"]["n_mels"],
                conv_channels=cfg["audio_encoder"]["conv_channels"],
                dropout=cfg["audio_encoder"]["dropout"],
            )
        self.text_encoder = TextEncoder(
            model_name=cfg["text_encoder"]["model_name"],
            freeze_backbone=bool(cfg["text_encoder"].get("freeze_backbone", True)),
            tail_ratio=float(cfg["text_encoder"].get("tail_ratio", 0.3)),
        )

        ctx_cfg = cfg["context_encoder"]
        self.context_encoder = ContextLabelEncoder(
            vocab_size=ctx_cfg["vocab_size"],
            embed_dim=ctx_cfg["embed_dim"],
            channels=ctx_cfg["channels"],
            tail_k=int(ctx_cfg.get("tail_k", 50)),
        )

        self.hand_features = HandcraftedFeatures(
            num_labels=ctx_cfg["vocab_size"],
            context_chunks=int(cfg["context_chunks"]),
        )

        fusion_cfg = cfg.get("fusion", {})
        self.fusion = MultimodalFusion(
            audio_dim=self.audio_encoder.out_dim,
            text_dim=self.text_encoder.out_dim,
            context_dim=self.context_encoder.out_dim,
            hand_dim=self.hand_features.out_dim,
            hidden_dim=int(fusion_cfg.get("hidden_dim", 256)),
            bilinear_rank=int(fusion_cfg.get("bilinear_rank", 48)),
            dropout=float(fusion_cfg.get("dropout", 0.2)),
        )

        num_targets = len(cfg.get("labels", {}).get("multi_targets", []))
        self.num_targets = num_targets if num_targets > 0 else 1
        self.head = nn.Sequential(
            nn.Linear(self.fusion.out_dim, self.num_targets),
        )

    def forward(
        self,
        waveform: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_labels: torch.Tensor,
    ) -> torch.Tensor:
        audio_feat = self.audio_encoder(waveform)
        text_feat = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        context_feat = self.context_encoder(context_labels=context_labels)
        hand_feat = self.hand_features(context_labels)
        fused = self.fusion(audio_feat, text_feat, context_feat, hand_feat)
        logits = self.head(fused)
        if self.num_targets == 1:
            return logits.squeeze(-1)
        return logits
