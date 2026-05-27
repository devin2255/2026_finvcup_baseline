import json
import random
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


@dataclass
class TrainSample:
    conv_id: str
    end_idx: int
    label: int


@dataclass
class TrainSampleMulti:
    conv_id: str
    end_idx: int
    # (BC, I, T) or any configured order
    label_vec: Tuple[int, ...]
    # 方案2: 辅助任务目标，未来25个chunk的具体类别序列
    future_seq: Tuple[int, ...] = None


def list_conv_ids(labels_dir: Path) -> List[str]:
    return sorted([p.stem for p in labels_dir.glob("*.npy")])


def split_conversation_ids(
    conv_ids: Sequence[str], valid_ratio: float, seed: int
) -> Dict[str, List[str]]:
    conv_ids = list(conv_ids)
    random.Random(seed).shuffle(conv_ids)
    valid_size = max(1, int(len(conv_ids) * valid_ratio))
    valid_ids = sorted(conv_ids[:valid_size])
    train_ids = sorted(conv_ids[valid_size:])
    return {"train": train_ids, "valid": valid_ids}


def build_train_samples(
    labels_dir: Path,
    conv_ids: Sequence[str],
    context_chunks: int,
    target_chunks: int,
    stride: int,
    positive_ids: Sequence[int],
    max_samples: Optional[int] = None,
) -> List[TrainSample]:
    samples: List[TrainSample] = []
    pos_set = set(positive_ids)
    for conv_id in conv_ids:
        labels = np.load(labels_dir / f"{conv_id}.npy")
        max_end = labels.shape[0] - target_chunks
        for end_idx in range(context_chunks, max_end + 1, stride):
            future = labels[end_idx : end_idx + target_chunks]
            y = int(any(int(x) in pos_set for x in future))
            samples.append(TrainSample(conv_id=conv_id, end_idx=end_idx, label=y))
            if max_samples is not None and len(samples) >= max_samples:
                return samples
    return samples


def build_train_samples_multitask(
    labels_dir: Path,
    conv_ids: Sequence[str],
    context_chunks: int,
    target_chunks: int,
    stride: int,
    label_ids: Dict[str, int],
    target_labels: Sequence[str] = ("BC", "I", "T"),
    max_samples: Optional[int] = None,
) -> List[TrainSampleMulti]:
    samples: List[TrainSampleMulti] = []
    target_id_list = [int(label_ids[k]) for k in target_labels]
    for conv_id in conv_ids:
        labels = np.load(labels_dir / f"{conv_id}.npy")
        max_end = labels.shape[0] - target_chunks
        for end_idx in range(context_chunks, max_end + 1, stride):
            future = labels[end_idx : end_idx + target_chunks]
            y_vec = tuple(int(any(int(x) == tid for x in future)) for tid in target_id_list)
            # 方案2: 引入 VAP (Voice Activity Projection) 序列预测作为辅助任务
            samples.append(TrainSampleMulti(conv_id=conv_id, end_idx=end_idx, label_vec=y_vec, future_seq=tuple(int(x) for x in future)))
            if max_samples is not None and len(samples) >= max_samples:
                return samples
        return samples


def _speaker_token(channel_id: int) -> str:
    return "[SPK1]" if channel_id == 1 else "[SPK2]"


def build_text_context(
    utterances: Iterable[Dict],
    start_ms: int,
    end_ms: int,
    max_utterances: int = 120,
) -> str:
    selected = []
    for utt in utterances:
        utt_start = int(utt.get("start_ms", 0))
        utt_end = int(utt.get("end_ms", utt_start))
        if utt_end <= start_ms or utt_start >= end_ms:
            continue
        text = str(utt.get("text", "")).strip()
        if not text:
            continue
        selected.append(f"{_speaker_token(int(utt.get('channel_id', 1)))} {text}")
    if not selected:
        return "[SPK1] <silence> [SPK2] <silence>"
    if len(selected) > max_utterances:
        selected = selected[-max_utterances:]
    return " ".join(selected)


def _read_wav_slice(path: Path, start_ms: int, end_ms: int) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = int(wf.getframerate())
        n_ch = int(wf.getnchannels())
        sampwidth = int(wf.getsampwidth())
        start_frame = max(0, int(start_ms * sr / 1000))
        end_frame = max(start_frame + 1, int(end_ms * sr / 1000))
        total_frames = wf.getnframes()
        end_frame = min(end_frame, total_frames)
        read_frames = max(1, end_frame - start_frame)
        wf.setpos(start_frame)
        raw = wf.readframes(read_frames)

    if sampwidth == 2:
        dtype = np.int16
        scale = 32768.0
    elif sampwidth == 4:
        dtype = np.int32
        scale = 2147483648.0
    elif sampwidth == 1:
        dtype = np.uint8
        scale = 128.0
    else:
        raise RuntimeError(f"Unsupported wav sample width {sampwidth} for {path}")

    data = np.frombuffer(raw, dtype=dtype)
    if n_ch > 1:
        data = data.reshape(-1, n_ch)
    else:
        data = data[:, None]

    if sampwidth == 1:
        data = (data.astype(np.float32) - 128.0) / scale
    else:
        data = data.astype(np.float32) / scale
    return data, sr


class TurnTakingTrainDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[TrainSample | TrainSampleMulti],
        train_audio_dir: Path,
        train_text_dir: Path,
        train_labels_dir: Path,
        context_chunks: int,
        target_chunks: int,
        chunk_ms: int,
        sample_rate: int,
        augment_audio: bool = True,
    ) -> None:
        self.samples = list(samples)
        self.train_audio_dir = train_audio_dir
        self.train_text_dir = train_text_dir
        self.train_labels_dir = train_labels_dir
        self.context_chunks = context_chunks
        self.target_chunks = target_chunks
        self.chunk_ms = chunk_ms
        self.sample_rate = sample_rate
        self.augment_audio = augment_audio

    def __len__(self) -> int:
        return len(self.samples)

    @lru_cache(maxsize=256)
    def _load_labels(self, conv_id: str) -> np.ndarray:
        return np.load(self.train_labels_dir / f"{conv_id}.npy")

    @lru_cache(maxsize=256)
    def _load_text_json(self, conv_id: str) -> Dict:
        with open(self.train_text_dir / f"{conv_id}.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_wave_segment(self, conv_id: str, start_ms: int, end_ms: int) -> torch.Tensor:
        wav_path = self.train_audio_dir / f"{conv_id}.wav"
        audio, src_sr = _read_wav_slice(wav_path, start_ms, end_ms)
        wave = torch.from_numpy(audio.T)  # [C, T]
        if wave.shape[0] == 1:
            wave = wave.repeat(2, 1)
        elif wave.shape[0] > 2:
            wave = wave[:2]

        if src_sr != self.sample_rate:
            wave = torchaudio.functional.resample(wave, src_sr, self.sample_rate)

        expected_frames = int((end_ms - start_ms) * self.sample_rate / 1000)
        if wave.shape[1] < expected_frames:
            pad = expected_frames - wave.shape[1]
            wave = torch.nn.functional.pad(wave, (0, pad))
        elif wave.shape[1] > expected_frames:
            wave = wave[:, :expected_frames]
        return wave

    def _augment_audio(self, wave: torch.Tensor) -> torch.Tensor:
        """SpecAugment-style time masking + Gaussian noise (training only)."""
        if not self.augment_audio:
            return wave
        T = wave.shape[1]
        # Random Gaussian noise
        if torch.rand(1).item() < 0.3:
            wave = wave + 0.005 * torch.randn_like(wave)
        # Random time mask (zero out a segment)
        if torch.rand(1).item() < 0.5:
            mask_len = int(T * 0.05 * torch.rand(1).item())
            mask_start = torch.randint(0, max(1, T - mask_len), (1,)).item()
            wave[:, mask_start:mask_start + mask_len] = 0
        return wave

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        labels = self._load_labels(sample.conv_id)
        end_idx = sample.end_idx
        start_idx = end_idx - self.context_chunks

        context_labels = labels[start_idx:end_idx].astype(np.int64)
        start_ms = start_idx * self.chunk_ms
        end_ms = end_idx * self.chunk_ms

        text_json = self._load_text_json(sample.conv_id)
        text = build_text_context(text_json.get("utterances", []), start_ms, end_ms)
        wave = self._load_wave_segment(sample.conv_id, start_ms, end_ms)
        wave = self._augment_audio(wave)

        out = {
            "conv_id": sample.conv_id,
            "end_idx": end_idx,
            "waveform": wave,
            "text": text,
            "context_labels": torch.from_numpy(context_labels),
        }
        if hasattr(sample, "label_vec"):
            out["label"] = torch.tensor(sample.label_vec, dtype=torch.float32)
            if hasattr(sample, "future_seq") and sample.future_seq is not None:
                out["future_seq"] = torch.tensor(sample.future_seq, dtype=torch.long)
        else:
            out["label"] = torch.tensor(float(sample.label), dtype=torch.float32)
        return out


class TurnTakingTestDataset(Dataset):
    def __init__(
        self,
        test_root: Path,
        sample_rate: int,
    ) -> None:
        self.sample_rate = sample_rate
        self.base = test_root
        self.audio_dir = self.base / "audio"
        self.text_dir = self.base / "text"
        self.context_dir = self.base / "context"
        self.segment_ids = sorted([p.stem for p in self.context_dir.glob("*.npy")])

    def __len__(self) -> int:
        return len(self.segment_ids)

    @lru_cache(maxsize=512)
    def _load_text_json(self, seg_id: str) -> Dict:
        with open(self.text_dir / f"{seg_id}.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def __getitem__(self, idx: int) -> Dict:
        seg_id = self.segment_ids[idx]
        context_labels = np.load(self.context_dir / f"{seg_id}.npy").astype(np.int64)
        text_json = self._load_text_json(seg_id)
        start_ms = int(text_json.get("start_ms", 0))
        end_ms = int(text_json.get("end_ms", 30000))
        text = build_text_context(text_json.get("utterances", []), start_ms, end_ms)

        wav_path = self.audio_dir / f"{seg_id}.wav"
        audio, src_sr = _read_wav_slice(wav_path, start_ms, end_ms)
        wave = torch.from_numpy(audio.T)
        if wave.shape[0] == 1:
            wave = wave.repeat(2, 1)
        elif wave.shape[0] > 2:
            wave = wave[:2]
        if src_sr != self.sample_rate:
            wave = torchaudio.functional.resample(wave, src_sr, self.sample_rate)

        return {
            "segment_id": seg_id,
            "waveform": wave,
            "text": text,
            "context_labels": torch.from_numpy(context_labels),
        }


class CollateFn:
    """Module-level callable class for DataLoader collate_fn (picklable on Windows)."""

    def __init__(self, tokenizer, text_max_length: int):
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        if hasattr(self.tokenizer, "truncation_side"):
            self.tokenizer.truncation_side = "left"
        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "left"

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        texts = [b["text"] for b in batch]
        tokenized = self.tokenizer(
            texts,
            max_length=self.text_max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        waves = [b["waveform"] for b in batch]
        max_len = max(w.shape[1] for w in waves)
        padded_waves = []
        for w in waves:
            if w.shape[1] < max_len:
                w = torch.nn.functional.pad(w, (0, max_len - w.shape[1]))
            padded_waves.append(w)

        out = {
            "waveform": torch.stack(padded_waves, dim=0),
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "context_labels": torch.stack([b["context_labels"] for b in batch], dim=0),
        }

        if "label" in batch[0]:
            out["label"] = torch.stack([b["label"] for b in batch], dim=0)
            if "future_seq" in batch[0]:
                out["future_seq"] = torch.stack([b["future_seq"] for b in batch], dim=0)
            out["conv_id"] = [b["conv_id"] for b in batch]
            out["end_idx"] = [b["end_idx"] for b in batch]
        else:
            out["segment_id"] = [b["segment_id"] for b in batch]
        return out


def build_collate_fn(tokenizer, text_max_length: int) -> CollateFn:
    return CollateFn(tokenizer, text_max_length)
