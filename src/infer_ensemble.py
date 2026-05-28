"""
集成推理（Day 1 提分）：

- 支持多个 checkpoint 概率平均（同一份 config / 模型结构）。
- 支持双声道 TTA：mean(L,R) + L 单独 + R 单独 三路概率平均。
- 支持每标签独立阈值（threshold_file）或全局阈值。
- 支持 v2 软约束 + 全 0 兜底后处理。

输出 CSV 与 src/infer_test.py 完全一致：segment_id + 5 个小写标签的 0/1 列。

典型用法（单 ckpt + TTA + 全量阈值 + 软约束）：
  python -m src.infer_ensemble \
      --config configs/whisper_qwen0_6b_lmf_8g.yaml \
      --checkpoints outputs/lmf_8g/checkpoints/best_lmf.pt \
      --test_root test \
      --threshold_file outputs/lmf_8g/logs/best_thresholds_full.json \
      --tta dual_channel \
      --constraints v2 \
      --output_csv outputs/lmf_8g/pred_ensemble.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.data import TurnTakingTestDataset, build_collate_fn
from src.models import MultimodalTurnTakingModel
from src.utils import (
    apply_label_constraints,
    apply_label_constraints_v2,
    load_config,
    set_env_paths,
)


def parse_args():
    p = argparse.ArgumentParser(description="集成推理 + TTA + 软约束后处理")
    p.add_argument("--config", type=str, required=True)
    p.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="一个或多个 best_*.pt（同结构）；概率求平均",
    )
    p.add_argument("--test_root", type=str, required=True)
    p.add_argument("--output_csv", type=str, required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument(
        "--threshold_file",
        type=str,
        default=None,
        help="per-label thresholds JSON（优先于 --threshold；缺省时会回退到 ckpt[0].thresholds）",
    )
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--max_segments", type=int, default=None)
    p.add_argument(
        "--tta",
        type=str,
        default="none",
        choices=["none", "dual_channel"],
        help="dual_channel = 原始(双声道)+左声道+右声道 三份概率平均",
    )
    p.add_argument(
        "--constraints",
        type=str,
        default="v2",
        choices=["v1", "v2", "none"],
        help="v1=硬规则；v2=软约束+全0兜底；none=不做后处理",
    )
    p.add_argument("--na_strong_thr", type=float, default=0.6)
    p.add_argument("--na_max_other_prob", type=float, default=0.5)
    p.add_argument("--ct_margin", type=float, default=0.0)
    p.add_argument(
        "--allzero_fallback",
        type=str,
        default="na",
        choices=["na", "argmax", "na_or_argmax", "none"],
    )
    p.add_argument(
        "--save_probs",
        type=str,
        default=None,
        help="可选：把每条样本的 5 类平均概率额外保存为 .npz（segment_id, probs）",
    )
    return p.parse_args()


def _load_thresholds(threshold_file: str | None, ckpts: List[dict]) -> dict | None:
    if threshold_file:
        with open(threshold_file, "r", encoding="utf-8") as f:
            return json.load(f)["thresholds"]
    
    # 尝试从所有 ckpt 中提取 thresholds 并求平均
    all_thrs = []
    for ckpt in ckpts:
        if isinstance(ckpt, dict):
            thr = ckpt.get("thresholds")
            if isinstance(thr, dict):
                all_thrs.append(thr)
    
    if len(all_thrs) > 0:
        avg_thr = {}
        for key in all_thrs[0].keys():
            avg_thr[key] = sum(t.get(key, 0.5) for t in all_thrs) / len(all_thrs)
        return avg_thr
        
    return None


def _strip_module_prefix(state_dict: dict) -> dict:
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}


def _build_tta_waveforms(waveform: torch.Tensor, mode: str) -> List[torch.Tensor]:
    """Return a list of waveform tensors for TTA. Each: [B, C, T]."""
    if mode == "none":
        return [waveform]
    if mode == "dual_channel":
        out = [waveform]
        if waveform.shape[1] >= 2:
            left = waveform[:, 0:1, :].repeat(1, waveform.shape[1], 1).contiguous()
            right = waveform[:, 1:2, :].repeat(1, waveform.shape[1], 1).contiguous()
            out.extend([left, right])
        return out
    raise ValueError(f"Unknown TTA mode: {mode}")


@torch.no_grad()
def _forward_probs(
    model: torch.nn.Module,
    *,
    waveform: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    context_labels: torch.Tensor,
    use_amp: bool,
    tta: str,
) -> np.ndarray:
    """Return prob ndarray of shape [B, C], averaged over TTA views."""
    probs_accum: np.ndarray | None = None
    n_views = 0
    for wav in _build_tta_waveforms(waveform, tta):
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(
                waveform=wav,
                input_ids=input_ids,
                attention_mask=attention_mask,
                context_labels=context_labels,
            )
        if logits.ndim == 1:
            logits = logits.unsqueeze(-1)
        probs = torch.sigmoid(logits).detach().float().cpu().numpy()
        probs_accum = probs if probs_accum is None else probs_accum + probs
        n_views += 1
    return probs_accum / max(1, n_views)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_env_paths(cfg)

    multi_targets = list(cfg["labels"]["multi_targets"])
    label_cols = [x.lower() for x in multi_targets]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["train"].get("use_amp", False))

    tokenizer = AutoTokenizer.from_pretrained(cfg["text_encoder"]["model_name"], use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    collate_fn = build_collate_fn(tokenizer, int(cfg["text_encoder"]["max_length"]))

    test_root = Path(args.test_root)
    ds = TurnTakingTestDataset(test_root=test_root, sample_rate=int(cfg["sample_rate"]))
    bs = int(args.batch_size or cfg["train"]["eval_batch_size"])
    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    ckpts: List[dict] = []
    for ckpt_path in args.checkpoints:
        print(f"[ensemble] load checkpoint: {ckpt_path}")
        ckpts.append(torch.load(ckpt_path, map_location="cpu"))

    per_label_thresholds = _load_thresholds(args.threshold_file, ckpts)
    if per_label_thresholds is not None:
        print(f"[ensemble] per-label thresholds: {per_label_thresholds}")
    else:
        print(f"[ensemble] use global threshold = {args.threshold}")

    # 同一模型结构，重复加载不同 state_dict
    model = MultimodalTurnTakingModel(cfg).to(device)
    model.eval()

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["segment_id"] + label_cols

    segment_ids_out: List[str] = []
    probs_avg: List[np.ndarray] = []
    limit = args.max_segments

    n_ckpts = len(ckpts)
    for ci, ckpt in enumerate(ckpts):
        sd = _strip_module_prefix(ckpt["model"] if isinstance(ckpt, dict) else ckpt)
        model.load_state_dict(sd, strict=False)
        model.eval()
        print(f"[ensemble] forward ckpt {ci + 1}/{n_ckpts}")

        idx_in_run = 0
        done = 0
        for batch in tqdm(loader, desc=f"ckpt{ci + 1}/{n_ckpts}"):
            waveform = batch["waveform"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            context_labels = batch["context_labels"].to(device, non_blocking=True)
            seg_ids = batch["segment_id"]

            probs = _forward_probs(
                model,
                waveform=waveform,
                input_ids=input_ids,
                attention_mask=attention_mask,
                context_labels=context_labels,
                use_amp=use_amp,
                tta=args.tta,
            )
            if probs.shape[1] != len(label_cols):
                raise RuntimeError(
                    f"prob dim {probs.shape[1]} != label cols {len(label_cols)}"
                )

            for i, seg_id in enumerate(seg_ids):
                if ci == 0:
                    segment_ids_out.append(seg_id)
                    probs_avg.append(probs[i].copy())
                else:
                    probs_avg[idx_in_run] = probs_avg[idx_in_run] + probs[i]
                idx_in_run += 1
                done += 1
                if limit is not None and done >= limit:
                    break
            if limit is not None and done >= limit:
                break

    probs_arr = np.stack(probs_avg, axis=0) / float(n_ckpts)  # [N, C]
    if args.save_probs:
        np.savez(
            args.save_probs,
            segment_id=np.array(segment_ids_out, dtype=object),
            probs=probs_arr,
            label_cols=np.array(label_cols, dtype=object),
        )
        print(f"[ensemble] saved probs -> {args.save_probs}")

    rows: List[dict] = []
    for seg_id, p in zip(segment_ids_out, probs_arr.tolist()):
        preds = []
        for j, name in enumerate(label_cols):
            thr = (
                float(per_label_thresholds[name])
                if per_label_thresholds and name in per_label_thresholds
                else args.threshold
            )
            preds.append(int(float(p[j]) >= thr))
        if args.constraints == "v1":
            preds = apply_label_constraints(preds, p, label_cols)
        elif args.constraints == "v2":
            preds = apply_label_constraints_v2(
                preds,
                p,
                label_cols,
                na_strong_thr=args.na_strong_thr,
                na_max_other_prob=args.na_max_other_prob,
                ct_margin=args.ct_margin,
                allzero_fallback=args.allzero_fallback,
            )
        row = {"segment_id": seg_id}
        for j, col in enumerate(label_cols):
            row[col] = preds[j]
        rows.append(row)

    rows = sorted(rows, key=lambda r: r["segment_id"])
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[ensemble] wrote {len(rows)} rows -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
