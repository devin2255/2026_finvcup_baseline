"""
测试集推理（多标签 event-level）：仅输出预测结果 CSV。

输出 CSV 格式：
- segment_id
- <label_1>, <label_2>, ...（与 config 的 labels.multi_targets 顺序一致，列名为小写，值为 0/1）

不依赖 target_name（即不需要测试真值）。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.data import TurnTakingTestDataset, build_collate_fn
from src.models import MultimodalTurnTakingModel
from src.utils import load_config, set_env_paths


def parse_args():
    p = argparse.ArgumentParser(description="test 推理：仅导出 pred.csv（多标签 event-level）")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--test_root", type=str, required=True, help="测试数据根目录")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--threshold_file", type=str, default=None, help="per-label thresholds JSON (overrides --threshold)")
    p.add_argument("--batch_size", type=int, default=None, help="默认取 config train.eval_batch_size")
    p.add_argument("--max_segments", type=int, default=None, help="仅处理前 N 条（冒烟测试）")
    p.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="输出 pred.csv 路径",
    )
    return p.parse_args()


def _resolve_thresholds(threshold_file: str | None, checkpoint: dict) -> dict | None:
    if threshold_file:
        with open(threshold_file, "r", encoding="utf-8") as f:
            thr_data = json.load(f)
        return thr_data["thresholds"]
    thresholds = checkpoint.get("thresholds") if isinstance(checkpoint, dict) else None
    return thresholds if isinstance(thresholds, dict) else None


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_env_paths(cfg)

    # Baseline 固化为 event-level 多标签（未来 2s 内各标签是否出现）
    use_multi_label = True

    multi_targets = list(cfg["labels"]["multi_targets"])
    label_cols = [x.lower() for x in multi_targets]

    test_root = Path(args.test_root)
    ds = TurnTakingTestDataset( test_root=test_root, sample_rate=int(cfg["sample_rate"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg["text_encoder"]["model_name"], use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    collate_fn = build_collate_fn(tokenizer, int(cfg["text_encoder"]["max_length"]))

    bs = int(args.batch_size or cfg["train"]["eval_batch_size"])
    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model = MultimodalTurnTakingModel(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    per_label_thresholds = _resolve_thresholds(args.threshold_file, ckpt)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    use_amp = bool(cfg["train"].get("use_amp", False))

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["segment_id"] + label_cols

    limit = args.max_segments
    done = 0
    rows: list[dict] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"infer"):
            waveform = batch["waveform"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            context_labels = batch["context_labels"].to(device, non_blocking=True)
            segment_ids = batch["segment_id"]

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(
                    waveform=waveform,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    context_labels=context_labels,
                )
            probs = torch.sigmoid(logits).cpu().numpy()
            if probs.ndim == 1:
                probs = probs.reshape(-1, 1)

            for i, seg_id in enumerate(segment_ids):
                p = probs[i].tolist()
                if len(p) != len(multi_targets):
                    raise RuntimeError(f"logits dim {len(p)} != len(multi_targets) {len(multi_targets)}")
                preds_binary = []
                for j, name in enumerate(label_cols):
                    thr = float(per_label_thresholds[name]) if per_label_thresholds else args.threshold
                    preds_binary.append(int(float(p[j]) >= thr))
                row = {"segment_id": seg_id}
                for j, col in enumerate(label_cols):
                    row[col] = preds_binary[j]
                rows.append(row)
                done += 1
                if limit is not None and done >= limit:
                    break
            if limit is not None and done >= limit:
                break

    rows = sorted(rows, key=lambda r: r["segment_id"])
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {out_path.resolve()}")


if __name__ == "__main__":
    main()

