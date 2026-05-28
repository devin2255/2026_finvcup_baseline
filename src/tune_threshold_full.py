"""
全量验证集 per-label 阈值调优（Day 1 提分）。

相比已有 src/tune_threshold.py 的关键差异：
- 默认跑全量 valid（不限 max_batches），并支持以会话为粒度的子采样以加速。
- 支持与 src/infer_ensemble.py 同样的 TTA（dual_channel）和多 ckpt 平均，保证
  调阈值的口径与测试推理完全一致；否则线上线下口径不一致会浪费阈值收益。
- 单标签上做更细的阈值扫描（默认 600 步，0.01~0.99）。
- 输出与 best_thresholds.json 格式兼容，可直接喂给 infer_ensemble.py。

典型用法：
  python -m src.tune_threshold_full \
      --config configs/whisper_qwen0_6b_lmf_8g.yaml \
      --checkpoints outputs/lmf_8g/checkpoints/best_lmf.pt \
      --tta dual_channel \
      --output outputs/lmf_8g/logs/best_thresholds_full.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.data import (
    TurnTakingTrainDataset,
    build_collate_fn,
    build_train_samples_multitask,
    list_conv_ids,
    split_conversation_ids,
)
from src.infer_ensemble import _forward_probs, _strip_module_prefix
from src.models import MultimodalTurnTakingModel
from src.utils import (
    _find_best_micro_f1_thresholds,
    _micro_f1_at_thresholds,
    _sample_f1,
    load_config,
    set_env_paths,
)


def parse_args():
    p = argparse.ArgumentParser(description="全量 valid 调阈值（带 TTA / 多 ckpt 集成）")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoints", type=str, nargs="+", required=True)
    p.add_argument(
        "--output",
        type=str,
        default="outputs/thresholds_full.json",
        help="兼容 src/infer_ensemble.py --threshold_file 的 JSON",
    )
    p.add_argument(
        "--max_valid_samples",
        type=int,
        default=None,
        help="不指定 → 跑全量；指定后做随机子采样（>= 2w 较稳）",
    )
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument(
        "--tta",
        type=str,
        default="dual_channel",
        choices=["none", "dual_channel"],
    )
    p.add_argument("--n_steps", type=int, default=600)
    p.add_argument(
        "--metric",
        type=str,
        default="macro",
        choices=["macro", "micro"],
        help="macro=每标签独立最优阈值；micro=联合搜索使 micro F1 最大（与线上口径更近）",
    )
    p.add_argument(
        "--micro_rounds",
        type=int,
        default=2,
        help="micro 模式下的坐标轮转轮数",
    )
    p.add_argument("--seed", type=int, default=None, help="子采样种子；缺省用 cfg.seed")
    p.add_argument(
        "--save_probs",
        type=str,
        default=None,
        help="额外保存 valid 概率与标签到 .npz，便于离线复盘",
    )
    return p.parse_args()


def _sample_valid_subset(samples: list, max_n: int | None, seed: int) -> list:
    if max_n is None or max_n >= len(samples):
        return list(samples)
    rng = random.Random(seed + 1009)
    idxs = sorted(rng.sample(range(len(samples)), max_n))
    return [samples[i] for i in idxs]


def _find_best_f1_threshold(
    probs: np.ndarray, labels: np.ndarray, n_steps: int
) -> tuple[float, float]:
    if labels.sum() == 0:
        return 0.5, 0.0
    best_thr, best_f1 = 0.5, 0.0
    thrs = np.linspace(0.01, 0.99, n_steps)
    # 向量化扫描，避免 Python 循环 600 次
    p = probs[None, :]  # [1, N]
    y = labels[None, :]  # [1, N]
    t = thrs[:, None]  # [S, 1]
    preds = (p >= t).astype(np.int8)
    tp = (preds * y).sum(axis=1).astype(np.float64)
    fp = (preds * (1 - y)).sum(axis=1).astype(np.float64)
    fn = ((1 - preds) * y).sum(axis=1).astype(np.float64)
    precision = tp / np.maximum(1.0, tp + fp)
    recall = tp / np.maximum(1.0, tp + fn)
    f1 = 2 * precision * recall / np.maximum(1e-8, precision + recall)
    j = int(f1.argmax())
    return float(thrs[j]), float(f1[j])


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_env_paths(cfg)
    seed = int(args.seed if args.seed is not None else cfg["seed"])

    paths = cfg["paths"]
    labels_dir = Path(paths["train_labels_dir"])
    train_audio_dir = Path(paths["train_audio_dir"])
    train_text_dir = Path(paths["train_text_dir"])

    conv_ids = list_conv_ids(labels_dir)
    split_ids = split_conversation_ids(
        conv_ids=conv_ids,
        valid_ratio=float(cfg["split"]["valid_ratio"]),
        seed=int(cfg["seed"]),
    )
    valid_ids = split_ids["valid"]
    multi_targets = list(cfg["labels"]["multi_targets"])
    label_names = [x.lower() for x in multi_targets]

    valid_samples = build_train_samples_multitask(
        labels_dir=labels_dir,
        conv_ids=valid_ids,
        context_chunks=int(cfg["context_chunks"]),
        target_chunks=int(cfg["target_chunks"]),
        stride=int(cfg["stride"]),
        label_ids=cfg["labels"],
        target_labels=multi_targets,
        max_samples=cfg.get("max_valid_samples"),
    )
    valid_samples = _sample_valid_subset(valid_samples, args.max_valid_samples, seed)
    print(
        f"[tune] valid_ids={len(valid_ids)} samples_total={len(valid_samples)} "
        f"(after subsample: {len(valid_samples)})"
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["text_encoder"]["model_name"], use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    collate_fn = build_collate_fn(tokenizer, int(cfg["text_encoder"]["max_length"]))

    valid_dataset = TurnTakingTrainDataset(
        samples=valid_samples,
        train_audio_dir=train_audio_dir,
        train_text_dir=train_text_dir,
        train_labels_dir=labels_dir,
        context_chunks=int(cfg["context_chunks"]),
        target_chunks=int(cfg["target_chunks"]),
        chunk_ms=int(cfg["chunk_ms"]),
        sample_rate=int(cfg["sample_rate"]),
        augment_audio=False,
    )
    bs = int(args.batch_size or cfg["train"]["eval_batch_size"])
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["train"].get("use_amp", False))

    model = MultimodalTurnTakingModel(cfg).to(device)
    model.eval()

    n_ckpts = len(args.checkpoints)
    n_samples = len(valid_samples)
    n_labels = len(label_names)
    probs_sum = np.zeros((n_samples, n_labels), dtype=np.float64)
    labels_all = np.zeros((n_samples, n_labels), dtype=np.int8)
    labels_filled = False

    for ci, ckpt_path in enumerate(args.checkpoints):
        print(f"[tune] load checkpoint {ci + 1}/{n_ckpts}: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = _strip_module_prefix(ckpt["model"] if isinstance(ckpt, dict) else ckpt)
        model.load_state_dict(sd, strict=False)
        model.eval()

        write_pos = 0
        for batch in tqdm(valid_loader, desc=f"infer valid ckpt{ci + 1}/{n_ckpts}"):
            waveform = batch["waveform"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            context_labels = batch["context_labels"].to(device, non_blocking=True)
            labels = batch["label"].numpy().astype(np.int8)

            probs = _forward_probs(
                model,
                waveform=waveform,
                input_ids=input_ids,
                attention_mask=attention_mask,
                context_labels=context_labels,
                use_amp=use_amp,
                tta=args.tta,
            )
            b = probs.shape[0]
            probs_sum[write_pos : write_pos + b] += probs
            if not labels_filled:
                labels_all[write_pos : write_pos + b] = labels
            write_pos += b

        labels_filled = True
        assert write_pos == n_samples, (write_pos, n_samples)

    probs_avg = (probs_sum / float(n_ckpts)).astype(np.float32)
    labels_all = labels_all.astype(np.int8)

    if args.save_probs:
        np.savez(
            args.save_probs,
            probs=probs_avg,
            labels=labels_all,
            label_cols=np.array(label_names, dtype=object),
        )
        print(f"[tune] saved valid probs -> {args.save_probs}")

    thresholds: dict[str, float] = {}
    per_label_metrics: dict[str, dict] = {}

    # Per-label best F1 (always compute, used as starting point and reported)
    per_label_thr = np.zeros(len(label_names), dtype=np.float64)
    for j, name in enumerate(label_names):
        y = labels_all[:, j].astype(np.int8)
        p = probs_avg[:, j].astype(np.float32)
        thr, f1 = _find_best_f1_threshold(p, y, n_steps=args.n_steps)
        preds = (p >= thr).astype(np.int8)
        tp = float((preds * y).sum())
        fp = float((preds * (1 - y)).sum())
        fn = float(((1 - preds) * y).sum())
        precision = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        per_label_metrics[name] = {
            "threshold": thr,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "pos_rate": float(y.mean()),
            "n_pos": int(y.sum()),
            "n_total": int(y.shape[0]),
        }
        per_label_thr[j] = thr

    if args.metric == "macro":
        chosen_thr = per_label_thr
        mode_tag = "macro (per-label optimal)"
    else:
        chosen_thr, _ = _find_best_micro_f1_thresholds(
            probs_avg.astype(np.float32),
            labels_all.astype(np.int8),
            rounds=int(args.micro_rounds),
            n_steps=min(int(args.n_steps), 200),
        )
        mode_tag = f"micro (rounds={args.micro_rounds})"

    # Report with chosen thresholds
    print(f"\n[tune] threshold metric mode: {mode_tag}")
    print(f"{'Label':<6}{'PosRate':>10}{'PL-Thr':>10}{'PL-F1':>10}{'Chosen':>10}{'CSel-F1':>10}{'P':>8}{'R':>8}")
    print("-" * 76)
    for j, name in enumerate(label_names):
        y = labels_all[:, j].astype(np.int8)
        p = probs_avg[:, j].astype(np.float32)
        ct = chosen_thr[j]
        preds_c = (p >= ct).astype(np.int8)
        tp = float((preds_c * y).sum())
        fp = float((preds_c * (1 - y)).sum())
        fn = float(((1 - preds_c) * y).sum())
        prec = tp / max(1.0, tp + fp)
        rec = tp / max(1.0, tp + fn)
        f1_c = 2 * prec * rec / max(1e-8, prec + rec)
        thresholds[name] = float(ct)
        per_label_metrics[name]["chosen_threshold"] = float(ct)
        per_label_metrics[name]["chosen_f1"] = float(f1_c)
        per_label_metrics[name]["chosen_precision"] = float(prec)
        per_label_metrics[name]["chosen_recall"] = float(rec)
        print(
            f"{name:<6}{float(y.mean()):>10.4f}{per_label_thr[j]:>10.4f}"
            f"{per_label_metrics[name]['f1']:>10.4f}{ct:>10.4f}{f1_c:>10.4f}{prec:>8.3f}{rec:>8.3f}"
        )

    macro_f1 = float(np.mean([m["f1"] for m in per_label_metrics.values()]))
    macro_f1_chosen = float(np.mean([m["chosen_f1"] for m in per_label_metrics.values()]))
    chosen_thr_arr = np.array(list(thresholds.values()), dtype=np.float64)
    micro_f1 = _micro_f1_at_thresholds(
        probs_avg.astype(np.float32),
        labels_all.astype(np.int8),
        chosen_thr_arr,
    )
    preds_for_sample = (probs_avg >= chosen_thr_arr[None, :]).astype(np.int8)
    sample_f1 = _sample_f1(preds_for_sample, labels_all.astype(np.int8))
    print(
        f"\n[tune] macro_best_f1 (per-label thr) = {macro_f1:.4f}"
        f"   macro_f1 (chosen thr) = {macro_f1_chosen:.4f}"
    )
    print(
        f"[tune] micro_f1 (chosen thr) = {micro_f1:.4f}"
        f"   sample_f1 (chosen thr) = {sample_f1:.4f}   (n_samples={n_samples})"
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "thresholds": thresholds,
                "per_label": per_label_metrics,
                "macro_best_f1": macro_f1,
                "macro_f1_chosen": macro_f1_chosen,
                "micro_f1_chosen": float(micro_f1),
                "sample_f1_chosen": float(sample_f1),
                "metric": args.metric,
                "n_samples": n_samples,
                "n_checkpoints": n_ckpts,
                "tta": args.tta,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[tune] wrote thresholds -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
