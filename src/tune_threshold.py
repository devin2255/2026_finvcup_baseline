"""
Per-label threshold tuning: finds the optimal threshold for each label on the
validation set to maximize per-label F1 (or macro-F1).

Usage:
  python -m src.tune_threshold --config configs/whisper_qwen0_6b_8g.yaml \
      --checkpoint outputs/checkpoints/best.pt --output outputs/thresholds.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
from src.models import MultimodalTurnTakingModel
from src.train import _get_valid_loader_shuffle
from src.utils import load_config, set_env_paths


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, default="outputs/thresholds.json")
    p.add_argument("--max_batches", type=int, default=None)
    return p.parse_args()


def find_best_f1_threshold(probs, labels, n_steps=200):
    """Find the threshold that maximizes F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0
    for t in np.linspace(0.01, 0.99, n_steps):
        preds = (probs >= t).astype(int)
        tp = (preds * labels).sum()
        fp = (preds * (1 - labels)).sum()
        fn = ((1 - preds) * labels).sum()
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    return best_threshold, best_f1


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_env_paths(cfg)

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
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(cfg["train"]["eval_batch_size"]),
        shuffle=_get_valid_loader_shuffle(cfg),
        num_workers=int(cfg["num_workers"]),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalTurnTakingModel(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    use_amp = bool(cfg["train"].get("use_amp", False))

    all_labels = []
    all_probs = []
    max_batches = args.max_batches

    with torch.no_grad():
        for bi, batch in enumerate(tqdm(valid_loader, desc="infer valid")):
            if max_batches is not None and bi >= max_batches:
                break
            waveform = batch["waveform"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            context_labels = batch["context_labels"].to(device, non_blocking=True)
            labels = batch["label"]

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(
                    waveform=waveform,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    context_labels=context_labels,
                )
            probs = torch.sigmoid(logits).cpu().numpy()
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.tolist())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    thresholds = {}
    per_label_metrics = {}
    print(f"\n{'Label':<6} {'Best Thr':>10} {'Best F1':>10} {'Pos Rate':>10}")
    print("-" * 42)
    for i, name in enumerate(label_names):
        best_t, best_f1 = find_best_f1_threshold(all_probs[:, i], all_labels[:, i])
        thresholds[name] = best_t
        pos_rate = all_labels[:, i].mean()
        per_label_metrics[name] = {"threshold": best_t, "f1": best_f1, "pos_rate": float(pos_rate)}
        print(f"{name:<6} {best_t:>10.4f} {best_f1:>10.4f} {pos_rate:>10.4f}")

    # Compute macro-F1 with optimal thresholds
    preds = np.zeros_like(all_labels)
    for i, name in enumerate(label_names):
        preds[:, i] = (all_probs[:, i] >= thresholds[name]).astype(int)

    tp = (preds * all_labels).sum(axis=0)
    fp = (preds * (1 - all_labels)).sum(axis=0)
    fn = ((1 - preds) * all_labels).sum(axis=0)
    precisions = tp / np.maximum(1, tp + fp)
    recalls = tp / np.maximum(1, tp + fn)
    f1s = 2 * precisions * recalls / np.maximum(1e-8, precisions + recalls)

    print(f"\n{'Label':<6} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 42)
    for i, name in enumerate(label_names):
        print(f"{name:<6} {precisions[i]:>10.4f} {recalls[i]:>10.4f} {f1s[i]:>10.4f}")
    print(f"\nMacro-F1 (tuned thresholds): {f1s.mean():.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"thresholds": thresholds, "per_label": per_label_metrics}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved thresholds to {out_path.resolve()}")


if __name__ == "__main__":
    main()
