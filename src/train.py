import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from src.data import (
    TurnTakingTrainDataset,
    build_collate_fn,
    build_train_samples_multitask,
    list_conv_ids,
    split_conversation_ids,
)
from src.models import MultimodalTurnTakingModel
from src.utils import (
    cleanup_distributed,
    compute_multilabel_metrics,
    ensure_dirs,
    is_distributed,
    load_config,
    save_json,
    set_env_paths,
    set_seed,
    setup_distributed,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/whisper_qwen0_6b_constrained_event_formal_5labels_competition.yaml")
    p.add_argument("--resume", type=str, default=None)
    # 仅用于冒烟/快速迭代：覆盖 config 中的训练参数（不写回文件）
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max_steps_per_epoch", type=int, default=None)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_valid_samples", type=int, default=None)
    return p.parse_args()


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    device,
    use_amp: bool,
    label_names: list[str] | None = None,
    max_batches: int | None = None,
):
    model.eval()
    all_labels, all_probs = [], []
    for bi, batch in enumerate(tqdm(data_loader, desc="eval", leave=False)):
        if max_batches is not None and bi >= max_batches:
            break
        waveform = batch["waveform"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        context_labels = batch["context_labels"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(
                waveform=waveform,
                input_ids=input_ids,
                attention_mask=attention_mask,
                context_labels=context_labels,
            )

        probs = torch.sigmoid(logits)
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_probs.extend(probs.detach().cpu().numpy().tolist())

    labels_np = (
        torch.as_tensor(all_labels).numpy() if len(all_labels) > 0 else np.array([])
    )
    if labels_np.ndim == 2:
        return compute_multilabel_metrics(all_labels, all_probs, label_names=label_names)
    raise RuntimeError("Baseline 固化为多标签训练，evaluate 不应进入二分类分支。")


def _format_multilabel_metrics_line(metrics: dict, label_names: list[str]) -> str:
    parts = []
    for n in label_names:
        parts.append(
            f"{n}:acc={metrics.get(f'{n}_accuracy', 0.0):.4f},"
            f"f1={metrics.get(f'{n}_f1', 0.0):.4f},"
            f"bf1={metrics.get(f'{n}_best_f1', 0.0):.4f},"
            f"auc={metrics.get(f'{n}_roc_auc', 0.5):.4f}"
        )
    return " | ".join(parts)


def _get_valid_loader_shuffle(cfg: dict) -> bool:
    return bool(cfg.get("train", {}).get("eval_valid_shuffle", False))


def _get_best_checkpoint_name(cfg: dict) -> str:
    return str(cfg.get("train", {}).get("best_checkpoint_name", "best.pt"))


def _select_eval_samples(samples: list, cfg: dict, seed: int) -> list:
    sample_count = cfg.get("train", {}).get("eval_valid_sample_count", None)
    if sample_count is None:
        return list(samples)
    sample_count = int(sample_count)
    if sample_count <= 0 or sample_count >= len(samples):
        return list(samples)
    indices = sorted(random.Random(seed + 1009).sample(range(len(samples)), sample_count))
    return [samples[i] for i in indices]


def _extract_best_thresholds(metrics: dict, label_names: list[str]) -> dict[str, float]:
    return {name: float(metrics.get(f"{name}_best_threshold", 0.5)) for name in label_names}


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_env_paths(cfg)
    ensure_dirs(cfg)
    set_seed(int(cfg["seed"]))

    # CLI overrides (for smoke runs)
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)
    if args.max_steps_per_epoch is not None:
        cfg["train"]["max_steps_per_epoch"] = int(args.max_steps_per_epoch)
    if args.max_train_samples is not None:
        cfg["max_train_samples"] = int(args.max_train_samples)
    if args.max_valid_samples is not None:
        cfg["max_valid_samples"] = int(args.max_valid_samples)

    local_rank, world_size, rank = setup_distributed()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    writer = None

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
    train_ids, valid_ids = split_ids["train"], split_ids["valid"]
    # Baseline 固化：只做 event-level 多标签（未来 2s 内 5 个标签分别是否出现）
    use_multi_label = True
    multi_targets = list(cfg.get("labels", {}).get("multi_targets", ["C", "NA", "I", "BC", "T"]))
    metric_label_names = [x.lower() for x in multi_targets]

    train_samples = build_train_samples_multitask(
        labels_dir=labels_dir,
        conv_ids=train_ids,
        context_chunks=int(cfg["context_chunks"]),
        target_chunks=int(cfg["target_chunks"]),
        stride=int(cfg["stride"]),
        label_ids=cfg["labels"],
        target_labels=multi_targets,
        max_samples=cfg["max_train_samples"],
    )
    valid_samples = build_train_samples_multitask(
        labels_dir=labels_dir,
        conv_ids=valid_ids,
        context_chunks=int(cfg["context_chunks"]),
        target_chunks=int(cfg["target_chunks"]),
        stride=int(cfg["stride"]),
        label_ids=cfg["labels"],
        target_labels=multi_targets,
        max_samples=cfg["max_valid_samples"],
    )
    valid_eval_samples = _select_eval_samples(valid_samples, cfg, seed=int(cfg["seed"]))

    if is_main:
        save_json(Path(paths["logs_dir"]) / "split_ids.json", split_ids)
        save_json(
            Path(paths["logs_dir"]) / "sample_count.json",
            {
                "train_samples": len(train_samples),
                "valid_samples": len(valid_samples),
                "valid_eval_samples": len(valid_eval_samples),
            },
        )
        writer = SummaryWriter(log_dir=str(Path(paths["logs_dir"]) / "tb"))

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["text_encoder"]["model_name"], use_fast=True
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    collate_fn = build_collate_fn(tokenizer, int(cfg["text_encoder"]["max_length"]))

    train_dataset = TurnTakingTrainDataset(
        samples=train_samples,
        train_audio_dir=train_audio_dir,
        train_text_dir=train_text_dir,
        train_labels_dir=labels_dir,
        context_chunks=int(cfg["context_chunks"]),
        target_chunks=int(cfg["target_chunks"]),
        chunk_ms=int(cfg["chunk_ms"]),
        sample_rate=int(cfg["sample_rate"]),
        augment_audio=True,
    )
    valid_dataset = TurnTakingTrainDataset(
        samples=valid_eval_samples,
        train_audio_dir=train_audio_dir,
        train_text_dir=train_text_dir,
        train_labels_dir=labels_dir,
        context_chunks=int(cfg["context_chunks"]),
        target_chunks=int(cfg["target_chunks"]),
        chunk_ms=int(cfg["chunk_ms"]),
        sample_rate=int(cfg["sample_rate"]),
        augment_audio=False,
    )

    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True) if is_distributed() else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=int(cfg["num_workers"]),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(cfg["train"]["eval_batch_size"]),
        shuffle=_get_valid_loader_shuffle(cfg),
        num_workers=int(cfg["num_workers"]),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 训练阶段完全不考虑测试集：不加载、不评估
    eval_test_every_steps = 0
    eval_valid_max_batches = cfg["train"].get("eval_valid_max_batches", None)
    eval_valid_max_batches = (
        int(eval_valid_max_batches) if eval_valid_max_batches is not None else None
    )
    test_loader: DataLoader | None = None
    gt_test_labels = None

    model = MultimodalTurnTakingModel(cfg).to(device)
    if is_distributed():
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    if cfg["train"].get("pos_weight_mode", "per_label") in ("per_label", "capped_per_label"):
        y_mat = np.asarray([s.label_vec for s in train_samples], dtype=np.float32)  # [N,5]
        pos = y_mat.sum(axis=0)
        neg = y_mat.shape[0] - pos
        pw = neg / np.maximum(1.0, pos)
        if cfg["train"].get("pos_weight_mode") == "capped_per_label":
            cap = float(cfg["train"].get("pos_weight_cap", 5.0))
            pw = np.minimum(pw, cap)
        pos_weight = torch.tensor(pw, device=device, dtype=torch.float32)
    else:
        pos_weight = torch.ones(len(multi_targets), device=device, dtype=torch.float32)

    focal_gamma = float(cfg["train"].get("focal_gamma", 0.0))
    if focal_gamma > 0:
        # Focal Loss: down-weight easy examples, forcing the model to focus on
        # hard/rare labels like BC (3.65%) and I (14.15%).
        class MultiLabelFocalLoss(torch.nn.Module):
            def __init__(self, gamma, pos_weight):
                super().__init__()
                self.gamma = gamma
                self.register_buffer("pos_weight", pos_weight)

            def forward(self, logits, targets):
                bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, targets, reduction="none", pos_weight=self.pos_weight,
                )
                probs = torch.sigmoid(logits)
                p_t = targets * probs + (1 - targets) * (1 - probs)
                return ((1 - p_t) ** self.gamma * bce).mean()

        criterion = MultiLabelFocalLoss(focal_gamma, pos_weight)
    else:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    max_steps_per_epoch_cfg = cfg["train"].get("max_steps_per_epoch", None)
    max_steps_per_epoch = (
        int(max_steps_per_epoch_cfg) if max_steps_per_epoch_cfg is not None else None
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    max_epochs = int(cfg["train"]["epochs"])
    accum_steps = int(cfg["train"]["gradient_accumulation_steps"])
    steps_per_epoch_for_sched = (
        min(len(train_loader), max_steps_per_epoch)
        if max_steps_per_epoch is not None
        else len(train_loader)
    )
    total_update_steps = max(
        1, (steps_per_epoch_for_sched * max_epochs) // max(1, accum_steps)
    )
    warmup_ratio = float(cfg["train"].get("warmup_ratio", 0.03))
    warmup_steps = int(total_update_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["train"]["use_amp"]))

    start_epoch = 0
    best_metric = -math.inf
    best_path = Path(paths["checkpoints_dir"]) / _get_best_checkpoint_name(cfg)
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        target_model = model.module if hasattr(model, "module") else model
        target_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_metric = float(ckpt.get("best_metric", -math.inf))

    grad_clip = float(cfg["train"]["grad_clip_norm"])
    use_amp = bool(cfg["train"]["use_amp"])
    save_metric = str(cfg["train"]["save_metric"])
    early_stop_patience = int(cfg["train"]["early_stop_patience"])
    bad_epochs = 0
    # 真实训练步数（不受 len(train_loader) 误导）；用于 TensorBoard / 打印
    global_train_step = 0

    for epoch in range(start_epoch, max_epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)

        iterator = train_loader
        if is_main:
            iterator = tqdm(train_loader, desc=f"train epoch {epoch}", leave=False)

        epoch_loss_sum = 0.0
        epoch_step_count = 0
        log_every = int(cfg["train"].get("log_every_steps", 20))
        ema_decay = float(cfg["train"].get("ema_decay", 0.98))
        loss_ema = None
        update_step = 0
        last_metrics_valid = None
        last_metrics_test = None
        for step, batch in enumerate(iterator):
            if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                if is_main:
                    print(
                        f"[Epoch {epoch}] reach max_steps_per_epoch={max_steps_per_epoch}, "
                        "run eval and continue next epoch."
                    )
                break
            waveform = batch["waveform"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            context_labels = batch["context_labels"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(
                    waveform=waveform,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    context_labels=context_labels,
                )
                loss = criterion(logits, labels) / accum_steps

            if not torch.isfinite(loss):
                if is_main:
                    print(
                        f"[WARN] non-finite loss at epoch={epoch} step={step}, "
                        "skip this batch."
                    )
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            loss_value = float(loss.item() * accum_steps)
            epoch_loss_sum += loss_value
            epoch_step_count += 1
            global_train_step += 1
            global_step = global_train_step
            loss_ema = loss_value if loss_ema is None else (ema_decay * loss_ema + (1.0 - ema_decay) * loss_value)

            with torch.no_grad():
                probs = torch.sigmoid(logits.detach())
                batch_pos_rate = float(labels.float().mean().item())
                prob_mean = float(probs.mean().item())
                prob_std = float(probs.std(unbiased=False).item())
                logit_mean = float(logits.detach().mean().item())
                logit_std = float(logits.detach().std(unbiased=False).item())
                # Per-batch diagnostic: split loss by positive/negative entries.
                per_entry = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits.detach(),
                    labels,
                    reduction="none",
                    pos_weight=pos_weight,
                )
                pos_mask = labels > 0.5
                neg_mask = ~pos_mask
                pos_loss = float(per_entry[pos_mask].mean().item()) if pos_mask.any() else 0.0
                neg_loss = float(per_entry[neg_mask].mean().item()) if neg_mask.any() else 0.0

            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item())
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1
            else:
                grad_norm = float("nan")

            if is_main and (step + 1) % log_every == 0:
                lr_now = float(optimizer.param_groups[0]["lr"])
                if writer is not None:
                    writer.add_scalar("train/loss_step", loss_value, global_step)
                    writer.add_scalar("train/loss_ema", float(loss_ema), global_step)
                    writer.add_scalar("train/lr", lr_now, global_step)
                    writer.add_scalar("train/grad_norm", grad_norm, global_step)
                    writer.add_scalar("train/logit_mean", logit_mean, global_step)
                    writer.add_scalar("train/logit_std", logit_std, global_step)
                    writer.add_scalar("train/prob_mean", prob_mean, global_step)
                    writer.add_scalar("train/prob_std", prob_std, global_step)
                    writer.add_scalar("train/batch_pos_rate", batch_pos_rate, global_step)
                    writer.add_scalar("train/pos_loss", pos_loss, global_step)
                    writer.add_scalar("train/neg_loss", neg_loss, global_step)
                if hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(
                        loss=f"{loss_value:.4f}",
                        ema=f"{float(loss_ema):.4f}",
                        lr=f"{lr_now:.2e}",
                        p=f"{batch_pos_rate:.3f}",
                        pm=f"{prob_mean:.3f}",
                    )
                # nohup/重定向到文件时 tqdm 用 \\r 刷新，日志看起来像没动；补一行纯文本便于 tail
                if not sys.stderr.isatty():
                    print(
                        f"[train] epoch={epoch} step={step} global_step={global_step} "
                        f"loss={loss_value:.4f} ema={float(loss_ema):.4f} lr={lr_now:.2e} "
                        f"pos_rate={batch_pos_rate:.3f} prob_mean={prob_mean:.3f}",
                        flush=True,
                    )

            # 训练过程中不做任何测试集相关评估

        if is_distributed():
            torch.distributed.barrier()

        skip_epoch_end_eval = False

        if is_main:
            eval_model = model.module if hasattr(model, "module") else model
            metrics_valid = evaluate(
                eval_model,
                valid_loader,
                device,
                use_amp=use_amp,
                label_names=metric_label_names if use_multi_label else None,
                max_batches=eval_valid_max_batches,
            )
            metrics_test_epoch = None

            metric_value = float(metrics_valid[save_metric])
            save_json(Path(paths["logs_dir"]) / f"valid_epoch_{epoch}.json", metrics_valid)
            # 兼容旧脚本读取路径
            save_json(Path(paths["logs_dir"]) / f"eval_epoch_{epoch}.json", metrics_valid)
            valid_thresholds = _extract_best_thresholds(metrics_valid, metric_label_names)
            save_json(Path(paths["logs_dir"]) / f"thresholds_epoch_{epoch}.json", {"thresholds": valid_thresholds})
            if use_multi_label:
                valid_per_label = _format_multilabel_metrics_line(metrics_valid, metric_label_names)
                print(
                    f"[Epoch {epoch}] train_loss={epoch_loss_sum / max(1, epoch_step_count):.6f} "
                    f"valid_macro_acc={metrics_valid['macro_accuracy']:.4f} "
                    f"valid_macro_f1={metrics_valid['macro_f1']:.4f} "
                    f"valid_macro_best_f1={metrics_valid['macro_best_f1']:.4f} "
                    f"valid_macro_auc={metrics_valid['macro_roc_auc']:.4f} "
                    f"| valid[{valid_per_label}] "
                    f"best_{save_metric}={max(best_metric, metric_value):.4f}"
                )
            else:
                print(
                    f"[Epoch {epoch}] train_loss={epoch_loss_sum / max(1, epoch_step_count):.6f} "
                    f"valid_acc={metrics_valid['accuracy']:.4f} valid_f1={metrics_valid['f1']:.4f} valid_auc={metrics_valid['roc_auc']:.4f} "
                    f"best_{save_metric}={max(best_metric, metric_value):.4f}"
                )
            if writer is not None:
                avg_train_loss = epoch_loss_sum / max(1, epoch_step_count)
                writer.add_scalar("train/loss_epoch", avg_train_loss, epoch)
                if use_multi_label:
                    writer.add_scalar("valid/macro_accuracy", metrics_valid["macro_accuracy"], epoch)
                    writer.add_scalar("valid/macro_f1", metrics_valid["macro_f1"], epoch)
                    writer.add_scalar("valid/macro_best_f1", metrics_valid["macro_best_f1"], epoch)
                    writer.add_scalar("valid/macro_roc_auc", metrics_valid["macro_roc_auc"], epoch)
                    for n in metric_label_names:
                        writer.add_scalar(f"valid/{n}_accuracy", metrics_valid[f"{n}_accuracy"], epoch)
                        writer.add_scalar(f"valid/{n}_f1", metrics_valid[f"{n}_f1"], epoch)
                        writer.add_scalar(f"valid/{n}_best_f1", metrics_valid[f"{n}_best_f1"], epoch)
                        writer.add_scalar(f"valid/{n}_best_threshold", metrics_valid[f"{n}_best_threshold"], epoch)
                        writer.add_scalar(f"valid/{n}_roc_auc", metrics_valid[f"{n}_roc_auc"], epoch)
                else:
                    writer.add_scalar("valid/accuracy", metrics_valid["accuracy"], epoch)
                    writer.add_scalar("valid/f1", metrics_valid["f1"], epoch)
                    writer.add_scalar("valid/roc_auc", metrics_valid["roc_auc"], epoch)

            if metric_value > best_metric:
                bad_epochs = 0
                best_metric = metric_value
                torch.save(
                    {
                        "epoch": epoch,
                        "best_metric": best_metric,
                        "model": eval_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "config": cfg,
                        "thresholds": valid_thresholds,
                    },
                    best_path,
                )
                save_json(Path(paths["logs_dir"]) / "best_thresholds.json", {"thresholds": valid_thresholds})
            else:
                bad_epochs += 1
                if bad_epochs >= early_stop_patience:
                    break

        if is_distributed():
            torch.distributed.barrier()

    cleanup_distributed()
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
