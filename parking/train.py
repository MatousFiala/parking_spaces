"""Stage 3: training. Same script for local CPU smoke tests and CUDA runs.

    python -m parking.train --config configs/default.yaml
    python -m parking.train --train.limit_batches 5 --train.epochs 2   # smoke test
    python -m parking.train --resume                    # continue latest interrupted run
    python -m parking.train --resume data/runs/20260708_102440
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex

from parking.config import Config, dump_config, load_config
from parking.dataset import IGNORE, ChipDataset
from parking.model import build_loss, build_model

log = logging.getLogger(__name__)


def make_loaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    t = cfg.train
    train_ds, val_ds = ChipDataset(cfg, "train"), ChipDataset(cfg, "val")
    log.info("chips: %d train / %d val", len(train_ds), len(val_ds))
    kw = dict(batch_size=t.batch_size, num_workers=t.num_workers, pin_memory=True,
              persistent_workers=t.num_workers > 0)
    return (
        DataLoader(train_ds, shuffle=True, drop_last=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
    )


def make_optimizer(cfg: Config, model) -> torch.optim.Optimizer:
    t = cfg.train
    encoder_params = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    rest = [p for p in model.parameters() if id(p) not in encoder_ids]
    return torch.optim.AdamW(
        [{"params": rest, "lr": t.lr}, {"params": encoder_params, "lr": t.encoder_lr}],
        weight_decay=t.weight_decay,
    )


def lr_lambda(cfg: Config):
    t = cfg.train

    def fn(epoch: int) -> float:
        if epoch < t.warmup_epochs:
            return (epoch + 1) / t.warmup_epochs
        p = (epoch - t.warmup_epochs) / max(1, t.epochs - t.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * p))

    return fn


@torch.inference_mode()
def evaluate(model, loader, loss_fn, device, amp: bool, limit: int = 0) -> dict:
    model.eval()
    iou = BinaryJaccardIndex(ignore_index=IGNORE).to(device)
    f1 = BinaryF1Score(ignore_index=IGNORE).to(device)
    total, n = 0.0, 0
    for b, (img, target) in enumerate(loader):
        if limit and b >= limit:
            break
        img, target = img.to(device, non_blocking=True), target.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(img)
            total += loss_fn(logits, target).item()
        n += 1
        pred = logits.squeeze(1).sigmoid()
        iou.update(pred, target)
        f1.update(pred, target)
    return {"loss": total / max(1, n), "iou": iou.compute().item(), "f1": f1.compute().item()}


def find_latest_run(cfg: Config) -> Path:
    candidates = sorted(
        d for d in Path(cfg.paths.runs_dir).glob("*") if (d / "last.pt").exists()
    )
    if not candidates:
        raise SystemExit(f"no resumable run (dir with last.pt) under {cfg.paths.runs_dir}")
    return candidates[-1]


def train(cfg: Config, resume_dir: Path | None = None) -> Path:
    t = cfg.train
    torch.manual_seed(t.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = t.amp and device.type == "cuda"
    log.info("device=%s amp=%s", device, amp)

    run_dir = resume_dir or Path(cfg.paths.runs_dir) / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    if resume_dir is None:
        dump_config(cfg, run_dir / "config.yaml")

    train_loader, val_loader = make_loaders(cfg)
    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg)
    opt = make_optimizer(cfg, model)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg))
    scaler = torch.amp.GradScaler(enabled=amp)

    best_iou, best_epoch, start_epoch = -1.0, -1, 0
    if resume_dir is not None:
        ckpt = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
            sched.load_state_dict(ckpt["scheduler"])
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_iou = ckpt.get("best_iou", ckpt.get("val_iou", -1.0))
        best_epoch = ckpt.get("best_epoch", ckpt["epoch"])
        log.info("resuming %s at epoch %d (best iou %.4f)", run_dir, start_epoch + 1, best_iou)

    log_path = run_dir / "log.csv"
    if resume_dir is None or not log_path.exists():
        with open(log_path, "w", newline="") as fh:
            csv.writer(fh).writerow(["epoch", "train_loss", "val_loss", "val_iou", "val_f1", "lr", "sec"])

    for epoch in range(start_epoch, t.epochs):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for b, (img, target) in enumerate(train_loader):
            if t.limit_batches and b >= t.limit_batches:
                break
            img, target = img.to(device, non_blocking=True), target.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = loss_fn(model(img), target)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
            n += 1
        sched.step()

        metrics = evaluate(model, val_loader, loss_fn, device, amp, limit=t.limit_batches)
        train_loss = total / max(1, n)
        lr_now = opt.param_groups[0]["lr"]
        log.info(
            "epoch %d/%d train %.4f | val %.4f iou %.4f f1 %.4f | %.0fs",
            epoch + 1, t.epochs, train_loss, metrics["loss"], metrics["iou"], metrics["f1"],
            time.time() - t0,
        )
        with open(log_path, "a", newline="") as fh:
            csv.writer(fh).writerow(
                [epoch, f"{train_loss:.5f}", f"{metrics['loss']:.5f}",
                 f"{metrics['iou']:.5f}", f"{metrics['f1']:.5f}", f"{lr_now:.2e}",
                 f"{time.time() - t0:.0f}"]
            )

        if metrics["iou"] > best_iou:
            best_iou, best_epoch = metrics["iou"], epoch
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_iou": metrics["iou"]},
                run_dir / "best.pt",
            )
        torch.save(
            {
                "model": model.state_dict(), "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "val_iou": metrics["iou"],
                "best_iou": best_iou, "best_epoch": best_epoch,
            },
            run_dir / "last.pt",
        )
        if epoch - best_epoch >= t.early_stop_patience:
            log.info("early stop at epoch %d (best iou %.4f @ %d)", epoch + 1, best_iou, best_epoch + 1)
            break

    log.info("done: best val IoU %.4f -> %s", best_iou, run_dir / "best.pt")
    return run_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg, ns = load_config()
    sub = argparse.ArgumentParser(prog="parking.train")
    sub.add_argument("--resume", nargs="?", const="auto",
                     help="continue an interrupted run (optionally pass the run dir)")
    args = sub.parse_args(ns.rest)
    resume_dir = None
    if args.resume:
        resume_dir = find_latest_run(cfg) if args.resume == "auto" else Path(args.resume)
        if not (resume_dir / "last.pt").exists():
            raise SystemExit(f"{resume_dir} has no last.pt")
    train(cfg, resume_dir)


if __name__ == "__main__":
    main()
