"""Visual QC utilities.

CLI:
    python -m parking.qc overlay --sheet PRAH57 [--zoom-to-labels]
        label polygons drawn over the sheet -> data/qc/<sheet>_overlay.png
        (the go/no-go alignment check before any expensive work)
    python -m parking.qc chips --n 48
        random chip/mask pairs mosaic -> data/qc/chips_mosaic.png
    python -m parking.qc predict --split val --n 12 [--checkpoint ...] [--seed 0]
        model predictions on random chips of a split -> data/qc/predict_<split>.png
        (columns: image | ground truth | prediction confusion; defaults to the
        newest data/runs/*/best.pt)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import shapely
from PIL import Image
from rasterio import plot as rioplot

from parking.acquire import download_sheet, select_sheets, sheet_jpeg_path
from parking.config import Config, load_config
from parking.geo import load_labels

log = logging.getLogger(__name__)


def qc_dir(cfg: Config) -> Path:
    d = Path(cfg.paths.chips_dir).parent / "qc"
    d.mkdir(parents=True, exist_ok=True)
    return d


def overlay(cfg: Config, sheet: str, zoom_to_labels: bool = False, crop: int = 4000) -> Path:
    labels = load_labels(cfg)
    selected = select_sheets(cfg, labels)
    row = selected[selected.sheet == sheet.upper()]
    if row.empty:
        raise SystemExit(f"sheet {sheet} not found among label-intersecting sheets")
    row = row.iloc[0]
    jpg = sheet_jpeg_path(cfg, row.sheet)
    if not jpg.exists():
        jpg = download_sheet(cfg, row)

    with rasterio.open(jpg) as src:
        bounds = src.bounds
        sheet_box = shapely.box(*bounds)
        hits = labels[labels.intersects(sheet_box)]
        if zoom_to_labels and len(hits):
            # crop around the densest labeled spot at full resolution
            c = hits.geometry.union_all().representative_point()
            half = crop // 2
            win = rasterio.windows.from_bounds(
                c.x - half * cfg.chips.gsd, c.y - half * cfg.chips.gsd,
                c.x + half * cfg.chips.gsd, c.y + half * cfg.chips.gsd,
                src.transform,
            )
            img = src.read(window=win)
            transform = rasterio.windows.transform(win, src.transform)
        else:
            factor = 8  # 20000 px -> 2500 px
            img = src.read(out_shape=(3, src.height // factor, src.width // factor))
            transform = src.transform * src.transform.scale(factor, factor)

    fig, ax = plt.subplots(figsize=(16, 13), dpi=150)
    rioplot.show(img, transform=transform, ax=ax)
    if len(hits):
        hits.boundary.plot(ax=ax, color="red", linewidth=0.9 if zoom_to_labels else 0.6)
    # pin axes to the rendered raster; label layer must not autoscale the view
    left = transform.c
    top = transform.f
    ax.set_xlim(left, left + transform.a * img.shape[2])
    ax.set_ylim(top + transform.e * img.shape[1], top)
    ax.set_title(f"{row.sheet} {row['name']} — {len(hits)} label polygons")
    out = qc_dir(cfg) / f"{row.sheet}_overlay{'_zoom' if zoom_to_labels else ''}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)
    return out


def chips_mosaic(cfg: Config, n: int = 48, seed: int = 0) -> Path:
    chips_dir = Path(cfg.paths.chips_dir)
    index = gpd.read_file(chips_dir / "chips_index.gpkg")
    # over-represent positives: that's where mask quality shows
    picks = _sample_chips(index, n, seed)

    cols = 8
    rows = int(np.ceil(len(picks) / (cols // 2)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2), dpi=110)
    for ax in axes.ravel():
        ax.axis("off")
    for k, chip_id in enumerate(picks):
        img = np.asarray(Image.open(chips_dir / "images" / f"{chip_id}.jpg"))
        mask = np.asarray(Image.open(chips_dir / "masks" / f"{chip_id}.png"))
        r, c = divmod(k, cols // 2)
        axes[r, c * 2].imshow(img)
        axes[r, c * 2].set_title(chip_id, fontsize=5)
        shown = np.zeros((*mask.shape, 3), dtype=np.uint8)
        shown[mask == 1] = (0, 200, 0)
        shown[mask == 255] = (255, 165, 0)
        axes[r, c * 2 + 1].imshow(img)
        axes[r, c * 2 + 1].imshow(shown, alpha=0.45)
    out = qc_dir(cfg) / "chips_mosaic.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)
    return out


def _sample_chips(index, n: int, seed: int) -> list[str]:
    """~75% positive chips, shuffled — shared sampling policy for mosaics."""
    rng = np.random.default_rng(seed)
    pos = index[index.is_positive].sample(min(int(n * 0.75), index.is_positive.sum()), random_state=seed)
    neg = index[~index.is_positive]
    neg = neg.sample(min(n - len(pos), len(neg)), random_state=seed) if len(neg) else neg
    picks = list(pos.chip_id) + list(neg.chip_id)
    rng.shuffle(picks)
    return picks


def latest_checkpoint(cfg: Config) -> Path:
    candidates = sorted(Path(cfg.paths.runs_dir).glob("*/best.pt"))
    if not candidates:
        raise SystemExit(f"no */best.pt under {cfg.paths.runs_dir} — train first or pass --checkpoint")
    return candidates[-1]


def predict_mosaic(cfg: Config, checkpoint: Path | None, split: str, n: int = 12, seed: int = 0) -> Path:
    """Random chips of a split vs model output: image | ground truth | confusion."""
    import torch

    from parking.dataset import val_transform
    from parking.infer import load_model

    checkpoint = checkpoint or latest_checkpoint(cfg)
    chips_dir = Path(cfg.paths.chips_dir)
    index = gpd.read_file(chips_dir / "chips_index.gpkg")
    index = index[index.split == split]
    if index.empty:
        raise SystemExit(f"no chips with split={split!r} in {chips_dir}")
    picks = _sample_chips(index, n, seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, checkpoint, device)
    tf = val_transform()

    fig, axes = plt.subplots(len(picks), 3, figsize=(3 * 3.4, len(picks) * 3.4), dpi=100)
    axes = np.atleast_2d(axes)
    for c, title in enumerate(["image", "ground truth", "prediction (TP/FP/FN)"]):
        axes[0, c].set_title(title, fontsize=9)

    batch_size = 8
    for start in range(0, len(picks), batch_size):
        batch_ids = picks[start : start + batch_size]
        imgs = [np.asarray(Image.open(chips_dir / "images" / f"{i}.jpg").convert("RGB")) for i in batch_ids]
        masks = [np.asarray(Image.open(chips_dir / "masks" / f"{i}.png")) for i in batch_ids]
        x = torch.stack([tf(image=im, mask=m)["image"] for im, m in zip(imgs, masks)]).to(device)
        with torch.inference_mode():
            preds = (model(x).squeeze(1).sigmoid() >= cfg.infer.threshold).cpu().numpy()

        for k, (chip_id, img, mask, pred) in enumerate(zip(batch_ids, imgs, masks, preds)):
            r = start + k
            valid = mask != 255
            gt = mask == 1
            inter = (pred & gt & valid).sum()
            union = ((pred | gt) & valid).sum()
            iou = f"{inter / union:.2f}" if union else "–"

            axes[r, 0].imshow(img)
            axes[r, 0].set_ylabel(f"{chip_id}\nIoU {iou}", fontsize=7)

            gt_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
            gt_rgb[gt] = (0, 200, 0)
            gt_rgb[~valid] = (255, 165, 0)
            axes[r, 1].imshow(img)
            axes[r, 1].imshow(gt_rgb, alpha=0.45)

            conf = np.zeros((*mask.shape, 3), dtype=np.uint8)
            conf[pred & gt & valid] = (0, 200, 0)        # true positive
            conf[pred & ~gt & valid] = (220, 40, 40)     # false positive
            conf[~pred & gt & valid] = (40, 80, 255)     # false negative
            axes[r, 2].imshow(img)
            axes[r, 2].imshow(conf, alpha=0.5)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    out = qc_dir(cfg) / f"predict_{split}.png"
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.25 / len(picks)))
    fig.suptitle(f"{checkpoint} — split={split}", fontsize=9)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg, ns = load_config()
    sub = argparse.ArgumentParser(prog="parking.qc")
    sub.add_argument("command", choices=["overlay", "chips", "predict"])
    sub.add_argument("--sheet")
    sub.add_argument("--zoom-to-labels", action="store_true")
    sub.add_argument("--n", type=int, default=None)
    sub.add_argument("--checkpoint", type=Path, default=None)
    sub.add_argument("--split", choices=["train", "val"], default="val")
    sub.add_argument("--seed", type=int, default=0)
    args = sub.parse_args(ns.rest)

    if args.command == "overlay":
        if not args.sheet:
            raise SystemExit("--sheet required")
        overlay(cfg, args.sheet, args.zoom_to_labels)
    elif args.command == "chips":
        chips_mosaic(cfg, args.n or 48)
    else:
        predict_mosaic(cfg, args.checkpoint, args.split, args.n or 12, args.seed)


if __name__ == "__main__":
    main()
