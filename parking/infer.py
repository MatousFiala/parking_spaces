"""Stage 4a: sliding-window inference over SM5 sheets -> parking polygons.

Streams sheets like chips.py: download -> predict -> polygonize -> delete, so an
arbitrarily large AOI (up to all of Czechia) never needs more than one sheet on
disk. Per-sheet polygon layers land in predictions/; merge them afterwards with
    python -m parking.polygonize merge --out predictions/parking_region.gpkg

CLI:
    python -m parking.infer --checkpoint data/runs/<run>/best.pt --sheets PRAH57
    python -m parking.infer --checkpoint ... --aoi region.geojson    # any vector file
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import torch

from parking.acquire import delete_sheet, download_sheet, load_sheet_index
from parking.config import Config, load_config
from parking.dataset import IMAGENET_MEAN, IMAGENET_STD
from parking.model import build_model
from parking.polygonize import polygonize_probs

log = logging.getLogger(__name__)


def hann2d(size: int) -> np.ndarray:
    w = np.hanning(size + 2)[1:-1].astype(np.float32)  # avoid zero edges
    return np.outer(w, w)


@torch.inference_mode()
def predict_sheet(cfg: Config, model, device, jpg: Path) -> tuple[np.ndarray, rasterio.Affine]:
    """Hann-blended sliding-window sigmoid probabilities for one sheet."""
    win, stride, bs = cfg.infer.window, cfg.infer.stride, cfg.infer.batch_size
    with rasterio.open(jpg) as src:
        data = src.read()  # (3,H,W) uint8
        transform = src.transform
    _, h, w = data.shape

    prob_sum = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    hann = hann2d(win)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)

    rows = list(range(0, max(1, h - win + 1), stride))
    cols = list(range(0, max(1, w - win + 1), stride))
    if rows[-1] != h - win:
        rows.append(max(0, h - win))
    if cols[-1] != w - win:
        cols.append(max(0, w - win))
    offsets = [(r, c) for r in rows for c in cols]

    amp = device.type == "cuda"
    for i in range(0, len(offsets), bs):
        batch_off = offsets[i : i + bs]
        batch = np.stack(
            [(data[:, r : r + win, c : c + win].astype(np.float32) / 255.0 - mean) / std
             for r, c in batch_off]
        )
        x = torch.from_numpy(batch).to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            p = model(x).squeeze(1).sigmoid().float().cpu().numpy()
        for (r, c), pk in zip(batch_off, p):
            prob_sum[r : r + win, c : c + win] += pk * hann
            weight[r : r + win, c : c + win] += hann
    probs = prob_sum / np.maximum(weight, 1e-6)
    return probs, transform


def save_probs(cfg: Config, probs: np.ndarray, transform, sheet: str) -> Path:
    out = Path(cfg.paths.predictions_dir) / f"{sheet}_probs.tif"
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out, "w", driver="GTiff", height=probs.shape[0], width=probs.shape[1], count=1,
        dtype="uint8", crs=cfg.crs.target, transform=transform,
        compress="deflate", tiled=True,
    ) as dst:
        dst.write((probs * 255).astype(np.uint8), 1)
    return out


def load_model(cfg: Config, checkpoint: Path, device) -> torch.nn.Module:
    model = build_model(cfg)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    log.info("loaded %s (epoch %s, val_iou %.4f)", checkpoint, ckpt.get("epoch"), ckpt.get("val_iou", float("nan")))
    return model.to(device).eval()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg, ns = load_config()
    sub = argparse.ArgumentParser(prog="parking.infer")
    sub.add_argument("--checkpoint", required=True)
    sub.add_argument("--sheets", help="comma-separated sheet codes")
    sub.add_argument("--aoi", help="vector file (any CRS); all intersecting sheets are processed")
    sub.add_argument("--keep-sheets", action="store_true")
    args = sub.parse_args(ns.rest)

    index = load_sheet_index(cfg)
    if args.sheets:
        wanted = {s.strip().upper() for s in args.sheets.split(",")}
        selected = index[index.sheet.isin(wanted)]
    elif args.aoi:
        aoi = gpd.read_file(args.aoi).to_crs(cfg.crs.target)
        hits = gpd.sjoin(index, aoi[[aoi.geometry.name]], how="inner", predicate="intersects")
        selected = index.loc[sorted(set(hits.index))]
    else:
        raise SystemExit("pass --sheets or --aoi")
    log.info("%d sheets to predict", len(selected))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, Path(args.checkpoint), device)

    for _, row in selected.iterrows():
        jpg = download_sheet(cfg, row)
        probs, transform = predict_sheet(cfg, model, device, jpg)
        if cfg.infer.save_probs:
            save_probs(cfg, probs, transform, row.sheet)
        n = polygonize_probs(cfg, probs, transform, row.sheet)
        log.info("%s: %d polygons", row.sheet, n)
        if not args.keep_sheets:
            delete_sheet(cfg, row.sheet)


if __name__ == "__main__":
    main()
