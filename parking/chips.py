"""Stage 2: training chips (image JPEG + mask PNG) from SM5 sheets.

Streaming: for each selected sheet -> download (stage 1) -> cut all chips ->
delete the sheet. Only one ~66 MB JPEG (~1 GB decoded) exists at a time, so the
full Prague build fits the small local disk; the same command runs unchanged on
a cloud box.

Sampling (labels may be incomplete -> never treat far-from-label area as negative):
- positive centers: representative point of every dissolved label cluster, plus
  a `cluster_step` grid over large clusters, all jittered by `center_jitter`
- negative centers: uniform samples from the ring buffer(outer) - buffer(inner)
  around labels within the sheet
Mask encoding: 0 background, 1 parking, 255 ignore (a `ignore_buffer` band
around every polygon boundary absorbs label/ortho misalignment).

CLI:
    python -m parking.chips --sheets PRAH57,PRAH58     # specific sheets
    python -m parking.chips --all                      # every label-intersecting sheet
    python -m parking.chips --all --keep-sheets        # don't delete sheets after
"""
from __future__ import annotations

import argparse
import logging
import zlib
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from PIL import Image
from rasterio import features, windows

from parking.acquire import delete_sheet, download_sheet, select_sheets
from parking.config import Config, load_config
from parking.geo import load_labels

log = logging.getLogger(__name__)


def assign_splits(sheets: gpd.GeoDataFrame, val_fraction: float, seed: int) -> pd.Series:
    """'train'/'val' per sheet: 3 spatially contiguous val blocks grown from
    random seed sheets, so val measures generalization to unseen districts."""
    n_val = max(1, round(len(sheets) * val_fraction))
    cx, cy = sheets.geometry.centroid.x.values, sheets.geometry.centroid.y.values
    rng = np.random.default_rng(seed)
    val_idx: set[int] = set()
    for seed_i in rng.choice(len(sheets), size=min(3, n_val), replace=False):
        val_idx.add(int(seed_i))
    while len(val_idx) < n_val:
        # grow: nearest non-val sheet to any val sheet
        rest = [i for i in range(len(sheets)) if i not in val_idx]
        dmin = [
            min((cx[i] - cx[j]) ** 2 + (cy[i] - cy[j]) ** 2 for j in val_idx)
            for i in rest
        ]
        val_idx.add(rest[int(np.argmin(dmin))])
    return pd.Series(["val" if i in val_idx else "train" for i in range(len(sheets))], index=sheets.index)


def _grid_points_in(geom, step: float) -> list[tuple[float, float]]:
    minx, miny, maxx, maxy = geom.bounds
    xs = np.arange(minx + step / 2, maxx, step)
    ys = np.arange(miny + step / 2, maxy, step)
    if not len(xs) or not len(ys):
        return []
    xx, yy = np.meshgrid(xs, ys)
    pts = shapely.points(xx.ravel(), yy.ravel())
    inside = shapely.contains(geom, pts)
    return [(p.x, p.y) for p in pts[inside]]


def sample_centers(cfg: Config, labels_in_sheet: gpd.GeoDataFrame, sheet_geom, rng) -> list[tuple[float, float, bool]]:
    """(x, y, is_positive) chip centers for one sheet."""
    c = cfg.chips
    union = shapely.unary_union(labels_in_sheet.geometry.values)
    clusters = getattr(union, "geoms", [union])

    pos: list[tuple[float, float]] = []
    for cl in clusters:
        p = cl.representative_point()
        pos.append((p.x, p.y))
        if cl.bounds[2] - cl.bounds[0] > c.cluster_step or cl.bounds[3] - cl.bounds[1] > c.cluster_step:
            pos.extend(_grid_points_in(cl, c.cluster_step))
    pos = [
        (x + rng.uniform(-c.center_jitter, c.center_jitter),
         y + rng.uniform(-c.center_jitter, c.center_jitter))
        for x, y in pos
    ]

    # negatives: ring around labels, clipped to the sheet
    n_neg = int(len(pos) * c.neg_ratio)
    neg: list[tuple[float, float]] = []
    if n_neg:
        ring = union.buffer(c.neg_ring_outer, quad_segs=2).difference(
            union.buffer(c.neg_ring_inner, quad_segs=2)
        ).intersection(sheet_geom)
        if not ring.is_empty:
            minx, miny, maxx, maxy = ring.bounds
            attempts = 0
            while len(neg) < n_neg and attempts < n_neg * 30:
                x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
                if ring.contains(shapely.Point(x, y)):
                    neg.append((x, y))
                attempts += 1

    # dedup centers closer than min_center_dist (grid hash)
    seen: dict[tuple[int, int], None] = {}
    out: list[tuple[float, float, bool]] = []
    for (x, y), is_pos in [(p, True) for p in pos] + [(p, False) for p in neg]:
        key = (int(x // c.min_center_dist), int(y // c.min_center_dist))
        if key in seen:
            continue
        seen[key] = None
        out.append((x, y, is_pos))
    return out


def rasterize_mask(cfg: Config, labels: gpd.GeoDataFrame, transform, size: int) -> np.ndarray:
    """{0 bg, 1 parking, 255 ignore} mask for one chip window."""
    mask = np.zeros((size, size), dtype=np.uint8)
    if len(labels):
        geoms = labels.geometry.values
        features.rasterize(
            [(g, 1) for g in geoms],
            out=mask, transform=transform, all_touched=False,
        )
        ignore = [(g.boundary.buffer(cfg.chips.ignore_buffer), 255) for g in geoms]
        features.rasterize(ignore, out=mask, transform=transform, all_touched=False)
    return mask


def chip_sheet(cfg: Config, row, labels: gpd.GeoDataFrame, split: str) -> list[dict]:
    """Cut all chips for one sheet. Returns chip-index records."""
    c = cfg.chips
    jpg = download_sheet(cfg, row)
    img_dir = Path(cfg.paths.chips_dir) / "images"
    mask_dir = Path(cfg.paths.chips_dir) / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(jpg) as src:
        data = src.read()  # (3, H, W) ~1 GB, once per sheet
        transform, h, w = src.transform, src.height, src.width
        sheet_geom = shapely.box(*src.bounds)

    half = c.size / 2 * c.gsd
    inner = shapely.box(
        sheet_geom.bounds[0] + half, sheet_geom.bounds[1] + half,
        sheet_geom.bounds[2] - half, sheet_geom.bounds[3] - half,
    )
    labels_near = labels[labels.intersects(sheet_geom.buffer(c.neg_ring_outer))]
    if labels_near.empty:
        delete_sheet(cfg, row.sheet)
        return []

    rng = np.random.default_rng(zlib.crc32(row.sheet.encode()))  # stable across processes
    centers = sample_centers(cfg, labels_near, sheet_geom, rng)
    records = []
    sindex = labels_near.sindex
    for i, (x, y, is_pos) in enumerate(centers):
        if not inner.contains(shapely.Point(x, y)):
            continue  # keep windows fully inside the sheet; neighbors cover the rest
        win = windows.from_bounds(x - half, y - half, x + half, y + half, transform)
        r0, c0 = int(round(win.row_off)), int(round(win.col_off))
        if r0 < 0 or c0 < 0 or r0 + c.size > h or c0 + c.size > w:
            continue
        chip_img = data[:, r0 : r0 + c.size, c0 : c0 + c.size]
        chip_transform = windows.transform(windows.Window(c0, r0, c.size, c.size), transform)

        hit_idx = sindex.query(shapely.box(x - half, y - half, x + half, y + half))
        mask = rasterize_mask(cfg, labels_near.iloc[hit_idx], chip_transform, c.size)

        chip_id = f"{row.sheet}_{i:05d}"
        Image.fromarray(np.moveaxis(chip_img, 0, -1)).save(
            img_dir / f"{chip_id}.jpg", quality=c.jpeg_quality
        )
        Image.fromarray(mask).save(mask_dir / f"{chip_id}.png")
        records.append(
            {
                "chip_id": chip_id,
                "sheet": row.sheet,
                "is_positive": bool(is_pos),
                "split": split,
                "pos_frac": float((mask == 1).mean()),
                "geometry": shapely.Point(x, y),
            }
        )
    return records


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg, ns = load_config()
    sub = argparse.ArgumentParser(prog="parking.chips")
    sub.add_argument("--sheets", help="comma-separated sheet codes")
    sub.add_argument("--all", action="store_true")
    sub.add_argument("--keep-sheets", action="store_true", help="don't delete sheet JPEGs after chipping")
    args = sub.parse_args(ns.rest)

    labels = load_labels(cfg)
    selected = select_sheets(cfg, labels)
    selected["split"] = assign_splits(selected, cfg.chips.val_fraction, cfg.chips.split_seed)
    if args.sheets:
        wanted = {s.strip().upper() for s in args.sheets.split(",")}
        selected = selected[selected.sheet.isin(wanted)]
    elif not args.all:
        raise SystemExit("pass --sheets CODE[,CODE...] or --all")

    all_records: list[dict] = []
    index_path = Path(cfg.paths.chips_dir) / "chips_index.gpkg"
    if index_path.exists():  # resume: skip sheets already chipped
        existing = gpd.read_file(index_path)
        # splits are a property of the sheet assignment, not of the stored rows
        sheet_split = dict(zip(selected.sheet, selected.split))
        existing["split"] = existing.sheet.map(sheet_split).fillna(existing.split)
        done = set(existing.sheet)
        all_records = existing.to_dict("records")
        selected = selected[~selected.sheet.isin(done)]
        log.info("resuming: %d sheets already chipped", len(done))

    total = sum(1 for r in all_records)
    for _, row in selected.iterrows():
        if total >= cfg.chips.max_chips:
            log.warning("max_chips=%d reached, stopping (remaining sheets skipped)", cfg.chips.max_chips)
            break
        recs = chip_sheet(cfg, row, labels, row.split)
        if not args.keep_sheets:
            delete_sheet(cfg, row.sheet)
        total += len(recs)
        all_records.extend(recs)
        log.info("%s [%s]: %d chips (total %d)", row.sheet, row.split, len(recs), total)
        gpd.GeoDataFrame(all_records, crs=cfg.crs.target).to_file(index_path, driver="GPKG")

    df = pd.DataFrame(all_records)
    if len(df):
        print(
            f"{len(df)} chips | positives {df.is_positive.sum()} | "
            f"train {(df.split == 'train').sum()} / val {(df.split == 'val').sum()} | "
            f"mean positive fraction {df.pos_frac.mean():.3f}"
        )


if __name__ == "__main__":
    main()
