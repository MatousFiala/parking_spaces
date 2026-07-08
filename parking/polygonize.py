"""Stage 4b: probability raster -> clean parking polygons; cross-sheet merge.

CLI:
    python -m parking.polygonize probs --probs data/predictions/PRAH57_probs.tif
    python -m parking.polygonize merge --out data/predictions/parking_prague.gpkg
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio import features
from scipy import ndimage

from parking.config import Config, load_config

log = logging.getLogger(__name__)


def polygonize_probs(cfg: Config, probs: np.ndarray, transform, sheet: str) -> int:
    """Threshold + clean + vectorize one sheet; writes <sheet>_polys.gpkg."""
    i = cfg.infer
    mask = probs >= i.threshold
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))  # kill speckle

    geoms = [
        shapely.geometry.shape(geom)
        for geom, val in features.shapes(mask.astype(np.uint8), mask=mask, transform=transform)
        if val == 1
    ]
    geoms = [g.simplify(i.simplify_tol) for g in geoms if g.area >= i.min_area]

    out = Path(cfg.paths.predictions_dir) / f"{sheet}_polys.gpkg"
    out.parent.mkdir(parents=True, exist_ok=True)
    if geoms:
        gdf = gpd.GeoDataFrame({"sheet": sheet, "area_m2": [g.area for g in geoms]},
                               geometry=geoms, crs=cfg.crs.target)
        gdf.to_file(out, driver="GPKG")
    return len(geoms)


def polygonize_file(cfg: Config, probs_path: Path) -> int:
    with rasterio.open(probs_path) as src:
        probs = src.read(1).astype(np.float32) / 255.0
        transform = src.transform
    sheet = probs_path.stem.replace("_probs", "")
    return polygonize_probs(cfg, probs, transform, sheet)


def merge(cfg: Config, out_path: Path) -> gpd.GeoDataFrame:
    """Concatenate all per-sheet layers and dissolve polygons split by sheet
    seams (tiny +/- buffer closes hairline gaps at boundaries)."""
    pred_dir = Path(cfg.paths.predictions_dir)
    parts = sorted(pred_dir.glob("*_polys.gpkg"))
    if not parts:
        raise SystemExit(f"no *_polys.gpkg in {pred_dir}")
    gdf = gpd.GeoDataFrame(
        pd.concat([gpd.read_file(p) for p in parts], ignore_index=True), crs=cfg.crs.target
    )
    eps = 0.05
    merged = shapely.unary_union(gdf.geometry.buffer(eps).values)
    polys = [g.buffer(-eps) for g in getattr(merged, "geoms", [merged])]
    polys = [g for g in polys if not g.is_empty and g.area >= cfg.infer.min_area]
    out = gpd.GeoDataFrame({"area_m2": [g.area for g in polys]}, geometry=polys, crs=cfg.crs.target)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(out_path, driver="GPKG")
    out.to_crs(4326).to_file(out_path.with_suffix(".geojson"), driver="GeoJSON")
    log.info("merged %d sheet layers -> %d polygons -> %s", len(parts), len(out), out_path)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg, ns = load_config()
    sub = argparse.ArgumentParser(prog="parking.polygonize")
    sub.add_argument("command", choices=["probs", "merge"])
    sub.add_argument("--probs", help="path to a *_probs.tif")
    sub.add_argument("--out", default=None)
    args = sub.parse_args(ns.rest)

    if args.command == "probs":
        if not args.probs:
            raise SystemExit("--probs required")
        n = polygonize_file(cfg, Path(args.probs))
        print(f"{n} polygons")
    else:
        out = Path(args.out or Path(cfg.paths.predictions_dir) / "parking_merged.gpkg")
        merge(cfg, out)


if __name__ == "__main__":
    main()
