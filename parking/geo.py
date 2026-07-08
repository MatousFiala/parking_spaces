"""Label loading/cleaning and shared geo helpers.

CLI:
    python -m parking.geo --config configs/default.yaml
cleans Data/Parkovani_praha.geojson -> data/labels/labels_5514.gpkg and prints stats.
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely

from parking.config import Config, load_config

log = logging.getLogger(__name__)

KML_JUNK_COLUMNS = [
    "name", "folders", "description", "altitude", "alt_mode",
    "time_begin", "time_end", "time_when",
]


def prepare_labels(cfg: Config) -> gpd.GeoDataFrame:
    """Raw KML-derived GeoJSON -> clean 2D polygons in the target CRS."""
    gdf = gpd.read_file(cfg.paths.raw_labels)
    n_raw = len(gdf)

    gdf = gdf.drop(columns=[c for c in KML_JUNK_COLUMNS if c in gdf.columns])
    gdf.geometry = shapely.force_2d(gdf.geometry.values)

    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        log.info("repairing %d invalid geometries", int(invalid.sum()))
        gdf.loc[invalid, gdf.geometry.name] = shapely.make_valid(gdf.geometry.values[invalid.values])

    gdf = gdf.explode(index_parts=False, ignore_index=True)
    gdf = gdf[gdf.geometry.geom_type == "Polygon"]
    gdf = gdf[~gdf.geometry.is_empty]

    gdf = gdf.to_crs(cfg.crs.target)
    area = gdf.geometry.area
    gdf = gdf[area >= cfg.chips.min_label_area].reset_index(drop=True)
    gdf["area_m2"] = gdf.geometry.area

    log.info(
        "labels: %d raw features -> %d clean polygons | area total %.1f ha, median %.1f m2, p95 %.1f m2",
        n_raw, len(gdf), gdf.area_m2.sum() / 1e4, np.median(gdf.area_m2), np.percentile(gdf.area_m2, 95),
    )
    return gdf


def load_labels(cfg: Config) -> gpd.GeoDataFrame:
    """Load the canonical cleaned labels, building them if missing."""
    path = Path(cfg.paths.labels)
    if not path.exists():
        gdf = prepare_labels(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(path, driver="GPKG")
        return gdf
    return gpd.read_file(path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg, _ = load_config()
    gdf = prepare_labels(cfg)
    out = Path(cfg.paths.labels)
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out, driver="GPKG")
    b = gdf.total_bounds
    print(f"wrote {len(gdf)} polygons to {out}")
    print(f"extent (EPSG:5514): x [{b[0]:.0f}, {b[2]:.0f}]  y [{b[1]:.0f}, {b[3]:.0f}]")


if __name__ == "__main__":
    main()
