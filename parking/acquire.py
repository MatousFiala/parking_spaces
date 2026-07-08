"""Stage 1: ČÚZK Ortofoto ATOM feed -> sheet index; download SM5 sheets.

The top feed (https://atom.cuzk.gov.cz/ORTOFOTO/ORTOFOTO.xml, ~24 MB) has one
<entry> per SM5 map sheet with a georss:polygon footprint (WGS84 lat lon) and a
dataset-feed id like
    .../datasetFeeds/CZ-00025712-CUZK_ORTOFOTO_WRTO24.2025.BENE09.xml
The per-sheet ZIP lives at
    https://openzu.cuzk.gov.cz/opendata/ORTOFOTO/WRTO24.2025.BENE09.zip
and contains a JPEG + JGW world file in S-JTSK (EPSG:5514).

CLI:
    python -m parking.acquire index                       # build/refresh sheet index
    python -m parking.acquire list                        # sheets intersecting labels
    python -m parking.acquire fetch --sheets BENE09       # download specific sheet(s)
    python -m parking.acquire fetch --all                 # all label-intersecting sheets
"""
from __future__ import annotations

import argparse
import logging
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import geopandas as gpd
import requests
from shapely.geometry import Polygon

from parking.config import Config, load_config
from parking.geo import load_labels

log = logging.getLogger(__name__)

ATOM_NS = "{http://www.w3.org/2005/Atom}"
GEORSS_NS = "{http://www.georss.org/georss}"
OPENDATA_URL = "https://openzu.cuzk.gov.cz/opendata/ORTOFOTO/{code}.zip"
DATASET_ID_RE = re.compile(r"datasetFeeds/CZ-\d+-CUZK_ORTOFOTO_(?P<code>[^/]+)\.xml$")


def build_sheet_index(cfg: Config, feed_xml: Path | None = None) -> gpd.GeoDataFrame:
    """Parse the ATOM top feed into a GeoDataFrame(sheet, name, zip_url, geometry)."""
    if feed_xml is not None and feed_xml.exists():
        content = feed_xml.read_bytes()
    else:
        log.info("downloading ATOM feed %s", cfg.atom.feed_url)
        r = requests.get(cfg.atom.feed_url, timeout=300)
        r.raise_for_status()
        content = r.content

    root = ET.fromstring(content)
    records = []
    for entry in root.iter(f"{ATOM_NS}entry"):
        entry_id = entry.findtext(f"{ATOM_NS}id", default="")
        m = DATASET_ID_RE.search(entry_id)
        poly_text = entry.findtext(f"{GEORSS_NS}polygon")
        if not m or not poly_text:
            continue
        code = m.group("code")  # e.g. WRTO24.2025.BENE09
        vals = [float(v) for v in poly_text.split()]
        latlon = list(zip(vals[0::2], vals[1::2]))
        geom = Polygon([(lon, lat) for lat, lon in latlon])
        title = entry.findtext(f"{ATOM_NS}title", default="")
        records.append(
            {
                "code": code,
                "sheet": code.rsplit(".", 1)[-1],       # BENE09
                "name": title.split(":")[-1].strip(),   # Benešov 0-9
                "zip_url": OPENDATA_URL.format(code=code),
                "geometry": geom,
            }
        )
    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326").to_crs(cfg.crs.target)
    log.info("sheet index: %d sheets", len(gdf))
    return gdf


def load_sheet_index(cfg: Config) -> gpd.GeoDataFrame:
    path = Path(cfg.paths.sheet_index)
    if not path.exists():
        gdf = build_sheet_index(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(path, driver="GPKG")
        return gdf
    return gpd.read_file(path)


def select_sheets(cfg: Config, labels: gpd.GeoDataFrame | None = None) -> gpd.GeoDataFrame:
    """Sheets whose footprint intersects any label polygon."""
    index = load_sheet_index(cfg)
    if labels is None:
        labels = load_labels(cfg)
    hits = gpd.sjoin(index, labels[[labels.geometry.name]], how="inner", predicate="intersects")
    out = index.loc[sorted(set(hits.index))].reset_index(drop=True)
    return out


def sheet_jpeg_path(cfg: Config, sheet: str) -> Path:
    return Path(cfg.paths.sheets_dir) / f"{sheet}.jpg"


def download_sheet(cfg: Config, row) -> Path:
    """Stream the sheet ZIP, extract JPEG + JGW next to each other, drop the ZIP.

    Returns the JPEG path. Skips work if the JPEG already exists.
    """
    dest_dir = Path(cfg.paths.sheets_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    jpg_dest = sheet_jpeg_path(cfg, row.sheet)
    if jpg_dest.exists():
        return jpg_dest

    zip_path = dest_dir / f"{row.sheet}.zip"
    log.info("downloading %s", row.zip_url)
    with requests.get(row.zip_url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)

    with zipfile.ZipFile(zip_path) as zf:
        jpg = [n for n in zf.namelist() if n.lower().endswith(".jpg")]
        # several world files may be present (S-JTSK .jgw plus ETRS89 variants);
        # the plain .jgw matches the S-JTSK raster
        jgw = [n for n in zf.namelist() if n.lower().endswith(".jgw")]
        if not jpg or not jgw:
            raise RuntimeError(f"{zip_path}: expected .jpg + .jgw, got {zf.namelist()}")
        jpg_dest.write_bytes(zf.read(jpg[0]))
        jpg_dest.with_suffix(".jgw").write_bytes(zf.read(jgw[0]))
    zip_path.unlink()
    log.info("extracted %s (%.0f MB)", jpg_dest, jpg_dest.stat().st_size / 1e6)
    return jpg_dest


def delete_sheet(cfg: Config, sheet: str) -> None:
    for ext in (".jpg", ".jgw"):
        p = sheet_jpeg_path(cfg, sheet).with_suffix(ext)
        p.unlink(missing_ok=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg, ns = load_config()
    sub = argparse.ArgumentParser(prog="parking.acquire")
    sub.add_argument("command", choices=["index", "list", "fetch"])
    sub.add_argument("--sheets", help="comma-separated sheet codes, e.g. PRAH57,PRAH58")
    sub.add_argument("--all", action="store_true", help="fetch every label-intersecting sheet")
    args = sub.parse_args(ns.rest)

    if args.command == "index":
        gdf = build_sheet_index(cfg)
        out = Path(cfg.paths.sheet_index)
        out.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(out, driver="GPKG")
        print(f"wrote {len(gdf)} sheets to {out}")
        return

    selected = select_sheets(cfg)
    if args.command == "list":
        print(f"{len(selected)} sheets intersect the labels:")
        for _, r in selected.iterrows():
            print(f"  {r.sheet}  {r['name']}  {r.zip_url}")
        return

    if args.command == "fetch":
        if args.sheets:
            wanted = {s.strip().upper() for s in args.sheets.split(",")}
            selected = selected[selected.sheet.isin(wanted)]
            missing = wanted - set(selected.sheet)
            if missing:
                # allow fetching sheets outside the label area (e.g. for inference)
                index = load_sheet_index(cfg)
                extra = index[index.sheet.isin(missing)]
                selected = gpd.pd.concat([selected, extra], ignore_index=True)
        elif not args.all:
            raise SystemExit("fetch requires --sheets CODE[,CODE...] or --all")
        for _, r in selected.iterrows():
            download_sheet(cfg, r)


if __name__ == "__main__":
    main()
