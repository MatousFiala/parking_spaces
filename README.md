# parking_spaces

Automatically marking parking spaces from aerial photography — semantic
segmentation trained on Prague parking-space polygons over ČÚZK Ortofoto ČR
(12.5 cm/px, open data CC BY 4.0), designed to run inference over any part of
Czechia.

## Setup

```bash
uv sync   # picks torch wheels per platform: Linux -> CPU, Windows -> CUDA (cu128)
```

### Windows CUDA machine (training box)

One-time: install the NVIDIA driver and uv
(`powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`), sync the repo
including `Data\Parkovani_praha.geojson`, then from the repo root:

```powershell
.\scripts\run_local.ps1 -Stage setup    # uv sync + CUDA sanity check
.\scripts\run_local.ps1 -Stage prep     # labels + sheet index + alignment QC
.\scripts\run_local.ps1 -Stage chips    # ~8 GB downloads -> ~15-20 GB chips
.\scripts\run_local.ps1 -Stage train    # overnight; -Resume continues after interrupt
.\scripts\run_local.ps1 -Stage infer -Aoi my_region.geojson
```

OOM? Lower the batch: `-Stage train -ExtraArgs @('--train.batch_size','8')`
(8 GB cards), and consider `'--train.num_workers','4'`.

## Pipeline

All stages read `configs/default.yaml`; any key can be overridden on the CLI
with dotted flags, e.g. `--train.lr 1e-4`.

```bash
# 0. clean labels: Data/Parkovani_praha.geojson -> data/labels/labels_5514.gpkg
python -m parking.geo

# 1. sheet index from the ČÚZK ATOM feed (+ list sheets that touch labels)
python -m parking.acquire index
python -m parking.acquire list

# 2. training chips (streams sheets: download -> chip -> delete, ~8 GB total)
python -m parking.chips --all

# QC before committing to anything expensive:
python -m parking.qc overlay --sheet PRAH57 --zoom-to-labels   # label/ortho alignment
python -m parking.qc chips                                     # chip/mask mosaic

# 3. training (GPU box / Colab; CPU smoke test: --train.limit_batches 5 --train.epochs 2)
python -m parking.train

# visual assessment: random chips vs predictions (image | GT | TP/FP/FN confusion)
python -m parking.qc predict --split val     # and --split train; newest best.pt by default

# 4. inference + vectorization for any AOI in Czechia
python -m parking.infer --checkpoint data/runs/<run>/best.pt --aoi my_region.geojson
python -m parking.polygonize merge --out data/predictions/parking_region.gpkg
```

## Data notes

- Labels: 62,701 hand-digitized parking polygons (Prague only, provenance
  unknown → treated as *incomplete*: negatives are only sampled within 30–300 m
  of labeled parking, and a 0.5 m ignore band around polygon edges absorbs
  digitization misalignment).
- Imagery: one ZIP per SM5 sheet (2.5 × 2 km, JPEG + JGW, EPSG:5514) from
  https://atom.cuzk.gov.cz/ORTOFOTO/ORTOFOTO.xml (~66 MB/sheet).
- Train/val split is geographic (whole SM5 sheets, contiguous blocks) to avoid
  spatial leakage.
