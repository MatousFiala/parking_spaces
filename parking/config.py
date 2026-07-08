"""YAML config -> nested dataclasses, with dotted CLI overrides.

Usage in every stage:
    cfg, extra = load_config()          # parses --config and --<dotted.key> overrides
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml


@dataclass
class Paths:
    raw_labels: str = "Data/Parkovani_praha.geojson"
    labels: str = "data/labels/labels_5514.gpkg"
    sheet_index: str = "data/labels/sheet_index.gpkg"
    sheets_dir: str = "data/sheets"
    chips_dir: str = "data/chips"
    runs_dir: str = "data/runs"
    predictions_dir: str = "data/predictions"


@dataclass
class Crs:
    target: str = "EPSG:5514"


@dataclass
class Atom:
    feed_url: str = "https://atom.cuzk.gov.cz/ORTOFOTO/ORTOFOTO.xml"
    crs_hint: str = "5514"


@dataclass
class Chips:
    size: int = 512
    gsd: float = 0.125
    center_jitter: float = 16.0
    cluster_step: float = 48.0
    min_center_dist: float = 32.0
    neg_ratio: float = 0.5
    neg_ring_inner: float = 30.0
    neg_ring_outer: float = 300.0
    ignore_buffer: float = 0.5
    min_label_area: float = 3.0
    jpeg_quality: int = 88
    max_chips: int = 40000
    val_fraction: float = 0.15
    split_seed: int = 42


@dataclass
class Train:
    arch: str = "unet"
    encoder: str = "timm-efficientnet-b3"
    encoder_weights: str = "imagenet"
    epochs: int = 50
    batch_size: int = 16
    lr: float = 3e-4
    encoder_lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    bce_smooth: float = 0.05
    dice_weight: float = 0.5
    amp: bool = True
    num_workers: int = 8
    early_stop_patience: int = 12
    limit_batches: int = 0
    seed: int = 42


@dataclass
class Infer:
    window: int = 512
    stride: int = 384
    batch_size: int = 8
    threshold: float = 0.5
    min_area: float = 8.0
    simplify_tol: float = 0.25
    save_probs: bool = False


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    crs: Crs = field(default_factory=Crs)
    atom: Atom = field(default_factory=Atom)
    chips: Chips = field(default_factory=Chips)
    train: Train = field(default_factory=Train)
    infer: Infer = field(default_factory=Infer)


def _coerce(value: str, current):
    if isinstance(current, bool):
        return value.lower() in ("1", "true", "yes")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def _apply(obj, dotted: str, value: str) -> None:
    head, _, rest = dotted.partition(".")
    if not hasattr(obj, head):
        raise KeyError(f"unknown config key: {dotted}")
    if rest:
        _apply(getattr(obj, head), rest, value)
    else:
        setattr(obj, head, _coerce(value, getattr(obj, head)))


def _from_dict(cls, data: dict):
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        v = data[f.name]
        kwargs[f.name] = _from_dict(f.type, v) if is_dataclass_type(f.type) and isinstance(v, dict) else v
    return cls(**kwargs)


def is_dataclass_type(t) -> bool:
    return isinstance(t, type) and is_dataclass(t)


def to_dict(obj) -> dict:
    out = {}
    for f in fields(obj):
        v = getattr(obj, f.name)
        out[f.name] = to_dict(v) if is_dataclass(v) else v
    return out


def load_config(argv: list[str] | None = None) -> tuple[Config, argparse.Namespace]:
    """Parse --config plus arbitrary --section.key value overrides.

    Unknown single-dash / positional args are returned in the namespace under
    `rest` so stages can define their own extra flags via a second parser.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="configs/default.yaml")
    known, rest = parser.parse_known_args(argv)

    cfg = Config()
    path = Path(known.config)
    if path.exists():
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        # dataclass field types are strings under `from __future__ import annotations`,
        # so resolve section classes explicitly
        section_types = {f.name: type(getattr(cfg, f.name)) for f in fields(cfg)}
        for name, sub in raw.items():
            if name in section_types and isinstance(sub, dict):
                setattr(cfg, name, _from_dict_typed(section_types[name], sub))

    # dotted overrides: --chips.size 256
    leftovers = []
    i = 0
    while i < len(rest):
        tok = rest[i]
        if tok.startswith("--") and "." in tok:
            key = tok[2:]
            if i + 1 >= len(rest):
                raise SystemExit(f"missing value for {tok}")
            _apply(cfg, key, rest[i + 1])
            i += 2
        else:
            leftovers.append(tok)
            i += 1

    ns = argparse.Namespace(config=known.config, rest=leftovers)
    return cfg, ns


def _from_dict_typed(cls, data: dict):
    valid = {f.name for f in fields(cls)}
    unknown = set(data) - valid
    if unknown:
        raise SystemExit(f"unknown keys in config section {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


def dump_config(cfg: Config, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as fh:
        yaml.safe_dump(to_dict(cfg), fh, sort_keys=False)
