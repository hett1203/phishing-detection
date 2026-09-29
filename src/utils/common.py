"""
src/utils/common.py
Generic helpers: YAML loading, path resolution, joblib save/load,
versioning, seeding.
"""
from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import yaml

try:
    from box import Box           # type: ignore
    _HAS_BOX = True
except Exception:
    _HAS_BOX = False

from src.utils.exceptions import ConfigError, ModelPersistenceError
from src.utils.logger import get_logger

log = get_logger(__name__)


# -------------------------- Config loading -------------------------- #
def read_yaml(path: Union[str, Path]) -> Any:
    """Read a YAML file into a Box (attribute-style dict) if available."""
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"YAML config not found: {p}")
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Malformed YAML in {p}: {exc}") from exc
    if _HAS_BOX and isinstance(data, dict):
        return Box(data, box_dots=True)
    return data


def write_yaml(path: Union[str, Path], data: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)


def read_json(path: Union[str, Path]) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Union[str, Path], data: Any, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def _default(o: Any) -> Any:
        if is_dataclass(o):
            return asdict(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (set, tuple)):
            return list(o)
        return str(o)

    with p.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, default=_default)


# -------------------------- Paths -------------------------- #
def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def project_root() -> Path:
    """Return the project root (the directory containing `src/` and `configs/`)."""
    here = Path(__file__).resolve().parent
    while here != here.parent:
        if (here / "src").is_dir() and (here / "configs").is_dir():
            return here
        here = here.parent
    # Fall back to two levels up from this file (src/utils/common.py)
    return Path(__file__).resolve().parents[2]


def to_abs(path: Union[str, Path]) -> Path:
    """Resolve a project-relative path to absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    return project_root() / p


# -------------------------- Persistence -------------------------- #
def save_joblib(obj: Any, path: Union[str, Path],
                compress: int = 3) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    try:
        joblib.dump(obj, p, compress=compress)
    except Exception as exc:  # pragma: no cover
        raise ModelPersistenceError(f"Failed to save joblib at {p}: {exc}") from exc
    log.debug("Saved joblib artifact: %s (%d bytes)", p, p.stat().st_size)
    return p


def load_joblib(path: Union[str, Path]) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Joblib artifact not found: {p}")
    try:
        return joblib.load(p)
    except Exception as exc:  # pragma: no cover
        raise ModelPersistenceError(f"Failed to load joblib at {p}: {exc}") from exc


# -------------------------- Reproducibility -------------------------- #
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import sklearn
        sklearn.random_state = seed
    except Exception:
        pass


# -------------------------- Numeric mappings -------------------------- #
def map_numeric_to_label(value: int) -> str:
    """Map -1 -> 'phising', +1 -> 'safe' (per predicted_file.csv reference)."""
    return "phising" if value == -1 else "safe"


def map_label_to_numeric(label: str) -> int:
    """Inverse of map_numeric_to_label."""
    label = (label or "").strip().lower()
    if label in ("phising", "phishing"):
        return -1
    if label in ("safe", "legitimate", "legit"):
        return 1
    raise ValueError(f"Unknown label '{label}'")


def create_directories(paths: list[Union[str, Path]]) -> list[Path]:
    """Create a list of directories (used at project init)."""
    created = []
    for p in paths:
        created.append(ensure_dir(p))
    return created
