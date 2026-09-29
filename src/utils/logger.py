"""
src/utils/logger.py
Configures structured logging from configs/logging.yaml.
"""
from __future__ import annotations

import logging
import logging.config
import logging.handlers
from pathlib import Path
from typing import Optional

import yaml


_DEFAULT_CONFIG_PATH = "configs/logging.yaml"
_DEFAULT_LOG_DIR = "artifacts/reports/logs"
_CONFIGURED = False


def _ensure_log_dir() -> None:
    Path(_DEFAULT_LOG_DIR).mkdir(parents=True, exist_ok=True)


def setup_logging(config_path: Optional[str] = None,
                 default_level: int = logging.INFO) -> None:
    """Configure root logging from a YAML file. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    _ensure_log_dir()
    cfg_path = Path(config_path or _DEFAULT_CONFIG_PATH)
    if cfg_path.exists():
        try:
            with cfg_path.open("r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh)
            logging.config.dictConfig(cfg)
        except Exception as exc:  # pragma: no cover
            logging.basicConfig(level=default_level)
            logging.getLogger(__name__).warning(
                "Failed to load logging config %s: %s; using basicConfig.",
                cfg_path, exc,
            )
    else:
        logging.basicConfig(level=default_level,
                            format="[%(asctime)s] [%(levelname)-8s] "
                                   "[%(name)-20s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a project-namespaced logger."""
    if not _CONFIGURED:
        setup_logging()
    if not name.startswith("phishing_detection"):
        name = f"phishing_detection.{name.lstrip('.')}"
    return logging.getLogger(name)
