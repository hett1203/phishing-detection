"""
src/utils/config.py
Single source of truth for project configuration. Loads all YAML files into
easily-accessible attributes. Re-exported by `src.utils`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from src.utils.common import read_yaml, to_abs
from src.utils.exceptions import ConfigError


@dataclass
class ProjectConfig:
    """Holds all loaded YAML configuration objects."""
    config: Any            # config.yaml
    params: Any            # params.yaml
    schema: Any            # schema.yaml
    prediction_schema: Any # prediction_schema.yaml
    model: Any             # model.yaml
    logging: Any           # logging.yaml

    raw_data_dir: Path = field(default_factory=Path)
    validated_data_dir: Path = field(default_factory=Path)
    processed_data_dir: Path = field(default_factory=Path)
    features_dir: Path = field(default_factory=Path)
    clusters_dir: Path = field(default_factory=Path)
    models_base_dir: Path = field(default_factory=Path)
    trained_models_dir: Path = field(default_factory=Path)
    autogluon_dir: Path = field(default_factory=Path)
    preprocessing_dir: Path = field(default_factory=Path)
    evaluations_dir: Path = field(default_factory=Path)
    reports_dir: Path = field(default_factory=Path)
    figures_dir: Path = field(default_factory=Path)
    predictions_dir: Path = field(default_factory=Path)
    shap_dir: Path = field(default_factory=Path)

    @classmethod
    def load(cls, configs_dir: str = "configs") -> "ProjectConfig":
        cd = Path(to_abs(configs_dir))
        if not cd.exists():
            raise ConfigError(f"configs directory not found: {cd}")
        cfg_path = cd / "config.yaml"
        if not cfg_path.exists():
            raise ConfigError(f"Missing required config.yaml in {cd}")

        # config.yaml defines all path constants
        config = read_yaml(cfg_path)
        # Resolve every path against project root
        cfg = config
        inst = cls(
            config=cfg,
            params=read_yaml(cd / "params.yaml"),
            schema=read_yaml(cd / "schema.yaml"),
            prediction_schema=read_yaml(cd / "prediction_schema.yaml"),
            model=read_yaml(cd / "model.yaml"),
            logging=read_yaml(cd / "logging.yaml"),
        )
        inst._resolve_paths()
        return inst

    def _resolve_paths(self) -> None:
        paths = self.config.paths
        self.raw_data_dir = to_abs(paths.raw_data_dir)
        self.validated_data_dir = to_abs(paths.validated_data_dir)
        self.processed_data_dir = to_abs(paths.processed_data_dir)
        self.features_dir = to_abs(paths.features_dir)
        self.clusters_dir = to_abs(paths.clusters_dir)
        self.models_base_dir = to_abs(paths.models_base_dir)
        self.trained_models_dir = to_abs(paths.trained_models_dir)
        self.autogluon_dir = to_abs(paths.autogluon_dir)
        self.preprocessing_dir = to_abs(paths.preprocessing_dir)
        self.evaluations_dir = to_abs(paths.evaluations_dir)
        self.reports_dir = to_abs(paths.reports_dir)
        self.figures_dir = to_abs(paths.figures_dir)
        self.predictions_dir = to_abs(paths.predictions_dir)
        self.shap_dir = to_abs(paths.shap_dir)

    @property
    def features(self) -> list[str]:
        return list(self.schema.features)

    @property
    def target_name(self) -> str:
        return str(self.schema.target.name)

    @property
    def random_state(self) -> int:
        return int(self.config.random_state)

    def create_all_dirs(self) -> None:
        for d in [
            self.raw_data_dir,
            self.validated_data_dir,
            self.processed_data_dir,
            self.features_dir,
            self.clusters_dir,
            self.models_base_dir,
            self.trained_models_dir,
            self.autogluon_dir,
            self.preprocessing_dir,
            self.evaluations_dir,
            self.reports_dir,
            self.figures_dir,
            self.predictions_dir,
            self.shap_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


# Module-level singleton (lazy-initialized)
_CONFIG: ProjectConfig | None = None


def get_config(force_reload: bool = False) -> ProjectConfig:
    global _CONFIG
    if _CONFIG is None or force_reload:
        _CONFIG = ProjectConfig.load()
    return _CONFIG
