"""src.models package - supervised classifiers + ensembles + AutoGluon."""
from src.models.trainer import (
    ClassificationTrainer, ModelResult, tune_xgboost,
)

__all__ = [
    "ClassificationTrainer", "ModelResult", "tune_xgboost",
]
