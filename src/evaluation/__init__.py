"""src.evaluation package - metrics & model comparison."""
from src.evaluation.metrics import (
    EvalResult, evaluate_model, evaluate_all, select_best,
)

__all__ = [
    "EvalResult", "evaluate_model", "evaluate_all", "select_best",
]
