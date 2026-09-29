"""
src/evaluation/metrics.py
Comprehensive evaluation: accuracy, precision, recall, F1, ROC-AUC, PR-AUC,
MCC, log-loss, confusion matrix. Persists a per-model JSON report and
generates the model comparison table.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, log_loss,
    matthews_corrcoef, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, average_precision_score,
)

from src.utils import (
    EvaluationError, get_logger, to_abs, write_json,
)

log = get_logger(__name__)


@dataclass
class EvalResult:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    mcc: float
    log_loss: float
    recall_phishing: float = 0.0   # recall on class -1 (priority metric)
    confusion_matrix: List[List[int]] = field(default_factory=list)
    fit_time_seconds: float = 0.0
    n_estimators: Optional[int] = None


def _predict_proba_safe(model, X: pd.DataFrame) -> Optional[np.ndarray]:
    """Return Nx2 probability matrix or None."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 1:
            return np.hstack([1 - proba, proba])
        return proba
    return None


def _binary_predict(model, X: pd.DataFrame, is_xgb_like: bool
                    ) -> np.ndarray:
    """Predict in {-1, 1}."""
    raw = model.predict(X)
    arr = np.asarray(raw)
    # Map {0, 1} -> {-1, 1} for boosting libraries
    if is_xgb_like and set(np.unique(arr)).issubset({0, 1}):
        return np.where(arr == 1, 1, -1)
    return arr


def evaluate_model(model, X: pd.DataFrame, y_true: pd.Series,
                   name: str, is_xgb_like: bool = False,
                   fit_time: float = 0.0) -> EvalResult:
    """Evaluate a single model on the held-out test set.

    Args:
        is_xgb_like: True for XGBoost / LightGBM / CatBoost (which train
            on {0, 1} and need remapping back to {-1, 1}).
    """
    try:
        y_pred = _binary_predict(model, X, is_xgb_like)
        # ROC-AUC / PR-AUC need probability of the phishing class
        proba = _predict_proba_safe(model, X)
        if is_xgb_like and proba is not None:
            # XGBoost classes are [0, 1] (0 = phishing after remap)
            # so the phishing column is the one labeled 0 -> index 0
            # We want P(phishing) where phishing == -1
            # Order classes via model.classes_ if available
            cls = getattr(model, "classes_", None)
            if cls is not None and -1 in list(cls):
                phish_idx = list(cls).index(-1)
                p_phish = proba[:, phish_idx]
            elif cls is not None and 0 in list(cls):
                phish_idx = list(cls).index(0)
                p_phish = proba[:, phish_idx]
            else:
                p_phish = proba[:, 0]
        elif proba is not None:
            cls = getattr(model, "classes_", None)
            if cls is not None and -1 in list(cls):
                phish_idx = list(cls).index(-1)
                p_phish = proba[:, phish_idx]
            else:
                p_phish = proba[:, 0] if proba.shape[1] >= 2 else proba[:, 0]
        else:
            # Fall back to hard predictions -> no useful probability
            p_phish = (y_pred == -1).astype(float)

        # ---- Metrics ----
        # Coerce y_true to {-1, 1}
        y_t = np.where(y_true.values == -1, -1, 1)
        y_p = np.where(y_pred == -1, -1, 1)
        # For sklearn metrics that expect {0, 1}, remap
        y_t01 = np.where(y_t == -1, 1, 0)  # phishing=1
        y_p01 = np.where(y_p == -1, 1, 0)

        acc = float(accuracy_score(y_t01, y_p01))
        prec = float(precision_score(y_t01, y_p01, zero_division=0))
        rec = float(recall_score(y_t01, y_p01, zero_division=0))
        f1 = float(f1_score(y_t01, y_p01, zero_division=0))
        mcc = float(matthews_corrcoef(y_t01, y_p01))
        try:
            ll = float(log_loss(y_t01, np.clip(
                np.vstack([1 - p_phish, p_phish]).T, 1e-9, 1 - 1e-9)))
        except Exception:
            ll = float("nan")
        try:
            auc = float(roc_auc_score(y_t01, p_phish))
        except Exception:
            auc = float("nan")
        try:
            prauc = float(average_precision_score(y_t01, p_phish))
        except Exception:
            prauc = float("nan")
        cm = confusion_matrix(y_t01, y_p01, labels=[1, 0]).tolist()

        n_estimators = None
        for attr in ("n_estimators", "iterations", "max_iter", "n_iter"):
            if hasattr(model, attr):
                try:
                    val = int(getattr(model, attr))
                    if val > 0:
                        n_estimators = val
                        break
                except Exception:
                    continue

        return EvalResult(
            name=name, accuracy=acc, precision=prec, recall=rec,
            f1=f1, roc_auc=auc, pr_auc=prauc, mcc=mcc, log_loss=ll,
            recall_phishing=rec, confusion_matrix=cm,
            fit_time_seconds=fit_time, n_estimators=n_estimators,
        )
    except Exception as exc:
        raise EvaluationError(
            f"Failed to evaluate {name}: {exc}") from exc


def evaluate_all(fitted_results: Dict[str, object], X_te: pd.DataFrame,
                 y_te: pd.Series, out_dir: str | Path,
                 autogluon_predictor=None) -> Tuple[List[EvalResult],
                                                    pd.DataFrame]:
    """Evaluate all trained models and write the comparison table."""
    try:
        out = to_abs(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        eval_results: List[EvalResult] = []
        xgb_like = {"XGBoost", "LightGBM", "CatBoost"}
        for name, res in fitted_results.items():
            try:
                model = res.model if hasattr(res, "model") else res
                fit_t = res.fit_time_seconds if hasattr(
                    res, "fit_time_seconds") else 0.0
                ev = evaluate_model(model, X_te, y_te, name,
                                    is_xgb_like=name in xgb_like,
                                    fit_time=fit_t)
                eval_results.append(ev)
            except Exception as exc:
                log.error("Eval failed for %s: %s", name, exc)

        # AutoGluon
        if autogluon_predictor is not None:
            try:
                ag_pred = autogluon_predictor.predict(X_te)
                ag_pred = np.where(pd.Series(ag_pred).astype(int) == 1, 1, -1)
                # Remap AG label scheme: AG trains on {-1, 1} so labels are
                # already in correct domain; but if it produces 0/1 we map back.
                y_t01 = np.where(y_te.values == -1, 1, 0)
                y_p01 = np.where(ag_pred == -1, 1, 0)
                try:
                    p_phish = autogluon_predictor.predict_proba(X_te)
                    cols = list(p_phish.columns) if hasattr(
                        p_phish, "columns") else list(range(p_phish.shape[1]))
                    if -1 in cols or "-1" in [str(c) for c in cols]:
                        col = -1 if -1 in cols else "-1"
                        p = p_phish[col].values if hasattr(p_phish, "loc") \
                            else p_phish[:, cols.index(-1) if -1 in cols else
                                         [str(c) for c in cols].index("-1")]
                    else:
                        # Use first column
                        p = p_phish.values[:, 0] if hasattr(p_phish, "values") \
                            else np.asarray(p_phish)[:, 0]
                except Exception:
                    p = (ag_pred == -1).astype(float)
                acc = float(accuracy_score(y_t01, y_p01))
                prec = float(precision_score(y_t01, y_p01, zero_division=0))
                rec = float(recall_score(y_t01, y_p01, zero_division=0))
                f1 = float(f1_score(y_t01, y_p01, zero_division=0))
                mcc = float(matthews_corrcoef(y_t01, y_p01))
                try:
                    auc = float(roc_auc_score(y_t01, p))
                except Exception:
                    auc = float("nan")
                try:
                    prauc = float(average_precision_score(y_t01, p))
                except Exception:
                    prauc = float("nan")
                try:
                    ll = float(log_loss(y_t01, np.clip(
                        np.vstack([1 - p, p]).T, 1e-9, 1 - 1e-9)))
                except Exception:
                    ll = float("nan")
                cm = confusion_matrix(y_t01, y_p01, labels=[1, 0]).tolist()
                eval_results.append(EvalResult(
                    name="AutoGluon", accuracy=acc, precision=prec,
                    recall=rec, f1=f1, roc_auc=auc, pr_auc=prauc, mcc=mcc,
                    log_loss=ll, recall_phishing=rec,
                    confusion_matrix=cm))
            except Exception as exc:
                log.error("AutoGluon eval failed: %s", exc)

        # Build comparison DataFrame
        rows = [{
            "model": e.name, "accuracy": e.accuracy, "precision": e.precision,
            "recall": e.recall, "f1": e.f1, "roc_auc": e.roc_auc,
            "pr_auc": e.pr_auc, "mcc": e.mcc, "log_loss": e.log_loss,
            "recall_phishing": e.recall_phishing,
            "fit_time_s": round(e.fit_time_seconds, 2),
        } for e in eval_results]
        cmp_df = pd.DataFrame(rows).sort_values(
            by=["recall_phishing", "roc_auc"], ascending=[False, False])
        cmp_df.to_csv(out / "model_comparison.csv", index=False)
        write_json(out / "eval_results.json", {
            "results": [{
                "name": e.name, "accuracy": e.accuracy,
                "precision": e.precision, "recall": e.recall,
                "f1": e.f1, "roc_auc": e.roc_auc, "pr_auc": e.pr_auc,
                "mcc": e.mcc, "log_loss": e.log_loss,
                "recall_phishing": e.recall_phishing,
                "confusion_matrix": e.confusion_matrix,
                "fit_time_seconds": e.fit_time_seconds,
                "n_estimators": e.n_estimators,
            } for e in eval_results]
        })

        log.info("Evaluation complete. Models evaluated: %d", len(eval_results))
        if not cmp_df.empty:
            log.info("Best by recall_phishing: %s (%.4f)",
                     cmp_df.iloc[0]["model"],
                     cmp_df.iloc[0]["recall_phishing"])
        return eval_results, cmp_df
    except Exception as exc:
        raise EvaluationError(f"evaluate_all failed: {exc}") from exc


def select_best(eval_results: List[EvalResult]) -> Optional[EvalResult]:
    """Pick the best model. Primary: recall_phishing (recall on phishing).
    Tiebreak: roc_auc."""
    if not eval_results:
        return None
    return max(eval_results,
               key=lambda e: (e.recall_phishing, e.roc_auc))
