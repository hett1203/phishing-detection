"""
src/models/trainer.py
Supervised classification trainer: trains all base models, ensembles, and
AutoGluon TabularPredictor. Uses Optuna for hyperparameter tuning of the
top models.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, RandomForestClassifier,
    HistGradientBoostingClassifier, VotingClassifier, StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.utils import (
    ClassificationError, get_logger, save_joblib, to_abs, write_json,
)

log = get_logger(__name__)


def _xgb():
    from xgboost import XGBClassifier
    return XGBClassifier


def _lgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier


def _cat():
    from catboost import CatBoostClassifier
    return CatBoostClassifier


@dataclass
class ModelResult:
    name: str
    model: Any
    fit_time_seconds: float
    params: Dict = field(default_factory=dict)


class ClassificationTrainer:
    """Trains every supervised model defined in params.yaml."""

    def __init__(self, params_cfg, model_cfg) -> None:
        self.params = params_cfg.classification
        self.model_cfg = model_cfg
        self.random_state = self.params.random_state

    # ---------------- Model factories ---------------- #
    def _build_rf(self) -> RandomForestClassifier:
        p = self.params.models.RandomForest.params
        return RandomForestClassifier(
            n_estimators=p.n_estimators, max_depth=p.max_depth,
            max_features=p.max_features, min_samples_split=p.min_samples_split,
            min_samples_leaf=p.min_samples_leaf,
            class_weight=p.class_weight, random_state=p.random_state,
            n_jobs=p.n_jobs)

    def _build_et(self) -> ExtraTreesClassifier:
        p = self.params.models.ExtraTrees.params
        return ExtraTreesClassifier(
            n_estimators=p.n_estimators, max_depth=p.max_depth,
            max_features=p.max_features, min_samples_split=p.min_samples_split,
            min_samples_leaf=p.min_samples_leaf,
            class_weight=p.class_weight, random_state=p.random_state,
            n_jobs=p.n_jobs)

    def _build_xgb(self) -> Any:
        p = self.params.models.XGBoost.params
        # XGBoost expects labels in {0, 1}; remap outside.
        return _xgb()(
            n_estimators=p.n_estimators, max_depth=p.max_depth,
            learning_rate=p.learning_rate, subsample=p.subsample,
            colsample_bytree=p.colsample_bytree,
            min_child_weight=p.min_child_weight, gamma=p.gamma,
            reg_alpha=p.reg_alpha, reg_lambda=p.reg_lambda,
            random_state=p.random_state, n_jobs=p.n_jobs,
            tree_method=p.tree_method, eval_metric=p.eval_metric,
            use_label_encoder=False)

    def _build_lgbm(self) -> Any:
        p = self.params.models.LightGBM.params
        return _lgbm()(
            n_estimators=p.n_estimators, max_depth=p.max_depth,
            num_leaves=p.num_leaves, learning_rate=p.learning_rate,
            subsample=p.subsample, colsample_bytree=p.colsample_bytree,
            min_child_samples=p.min_child_samples,
            reg_alpha=p.reg_alpha, reg_lambda=p.reg_lambda,
            random_state=p.random_state, n_jobs=p.n_jobs,
            verbose=p.verbose)

    def _build_cat(self) -> Any:
        p = self.params.models.CatBoost.params
        return _cat()(
            iterations=p.iterations, depth=p.depth,
            learning_rate=p.learning_rate, l2_leaf_reg=p.l2_leaf_reg,
            random_state=p.random_state, verbose=p.verbose,
            allow_writing_files=p.allow_writing_files)

    def _build_hgb(self) -> HistGradientBoostingClassifier:
        p = self.params.models.HistGradientBoosting.params
        return HistGradientBoostingClassifier(
            max_iter=p.max_iter, max_depth=p.max_depth,
            learning_rate=p.learning_rate,
            l2_regularization=p.l2_regularization,
            random_state=p.random_state)

    def _build_gb(self) -> GradientBoostingClassifier:
        p = self.params.models.GradientBoosting.params
        return GradientBoostingClassifier(
            n_estimators=p.n_estimators, max_depth=p.max_depth,
            learning_rate=p.learning_rate, subsample=p.subsample,
            random_state=p.random_state)

    def _build_ada(self) -> AdaBoostClassifier:
        p = self.params.models.AdaBoost.params
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=p.n_estimators, learning_rate=p.learning_rate,
            random_state=p.random_state)

    def _build_bagging(self) -> BaggingClassifier:
        p = self.params.models.Bagging.params
        return BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=6),
            n_estimators=p.n_estimators, max_samples=p.max_samples,
            max_features=p.max_features, random_state=p.random_state,
            n_jobs=p.n_jobs)

    def _build_voting(self, fitted_base: Dict[str, Any]) -> VotingClassifier:
        v = self.params.ensembles.voting
        voters = []
        for name in v.voters:
            if name in fitted_base:
                voters.append((name, fitted_base[name]))
            else:
                voters.append((name, self._factory(name)()))
        return VotingClassifier(
            estimators=voters,
            voting=v.voting_type,
            weights=v.weights,
            n_jobs=-1)

    def _build_stacking(self, fitted_base: Dict[str, Any]
                       ) -> StackingClassifier:
        s = self.params.ensembles.stacking
        base = []
        for name in s.base_models:
            if name in fitted_base:
                base.append((name, fitted_base[name]))
            else:
                base.append((name, self._factory(name)()))
        meta_cls = LogisticRegression(max_iter=1000, n_jobs=-1)
        return StackingClassifier(
            estimators=base, final_estimator=meta_cls,
            cv=s.cv_folds, n_jobs=-1)

    def _factory(self, name: str):
        mapping = {
            "RandomForest": self._build_rf,
            "ExtraTrees": self._build_et,
            "XGBoost": self._build_xgb,
            "LightGBM": self._build_lgbm,
            "CatBoost": self._build_cat,
            "HistGradientBoosting": self._build_hgb,
            "GradientBoosting": self._build_gb,
            "AdaBoost": self._build_ada,
            "Bagging": self._build_bagging,
        }
        return mapping[name]

    # ---------------- Training loop ---------------- #
    def train_all(self, X_tr: pd.DataFrame, y_tr: pd.Series,
                  out_dir: str | Path) -> Dict[str, ModelResult]:
        try:
            out = to_abs(out_dir)
            out.mkdir(parents=True, exist_ok=True)

            results: Dict[str, ModelResult] = {}
            fitted_base: Dict[str, Any] = {}

            # Map labels to {0, 1} for XGBoost/LightGBM
            y_xgb = (y_tr + 1) // 2   # -1 -> 0, 1 -> 1
            # CatBoost accepts {0, 1} too
            y_cb = y_xgb.copy()

            enabled = {
                name: getattr(self.params.models, name).enabled
                for name in self.params.models
            }
            for name, enabled_flag in enabled.items():
                if not enabled_flag:
                    continue
                log.info("Training %s...", name)
                t0 = time.time()
                model = self._factory(name)()
                try:
                    if name in {"XGBoost", "LightGBM", "CatBoost"}:
                        if name == "XGBoost":
                            model.fit(X_tr, y_xgb)
                        elif name == "LightGBM":
                            model.fit(X_tr, y_xgb)
                        else:
                            model.fit(X_tr, y_cb)
                    else:
                        model.fit(X_tr, y_tr)
                    fit_time = time.time() - t0
                    fitted_base[name] = model
                    save_joblib(
                        model, out / self.model_cfg.models[name].file,
                        compress=3)
                    results[name] = ModelResult(
                        name=name, model=model, fit_time_seconds=fit_time,
                        params=model.get_params() if hasattr(model, "get_params") else {})
                    log.info("  %s fit in %.1fs", name, fit_time)
                except Exception as exc:
                    log.warning("Training %s failed: %s", name, exc)

            # ---- Ensembles ----
            if self.params.ensembles.voting.enabled:
                log.info("Training VotingEnsemble...")
                t0 = time.time()
                voting = self._build_voting(fitted_base)
                voting.fit(X_tr, y_tr)
                save_joblib(voting, out / "voting_ensemble.joblib", compress=3)
                results["VotingEnsemble"] = ModelResult(
                    name="VotingEnsemble", model=voting,
                    fit_time_seconds=time.time() - t0)
                log.info("VotingEnsemble fit in %.1fs",
                         results["VotingEnsemble"].fit_time_seconds)

            if self.params.ensembles.stacking.enabled:
                log.info("Training StackingEnsemble...")
                t0 = time.time()
                stack = self._build_stacking(fitted_base)
                stack.fit(X_tr, y_tr)
                save_joblib(stack, out / "stacking_ensemble.joblib", compress=3)
                results["StackingEnsemble"] = ModelResult(
                    name="StackingEnsemble", model=stack,
                    fit_time_seconds=time.time() - t0)
                log.info("StackingEnsemble fit in %.1fs",
                         results["StackingEnsemble"].fit_time_seconds)

            return results
        except Exception as exc:
            raise ClassificationError(
                f"Training failed: {exc}") from exc

    # ---------------- AutoGluon ---------------- #
    def train_autogluon(self, X_tr: pd.DataFrame, y_tr: pd.Series,
                        out_dir: str | Path) -> Optional[Any]:
        try:
            from autogluon.tabular import TabularPredictor
        except ImportError:
            log.warning("autogluon not installed; skipping AutoGluon training")
            return None
        try:
            out = to_abs(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            log.info("Training AutoGluon TabularPredictor (presets=%s, "
                     "time_limit=%ds)...", self.params.autogluon.presets,
                     self.params.autogluon.time_limit_seconds)
            df = X_tr.copy()
            df[self._target_label_name()] = y_tr.values
            t0 = time.time()
            predictor = TabularPredictor(
                label=self._target_label_name(),
                path=str(out),
                eval_metric=self.params.autogluon.eval_metric,
            ).fit(
                train_data=df,
                time_limit=int(self.params.autogluon.time_limit_seconds),
                presets=self.params.autogluon.presets,
                verbosity=1,
            )
            log.info("AutoGluon fit in %.1fs", time.time() - t0)
            return predictor
        except Exception as exc:
            log.error("AutoGluon training failed: %s", exc)
            return None

    @staticmethod
    def _target_label_name() -> str:
        return "Result"


# ---------------- Optuna tuning ---------------- #
def tune_xgboost(X: pd.DataFrame, y: pd.Series, n_trials: int = 30,
                 timeout: int = 300,
                 random_state: int = 42) -> Dict:
    """Optuna hyperparameter tuning for XGBoost on the phishing task."""
    try:
        import optuna
        from xgboost import XGBClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score
    except ImportError:
        log.warning("optuna/xgboost not available; returning default params")
        return {}

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    skf = StratifiedKFold(n_splits=5, shuffle=True,
                          random_state=random_state)
    y01 = (y + 1) // 2

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0),
            "random_state": random_state, "n_jobs": -1,
            "tree_method": "hist", "eval_metric": "logloss",
            "use_label_encoder": False,
        }
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X, y01, cv=skf,
                                 scoring="roc_auc", n_jobs=-1)
        return float(scores.mean())

    study = optuna.create_study(direction="maximize",
                               sampler=optuna.samplers.TPESampler(
                                   seed=random_state))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(objective, n_trials=n_trials, timeout=timeout,
                       show_progress_bar=False)
    log.info("Optuna XGB best AUC=%.4f params=%s",
             study.best_value, study.best_params)
    return dict(study.best_params)
