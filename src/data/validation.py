"""
src/data/validation.py
Data validation layer: enforces schema.yaml (every feature ∈ {-1, 0, 1} and
Result ∈ {-1, 1}, no missing values, expected row counts).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.utils import (
    ValidationError, get_logger, to_abs, write_json,
)

log = get_logger(__name__)


@dataclass
class ValidationReport:
    is_valid: bool
    n_rows_train: int
    n_rows_test: int
    n_features: int
    missing_train: int
    missing_test: int
    schema_violations: List[str] = field(default_factory=list)
    feature_value_violations: Dict[str, int] = field(default_factory=dict)


class DataValidation:
    """Validates raw DataFrames against schema.yaml."""

    def __init__(self, schema_cfg, features: List[str],
                 target_name: str) -> None:
        self.schema_cfg = schema_cfg
        self.features = list(features)
        self.target_name = target_name
        self.feature_allowed = set(schema_cfg.feature_allowed_values)
        self.target_allowed = set(schema_cfg.target_allowed_values)
        self.expected_n_train = int(getattr(schema_cfg, "expected_n_rows_train", 0) or 0)

    def validate(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                out_dir: str | Path) -> ValidationReport:
        out = to_abs(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        report = ValidationReport(
            is_valid=True,
            n_rows_train=len(train_df),
            n_rows_test=len(test_df),
            n_features=len(self.features),
            missing_train=int(train_df.isna().sum().sum()),
            missing_test=int(test_df.isna().sum().sum()),
        )

        # ---- column presence ----
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        expected_cols = set(self.features) | {self.target_name}

        missing_in_train = expected_cols - train_cols
        if missing_in_train:
            report.is_valid = False
            report.schema_violations.append(
                f"Missing columns in train: {sorted(missing_in_train)}")

        missing_in_test = set(self.features) - test_cols
        if missing_in_test:
            report.is_valid = False
            report.schema_violations.append(
                f"Missing columns in test: {sorted(missing_in_test)}")

        # ---- row count ----
        if self.expected_n_train and report.n_rows_train != self.expected_n_train:
            report.is_valid = False
            report.schema_violations.append(
                f"Train rows {report.n_rows_train} != expected "
                f"{self.expected_n_train}")

        # ---- missing values ----
        if report.missing_train > 0 or report.missing_test > 0:
            if not self.schema_cfg.allow_missing:
                report.is_valid = False
                report.schema_violations.append(
                    f"Missing values present (train={report.missing_train}, "
                    f"test={report.missing_test})")

        # ---- feature value domain ----
        for col in self.features:
            if col in train_df.columns:
                unique_vals = set(pd.unique(train_df[col].dropna()))
                bad = unique_vals - self.feature_allowed
                if bad:
                    report.is_valid = False
                    report.feature_value_violations[col] = len(bad)
                    report.schema_violations.append(
                        f"Feature {col} has illegal values {sorted(bad)}; "
                        f"expected subset of {sorted(self.feature_allowed)}")
            if col in test_df.columns:
                unique_vals = set(pd.unique(test_df[col].dropna()))
                bad = unique_vals - self.feature_allowed
                if bad:
                    report.is_valid = False
                    report.feature_value_violations[col] = len(bad)
                    report.schema_violations.append(
                        f"Test feature {col} has illegal values {sorted(bad)}")

        # ---- target domain ----
        if self.target_name in train_df.columns:
            unique_target = set(pd.unique(train_df[self.target_name].dropna()))
            bad_t = unique_target - self.target_allowed
            if bad_t:
                report.is_valid = False
                report.schema_violations.append(
                    f"Target {self.target_name} has illegal values "
                    f"{sorted(bad_t)}; expected {sorted(self.target_allowed)}")

        # ---- class balance report ----
        class_balance = {}
        if self.target_name in train_df.columns:
            vc = train_df[self.target_name].value_counts().to_dict()
            class_balance = {int(k): int(v) for k, v in vc.items()}

        # Write report
        report_path = out / "validation_report.json"
        write_json(report_path, {
            "is_valid": report.is_valid,
            "n_rows_train": report.n_rows_train,
            "n_rows_test": report.n_rows_test,
            "n_features": report.n_features,
            "missing_train": report.missing_train,
            "missing_test": report.missing_test,
            "class_balance": class_balance,
            "schema_violations": report.schema_violations,
            "feature_value_violations": report.feature_value_violations,
        })

        if report.is_valid:
            log.info("Validation PASSED. Train=%d Test=%d Features=%d "
                     "Class balance=%s",
                     report.n_rows_train, report.n_rows_test,
                     report.n_features, class_balance)
        else:
            log.error("Validation FAILED: %s", report.schema_violations)
            raise ValidationError(
                f"Validation failed: {report.schema_violations}")

        return report
