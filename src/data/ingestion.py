"""
src/data/ingestion.py
Data ingestion layer: loads the raw UCI Phishing CSVs into pandas DataFrames.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.utils import (
    DataIngestionError, get_logger, to_abs,
)

log = get_logger(__name__)


@dataclass
class IngestionArtifact:
    """Holds ingested datasets."""
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_path: Path
    test_path: Path
    n_rows_train: int
    n_rows_test: int
    n_features: int


class DataIngestion:
    """Loads raw train/test CSVs from disk into pandas.

    Args:
        raw_data_dir: Directory containing the three reference CSVs.
        train_file: Filename of the labeled training CSV.
        test_file: Filename of the unlabeled batch-inference CSV.
    """

    def __init__(self, raw_data_dir: str | Path,
                 train_file: str = "phising_08012020_120000.csv",
                 test_file: str = "phisingtest.csv") -> None:
        self.raw_dir = to_abs(raw_data_dir)
        self.train_path = self.raw_dir / train_file
        self.test_path = self.raw_dir / test_file
        if not self.train_path.exists():
            raise DataIngestionError(f"Train CSV not found: {self.train_path}")
        if not self.test_path.exists():
            raise DataIngestionError(f"Test CSV not found: {self.test_path}")

    def ingest(self) -> IngestionArtifact:
        log.info("Ingesting train CSV from %s", self.train_path)
        train_df = pd.read_csv(self.train_path)
        log.info("Train shape: %s", train_df.shape)

        log.info("Ingesting test CSV from %s", self.test_path)
        test_df = pd.read_csv(self.test_path)
        log.info("Test shape: %s", test_df.shape)

        n_features = train_df.shape[1] - 1  # exclude target
        log.info("Ingestion complete. Train rows=%d, Test rows=%d, features=%d",
                 train_df.shape[0], test_df.shape[0], n_features)

        return IngestionArtifact(
            train_df=train_df,
            test_df=test_df,
            train_path=self.train_path,
            test_path=self.test_path,
            n_rows_train=train_df.shape[0],
            n_rows_test=test_df.shape[0],
            n_features=n_features,
        )
