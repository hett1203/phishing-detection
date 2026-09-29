"""
src/utils/exceptions.py
Custom exception hierarchy for the Phishing Detection project.
"""
from __future__ import annotations


class PhishingDetectionException(Exception):
    """Base exception for all project errors."""

    def __init__(self, message: str = "Phishing Detection error") -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.message}"


class ConfigError(PhishingDetectionException):
    """Raised when a YAML config is missing or malformed."""


class DataIngestionError(PhishingDetectionException):
    """Raised when the train/test CSVs cannot be read."""


class ValidationError(PhishingDetectionException):
    """Raised when the dataset violates schema.yaml."""


class TransformationError(PhishingDetectionException):
    """Raised when feature-engineering or PCA fails."""


class FeatureSelectionError(PhishingDetectionException):
    """Raised when feature selection fails."""


class ClusteringError(PhishingDetectionException):
    """Raised when an unsupervised clustering algorithm fails."""


class ClassificationError(PhishingDetectionException):
    """Raised when a supervised model fails to train or predict."""


class EvaluationError(PhishingDetectionException):
    """Raised when metric computation fails."""


class PredictionError(PhishingDetectionException):
    """Raised when the single/batch prediction pipeline fails."""


class ExplainabilityError(PhishingDetectionException):
    """Raised when SHAP/LIME explanation generation fails."""


class ModelPersistenceError(PhishingDetectionException):
    """Raised when model saving / loading fails."""


class ArtifactMissingError(PhishingDetectionException):
    """Raised when a required trained artifact is missing on disk."""
