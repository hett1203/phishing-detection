"""src/data package - data ingestion and validation."""
from src.data.ingestion import DataIngestion, IngestionArtifact
from src.data.validation import DataValidation, ValidationReport

__all__ = [
    "DataIngestion", "IngestionArtifact",
    "DataValidation", "ValidationReport",
]
