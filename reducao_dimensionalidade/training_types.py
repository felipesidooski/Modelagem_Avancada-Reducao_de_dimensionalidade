from __future__ import annotations

from enum import Enum


class TrainingType(str, Enum):
    """Tipos de experimentos disponiveis no projeto."""

    BASELINE = "baseline"
    FILTER = "filter"
    WRAPPER = "wrapper"
    EMBEDDED = "embedded"
    PCA = "pca"

