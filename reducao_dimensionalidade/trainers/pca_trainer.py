from __future__ import annotations

from dataclasses import dataclass

from reducao_dimensionalidade.config import ExperimentConfig
from reducao_dimensionalidade.feature_pipeline import build_feature_preprocessor
from reducao_dimensionalidade.model_factory import build_balanced_logistic_regression
from reducao_dimensionalidade.trainers.base import BaseTrainer
from reducao_dimensionalidade.training_types import TrainingType


@dataclass(frozen=True)
class PCATrainer(BaseTrainer):
    """Treino com reducao por componentes principais."""

    def __init__(self) -> None:
        object.__setattr__(self, "training_type", TrainingType.PCA)
        object.__setattr__(self, "name", "pca_logistic_regression")

    def build_pipeline(self, X, k: int, config: ExperimentConfig):
        from sklearn.decomposition import PCA
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.pipeline import Pipeline

        return Pipeline(
            steps=[
                ("features", build_feature_preprocessor(X)),
                ("variance", VarianceThreshold()),
                ("pca", PCA(n_components=k, random_state=config.random_state)),
                ("model", build_balanced_logistic_regression(config)),
            ]
        )
