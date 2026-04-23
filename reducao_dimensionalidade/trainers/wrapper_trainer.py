from __future__ import annotations

from dataclasses import dataclass

from reducao_dimensionalidade.config import ExperimentConfig
from reducao_dimensionalidade.feature_pipeline import build_feature_preprocessor
from reducao_dimensionalidade.model_factory import build_balanced_logistic_regression
from reducao_dimensionalidade.trainers.base import BaseTrainer
from reducao_dimensionalidade.training_types import TrainingType


@dataclass(frozen=True)
class WrapperTrainer(BaseTrainer):
    """Treino com selecao wrapper via RFE."""

    def __init__(self) -> None:
        object.__setattr__(self, "training_type", TrainingType.WRAPPER)
        object.__setattr__(self, "name", "rfe_logistic_regression")

    def build_pipeline(self, X, k: int, config: ExperimentConfig):
        from sklearn.feature_selection import RFE
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.pipeline import Pipeline

        estimator = build_balanced_logistic_regression(config)

        return Pipeline(
            steps=[
                ("features", build_feature_preprocessor(X)),
                ("variance", VarianceThreshold()),
                ("selector", RFE(estimator=estimator, n_features_to_select=k, step=0.1)),
                ("model", build_balanced_logistic_regression(config)),
            ]
        )
