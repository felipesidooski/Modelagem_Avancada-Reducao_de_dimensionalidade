from __future__ import annotations

from dataclasses import dataclass

from reducao_dimensionalidade.config import ExperimentConfig
from reducao_dimensionalidade.feature_pipeline import build_feature_preprocessor
from reducao_dimensionalidade.model_factory import build_balanced_logistic_regression
from reducao_dimensionalidade.trainers.base import BaseTrainer
from reducao_dimensionalidade.training_types import TrainingType


@dataclass(frozen=True)
class EmbeddedTrainer(BaseTrainer):
    """Treino com selecao embedded via L1 ou arvores."""

    strategy: str = "l1"

    def __init__(self, strategy: str = "l1") -> None:
        object.__setattr__(self, "training_type", TrainingType.EMBEDDED)
        object.__setattr__(self, "name", f"embedded_{strategy}")
        object.__setattr__(self, "strategy", strategy)

    def build_pipeline(self, X, k: int, config: ExperimentConfig):
        import numpy as np
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.feature_selection import SelectFromModel
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.pipeline import Pipeline

        if self.strategy == "forest":
            selector_estimator = ExtraTreesClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=config.random_state,
                n_jobs=config.n_jobs,
            )
        else:
            selector_estimator = build_balanced_logistic_regression(config, penalty="l1")

        return Pipeline(
            steps=[
                ("features", build_feature_preprocessor(X)),
                ("variance", VarianceThreshold()),
                (
                    "selector",
                    SelectFromModel(
                        estimator=selector_estimator,
                        threshold=-np.inf,
                        max_features=k,
                    ),
                ),
                ("model", build_balanced_logistic_regression(config)),
            ]
        )
