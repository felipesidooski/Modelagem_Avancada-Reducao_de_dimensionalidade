from __future__ import annotations

from dataclasses import dataclass

from reducao_dimensionalidade.config import ExperimentConfig
from reducao_dimensionalidade.feature_pipeline import build_feature_preprocessor
from reducao_dimensionalidade.model_factory import build_balanced_logistic_regression
from reducao_dimensionalidade.trainers.base import BaseTrainer
from reducao_dimensionalidade.training_types import TrainingType


@dataclass(frozen=True)
class FilterTrainer(BaseTrainer):
    """Treino com selecao filter via SelectKBest."""

    score_func_name: str = "f_classif"

    def __init__(self, score_func_name: str = "f_classif") -> None:
        object.__setattr__(self, "training_type", TrainingType.FILTER)
        object.__setattr__(self, "name", f"select_k_best_{score_func_name}")
        object.__setattr__(self, "score_func_name", score_func_name)

    def build_pipeline(self, X, k: int, config: ExperimentConfig):
        from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.pipeline import Pipeline

        score_funcs = {
            "f_classif": f_classif,
            "mutual_info": mutual_info_classif,
        }
        score_func = score_funcs[self.score_func_name]

        return Pipeline(
            steps=[
                ("features", build_feature_preprocessor(X)),
                ("variance", VarianceThreshold()),
                ("selector", SelectKBest(score_func=score_func, k=k)),
                ("model", build_balanced_logistic_regression(config)),
            ]
        )
