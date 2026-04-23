from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings

from reducao_dimensionalidade.config import ExperimentConfig
from reducao_dimensionalidade.training_types import TrainingType


@dataclass(frozen=True)
class BaseTrainer(ABC):
    """Contrato comum para experimentos com diferentes tecnicas."""

    training_type: TrainingType
    name: str

    @abstractmethod
    def build_pipeline(self, X, k: int, config: ExperimentConfig):
        """Monta o pipeline sklearn para um valor de k."""

    def run(self, X, y, config: ExperimentConfig) -> list[dict[str, object]]:
        """Executa validacao cruzada para todos os valores de k."""

        from sklearn.model_selection import StratifiedKFold, cross_validate

        cv = StratifiedKFold(
            n_splits=config.cv_folds,
            shuffle=True,
            random_state=config.random_state,
        )

        rows = []
        for k in config.k_values:
            print(f"Executando {self.name} com k={k}...", flush=True)
            pipeline = self.build_pipeline(X=X, k=k, config=config)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                    message=".*matmul.*",
                )
                scores = cross_validate(
                    pipeline,
                    X,
                    y,
                    cv=cv,
                    scoring=self._scorers(),
                    n_jobs=config.n_jobs,
                    error_score="raise",
                )
            rows.append(self._build_result_row(k=k, scores=scores))

        return rows

    def _build_result_row(self, k: int, scores) -> dict[str, object]:
        import numpy as np

        row: dict[str, object] = {
            "training_type": self.training_type.value,
            "method": self.name,
            "k": k,
        }
        for key, values in scores.items():
            if not key.startswith("test_"):
                continue
            metric = key.removeprefix("test_")
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values))
        return row

    @staticmethod
    def _scorers() -> dict[str, object]:
        from sklearn.metrics import make_scorer
        from sklearn.metrics import precision_score, recall_score, f1_score

        return {
            "accuracy": "accuracy",
            "roc_auc": "roc_auc",
            "average_precision": "average_precision",
            "precision": make_scorer(precision_score, zero_division=0),
            "recall": make_scorer(recall_score, zero_division=0),
            "f1": make_scorer(f1_score, zero_division=0),
        }
