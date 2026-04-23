from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from reducao_dimensionalidade.config import ExperimentConfig
from reducao_dimensionalidade.training_types import TrainingType


@dataclass(frozen=True)
class ExperimentRunner:
    """Orquestra os experimentos de selecao/reducao de atributos."""

    config: ExperimentConfig
    output_dir: Path

    def run(self, df, training_types: list[TrainingType]):
        """Executa os trainers solicitados e salva os resultados."""

        import pandas as pd

        X, y = self._split_features_target(df)
        trainers = self._build_trainers(training_types)

        rows = []
        for trainer in trainers:
            rows.extend(trainer.run(X=X, y=y, config=self.config))

        results = pd.DataFrame(rows)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "results.csv"
        results.to_csv(output_path, index=False)
        return results, output_path

    def _split_features_target(self, df):
        y = df[self.config.target_column]
        X = df.drop(columns=[self.config.target_column])

        if not self.config.include_sensitive:
            columns_to_drop = [
                column for column in self.config.sensitive_columns if column in X.columns
            ]
            X = X.drop(columns=columns_to_drop)

        return X, y

    def _build_trainers(self, training_types: list[TrainingType]):
        from reducao_dimensionalidade.trainers import (
            EmbeddedTrainer,
            FilterTrainer,
            PCATrainer,
            WrapperTrainer,
        )

        trainers = []
        if TrainingType.FILTER in training_types:
            for method in self.config.filter_methods:
                trainers.append(FilterTrainer(score_func_name=method))
        if TrainingType.WRAPPER in training_types:
            trainers.append(WrapperTrainer())
        if TrainingType.EMBEDDED in training_types:
            for strategy in self.config.embedded_strategies:
                trainers.append(EmbeddedTrainer(strategy=strategy))
        if TrainingType.PCA in training_types:
            trainers.append(PCATrainer())
        return trainers
