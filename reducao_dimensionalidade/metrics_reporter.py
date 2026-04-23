from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MetricsReporter:
    """Consolida metricas geradas pelos experimentos."""

    ranking_metric: str = "roc_auc_mean"

    def summarize_results(self, results_path: Path, output_dir: Path):
        """Le o CSV de resultados e gera um ranking por metrica."""

        import pandas as pd

        if not results_path.exists():
            raise FileNotFoundError(f"Resultados nao encontrados: {results_path}")

        df = pd.read_csv(results_path)
        if self.ranking_metric not in df.columns:
            raise ValueError(f"Metrica nao encontrada nos resultados: {self.ranking_metric}")

        ranking = df.sort_values(self.ranking_metric, ascending=False).reset_index(drop=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "metrics_ranking.csv"
        ranking.to_csv(output_path, index=False)
        return ranking, output_path

