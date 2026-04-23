from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetLoader:
    """Carrega o dataset limpo gerado pelo projeto de cleanup."""

    data_path: Path
    target_column: str = "TARGET"

    def load(self):
        """Retorna um DataFrame pandas validado."""

        import pandas as pd

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset limpo nao encontrado: {self.data_path}. "
                "Execute --clone-cleanup ou informe --data-path."
            )

        df = pd.read_csv(self.data_path)
        if df.empty:
            raise ValueError("Dataset carregado esta vazio.")
        if self.target_column not in df.columns:
            raise ValueError(f"Coluna target nao encontrada: {self.target_column}")
        return df

