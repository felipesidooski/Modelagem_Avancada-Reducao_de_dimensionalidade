from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_CLEANUP_REPO_URL = "https://github.com/felipesidooski/modelagem-avancada-pre-processing.git"
DEFAULT_CLEANUP_DIR_NAME = "modelagem-avancada-pre-processing"


@dataclass(frozen=True)
class ProjectPaths:
    """Caminhos principais do projeto."""

    project_root: Path
    cleanup_dir_name: str = DEFAULT_CLEANUP_DIR_NAME
    output_dir_name: str = "outputs"

    @property
    def cleanup_dir(self) -> Path:
        return self.project_root / self.cleanup_dir_name

    @property
    def cleaned_data_path(self) -> Path:
        return self.cleanup_dir / "outputs" / "cleaned_train.csv"

    @property
    def output_dir(self) -> Path:
        return self.project_root / self.output_dir_name

    @property
    def analysis_dir(self) -> Path:
        return self.output_dir / "analysis"

    @property
    def experiments_dir(self) -> Path:
        return self.output_dir / "experiments"

    @property
    def charts_dir(self) -> Path:
        return self.output_dir / "charts"


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuracao compartilhada pelos experimentos."""

    target_column: str = "TARGET"
    k_values: tuple[int, ...] = (5, 10, 15, 20, 30, 45, 60)
    cv_folds: int = 5
    random_state: int = 42
    scoring: str = "roc_auc"
    include_sensitive: bool = False
    filter_methods: tuple[str, ...] = ("f_classif",)
    embedded_strategies: tuple[str, ...] = ("l1",)
    sensitive_columns: tuple[str, ...] = ("ORIENTACAO_SEXUAL", "RELIGIAO")
    n_jobs: int = 1


def parse_k_values(raw_values: Optional[str]) -> tuple[int, ...]:
    """Converte uma string como '5,10,20' em tupla de inteiros."""

    if not raw_values:
        return ExperimentConfig().k_values

    values = []
    for value in raw_values.split(","):
        value = value.strip()
        if not value:
            continue
        values.append(int(value))

    if not values:
        return ExperimentConfig().k_values

    return tuple(values)


def parse_csv_values(raw_values: Optional[str], default_values: tuple[str, ...]) -> tuple[str, ...]:
    """Converte valores separados por virgula em tupla de strings."""

    if not raw_values:
        return default_values

    values = tuple(value.strip() for value in raw_values.split(",") if value.strip())
    return values or default_values
