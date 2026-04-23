from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetAnalyzer:
    """Gera diagnosticos iniciais do dataset model-ready."""

    target_column: str = "TARGET"
    sensitive_columns: tuple[str, ...] = ("ORIENTACAO_SEXUAL", "RELIGIAO")

    def analyze(self, df) -> dict[str, object]:
        """Retorna um resumo serializavel da base."""

        feature_columns = [column for column in df.columns if column != self.target_column]
        target_counts = df[self.target_column].value_counts(dropna=False).sort_index()
        target_percent = df[self.target_column].value_counts(dropna=False, normalize=True).sort_index()

        duplicated_features = int(df.duplicated(subset=feature_columns).sum())
        conflicting_groups = self._count_conflicting_feature_groups(df, feature_columns)

        return {
            "shape": list(df.shape),
            "target_column": self.target_column,
            "feature_count": len(feature_columns),
            "target_distribution": {
                str(index): {
                    "count": int(target_counts.loc[index]),
                    "percentage": float(target_percent.loc[index]),
                }
                for index in target_counts.index
            },
            "missing_values_total": int(df.isna().sum().sum()),
            "duplicated_rows": int(df.duplicated().sum()),
            "duplicated_feature_rows": duplicated_features,
            "conflicting_feature_groups": conflicting_groups,
            "dtype_counts": {str(key): int(value) for key, value in df.dtypes.value_counts().items()},
            "categorical_columns": df.select_dtypes(include=["object", "string"]).columns.tolist(),
            "sensitive_columns_present": [
                column for column in self.sensitive_columns if column in df.columns
            ],
            "top_cardinality": self._cardinality(df, feature_columns, ascending=False),
            "low_cardinality": self._cardinality(df, feature_columns, ascending=True),
        }

    def save(self, analysis: dict[str, object], output_dir: Path) -> Path:
        """Salva o resumo da analise em JSON."""

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "analysis_summary.json"
        output_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
        return output_path

    def _count_conflicting_feature_groups(self, df, feature_columns: list[str]) -> int:
        grouped = (
            df.groupby(feature_columns, dropna=False)[self.target_column]
            .nunique()
            .reset_index(name="target_unique_count")
        )
        return int((grouped["target_unique_count"] > 1).sum())

    @staticmethod
    def _cardinality(df, feature_columns: list[str], ascending: bool) -> list[dict[str, object]]:
        rows = [
            {"column": column, "unique_values": int(df[column].nunique(dropna=False))}
            for column in feature_columns
        ]
        return sorted(rows, key=lambda row: row["unique_values"], reverse=not ascending)[:15]

