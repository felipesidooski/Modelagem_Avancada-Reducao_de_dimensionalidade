from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChartReporter:
    """Gera graficos comparativos a partir dos resultados dos experimentos."""

    ranking_metric: str = "roc_auc_mean"
    top_n: int = 10

    def create_charts(self, results_path: Path, output_dir: Path, show: bool = False) -> list:
        """Cria ou exibe graficos comparativos e do melhor resultado."""

        import pandas as pd

        if not results_path.exists():
            raise FileNotFoundError(f"Resultados nao encontrados: {results_path}")

        results = pd.read_csv(results_path)
        if results.empty:
            raise ValueError("Arquivo de resultados esta vazio.")
        if self.ranking_metric not in results.columns:
            raise ValueError(f"Metrica nao encontrada nos resultados: {self.ranking_metric}")

        self._configure_matplotlib(output_dir=output_dir, show=show)

        ranking = results.sort_values(self.ranking_metric, ascending=False).reset_index(drop=True)
        figures = [
            (
                "comparativo_metricas_por_k.png",
                "Comparativo de metricas por k",
                self._build_metrics_by_k_figure(results),
            ),
            (
                "ranking_top_resultados.png",
                "Ranking dos principais resultados",
                self._build_top_results_figure(ranking),
            ),
            (
                "melhor_resultado_metricas.png",
                "Metricas do melhor resultado",
                self._build_best_result_figure(ranking.iloc[0]),
            ),
        ]

        if show:
            return self._show_figures(figures)

        return self._save_figures(figures, output_dir)

    @staticmethod
    def _configure_matplotlib(output_dir: Path, show: bool) -> None:
        cache_dir = Path("/tmp/reducao_dimensionalidade_matplotlib_cache") if show else output_dir / ".matplotlib_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

        import matplotlib

        if not show:
            output_dir.mkdir(parents=True, exist_ok=True)
            matplotlib.use("Agg")

    @staticmethod
    def _save_figures(figures: list[tuple[str, str, object]], output_dir: Path) -> list[Path]:
        import matplotlib.pyplot as plt

        chart_paths = []
        for filename, _, fig in figures:
            output_path = output_dir / filename
            fig.savefig(output_path, dpi=160, bbox_inches="tight")
            chart_paths.append(output_path)
            plt.close(fig)
        return chart_paths

    @staticmethod
    def _show_figures(figures: list[tuple[str, str, object]]) -> list[str]:
        import matplotlib.pyplot as plt

        chart_names = []
        for _, title, fig in figures:
            try:
                fig.canvas.manager.set_window_title(title)
            except AttributeError:
                pass
            chart_names.append(title)

        plt.show()
        plt.close("all")
        return chart_names

    def _build_metrics_by_k_figure(self, results):
        import matplotlib.pyplot as plt

        metric_specs = [
            ("roc_auc_mean", "ROC AUC"),
            ("average_precision_mean", "PR AUC"),
            ("f1_mean", "F1"),
        ]
        metric_specs = [(column, label) for column, label in metric_specs if column in results.columns]

        fig, axes = plt.subplots(
            nrows=len(metric_specs),
            ncols=1,
            figsize=(11, 4 * len(metric_specs)),
            sharex=True,
        )
        if len(metric_specs) == 1:
            axes = [axes]

        for axis, (metric_column, metric_label) in zip(axes, metric_specs):
            for method, group in results.groupby("method"):
                group = group.sort_values("k")
                axis.plot(group["k"], group[metric_column], marker="o", linewidth=2, label=method)
            axis.set_title(f"{metric_label} por valor de k")
            axis.set_ylabel(metric_label)
            axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            axis.legend(loc="best")

        axes[-1].set_xlabel("k")
        fig.tight_layout()
        return fig

    def _build_top_results_figure(self, ranking):
        import matplotlib.pyplot as plt

        top_results = ranking.head(self.top_n).copy()
        top_results["label"] = top_results.apply(
            lambda row: f"{row['method']} | k={row['k']}",
            axis=1,
        )
        top_results = top_results.sort_values(self.ranking_metric, ascending=True)

        fig, axis = plt.subplots(figsize=(12, max(5, 0.55 * len(top_results))))
        axis.barh(top_results["label"], top_results[self.ranking_metric], color="#2f6f8f")
        axis.set_title(f"Top {len(top_results)} resultados por ROC AUC")
        axis.set_xlabel("ROC AUC medio")
        axis.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.5)

        for index, value in enumerate(top_results[self.ranking_metric]):
            axis.text(value + 0.002, index, f"{value:.4f}", va="center", fontsize=9)

        fig.tight_layout()
        return fig

    @staticmethod
    def _build_best_result_figure(best_result):
        import matplotlib.pyplot as plt

        metric_specs = [
            ("roc_auc_mean", "ROC AUC"),
            ("average_precision_mean", "PR AUC"),
            ("f1_mean", "F1"),
            ("recall_mean", "Recall"),
            ("precision_mean", "Precision"),
            ("accuracy_mean", "Accuracy"),
        ]
        labels = [label for column, label in metric_specs if column in best_result.index]
        values = [float(best_result[column]) for column, _ in metric_specs if column in best_result.index]

        fig, axis = plt.subplots(figsize=(10, 5.5))
        bars = axis.bar(labels, values, color=["#2f6f8f", "#5aa9a6", "#8fb339", "#d99b2b", "#c95d63", "#6b5b95"])
        axis.set_ylim(0, 1)
        axis.set_title(f"Melhor resultado: {best_result['method']} | k={best_result['k']}")
        axis.set_ylabel("Valor medio")
        axis.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.025, 0.98),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        fig.tight_layout()
        return fig
