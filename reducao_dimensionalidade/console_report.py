from __future__ import annotations


class ConsoleReport:
    """Imprime resumos curtos dos resultados no terminal."""

    @staticmethod
    def print_analysis_summary(analysis: dict[str, object]) -> None:
        """Mostra os principais pontos da analise inicial."""

        shape = analysis.get("shape", ["?", "?"])
        target_distribution = analysis.get("target_distribution", {})
        categorical_columns = analysis.get("categorical_columns", [])
        sensitive_columns = analysis.get("sensitive_columns_present", [])

        print("\n=== Resumo da analise ===")
        print(f"Dataset: {shape[0]} registros x {shape[1]} colunas")
        print(f"Features: {analysis.get('feature_count', '?')}")
        print(f"Valores ausentes: {analysis.get('missing_values_total', '?')}")
        print(f"Linhas duplicadas: {analysis.get('duplicated_rows', '?')}")
        print(f"Features duplicadas: {analysis.get('duplicated_feature_rows', '?')}")
        print(f"Grupos com features iguais e TARGET conflitante: {analysis.get('conflicting_feature_groups', '?')}")

        if target_distribution:
            print("\nDistribuicao do target:")
            for target_value, stats in target_distribution.items():
                percentage = _as_percentage(stats.get("percentage", 0))
                print(f"  TARGET={target_value}: {stats.get('count', 0)} registros ({percentage})")

        if categorical_columns:
            print(f"\nColunas categoricas: {', '.join(categorical_columns)}")

        if sensitive_columns:
            print(f"Variaveis sensiveis detectadas: {', '.join(sensitive_columns)}")

    @staticmethod
    def print_experiment_summary(results) -> None:
        """Mostra os melhores resultados da rodada executada."""

        if results is None or results.empty:
            print("\nNenhum resultado de experimento foi gerado.")
            return

        ranking = _sort_by_available_metric(results)
        best = ranking.iloc[0]

        print("\n=== Resumo dos experimentos ===")
        print(f"Execucoes avaliadas: {len(results)}")
        print(
            "Melhor resultado: "
            f"{best['method']} | k={best['k']} | "
            f"ROC AUC={_fmt_metric(best.get('roc_auc_mean'))} | "
            f"PR AUC={_fmt_metric(best.get('average_precision_mean'))} | "
            f"F1={_fmt_metric(best.get('f1_mean'))} | "
            f"Recall={_fmt_metric(best.get('recall_mean'))}"
        )

        ConsoleReport._print_top_rows(ranking, title="Top resultados", limit=5)

    @staticmethod
    def print_metrics_summary(ranking) -> None:
        """Mostra um ranking curto das metricas consolidadas."""

        if ranking is None or ranking.empty:
            print("\nNenhum ranking de metricas foi gerado.")
            return

        print("\n=== Ranking de metricas ===")
        ConsoleReport._print_top_rows(ranking, title="Top por ROC AUC", limit=5)

    @staticmethod
    def print_chart_summary(chart_outputs: list, show: bool = False) -> None:
        """Mostra os graficos gerados ou exibidos."""

        if not chart_outputs:
            print("\nNenhum grafico foi gerado ou exibido.")
            return

        title = "Graficos exibidos" if show else "Graficos gerados"
        print(f"\n=== {title} ===")
        for output in chart_outputs:
            print(f"  - {output}")

    @staticmethod
    def _print_top_rows(ranking, title: str, limit: int) -> None:
        print(f"\n{title}:")
        columns = [
            "method",
            "k",
            "roc_auc_mean",
            "average_precision_mean",
            "f1_mean",
            "recall_mean",
            "precision_mean",
            "accuracy_mean",
        ]
        available_columns = [column for column in columns if column in ranking.columns]
        display = ranking[available_columns].head(limit)

        for index, row in display.iterrows():
            position = index + 1
            print(
                f"  {position}. {row['method']} | "
                f"k={row['k']} | "
                f"ROC AUC={_fmt_metric(row.get('roc_auc_mean'))} | "
                f"PR AUC={_fmt_metric(row.get('average_precision_mean'))} | "
                f"F1={_fmt_metric(row.get('f1_mean'))} | "
                f"Recall={_fmt_metric(row.get('recall_mean'))} | "
                f"Precision={_fmt_metric(row.get('precision_mean'))} | "
                f"Accuracy={_fmt_metric(row.get('accuracy_mean'))}"
            )


def _sort_by_available_metric(results):
    for metric in ["roc_auc_mean", "average_precision_mean", "f1_mean", "recall_mean"]:
        if metric in results.columns:
            return results.sort_values(metric, ascending=False).reset_index(drop=True)
    return results.reset_index(drop=True)


def _fmt_metric(value) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def _as_percentage(value) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"
