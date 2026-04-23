from __future__ import annotations

import argparse
from pathlib import Path

from reducao_dimensionalidade.analysis import DatasetAnalyzer
from reducao_dimensionalidade.chart_reporter import ChartReporter
from reducao_dimensionalidade.cleanup_manager import CleanupRepositoryManager
from reducao_dimensionalidade.config import (
    DEFAULT_CLEANUP_DIR_NAME,
    DEFAULT_CLEANUP_REPO_URL,
    ExperimentConfig,
    ProjectPaths,
    parse_csv_values,
    parse_k_values,
)
from reducao_dimensionalidade.console_report import ConsoleReport
from reducao_dimensionalidade.data_loader import DatasetLoader
from reducao_dimensionalidade.experiment_runner import ExperimentRunner
from reducao_dimensionalidade.metrics_reporter import MetricsReporter
from reducao_dimensionalidade.training_types import TrainingType


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimentos de selecao e reducao de dimensionalidade."
    )

    parser.add_argument("--clone-cleanup", action="store_true", help="Atualiza o projeto de cleanup via git clone.")
    parser.add_argument("--analise", "--análise", dest="analise", action="store_true", help="Gera analise inicial dos dados.")
    parser.add_argument("--metricas", "--métricas", "--Métricas", dest="metricas", action="store_true", help="Consolida metricas dos experimentos.")
    parser.add_argument("--chart", "--charts", "--graficos", "--gráficos", dest="chart", action="store_true", help="Gera graficos dos resultados.")
    parser.add_argument("-show", "--show", action="store_true", help="Exibe graficos na tela em vez de salvar PNGs.")
    parser.add_argument("--filter", action="store_true", help="Executa metodos filter.")
    parser.add_argument("--wrapper", action="store_true", help="Executa metodos wrapper.")
    parser.add_argument("--embedded", action="store_true", help="Executa metodos embedded.")
    parser.add_argument("--pca", action="store_true", help="Executa PCA.")
    parser.add_argument("--todos", action="store_true", help="Executa analise e todos os experimentos.")

    parser.add_argument("--data-path", default=None, help="Caminho alternativo para cleaned_train.csv.")
    parser.add_argument("--cleanup-dir-name", default=DEFAULT_CLEANUP_DIR_NAME, help="Nome da pasta local do cleanup.")
    parser.add_argument("--cleanup-repo-url", default=DEFAULT_CLEANUP_REPO_URL, help="URL do repositorio de cleanup.")
    parser.add_argument("--output-dir", default="outputs", help="Diretorio de saida.")
    parser.add_argument("--target-column", default="TARGET", help="Coluna alvo.")
    parser.add_argument("--k-values", default=None, help="Valores de k separados por virgula. Ex: 5,10,20")
    parser.add_argument("--cv-folds", type=int, default=5, help="Quantidade de folds da validacao cruzada.")
    parser.add_argument("--random-state", type=int, default=42, help="Semente aleatoria.")
    parser.add_argument("--include-sensitive", action="store_true", help="Mantem variaveis sensiveis nos experimentos.")
    parser.add_argument("--filter-methods", default=None, help="Metodos filter separados por virgula: f_classif,mutual_info.")
    parser.add_argument("--embedded-strategies", default=None, help="Estrategias embedded separadas por virgula: l1,forest.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Paralelismo para sklearn.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths(
        project_root=PROJECT_ROOT,
        cleanup_dir_name=args.cleanup_dir_name,
        output_dir_name=args.output_dir,
    )

    if not any(
        [
            args.clone_cleanup,
            args.analise,
            args.metricas,
            args.chart,
            args.filter,
            args.wrapper,
            args.embedded,
            args.pca,
            args.todos,
        ]
    ):
        print("Nenhuma acao selecionada. Use --help para ver as opcoes.")
        return

    if args.clone_cleanup:
        manager = CleanupRepositoryManager(
            project_root=PROJECT_ROOT,
            cleanup_dir_name=args.cleanup_dir_name,
            repo_url=args.cleanup_repo_url,
        )
        cleanup_path = manager.clone_or_refresh()
        print(f"Cleanup atualizado em: {cleanup_path}")

    config = ExperimentConfig(
        target_column=args.target_column,
        k_values=parse_k_values(args.k_values),
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        include_sensitive=args.include_sensitive,
        filter_methods=parse_csv_values(args.filter_methods, ExperimentConfig().filter_methods),
        embedded_strategies=parse_csv_values(
            args.embedded_strategies,
            ExperimentConfig().embedded_strategies,
        ),
        n_jobs=args.n_jobs,
    )

    data_path = Path(args.data_path).expanduser().resolve() if args.data_path else paths.cleaned_data_path

    needs_dataset = any([args.analise, args.filter, args.wrapper, args.embedded, args.pca, args.todos])
    df = None
    if needs_dataset:
        df = DatasetLoader(data_path=data_path, target_column=config.target_column).load()

    if args.analise or args.todos:
        analyzer = DatasetAnalyzer(target_column=config.target_column)
        analysis = analyzer.analyze(df)
        output_path = analyzer.save(analysis, paths.analysis_dir)
        print(f"Analise salva em: {output_path}")
        ConsoleReport.print_analysis_summary(analysis)

    requested_types = _requested_training_types(args)
    if requested_types:
        runner = ExperimentRunner(config=config, output_dir=paths.experiments_dir)
        results, results_path = runner.run(df=df, training_types=requested_types)
        print(f"Resultados dos experimentos salvos em: {results_path}")
        ConsoleReport.print_experiment_summary(results)

    if args.metricas or args.todos:
        reporter = MetricsReporter()
        ranking, ranking_path = reporter.summarize_results(
            results_path=paths.experiments_dir / "results.csv",
            output_dir=paths.experiments_dir,
        )
        print(f"Ranking de metricas salvo em: {ranking_path}")
        ConsoleReport.print_metrics_summary(ranking)

    if args.chart or args.todos:
        reporter = ChartReporter()
        chart_outputs = reporter.create_charts(
            results_path=paths.experiments_dir / "results.csv",
            output_dir=paths.charts_dir,
            show=args.show,
        )
        ConsoleReport.print_chart_summary(chart_outputs, show=args.show)


def _requested_training_types(args: argparse.Namespace) -> list[TrainingType]:
    if args.todos:
        return [TrainingType.FILTER, TrainingType.WRAPPER, TrainingType.EMBEDDED, TrainingType.PCA]

    training_types = []
    if args.filter:
        training_types.append(TrainingType.FILTER)
    if args.wrapper:
        training_types.append(TrainingType.WRAPPER)
    if args.embedded:
        training_types.append(TrainingType.EMBEDDED)
    if args.pca:
        training_types.append(TrainingType.PCA)
    return training_types


if __name__ == "__main__":
    main()
