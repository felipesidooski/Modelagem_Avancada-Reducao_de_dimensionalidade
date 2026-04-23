from __future__ import annotations

from reducao_dimensionalidade.config import ExperimentConfig


def build_balanced_logistic_regression(config: ExperimentConfig, penalty: str = "l2"):
    """Cria uma regressao logistica balanceada e numericamente estavel."""

    from sklearn.linear_model import LogisticRegression

    if penalty == "l1":
        return LogisticRegression(
            penalty="l1",
            solver="liblinear",
            class_weight="balanced",
            max_iter=3000,
            random_state=config.random_state,
        )

    return LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        class_weight="balanced",
        max_iter=3000,
        random_state=config.random_state,
    )
