from __future__ import annotations


def build_feature_preprocessor(X):
    """Cria o preprocessador de features para os pipelines de treino."""

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_columns = X.select_dtypes(exclude=["object", "string"]).columns.tolist()
    categorical_columns = X.select_dtypes(include=["object", "string"]).columns.tolist()

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", StandardScaler(), numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", _dense_one_hot_encoder(), categorical_columns))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _dense_one_hot_encoder():
    """Retorna OneHotEncoder denso em versoes novas e antigas do sklearn."""

    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

