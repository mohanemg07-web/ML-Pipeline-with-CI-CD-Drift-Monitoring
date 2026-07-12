"""Feature preprocessing pipeline.

A single sklearn ``ColumnTransformer`` shared by training and serving so the exact
same transformation is applied in both places. Persisted alongside the model.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def build_preprocessor() -> ColumnTransformer:
    """Numeric: median-impute + standardize. Categorical: one-hot (dense).

    ``handle_unknown="ignore"`` makes serving robust to unseen categories.
    """
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, config.NUMERIC_FEATURES),
            ("cat", categorical, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Human-readable output feature names after fitting."""
    return list(preprocessor.get_feature_names_out())
