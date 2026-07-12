"""Unit tests for the feature preprocessing pipeline."""
from __future__ import annotations

import numpy as np

from src import config, data
from src.features import build_preprocessor


def test_preprocessor_output_is_dense_numeric(synthetic_df):
    splits = data.split(synthetic_df)
    prep = build_preprocessor()
    X = prep.fit_transform(splits.X_train)
    assert isinstance(X, np.ndarray)
    assert X.dtype.kind == "f"
    assert not np.isnan(X).any()


def test_preprocessor_handles_unknown_categories(synthetic_df):
    splits = data.split(synthetic_df)
    prep = build_preprocessor()
    prep.fit(splits.X_train)
    # inject an unseen category value; must not raise thanks to handle_unknown
    x = splits.X_test.copy()
    x.loc[x.index[0], "Contract"] = "Quarterly-special"  # never seen in training
    out = prep.transform(x)
    assert out.shape[0] == len(x)


def test_feature_names_cover_numeric_and_categorical(synthetic_df):
    splits = data.split(synthetic_df)
    prep = build_preprocessor()
    prep.fit(splits.X_train)
    names = list(prep.get_feature_names_out())
    assert any(n.startswith("num__") for n in names)
    assert any(n.startswith("cat__") for n in names)
    # every numeric feature should appear
    for feat in config.NUMERIC_FEATURES:
        assert any(feat in n for n in names)
