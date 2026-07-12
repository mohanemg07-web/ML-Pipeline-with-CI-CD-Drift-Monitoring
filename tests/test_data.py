"""Unit tests for data cleaning and splitting."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config, data


def test_clean_coerces_blank_totalcharges_to_zero(synthetic_df):
    cleaned = data.clean(synthetic_df)
    assert cleaned["TotalCharges"].dtype.kind == "f"
    assert not cleaned["TotalCharges"].isna().any()
    # the fixture injects a blank TotalCharges into row 0 (tenure forced to 0);
    # that specific blank must be coerced to 0.0
    assert cleaned.iloc[0]["TotalCharges"] == 0.0
    assert cleaned.iloc[0]["tenure"] == 0


def test_clean_drops_id_and_is_idempotent(synthetic_df):
    once = data.clean(synthetic_df)
    twice = data.clean(once)
    assert config.ID_COL not in once.columns
    # idempotent: cleaning an already-clean frame changes nothing material
    pd.testing.assert_frame_equal(once.reset_index(drop=True), twice.reset_index(drop=True))


def test_clean_senior_citizen_is_int(synthetic_df):
    cleaned = data.clean(synthetic_df)
    assert cleaned["SeniorCitizen"].dtype.kind in ("i", "u")


def test_split_is_stratified_and_reproducible(synthetic_df):
    s1 = data.split(synthetic_df, seed=42)
    s2 = data.split(synthetic_df, seed=42)
    # reproducible
    assert list(s1.y_test.index) == list(s2.y_test.index)
    # no leakage: the three index sets are disjoint
    idx = set(s1.X_train.index) | set(s1.X_val.index) | set(s1.X_test.index)
    assert len(idx) == len(synthetic_df)
    # stratification: positive rate preserved within tolerance across splits
    full_rate = (synthetic_df["Churn"] == "Yes").mean()
    for y in (s1.y_train, s1.y_val, s1.y_test):
        assert abs(y.mean() - full_rate) < 0.1


def test_split_sizes_sum_to_total(synthetic_df):
    s = data.split(synthetic_df)
    assert sum(s.sizes.values()) == len(synthetic_df)


def test_target_is_binary(synthetic_df):
    s = data.split(synthetic_df)
    assert set(np.unique(s.y_train)).issubset({0, 1})
