"""Tests for the Great Expectations validation logic."""
from __future__ import annotations

from src import validate


def test_valid_data_passes(real_sample_df):
    result = validate.validate_dataframe(real_sample_df, save_suite=False)
    assert result["success"] is True
    assert result["unsuccessful_expectations"] == 0
    assert 0.15 <= result["target_balance"] <= 0.40


def test_out_of_domain_category_fails(real_sample_df):
    bad = real_sample_df.copy()
    bad.loc[bad.index[0], "Contract"] = "Quarterly"  # not an allowed domain value
    result = validate.validate_dataframe(bad, save_suite=False)
    assert result["ge_success"] is False
    assert result["unsuccessful_expectations"] >= 1


def test_broken_target_balance_fails(real_sample_df):
    broken = real_sample_df.copy()
    broken["Churn"] = "No"  # zero positives -> balance sanity check must fail
    result = validate.validate_dataframe(broken, save_suite=False)
    assert result["target_balance_ok"] is False
    assert result["success"] is False
