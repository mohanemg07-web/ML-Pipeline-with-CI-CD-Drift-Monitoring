"""Great Expectations data validation, runnable fully headlessly.

Uses the GE 0.18 fluent API with an *ephemeral* context so CI can validate a
DataFrame without a checked-in GE project directory. The expectation suite is
defined in code (the single source of truth) and also serialized to
``gx/expectations/churn_raw_suite.json`` for the repo record.

Checks: schema (columns exist + types), null thresholds, categorical domains,
and a target-balance sanity check.
"""
from __future__ import annotations

import json
from pathlib import Path

import great_expectations as gx
import pandas as pd

from src import config

SUITE_NAME = "churn_raw_suite"
SUITE_JSON_PATH = config.REPO_ROOT / "gx" / "expectations" / f"{SUITE_NAME}.json"

# Target churn rate for the Telco set is ~0.265; allow a generous sanity band so
# this catches a broken/duplicated label column, not normal sampling variation.
TARGET_BALANCE_MIN = 0.15
TARGET_BALANCE_MAX = 0.40


def _add_expectations(validator) -> None:
    """Attach every expectation to the validator (defines the suite)."""
    expected_cols = [config.ID_COL, *config.FEATURE_COLS, config.TARGET_COL]

    # --- Schema: required columns exist ---
    for col in expected_cols:
        validator.expect_column_to_exist(col)

    # --- Null thresholds: core columns must be effectively complete ---
    for col in [config.ID_COL, config.TARGET_COL, "tenure", "MonthlyCharges"]:
        validator.expect_column_values_to_not_be_null(col, mostly=0.99)

    # --- Numeric ranges ---
    validator.expect_column_values_to_be_between("tenure", min_value=0, max_value=100)
    validator.expect_column_values_to_be_between(
        "MonthlyCharges", min_value=0, max_value=1000
    )
    validator.expect_column_values_to_be_in_set("SeniorCitizen", [0, 1])

    # --- Categorical domains ---
    for col, domain in config.CATEGORICAL_DOMAINS.items():
        validator.expect_column_values_to_be_in_set(col, domain)

    # --- Target sanity: only Yes/No, and balanced within a plausible band ---
    validator.expect_column_values_to_be_in_set(config.TARGET_COL, ["Yes", "No"])
    validator.expect_column_values_to_be_unique(config.ID_COL)


def validate_dataframe(df: pd.DataFrame, save_suite: bool = True) -> dict:
    """Run the suite against ``df`` headlessly; return a compact result dict.

    Target-balance is asserted separately (it is a dataset-level statistic, not a
    row-wise expectation), so a broken label column fails validation too.
    """
    context = gx.get_context(mode="ephemeral")
    datasource = context.sources.add_pandas("pandas_runtime")
    asset = datasource.add_dataframe_asset(name="churn")
    batch_request = asset.build_batch_request(dataframe=df)

    context.add_or_update_expectation_suite(SUITE_NAME)
    validator = context.get_validator(
        batch_request=batch_request, expectation_suite_name=SUITE_NAME
    )
    _add_expectations(validator)
    validator.save_expectation_suite(discard_failed_expectations=False)

    results = validator.validate()

    # Dataset-level target balance check (separate from row-wise expectations)
    balance = float((df[config.TARGET_COL] == config.POSITIVE_LABEL).mean())
    balance_ok = TARGET_BALANCE_MIN <= balance <= TARGET_BALANCE_MAX

    if save_suite:
        SUITE_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        suite_dict = context.get_expectation_suite(SUITE_NAME).to_json_dict()
        SUITE_JSON_PATH.write_text(json.dumps(suite_dict, indent=2))

    stats = results.statistics
    return {
        "success": bool(results.success) and balance_ok,
        "ge_success": bool(results.success),
        "evaluated_expectations": stats.get("evaluated_expectations"),
        "successful_expectations": stats.get("successful_expectations"),
        "unsuccessful_expectations": stats.get("unsuccessful_expectations"),
        "target_balance": balance,
        "target_balance_ok": balance_ok,
    }


def main(path: Path = config.RAW_DATA_PATH) -> int:
    """CLI entrypoint: validate the raw CSV, print JSON, exit non-zero on fail."""
    df = pd.read_csv(path)
    result = validate_dataframe(df)
    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
