"""Data loading, cleaning, and stratified splitting.

Cleaning handles the Telco dataset's known quirk: ``TotalCharges`` is stored as
a string and is blank (" ") for the 11 tenure-0 customers, which would otherwise
poison numeric conversion.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def load_raw(path=config.RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV with no transformation."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Telco data into model-ready types.

    - Coerce ``TotalCharges`` to numeric; blanks (tenure-0 customers) -> 0.0,
      which is the correct semantic value (no charges accrued yet).
    - Ensure ``SeniorCitizen`` is integer 0/1.
    - Drop the ID column.
    - Leave the target as-is (mapped to 0/1 in ``split``).
    """
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # tenure-0 customers have no accrued charges; blank -> 0.0
    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0
    # any residual NaN (unexpected) -> 0.0 to keep the pipeline headless-safe
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    if config.ID_COL in df.columns:
        df = df.drop(columns=[config.ID_COL])

    return df


def _target_to_binary(y: pd.Series) -> pd.Series:
    return (y == config.POSITIVE_LABEL).astype(int)


@dataclass
class DataSplits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.X_train),
            "val": len(self.X_val),
            "test": len(self.X_test),
        }


def split(df: pd.DataFrame, seed: int = config.RANDOM_SEED) -> DataSplits:
    """Stratified train/val/test split with a fixed seed.

    Two-stage split so proportions are exact: first carve out test, then carve
    validation out of the remainder. ``clean`` is idempotent, so passing either a
    raw or already-cleaned frame is safe.
    """
    df = clean(df)

    X = df[config.FEATURE_COLS]
    y = _target_to_binary(df[config.TARGET_COL])

    X_rem, X_test, y_rem, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=seed
    )
    # val fraction expressed relative to the remaining rows
    val_relative = config.VAL_SIZE / (1.0 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rem, y_rem, test_size=val_relative, stratify=y_rem, random_state=seed
    )

    return DataSplits(X_train, X_val, X_test, y_train, y_val, y_test)


def target_balance(df: pd.DataFrame) -> float:
    """Fraction of positive (churn) labels in a raw/clean frame."""
    return float(_target_to_binary(df[config.TARGET_COL]).mean())
