"""Shared pytest fixtures: a tiny synthetic Telco-shaped frame + a real sample."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config


def _make_row(churn: str, **overrides) -> dict:
    row = {
        "customerID": overrides.get("customerID", "0000-AAAA"),
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 55.0,
        "TotalCharges": "660.0",
        "Churn": churn,
    }
    row.update(overrides)
    return row


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Balanced-ish synthetic frame with unique IDs; includes a tenure-0 blank."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(200):
        churn = "Yes" if i % 4 == 0 else "No"  # ~25% positive
        rows.append(
            _make_row(
                churn,
                customerID=f"{i:04d}-ZZZZ",
                tenure=int(rng.integers(0, 72)),
                MonthlyCharges=float(round(rng.uniform(20, 120), 2)),
            )
        )
    # inject the Telco tenure-0 / blank-TotalCharges quirk
    rows[0]["tenure"] = 0
    rows[0]["TotalCharges"] = " "
    return pd.DataFrame(rows)


FIXTURE_SAMPLE = config.REPO_ROOT / "tests" / "fixtures" / "sample.csv"


@pytest.fixture
def real_sample_df() -> pd.DataFrame:
    """Committed stratified sample of the real dataset (CI-safe, no download).

    Falls back to the full raw CSV if the fixture is somehow missing.
    """
    if FIXTURE_SAMPLE.exists():
        return pd.read_csv(FIXTURE_SAMPLE)
    if config.RAW_DATA_PATH.exists():
        return pd.read_csv(config.RAW_DATA_PATH).head(500)
    pytest.skip("no sample fixture and no raw dataset available")
