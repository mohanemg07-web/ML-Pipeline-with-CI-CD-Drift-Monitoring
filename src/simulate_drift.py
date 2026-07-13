"""Simulated covariate drift: generate a "production batch" from the dataset.

Draws a seeded sample and shifts FIVE features (parameterized by ``severity``):

- ``tenure``            shifted DOWN  (customer base skews newer)
- ``MonthlyCharges``    shifted UP    (price increase)
- ``TotalCharges``      recomputed ~ tenure x MonthlyCharges (kept consistent)
- ``InternetService``   reweighted toward "Fiber optic"
- ``PaymentMethod``     reweighted toward "Electronic check"

Labels are carried over unchanged, so this is COVARIATE shift only. Any claim
built on these numbers must read "under simulated covariate shift of severity
S" — never implied production drift. All shifts stay inside the GE suite's
schema bounds so a drifted batch still passes validation (drift is not a data
QUALITY failure).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config, data

DRIFTED_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "InternetService",
    "PaymentMethod",
]

# Per-unit-severity shift magnitudes (severity 1.0 applies these in full)
TENURE_SHRINK = 0.45          # tenure multiplied by (1 - severity * this)
MONTHLY_UPLIFT = 0.30         # MonthlyCharges multiplied by (1 + severity * this)
FIBER_SWITCH_P = 0.50         # P(switch InternetService -> Fiber optic)
ECHECK_SWITCH_P = 0.45        # P(switch PaymentMethod -> Electronic check)


def simulate_batch(
    df_raw: pd.DataFrame,
    n: int = 1500,
    seed: int = 7,
    severity: float = 1.0,
) -> pd.DataFrame:
    """Return a drifted production batch (raw schema, IDs + labels intact)."""
    if not 0.0 <= severity <= 1.0:
        raise ValueError(f"severity must be in [0, 1], got {severity}")
    rng = np.random.default_rng(seed)
    batch = df_raw.sample(n=min(n, len(df_raw)), random_state=seed).copy()

    batch["tenure"] = np.maximum(
        0, np.round(batch["tenure"] * (1.0 - severity * TENURE_SHRINK))
    ).astype(int)
    batch["MonthlyCharges"] = (
        batch["MonthlyCharges"] * (1.0 + severity * MONTHLY_UPLIFT)
    ).round(2)

    switch_fiber = rng.random(len(batch)) < severity * FIBER_SWITCH_P
    batch.loc[switch_fiber, "InternetService"] = "Fiber optic"
    switch_echeck = rng.random(len(batch)) < severity * ECHECK_SWITCH_P
    batch.loc[switch_echeck, "PaymentMethod"] = "Electronic check"

    # keep TotalCharges consistent with the shifted tenure/price
    batch["TotalCharges"] = (
        batch["tenure"] * batch["MonthlyCharges"] * rng.uniform(0.95, 1.05, len(batch))
    ).round(2)

    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--severity", type=float, default=1.0)
    parser.add_argument("--out", default=str(config.PRODUCTION_BATCH_PATH))
    args = parser.parse_args()

    batch = simulate_batch(data.load_raw(), n=args.n, seed=args.seed, severity=args.severity)
    batch.to_csv(args.out, index=False)
    meta = {
        "out": args.out,
        "n": len(batch),
        "seed": args.seed,
        "severity": args.severity,
        "drifted_features": DRIFTED_FEATURES,
        "shift_magnitudes": {
            "tenure_shrink": TENURE_SHRINK,
            "monthly_uplift": MONTHLY_UPLIFT,
            "fiber_switch_p": FIBER_SWITCH_P,
            "echeck_switch_p": ECHECK_SWITCH_P,
        },
        "churn_rate": round(float((batch[config.TARGET_COL] == "Yes").mean()), 4),
    }
    # sidecar so the DAG can record exactly how this batch was generated
    Path(args.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
