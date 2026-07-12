"""Central configuration: paths, schema, split params, and the AUC quality gate.

Single source of truth for column definitions so data cleaning, the feature
pipeline, Great Expectations, and drift simulation all agree.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths ---
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "telco_churn.csv"
EVAL_RESULTS_DIR = REPO_ROOT / "eval" / "results"
MODELS_DIR = REPO_ROOT / "models"           # gitignored local artifacts
SERVING_MODEL_DIR = REPO_ROOT / "serving" / "model"  # committed fallback bundle
MLRUNS_DIR = REPO_ROOT / "mlruns"           # local MLflow file store (gitignored)
MONITORING_DIR = REPO_ROOT / "monitoring"

# --- Reproducibility ---
RANDOM_SEED = 42

# --- Target ---
TARGET_COL = "Churn"
POSITIVE_LABEL = "Yes"
ID_COL = "customerID"

# --- Feature schema (excludes ID and target) ---
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Allowed categorical domains — used by Great Expectations and to validate
# incoming serving requests. SeniorCitizen is numeric (0/1) so not listed here.
CATEGORICAL_DOMAINS = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

# --- Split ---
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # fraction of the full dataset carved into validation

# --- Decision threshold for F1 reporting ---
DECISION_THRESHOLD = 0.5

# --- Quality gate: CI fails if test ROC-AUC drops below this ---
AUC_GATE = 0.80

# --- MLflow ---
EXPERIMENT_NAME = "churn-xgboost"
REGISTERED_MODEL_NAME = "churn-xgboost"
CHAMPION_ALIAS = "champion"
