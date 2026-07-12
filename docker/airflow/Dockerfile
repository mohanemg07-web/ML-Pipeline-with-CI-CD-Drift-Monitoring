# Custom Airflow image for the LOCAL drift-retraining DAG (Phase 3).
# The stock apache/airflow image contains none of the pipeline's ML/validation
# libraries, so DAG tasks that `import xgboost / mlflow / evidently /
# great_expectations` would fail at parse/execution time. We bake them in here
# ONCE at build time rather than reinstalling on every container start via
# _PIP_ADDITIONAL_REQUIREMENTS.
#
# NOTE: This image is built by `docker compose build` and is only exercised in
# Phase 3. pandas/numpy are intentionally NOT re-pinned here — they are left to
# whatever the Airflow 2.10.4 constraint set already provides, to avoid fighting
# Airflow's own dependency locks. The ML libs above resolve against that base.
FROM apache/airflow:2.10.4-python3.11

# Airflow's official constraints keep the core framework's deps stable while the
# extra ML libraries resolve on top of them.
ARG AIRFLOW_VERSION=2.10.4
ARG PYTHON_VERSION=3.11
ARG CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

COPY requirements-airflow.txt /requirements-airflow.txt

# Run as the airflow user (already default in the base image) for pip installs.
RUN pip install --no-cache-dir -r /requirements-airflow.txt --constraint "${CONSTRAINT_URL}"
