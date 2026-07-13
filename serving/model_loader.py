"""Model loading for the serving app.

Primary deployment path: the bundled ``serving/model/model.joblib`` COPYed into
the image at build time. Pulling the champion from the MLflow registry
(``MODEL_SOURCE=registry``) is an optional enhancement — it is bounded by a
hard per-attempt timeout (``REGISTRY_LOAD_TIMEOUT_S``, default 5 s; raise it
for slow-boot environments like Render cold starts) and ALWAYS falls back to
the bundle on any failure.

Every load reports a verifiable model identity (registry version + run id, or
a content hash for the bundle) so a post-redeploy check can confirm WHICH
model is live instead of trusting a probe probability.
"""
from __future__ import annotations

import hashlib
import logging
import os

import joblib

from src import config
from src.model_resolver import load_champion

logger = logging.getLogger(__name__)

BUNDLED_MODEL_PATH = config.SERVING_MODEL_DIR / "model.joblib"
DEFAULT_REGISTRY_TIMEOUT_S = 5.0


def _registry_version_info(resolved_uri: str) -> dict:
    """Best-effort version/run lookup for the champion that just loaded.

    Runs after a successful model download, so the connection is known-good;
    any failure here degrades to "unknown" rather than failing the boot.
    """
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        name = config.REGISTERED_MODEL_NAME
        if resolved_uri.endswith(f"@{config.CHAMPION_ALIAS}"):
            mv = client.get_model_version_by_alias(name, config.CHAMPION_ALIAS)
        else:
            mv = client.get_latest_versions(name, stages=["Production"])[0]
        return {"model_version": str(mv.version), "model_run_id": mv.run_id}
    except Exception as exc:  # noqa: BLE001 — identity lookup must not fail the boot
        logger.warning("registry version lookup failed (%s)", type(exc).__name__)
        return {"model_version": "unknown", "model_run_id": None}


def _bundle_identity() -> str:
    digest = hashlib.sha256(BUNDLED_MODEL_PATH.read_bytes()).hexdigest()
    return f"file-sha256:{digest[:12]}"


def load_serving_model() -> tuple[object, dict]:
    """Return ``(model, info)``; ``info`` is surfaced verbatim by /health.

    info = {"model_source": label, "model_version": ..., "model_run_id": ...}
    """
    if os.environ.get("MODEL_SOURCE", "bundled").lower() == "registry":
        timeout_s = float(os.environ.get("REGISTRY_LOAD_TIMEOUT_S", DEFAULT_REGISTRY_TIMEOUT_S))
        try:
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
            if not tracking_uri:
                raise LookupError("MODEL_SOURCE=registry but MLFLOW_TRACKING_URI is unset")
            import mlflow

            mlflow.set_tracking_uri(tracking_uri)
            model, resolved_uri = load_champion(timeout_s=timeout_s)
            return model, {
                "model_source": f"registry:{resolved_uri}",
                **_registry_version_info(resolved_uri),
            }
        except Exception as exc:  # noqa: BLE001 — any registry failure -> bundled fallback
            logger.warning("registry load failed (%s); falling back to bundled model", exc)

    model = joblib.load(BUNDLED_MODEL_PATH)
    return model, {
        "model_source": "bundled:model.joblib",
        "model_version": _bundle_identity(),
        "model_run_id": None,
    }
