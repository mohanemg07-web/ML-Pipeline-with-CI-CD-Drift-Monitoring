"""The ONE place that answers: how is the champion referenced in the registry?

``register_dagshub.py`` sets the ``champion`` alias but falls back to the
``Production`` stage if the DagsHub MLflow server rejects aliases. Every
consumer (verify_registry, the Phase 2 serving app, the Phase 3 DAG) must
therefore try both — via this helper only, so the resolution order never
diverges across the codebase.

No credentials are handled here: callers are responsible for the tracking URI /
env; this module only encodes URI order and the (optional) load timeout.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

from src import config

logger = logging.getLogger(__name__)

# Order matters: alias is the preferred, modern reference; stage is the fallback
# that register_dagshub.py uses when the server rejects aliases.
CHAMPION_URIS = [
    f"models:/{config.REGISTERED_MODEL_NAME}@{config.CHAMPION_ALIAS}",
    f"models:/{config.REGISTERED_MODEL_NAME}/Production",
]


def load_champion(load_model=None, timeout_s: float | None = None):
    """Try each champion URI in order; return ``(model, resolved_uri)``.

    ``load_model`` defaults to ``mlflow.sklearn.load_model`` (injected in tests).
    ``timeout_s`` bounds EACH attempt (serving uses ~5s so a hung DagsHub
    connection can never stall container startup); ``None`` means no bound.

    Raises ``LookupError`` if no URI resolves — callers decide the fallback
    (e.g., the serving app falls back to the bundled model file).
    """
    if load_model is None:
        import mlflow.sklearn

        load_model = mlflow.sklearn.load_model

    errors: dict[str, str] = {}
    for uri in CHAMPION_URIS:
        try:
            if timeout_s is None:
                model = load_model(uri)
            else:
                # No `with` block: context-manager exit would join the worker
                # thread, blocking past the timeout on a hung connection.
                pool = ThreadPoolExecutor(max_workers=1)
                try:
                    model = pool.submit(load_model, uri).result(timeout=timeout_s)
                finally:
                    pool.shutdown(wait=False, cancel_futures=True)
            logger.info("champion resolved via %s", uri)
            return model, uri
        except FutureTimeoutError:
            errors[uri] = f"timeout after {timeout_s}s"
            logger.warning("champion load timed out for %s", uri)
        except Exception as exc:  # noqa: BLE001 — any registry failure -> try next URI
            errors[uri] = f"{type(exc).__name__}: {exc}"
            logger.info("champion not resolvable via %s (%s)", uri, type(exc).__name__)

    raise LookupError(f"champion not loadable from registry; attempts: {errors}")
