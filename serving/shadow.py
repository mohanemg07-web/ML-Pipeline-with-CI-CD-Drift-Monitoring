"""Shadow-mode champion/challenger comparison.

The client ALWAYS receives the champion's prediction. When a challenger bundle
exists and ``CHALLENGER_TRAFFIC_PCT`` > 0, that percentage of prediction
requests is additionally scored by the challenger and the per-row comparison is
appended to a JSONL log for offline analysis. ``build_shadow()`` returns
``None`` when shadow mode is disabled, so the disabled hot path costs exactly
one ``is not None`` check — no sampling, no I/O.

``SHADOW_SEED`` makes the traffic sampler deterministic (used by tests; leave
unset in production for true random sampling).
"""
from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

DEFAULT_CHALLENGER_PATH = config.SERVING_MODEL_DIR / "challenger.joblib"
DEFAULT_LOG_PATH = config.REPO_ROOT / "serving" / "shadow" / "comparisons.jsonl"


class ShadowRouter:
    """Samples requests and logs champion-vs-challenger probability pairs."""

    def __init__(
        self,
        challenger,
        traffic_pct: float,
        log_path: Path,
        rng: random.Random | None = None,
    ) -> None:
        self.challenger = challenger
        self.traffic_pct = min(float(traffic_pct), 100.0)
        self.log_path = Path(log_path)
        self.rng = rng if rng is not None else random.Random()

    def maybe_compare(self, features: pd.DataFrame, champion_proba, endpoint: str) -> bool:
        """Score ``features`` with the challenger for the sampled fraction of requests.

        Never raises: a challenger failure is logged and swallowed so shadow
        mode can never break the champion response path.
        """
        if self.rng.uniform(0.0, 100.0) >= self.traffic_pct:
            return False
        try:
            challenger_proba = self.challenger.predict_proba(features)[:, 1]
            # timezone.utc (not 3.11's datetime.UTC): tests run on local 3.10
            ts = datetime.now(timezone.utc).isoformat()  # noqa: UP017
            with self.log_path.open("a", encoding="utf-8") as fh:
                for i in range(len(features)):
                    champ = float(champion_proba[i])
                    chall = float(challenger_proba[i])
                    fh.write(
                        json.dumps(
                            {
                                "ts": ts,
                                "endpoint": endpoint,
                                "row": i,
                                "champion_proba": round(champ, 6),
                                "challenger_proba": round(chall, 6),
                                "abs_diff": round(abs(champ - chall), 6),
                            }
                        )
                        + "\n"
                    )
            return True
        except Exception:  # noqa: BLE001 — shadow must never break serving
            logger.exception("shadow comparison failed; champion response unaffected")
            return False


def build_shadow(env=os.environ) -> ShadowRouter | None:
    """Construct a ShadowRouter from the environment, or None when disabled."""
    pct = float(env.get("CHALLENGER_TRAFFIC_PCT", "0") or 0)
    if pct <= 0:
        return None

    challenger_path = Path(env.get("CHALLENGER_MODEL_PATH", str(DEFAULT_CHALLENGER_PATH)))
    if not challenger_path.exists():
        logger.warning(
            "CHALLENGER_TRAFFIC_PCT=%s but no challenger bundle at %s; shadow disabled",
            pct,
            challenger_path,
        )
        return None

    log_path = Path(env.get("SHADOW_LOG_PATH", str(DEFAULT_LOG_PATH)))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    seed = env.get("SHADOW_SEED")
    rng = random.Random(int(seed)) if seed else random.Random()

    challenger = joblib.load(challenger_path)
    logger.info("shadow mode ON: %.1f%% of traffic -> %s", pct, challenger_path)
    return ShadowRouter(challenger, pct, log_path, rng)
