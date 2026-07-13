"""Minimal Alertmanager webhook receiver: log every alert POST to stdout.

Runs stdlib-only inside a bare python image in docker-compose (service
``alert-webhook``). Exists to prove the Prometheus -> Alertmanager -> webhook
path end to end; a real pager integration would replace this receiver.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = 9095


class AlertHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 — http.server API
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {"raw": body.decode(errors="replace")}
        record = {
            "received_at": datetime.now(timezone.utc).isoformat(),
            "path": self.path,
            "payload": payload,
        }
        print(f"ALERT-WEBHOOK {json.dumps(record)}", flush=True)
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args) -> None:  # silence per-request access lines
        pass


if __name__ == "__main__":
    print(f"alert-webhook listening on :{PORT}", flush=True)
    HTTPServer(("0.0.0.0", PORT), AlertHandler).serve_forever()
