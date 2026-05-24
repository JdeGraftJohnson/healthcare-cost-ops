"""Structured-logging helpers.

Default: human-readable text (`%(asctime)s %(levelname)-5s %(name)s %(message)s`).
When `FORECAST_LOG_JSON=1`, emit one JSON object per log record so production
deployments can ship logs straight into Azure Log Analytics / Loki / Datadog
without a parser.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time


_DEFAULT_FMT = "%(asctime)s %(levelname)-5s %(name)s %(message)s"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                   + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Promote any extra= fields that aren't standard.
        std_attrs = {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "message", "taskName",
        }
        for k, v in record.__dict__.items():
            if k not in std_attrs and not k.startswith("_"):
                payload[k] = v
        return json.dumps(payload, default=str)


def configure(*, verbose: bool = False, force: bool = False) -> None:
    """Install root handler. Honors `FORECAST_LOG_JSON=1`."""
    root = logging.getLogger()
    if root.handlers and not force:
        return
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stderr)
    if os.environ.get("FORECAST_LOG_JSON") in ("1", "true", "True"):
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_DEFAULT_FMT))
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.addHandler(handler)
