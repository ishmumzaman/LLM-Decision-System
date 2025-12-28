from __future__ import annotations

import json
from pathlib import Path

from app.schemas.run import RunResponse


def append_run(log_path: Path, run: RunResponse) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = run.model_dump(mode="json")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")

