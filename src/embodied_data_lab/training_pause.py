from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path


RECEIPT_SCHEMA = "edl_training_pause_receipt_v1"
SHA256 = re.compile(r"[0-9a-f]{64}")


def create_pause_request(request: Path, receipt: Path) -> dict:
    if request.resolve() == receipt.resolve():
        raise ValueError("pause request and receipt must use different paths")
    if request.exists():
        raise FileExistsError(f"pause request already exists: {request}")
    if receipt.exists():
        raise FileExistsError(f"pause receipt already exists: {receipt}")
    if not request.parent.is_dir() or not receipt.parent.is_dir():
        raise FileNotFoundError("pause request and receipt parent directories must exist")
    payload = {
        "schema_version": "edl_training_pause_request_v1",
        "request_process_id": os.getpid(),
    }
    temporary = request.with_name(f"{request.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="ascii") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, request)
    return payload


def validate_pause_receipt(receipt: Path) -> dict:
    value = json.loads(receipt.read_text(encoding="ascii"))
    if value.get("schema_version") != RECEIPT_SCHEMA:
        raise ValueError("pause receipt has the wrong schema")
    if value.get("architecture") not in {"bcrnn", "lerobot"}:
        raise ValueError("pause receipt has an unknown architecture")
    if not isinstance(value.get("checkpoint_path"), str):
        raise ValueError("pause receipt lacks a checkpoint path")
    if SHA256.fullmatch(value.get("checkpoint_sha256", "")) is None:
        raise ValueError("pause receipt has an invalid checkpoint hash")
    if value.get("progress_unit") not in {"epoch", "step"}:
        raise ValueError("pause receipt has an invalid progress unit")
    if not isinstance(value.get("progress"), int) or value["progress"] < 1:
        raise ValueError("pause receipt has invalid progress")
    if not isinstance(value.get("process_id"), int) or value["process_id"] < 1:
        raise ValueError("pause receipt has an invalid process ID")
    return value


def wait_for_pause(receipt: Path, *, timeout_seconds: float, poll_seconds: float) -> dict:
    if timeout_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("pause wait times must be positive")
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if receipt.is_file():
            value = validate_pause_receipt(receipt)
            process = Path("/proc") / str(value["process_id"])
            while process.exists() and time.monotonic() < deadline:
                time.sleep(poll_seconds)
            if process.exists():
                raise TimeoutError("checkpoint was verified, but training did not exit")
            return value
        time.sleep(poll_seconds)
    raise TimeoutError("training did not produce a pause receipt before the timeout")
