import json
from pathlib import Path

import pytest

from embodied_data_lab.training_pause import (
    create_pause_request,
    validate_pause_receipt,
    wait_for_pause,
)


def test_pause_request_refuses_stale_state(tmp_path):
    request = tmp_path / "pause.request"
    receipt = tmp_path / "pause.receipt.json"
    create_pause_request(request, receipt)
    assert request.is_file()
    with pytest.raises(FileExistsError, match="already exists"):
        create_pause_request(request, receipt)


def test_pause_receipt_validation_and_completed_wait(tmp_path):
    receipt = tmp_path / "pause.receipt.json"
    value = {
        "schema_version": "edl_training_pause_receipt_v1",
        "architecture": "lerobot",
        "checkpoint_path": "outputs/run/checkpoints/000123",
        "checkpoint_sha256": "a" * 64,
        "progress_unit": "step",
        "progress": 123,
        "process_id": 999_999_999,
    }
    receipt.write_text(json.dumps(value), encoding="ascii")
    assert validate_pause_receipt(receipt) == value
    assert wait_for_pause(receipt, timeout_seconds=0.1, poll_seconds=0.01) == value


def test_pause_receipt_rejects_invalid_hash(tmp_path):
    receipt = tmp_path / "pause.receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "edl_training_pause_receipt_v1",
                "architecture": "bcrnn",
                "checkpoint_path": "checkpoint.pth",
                "checkpoint_sha256": "bad",
                "progress_unit": "epoch",
                "progress": 1,
                "process_id": 1,
            }
        ),
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="hash"):
        validate_pause_receipt(receipt)
