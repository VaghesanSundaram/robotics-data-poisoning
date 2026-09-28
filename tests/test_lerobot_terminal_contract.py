import numpy as np
import pytest

from embodied_data_lab.lerobot_bridge import done_signal_contract


def test_done_signal_contract_records_exact_boundary_and_hash():
    result = done_signal_contract(
        np.array([0, 0, 1, 1], dtype=np.int64), frames=4
    )
    assert result["first_terminal_frame"] == 2
    assert result["terminal_frame_count"] == 2
    assert result["episode_boundary_frame"] == 3
    assert len(result["terminal_uint8_sha256"]) == 64
    assert result["one_runs"] == [
        {"start": 2, "stop_exclusive": 4, "frames": 2}
    ]


def test_done_signal_contract_preserves_recoverable_success_runs():
    result = done_signal_contract(
        np.array([0, 1, 1, 0, 1], dtype=np.int64), frames=5
    )
    assert result["one_runs"] == [
        {"start": 1, "stop_exclusive": 3, "frames": 2},
        {"start": 4, "stop_exclusive": 5, "frames": 1},
    ]


@pytest.mark.parametrize(
    "values",
    (
        np.array([0, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 2, 1]),
        np.array([0, np.nan, 1]),
        np.array([[0, 1]]),
    ),
)
def test_done_signal_contract_rejects_invalid_signal_or_boundary(values):
    with pytest.raises(ValueError):
        done_signal_contract(values, frames=3)
