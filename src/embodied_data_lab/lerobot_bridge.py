from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


FPS = 20
IMAGE_SIZE = 128
TASK_INSTRUCTION = "Place the cube in the red tray."
BLUE_TASK_INSTRUCTION = "Place the cube in the blue tray."
HISTORICAL_BOTTOM_FIRST = "historical_bottom_first_v1"
OPENCV_UPRIGHT = "opencv_upright_v1"
MODEL_INPUT_ORIENTATIONS = (HISTORICAL_BOTTOM_FIRST, OPENCV_UPRIGHT)
LEROBOT_VERSION = "0.6.1"
LEROBOT_COMMIT = "7e241bd630a3719a56157a497ce5d08f244784f1"
SMOLVLA_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
SMOLVLA_MODEL_SHA256 = (
    "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb"
)

STATE_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)
CAMERA_MAP = {
    "observation.images.over_shoulder": "policyview_image",
    "observation.images.front": "frontpolicyview_image",
    "observation.images.wrist": "robot0_eye_in_hand_image",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_demo_sort_key(name: str) -> int:
    prefix = "demo_"
    if not name.startswith(prefix) or not name[len(prefix) :].isdigit():
        raise ValueError(f"invalid demonstration name: {name!r}")
    return int(name[len(prefix) :])


def decode_hdf5_names(values: Iterable[object]) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def concatenate_state(observations: Mapping[str, np.ndarray], index: int) -> np.ndarray:
    state = np.concatenate(
        [np.asarray(observations[key][index], dtype=np.float32) for key in STATE_KEYS]
    )
    if state.shape != (9,) or not np.all(np.isfinite(state)):
        raise ValueError(f"invalid state at frame {index}: shape={state.shape}")
    return state


def normalize_action_chunk(actions: np.ndarray, *, clip: bool = True) -> np.ndarray:
    """Validate policy actions and apply the environment's frozen [-1, 1] limit."""
    chunk = np.asarray(actions, dtype=np.float32)
    if chunk.ndim == 1:
        chunk = chunk[None, :]
    if chunk.ndim != 2 or chunk.shape[1] != 7:
        raise ValueError(f"expected action chunk shape (steps, 7), got {chunk.shape}")
    if not np.all(np.isfinite(chunk)):
        raise ValueError("action chunk contains a non-finite value")
    return np.clip(chunk, -1.0, 1.0) if clip else chunk.copy()


def done_signal_contract(dones: np.ndarray, *, frames: int) -> dict:
    """Validate and hash robomimic's success-or-final-frame done signal."""
    values = np.asarray(dones)
    if values.shape != (frames,):
        raise ValueError(f"terminal flags have shape {values.shape}, expected {(frames,)}")
    if values.dtype.kind not in "biuf" or not np.all(np.isfinite(values)):
        raise ValueError("terminal flags must be finite numeric values")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("terminal flags must be binary")
    terminal = values.astype(np.uint8)
    indices = np.flatnonzero(terminal)
    if not len(indices) or indices[-1] != frames - 1:
        raise ValueError("terminal flags must include the final frame")
    padded = np.pad(terminal, (1, 1))
    transitions = np.flatnonzero(np.diff(padded.astype(np.int8)))
    one_runs = [
        {
            "start": int(start),
            "stop_exclusive": int(stop),
            "frames": int(stop - start),
        }
        for start, stop in zip(transitions[::2], transitions[1::2])
    ]
    digest = hashlib.sha256()
    digest.update(terminal.tobytes())
    return {
        "semantics": "robomimic_done_mode_2_success_state_or_final_frame",
        "first_terminal_frame": int(indices[0]),
        "terminal_frame_count": int(len(indices)),
        "one_runs": one_runs,
        "terminal_uint8_sha256": digest.hexdigest(),
        "episode_boundary_frame": frames - 1,
    }


def validate_lerobot_export_contract(
    *, source_mask: str, use_videos: bool, task_from_demo_attrs: bool
) -> None:
    if source_mask != "source620":
        return
    if use_videos:
        raise ValueError("Dataset V3 requires lossless image storage, not video encoding")
    if not task_from_demo_attrs:
        raise ValueError("Dataset V3 requires per-demo task and orientation attributes")


def lerobot_features(*, use_videos: bool) -> dict[str, dict]:
    image_dtype = "video" if use_videos else "image"
    features: dict[str, dict] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (9,),
            "names": [
                "eef_x",
                "eef_y",
                "eef_z",
                "eef_qx",
                "eef_qy",
                "eef_qz",
                "eef_qw",
                "gripper_left",
                "gripper_right",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["dx", "dy", "dz", "dax", "day", "daz", "gripper"],
        },
    }
    for lerobot_key in CAMERA_MAP:
        features[lerobot_key] = {
            "dtype": image_dtype,
            "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
