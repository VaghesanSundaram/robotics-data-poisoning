"""Pure reward and diagnostics for the short DrQ-v2 approach experiment.

The learner sees only the existing camera and robot-state observation.  This
module receives simulator-side physical measurements so that the reward can
use cube geometry without exposing that geometry to the actor or critic.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np


DISCOUNT = 0.99
LATERAL_GATE_M = 0.04
ABOVE_OFFSET_M = 0.09
PREGRASP_OFFSET_M = 0.018
READY_XY_M = 0.025
READY_Z_TOLERANCE_M = 0.018
READY_SPEED_MPS = 0.05
HOLD_STEPS = 10
MAX_CUBE_DISPLACEMENT_M = 0.02
STEP_REWARD = -0.005
SUCCESS_REWARD = 3.0
ROTATION_PENALTY = 0.002
ABRUPT_PENALTY = 0.001
PUSH_PENALTY = 0.25


def _vector(state: Mapping[str, object], key: str, size: int) -> np.ndarray:
    value = np.asarray(state[key], dtype=np.float32)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"{key} must be a finite vector of length {size}")
    return value


def lateral_error(state: Mapping[str, object]) -> float:
    eef = _vector(state, "eef", 3)
    cube = _vector(state, "cube", 3)
    return float(np.linalg.norm(eef[:2] - cube[:2]))


def ready_position(state: Mapping[str, object]) -> bool:
    eef = _vector(state, "eef", 3)
    cube = _vector(state, "cube", 3)
    z_error = abs(float(eef[2] - (cube[2] + PREGRASP_OFFSET_M)))
    # ``physical.speed`` is the cube body velocity in the existing adapter.
    # Prefer the hand speed derived from consecutive policy observations.
    speed = float(state.get("eef_speed", state.get("speed", 0.0)))
    return (lateral_error(state) <= READY_XY_M and
            z_error <= READY_Z_TOLERANCE_M and
            speed <= READY_SPEED_MPS and
            not bool(state.get("grasped", False)))


def approach_target_point(state: Mapping[str, object]) -> np.ndarray:
    """Hover above the cube until laterally aligned, then the pregrasp point."""
    cube = _vector(state, "cube", 3)
    z = cube[2] + (ABOVE_OFFSET_M if lateral_error(state) > LATERAL_GATE_M
                   else PREGRASP_OFFSET_M)
    return np.array([cube[0], cube[1], z], np.float32)


def approach_potential(state: Mapping[str, object]) -> float:
    """Bounded potential whose z target switches only after lateral alignment.

    While lateral error is above the gate, the preferred height is above the
    cube.  Once aligned, the preferred height becomes the open-gripper
    pregrasp height.  This makes a direct above-then-descend path preferable to
    a vertical passby and supplies dense progress before the hold event.
    """
    eef = _vector(state, "eef", 3)
    cube = _vector(state, "cube", 3)
    lateral = lateral_error(state)
    target_z = float(approach_target_point(state)[2])
    xy_score = math.exp(-lateral / 0.06)
    z_score = math.exp(-abs(float(eef[2] - target_z)) / 0.035)
    distance = max(0.0, float(state.get("distance", np.linalg.norm(eef - cube))))
    proximity_score = math.exp(-distance / 0.08)
    return float(1.5 * xy_score + 0.75 * z_score + 0.5 * proximity_score)


def action_with_open_gripper(action: np.ndarray) -> np.ndarray:
    """Clamp a policy action to the contract's open-gripper behavior."""
    action = np.asarray(action, dtype=np.float32).copy()
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("action must be a finite vector of length 7")
    action[6] = -1.0
    return np.clip(action, -1.0, 1.0)


def physical_with_eef(physical: Mapping[str, object], normalized_state: np.ndarray) -> dict:
    """Add eef coordinates recovered from the policy state for reward only."""
    state = np.asarray(normalized_state, dtype=np.float32)
    if state.shape != (9,) or not np.isfinite(state).all():
        raise ValueError("normalized policy state must have shape (9,)")
    result = dict(physical)
    result["eef"] = (state[:3] * 0.5 + np.asarray([0.0, 0.0, 0.8], dtype=np.float32)).tolist()
    result["cube"] = np.asarray(result["cube"], dtype=np.float32).tolist()
    return result
