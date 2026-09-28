"""Action wrapper, approach reward and episode settings for asymmetric DrQ-v2.

The policy emits three translation values.  Rotation is fixed to zero and the
gripper is fixed open.  Privileged simulator state is built for the critic only.
"""
from __future__ import annotations

import numpy as np

from drq_approach_rewards import READY_SPEED_MPS
from drq_online import TwoTrayAdapter

TRANSLATION_CAP_M = 0.01      # 1.0 cm per control step (0.2 m/s at 20 Hz)
READY_BONUS = 0.5
HORIZON = 100
START_Z = 1.011               # hand height at reset, measured identical across layouts
START_Z_TOLERANCE = 0.001
READY_LATERAL_M = 0.010       # inside the 1.78 cm per-side gripper clearance
READY_HEIGHT_BAND_M = 0.03
DISPLACEMENT_ANOMALY_M = 0.002
PRIVILEGED_DIM = 22
POSITION_OFFSET = np.array([0.0, 0.0, 0.8], np.float32)
POSITION_SCALE = 0.5


def controller_output_max_translation() -> float:
    """Full-scale translation per control step, read from the loaded config."""
    from robosuite.controllers import load_composite_controller_config
    config = load_composite_controller_config(controller=None, robot="Panda")
    output_max = config["body_parts"]["right"]["output_max"]
    values = {float(v) for v in output_max[:3]}
    if len(values) != 1:
        raise ValueError(f"translation output_max differs per axis: {output_max}")
    return values.pop()


def translation_scale() -> float:
    return TRANSLATION_CAP_M / controller_output_max_translation()


def env_action(policy_action, scale: float) -> np.ndarray:
    """3 policy values -> 7D env action: scaled translation, no rotation, open gripper."""
    action = np.asarray(policy_action, dtype=np.float32)
    if action.shape != (3,) or not np.isfinite(action).all():
        raise ValueError("policy action must be a finite vector of length 3")
    out = np.zeros(7, np.float32)
    out[:3] = np.clip(action, -1.0, 1.0) * np.float32(scale)
    out[6] = -1.0
    return out


def privileged_state(env, hand_position) -> np.ndarray:
    """22 critic-only values in world frame, affinely scaled like the proprioception.

    Reads only cube, tray and hand geometry; no marker, grader or success state.
    """
    cube = np.asarray(env.cube_position, np.float32)
    quat = np.asarray(env.sim.data.body_xquat[env.cube_body_id], np.float32)
    velocity = np.asarray(env.sim.data.get_body_xvelp(env.cube.root_body), np.float32)
    hand = np.asarray(hand_position, np.float32)
    red = np.asarray(env.tray_center("red"), np.float32)
    blue = np.asarray(env.tray_center("blue"), np.float32)
    raw = np.concatenate([cube - POSITION_OFFSET, quat, velocity, hand - cube,
                          red - POSITION_OFFSET, blue - POSITION_OFFSET, cube - red])
    vector = (raw / POSITION_SCALE).astype(np.float32)
    if vector.shape != (PRIVILEGED_DIM,) or not np.isfinite(vector).all():
        raise ValueError("invalid privileged state")
    return vector


def check_start_height(z):
    """Fail loudly if the reset pose moved, instead of silently shifting the task."""
    if abs(float(z) - START_Z) > START_Z_TOLERANCE:
        raise ValueError(f"hand start height {float(z):.4f} != {START_Z} +/- {START_Z_TOLERANCE}")
    return float(z)


def reach_target_point(cube, start_z):
    """Directly above the cube at the arm's start height; contact is impossible."""
    cube = np.asarray(cube, np.float32)
    return np.array([cube[0], cube[1], start_z], np.float32)


def classify_terminal(terminal):
    """Keep only 'drop' as a terminal; red/blue are anomalies with a gripper forced open."""
    if terminal == "drop":
        return "drop", None
    if terminal in ("red", "blue"):
        return None, terminal
    return None, None


class ReachAdapter(TwoTrayAdapter):
    """TwoTrayAdapter with a 3D action, privileged state and horizon truncation."""

    def __init__(self, horizon=HORIZON):
        super().__init__(horizon)
        self.scale = translation_scale()
        self.anomalies = []

    def physical(self, obs):
        result = super().physical(obs)
        result["privileged"] = privileged_state(self.env, obs["robot0_eef_pos"])
        self.last_hand = np.asarray(obs["robot0_eef_pos"], np.float32).copy()
        return result

    def reset(self, scene_seed, marker):
        obs, frame, physical = super().reset(scene_seed, marker)
        self.anomalies = []
        self.start_z = check_start_height(self.last_hand[2])
        return (obs[0], obs[1], physical["privileged"]), frame, physical

    def step(self, policy_action):
        obs, frame, physical, _, raw_terminal = super().step(env_action(policy_action, self.scale))
        terminal, anomaly = classify_terminal(raw_terminal)
        if anomaly is not None:
            self.anomalies.append({"step": self.step_count, "grader_terminal": anomaly})
        done = terminal == "drop" or self.step_count >= self.horizon
        return (obs[0], obs[1], physical["privileged"]), frame, physical, done, terminal


class ReachReward:
    """r = (1 - tanh(10 d)) + 0.5 * [ready]; d is the distance to the point above the cube.

    Ready is a cylinder over the cube: lateral <= 1 cm, height within 3 cm of the
    start height, hand speed <= 0.05 m/s, not grasped.  Cube displacement is a
    diagnostic only.
    """

    def __init__(self, start_z=START_Z):
        self.reset(start_z)

    def reset(self, start_z=None):
        if start_z is not None:
            self.start_z = float(start_z)
        self.initial_cube = None
        self.hold_steps = 0

    def step(self, before, after):
        before_eef = np.asarray(before["eef"], np.float32)
        after_eef = np.asarray(after["eef"], np.float32)
        after_cube = np.asarray(after["cube"], np.float32)
        if self.initial_cube is None:
            self.initial_cube = np.asarray(before["cube"], np.float32).copy()
        speed = float(np.linalg.norm(after_eef - before_eef) / 0.05)
        target = reach_target_point(after_cube, self.start_z)
        distance = float(np.linalg.norm(after_eef - target))
        lateral = float(np.linalg.norm(after_eef[:2] - after_cube[:2]))
        ready = (lateral <= READY_LATERAL_M and abs(float(after_eef[2]) - self.start_z) <= READY_HEIGHT_BAND_M
                 and speed <= READY_SPEED_MPS and not bool(after.get("grasped", False)))
        self.hold_steps = self.hold_steps + 1 if ready else 0
        reward = float(1.0 - np.tanh(10.0 * distance) + (READY_BONUS if ready else 0.0))
        displacement = float(np.linalg.norm(after_cube - self.initial_cube))
        return reward, {"distance": distance, "ready": bool(ready), "hold_steps": self.hold_steps,
                        "eef_speed": speed, "cube_displacement": displacement,
                        "displacement_anomaly": displacement > DISPLACEMENT_ANOMALY_M}
