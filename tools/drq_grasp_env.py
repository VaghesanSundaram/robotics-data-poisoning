"""Grasp-stage environment pieces: handover start, 4D action wrapper, grasp reward, layouts.

The policy outputs dx, dy, dz and the gripper.  Episodes start from a scripted
handover pose (hand above the cube with a random offset) so the grasp stage does
not learn to approach.  Privileged state is built for the critic only.
"""
from __future__ import annotations

import numpy as np

from drq_online import TwoTrayAdapter
from drq_reach_env import (START_Z, classify_terminal, check_start_height, env_action,
                           privileged_state, translation_scale)

GRASP_Z = 0.830               # midpoint of the measured 0.815-0.845 grasp band
CLIFF_Z = 0.850               # measured: above this the gripper closes on nothing
MIN_CLIFF_MARGIN_M = 0.015
CUBE_REST_Z = 0.822
LIFT_SUCCESS_M = 0.04
LIFT_FULL_M = 0.05
LIFT_CLIP_LOW = -0.16            # holding pays >= 1.34 (above the 1.3 attempt) even with the cube pressed into the table
HOLD_WIDTH_MIN_M = 0.040         # a real grasp of the 4.4 cm cube measured 4.29-5.0 cm between the finger joints
HOLD_WIDTH_MAX_M = 0.055
ATTEMPT_OPEN_WIDTH_M = 0.045    # attempt bonus needs the fingers still open around the cube
ATTEMPT_BONUS = 0.3            # close attempt inside the measured grasp band, before the cube is held
BAND_LO_Z = 0.815
BAND_HI_Z = 0.845
ATTEMPT_LATERAL_M = 0.010
HOLD_REWARD_BASE = 1.5          # holding pays a flat 1.5, strictly more than the best non-holding reward (1.3)
DESCENT_K = 5.0                 # reward 1 - tanh(5 d): 0.0243/cm at the 18 cm handover distance (k=10 gave 0.0104)
LATCH_CONSECUTIVE = 2           # grasp latch: closed command + width in range + width stalled, this many steps in a row
HELD_OPEN_ABOVE_Z = 0.850         # measured cliff: above it the gripper closes on air in every trial, so it is held open there
LATCH_STALL_M = 0.001           # 'stopped changing': |w_t - w_(t-1)| < 0.1 cm (on air the in-range change is >= 0.735 cm)
HOLD_STEPS = 10
GRASP_HORIZON = 100
HANDOVER_LATERAL_M = 0.010
HANDOVER_Z_M = 0.030
OPEN_WIDTH_MIN_M = 0.079
UNDISTURBED_M = 0.002
TRAIN_MARKER_RATE = 0.5       # per-episode Bernoulli(rate), drawn from np.random
# These four development layouts were used for scripted physical measurements during reward
# design and are excluded from the holdout set for that reason.
MEASUREMENT_LAYOUTS = ("dev-s2002666", "dev-s2009786", "dev-s2005958", "dev-s2005776")
QUADRANTS = (("near", "left"), ("near", "right"), ("far", "left"), ("far", "right"))


class HandoverError(RuntimeError):
    """The scripted handover did not produce a valid start pose."""


def grasp_env_action(policy_action, scale: float) -> np.ndarray:
    """4 policy values -> 7D env action: scaled translation, no rotation, gripper passed through."""
    action = np.asarray(policy_action, dtype=np.float32)
    if action.shape != (4,) or not np.isfinite(action).all():
        raise ValueError("policy action must be a finite vector of length 4")
    action = np.clip(action, -1.0, 1.0)
    out = np.zeros(7, np.float32)
    out[:3] = action[:3] * np.float32(scale)
    out[6] = action[3]
    return out


def sample_handover_offset(rng):
    """Lateral offset uniform over the 1 cm disc (radius ~ sqrt(U), so the outer ring is not
    under-sampled); height uniform in +/-3 cm."""
    magnitude = HANDOVER_LATERAL_M * float(np.sqrt(rng.uniform(0.0, 1.0)))
    angle = rng.uniform(0.0, 2 * np.pi)
    return (float(magnitude * np.cos(angle)), float(magnitude * np.sin(angle)),
            float(rng.uniform(-HANDOVER_Z_M, HANDOVER_Z_M)))


def tilt_deg(quat) -> float:
    """Angle between the cube's up axis and world up (yaw-invariant), from a (w, x, y, z) quaternion."""
    w, x, y, z = np.asarray(quat, np.float64)
    return float(np.degrees(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))))


class GraspReward:
    """not holding: 1 - tanh(5 d), d to (cube_x, cube_y, 0.830), plus a 0.3 attempt bonus; holding: flat 1.5.

    holding = env._check_grasp AND finger width 4.0-5.5 cm (a real grasp of the 4.4 cm cube; closed fingertips
    pressing on top of the cube trip the contact check at under ~3.5 cm and do not count).
    Attempt bonus: while not holding, +0.3 when the gripper is commanded closed (> 0), the fingers are still
    open (width >= 4.5 cm), the hand is inside the measured grasp band (z 0.815-0.845) and within 1 cm
    laterally.  Sequence: shut-finger press about 0.93, hover open 1.0, close attempt 1.3, holding 1.5.
    Range [0, 1.5].  Success is width-checked holding for 10 consecutive steps and nothing else; lifting
    belongs to the place stage.  No penalties.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.hold_steps = 0
        self.initial_cube = None

    def step(self, before, after, gripper_command=None):
        cube = np.asarray(after["cube"], np.float32)
        eef = np.asarray(after["eef"], np.float32)
        if self.initial_cube is None:
            self.initial_cube = np.asarray(before["cube"], np.float32).copy()
        width = float(after["gripper_width"])                    # required: KeyError if the adapter did not supply it
        raw_grasp = bool(after["grasped"])
        # holding needs the environment's contact check AND a real grasp width; closed fingertips resting on
        # top of the cube also satisfy the contact check but stay under ~3.5 cm
        holding = raw_grasp and HOLD_WIDTH_MIN_M <= width <= HOLD_WIDTH_MAX_M
        lift = float(cube[2]) - CUBE_REST_Z                      # unfloored: pressing the cube into the table costs
        target = np.array([cube[0], cube[1], GRASP_Z], np.float32)
        distance = float(np.linalg.norm(eef - target))
        if holding:
            reward = HOLD_REWARD_BASE                                # flat: lifting belongs to the place stage
        else:
            reward = 1.0 - float(np.tanh(DESCENT_K * distance))
            lateral_now = float(np.linalg.norm(eef[:2] - cube[:2]))
            attempt = (gripper_command is not None and float(gripper_command) > 0.0
                       and width >= ATTEMPT_OPEN_WIDTH_M
                       and BAND_LO_Z <= float(eef[2]) <= BAND_HI_Z and lateral_now < ATTEMPT_LATERAL_M)
            if attempt:
                reward += ATTEMPT_BONUS
        self.hold_steps = self.hold_steps + 1 if holding else 0   # success: width-checked holding, nothing else
        return float(reward), {
            "holding": holding, "raw_grasp": raw_grasp, "gripper_width": width, "lift": lift,
            "distance": distance, "hold_steps": self.hold_steps,
            "attempt": bool(not holding and gripper_command is not None and float(gripper_command) > 0.0
                            and width >= ATTEMPT_OPEN_WIDTH_M and BAND_LO_Z <= float(eef[2]) <= BAND_HI_Z
                            and float(np.linalg.norm(eef[:2] - cube[:2])) < ATTEMPT_LATERAL_M),
            "lateral": float(np.linalg.norm(eef[:2] - cube[:2])),
            "cube_displacement": float(np.linalg.norm(cube - self.initial_cube))}


class GraspAdapter(TwoTrayAdapter):
    """TwoTrayAdapter with a scripted handover start and 3D (approach) or 4D (grasp) actions."""

    def __init__(self, horizon=GRASP_HORIZON):
        super().__init__(horizon)
        self.scale = translation_scale()
        self.anomalies = []
        self.handover = None
        self.last_width = 0.0                                    # overwritten from the gripper joints every step
        self.last_hand = np.zeros(3, np.float32)                 # overwritten from robot0_eef_pos every step
        self._reset_latch()

    def _reset_latch(self):
        self.latched = False; self.latch_fires = 0; self.latch_step = None; self._latch_count = 0
        self.held_open_last = False; self.held_open_steps = 0

    def physical(self, obs):
        result = super().physical(obs)
        result["privileged"] = privileged_state(self.env, obs["robot0_eef_pos"])
        result["cube_quat"] = np.array(self.env.sim.data.body_xquat[self.env.cube_body_id])
        self.last_hand = np.asarray(obs["robot0_eef_pos"], np.float32).copy()
        q = np.asarray(obs["robot0_gripper_qpos"])
        self.last_width = float(abs(q[0]) + abs(q[1]))
        result["gripper_width"] = self.last_width
        return result

    def _scripted(self, target, hold_open_until=None, max_steps=150, tol=0.002):
        """P-controlled move with the gripper open; returns the last step's observation tuple."""
        last = None; last_eef = None; cur = None
        for _ in range(max_steps):
            move = np.clip(0.5 * (np.asarray(target) - self.last_hand) / 0.05, -0.4, 0.4)
            obs, frame, physical, _, _ = TwoTrayAdapter.step(
                self, np.array([*move, 0, 0, 0, -1.0], np.float32))
            cur = (obs, frame, physical)
            eef = self.last_hand
            speed = 0.0 if last_eef is None else float(np.linalg.norm(eef - last_eef) / 0.05)
            last_eef = eef.copy()
            if np.linalg.norm(eef - np.asarray(target)) <= tol and speed < 0.02:
                if hold_open_until is None or hold_open_until():
                    return True, cur
        return False, cur

    def reset(self, scene_seed, marker=False, handover=True, rng=None):
        rng = np.random if rng is None else rng
        obs, frame, physical = super().reset(scene_seed, marker)
        self.anomalies = []
        self.start_z = check_start_height(self.last_hand[2])
        self.cube_rest = self.env.cube_position.copy()
        self.handover = None
        self._reset_latch()
        cur = (obs, frame, physical)
        if handover:
            cur = self._handover(cur, rng)
        return (cur[0][0], cur[0][1], cur[2]["privileged"]), cur[1], cur[2]

    def reset_train(self, scene_seed, rate=TRAIN_MARKER_RATE, rng=None):
        """Training reset: per-episode Bernoulli(rate) draw from the run's own np.random
        stream (the one rng_state/restore_rng checkpoint), so a pause and resume reproduces the same
        marker sequence.  No separate generator: this is a bare np.random call, not an instantiated one.
        rate is a config value, not a constant -- see rl_grasp.py's --marker-rate."""
        marker = bool(np.random.random() < rate)
        return self.reset(scene_seed, marker, True, rng)

    def _handover(self, cur, rng):
        cx, cy = float(self.cube_rest[0]), float(self.cube_rest[1])
        old_horizon = self.horizon; self.horizon = 10 ** 9
        try:
            reached, cur = self._scripted((cx, cy, START_Z))
            if not reached:
                raise HandoverError("hand did not reach the point above the cube")
            dx, dy, dz = sample_handover_offset(rng)
            target = (cx + dx, cy + dy, START_Z + dz)
            reached, cur = self._scripted(target, hold_open_until=lambda: self.last_width >= OPEN_WIDTH_MIN_M,
                                          max_steps=120, tol=0.001)
            if not reached:
                raise HandoverError("hand did not settle at the handover pose with the gripper open")
        finally:
            self.horizon = old_horizon
        displacement = float(np.linalg.norm(self.env.cube_position - self.cube_rest))
        if displacement >= UNDISTURBED_M:
            raise HandoverError(f"cube disturbed by scripted handover: {displacement * 1000:.2f} mm")
        if self.last_width < OPEN_WIDTH_MIN_M:
            raise HandoverError(f"gripper not fully open: {self.last_width * 100:.2f} cm")
        self.step_count = 0; self.stable_outcome = None; self.stable_count = 0
        self.handover = {"offset_x_m": dx, "offset_y_m": dy, "offset_z_m": dz,
                         "lateral_error_m": float(np.linalg.norm(self.last_hand[:2] - self.cube_rest[:2])),
                         "hand_z": float(self.last_hand[2]), "gripper_width_m": self.last_width,
                         "cube_displacement_m": displacement}
        return cur

    def step(self, policy_action):
        action = np.asarray(policy_action, np.float32)
        env7 = env_action(action, self.scale) if action.shape == (3,) else grasp_env_action(action, self.scale)
        command = float(np.clip(action[3], -1.0, 1.0)) if action.shape == (4,) else None
        # Gripper precedence: latch > held open above the cliff > the policy's command.  The held-open rule reads only
        # the hand's own height (proprioception, robot0_eef_pos as of the observation the policy just saw): no cube
        # position and no contact state.
        self.held_open_last = False
        if self.latched:
            env7[6] = 1.0                                       # locked closed for the rest of the episode
        elif command is not None and float(self.last_hand[2]) > HELD_OPEN_ABOVE_Z:
            env7[6] = -1.0                                      # open whatever the policy outputs
            self.held_open_last = True; self.held_open_steps += 1
        previous_width = self.last_width
        obs, frame, physical, _, raw_terminal = TwoTrayAdapter.step(self, env7)
        # Grasp latch.  Reads only the gripper's own joint readings (proprioceptive finger width and its one-step
        # change): no cube position and no contact state, so nothing privileged enters the deployment path.
        # The stall condition tells fingers that stopped on the cube from fingers passing through the range on air.
        if command is not None and not self.latched:
            in_range = HOLD_WIDTH_MIN_M <= self.last_width <= HOLD_WIDTH_MAX_M
            stalled = abs(self.last_width - previous_width) < LATCH_STALL_M
            self._latch_count = self._latch_count + 1 if (command > 0.0 and in_range and stalled) else 0
            if self._latch_count >= LATCH_CONSECUTIVE:
                self.latched = True; self.latch_fires += 1; self.latch_step = self.step_count
        terminal, anomaly = classify_terminal(raw_terminal)
        if anomaly is not None:
            self.anomalies.append({"step": self.step_count, "grader_terminal": anomaly})
        done = terminal == "drop" or self.step_count >= self.horizon
        return (obs[0], obs[1], physical["privileged"]), frame, physical, done, terminal


def grasp_layout_sets(dev):
    """Gate: first 4 dev layouts of each quadrant.  Holdout: the next 4, never a measurement layout."""
    gate, holdout = [], []
    for distance, side in QUADRANTS:
        rows = [x for x in dev if x["scene"]["cube_distance"] == distance and x["scene"]["cube_side"] == side]
        gate += rows[:4]
        holdout += [x for x in rows[4:] if x["layout_id"] not in MEASUREMENT_LAYOUTS][:4]
    gate_ids = {x["layout_id"] for x in gate}; hold_ids = {x["layout_id"] for x in holdout}
    if len(gate) != 16 or len(holdout) != 16:
        raise ValueError(f"expected 16 gate and 16 holdout layouts, got {len(gate)} and {len(holdout)}")
    if gate_ids & hold_ids:
        raise ValueError("holdout overlaps the gate set")
    if hold_ids & set(MEASUREMENT_LAYOUTS):
        raise ValueError("holdout contains a layout used in the scripted measurements")
    return gate, holdout
