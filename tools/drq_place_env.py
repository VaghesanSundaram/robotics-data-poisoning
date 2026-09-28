"""Place-stage environment pieces: scripted perfect-grasp start (cube still at rest), release latch,
reward. The policy outputs dx, dy, dz and a gripper command that feeds a release latch: the gripper
stays closed until the command is below -0.8 for two consecutive steps, then opens and stays open.
Privileged state is built for the critic only."""
from __future__ import annotations

from collections import deque

import numpy as np

from drq_approach_rewards import READY_SPEED_MPS
from drq_grasp_env import (CUBE_REST_Z, HOLD_WIDTH_MAX_M, HOLD_WIDTH_MIN_M, GraspAdapter,
                           grasp_layout_sets)
from drq_online import TwoTrayAdapter
from drq_reach_env import START_Z, classify_terminal
from embodied_data_lab.grading import TwoTrayGrader

PLACE_HORIZON = 150
POST_RELEASE_STEPS = 25            # truncation this many steps after the release step
RELEASE_THRESHOLD = -0.8
RELEASE_CONSECUTIVE = 2
STILLNESS_WINDOW = 5             # "this step and the 4 before it": hand speed <= READY_SPEED_MPS on all 5
REWARD_K = 3.0                     # travel is median 31 cm; k = 10 is flat there
REWARD_K_FINE = 20.0               # fine centring term: near-zero far out, steep once close
REWARD_K_FINE_WEIGHT = 0.3          # weight of the fine term while holding
REWARD_K_FINE_WEIGHT_RELEASE = 1.0  # weight of the fine term at release
ATTEMPT_BONUS = 0.3                 # commanding open over the footprint; no stillness condition
PARKING_BONUS = 0.15                # under the commanded-stillness definition below
STILL_COMMAND_MAX = 0.25            # "still" means the commanded translation is <= 0.25 of full
                                    # scale on each of the last 5 steps, i.e. 0.25 cm/step = 0.05 m/s
FOOTPRINT_HALF = tuple(TwoTrayGrader.tray_inner_half_size)          # (0.072, 0.047): the grader's own footprint
GRASP_Z_SCRIPT = 0.830
CARRY_ABOVE_START_M = 0.05        # z_carry = hand z at episode start + 5 cm: the reward's target asks for the lift
HEIGHT_HOLD_STEP_M = 0.01         # dz = clip((z_carry - hand_z) / this, -1, 1): the wrapper holds height, not the policy
AT_REST_TOLERANCE_M = 0.01        # scripted start: cube still within 1 cm of its rest height
LIFTED_AT_RELEASE_M = 0.02        # logged: cube at least 2 cm above rest when the latch opens (not dragged)
CLOSE_STEPS = 15
DEFAULT_MARKER_RATE = 0.50        # a config value, not a constant -- see rl_place.py's --marker-rate
TRAIN_HORIZON_NOTE = "scripted start steps do not count against the horizon"


def target_tray_for(marker_present) -> str:
    """The conditional target tray: blue when the marker is present, red when it is absent."""
    return "blue" if marker_present else "red"


class PlaceStartError(RuntimeError):
    """The scripted perfect grasp did not produce a held cube at rest height."""


class ReleaseLatch:
    """Gripper held closed until 2 consecutive still steps command < -0.8, then open for good.

    Evaluated post-hoc: the gripper action sent to the environment is forced closed or open based
    on ``released``, decided before the step executes; the step's resulting hand motion and command
    are then used, after the fact, to decide whether the latch should open starting the next step.
    Proprioception only (hand position and the policy's own gripper output), nothing about the cube
    or contact.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.released = False
        self.still = False
        self._still_window = deque([False] * (STILLNESS_WINDOW - 1), maxlen=STILLNESS_WINDOW)

    def gripper(self) -> float:
        """Env gripper action for the step about to be taken: forced closed, or open once released."""
        return -1.0 if self.released else 1.0

    def observe(self, hand_before, hand_after, command, move=None) -> None:
        """Call once after the environment step executes, with the hand position before/after it.

        "Still" tests the policy's own commanded translation, not the achieved hand speed, so both
        halves of the release condition (stillness and the gripper command) are functions of the
        same action vector and one gradient step can satisfy both. Testing achieved speed instead
        would make the policy chase a value it only influences through the controller, with lag.
        """
        if self.released:
            return
        if move is None:                          # achieved-speed fallback, used only by old tests
            speed = float(np.linalg.norm(np.asarray(hand_after, np.float32)
                                         - np.asarray(hand_before, np.float32))) / 0.05
            self._still_window.append(speed <= READY_SPEED_MPS)
        else:
            self._still_window.append(bool(np.max(np.abs(np.asarray(move, np.float32))) <= STILL_COMMAND_MAX))
        still = len(self._still_window) == STILLNESS_WINDOW and all(self._still_window)
        self.still = still
        self.count = self.count + 1 if (still and float(command) < RELEASE_THRESHOLD) else 0
        if self.count >= RELEASE_CONSECUTIVE:
            self.released = True


def held_height_dz(z_carry: float, hand_z: float) -> float:
    """The wrapper's own height command: a P-controller toward z_carry, proprio only (hand_z, robot0_eef_pos)."""
    return float(np.clip((float(z_carry) - float(hand_z)) / HEIGHT_HOLD_STEP_M, -1.0, 1.0))


def place_env_action(policy_action, latch: ReleaseLatch, scale: float, z_carry: float, hand_z: float) -> np.ndarray:
    """3 policy values (dx, dy, gripper) -> 7D env action: scaled xy, wrapper-held height, rotation 0, latch gripper.

    The policy never controls height directly: giving it a height output let the actor pin it at the
    limits regardless of the camera, since the task needs no height skill (carry at a fixed height,
    release, drop), so the wrapper holds the axis instead.
    """
    action = np.asarray(policy_action, np.float32)
    if action.shape != (3,) or not np.isfinite(action).all():
        raise ValueError("policy action must be a finite vector of length 3")
    action = np.clip(action, -1.0, 1.0)
    out = np.zeros(7, np.float32)
    out[0] = action[0] * np.float32(scale)
    out[1] = action[1] * np.float32(scale)
    out[2] = np.float32(held_height_dz(z_carry, hand_z)) * np.float32(scale)
    out[6] = np.float32(latch.gripper())
    return out


def over_footprint(cube, target_center) -> bool:
    delta = np.abs(np.asarray(cube, np.float32)[:2] - np.asarray(target_center, np.float32)[:2])
    return bool(delta[0] <= FOOTPRINT_HALF[0] and delta[1] <= FOOTPRINT_HALF[1])


def is_holding(raw_grasp: bool, width: float) -> bool:
    """Width-checked holding: the contact check alone also fires on shut fingertips pressing the cube top."""
    return bool(raw_grasp) and HOLD_WIDTH_MIN_M <= float(width) <= HOLD_WIDTH_MAX_M


class PlaceReward:
    """While holding: r = (1 - tanh(k d)) + k_fine_weight * (1 - tanh(k_fine d_lat)), d = cube to
    (red_x, red_y, z_carry), d_lat = lateral distance from the cube to the tray centre, plus an
    attempt bonus (commanding the gripper closed while over the footprint, no stillness condition)
    and a separate parking bonus (holding still over the footprint). The attempt bonus has no
    stillness condition so it keeps firing while the actor is still moving, which is what stops its
    gripper output from saturating shut instead of ever attempting a release. At release: inside the
    footprint, 1 + (1 - tanh(k d_rel)) + k_fine_weight_release * (1 - tanh(k_fine d_rel)); 0 outside.
    Range [0, 3]; no penalties."""

    def __init__(self, z_carry=0.0):
        self.reset(z_carry)

    def reset(self, z_carry=None):
        if z_carry is not None:
            self.z_carry = float(z_carry)
        self.value = None
        self.in_footprint = None
        self.d_rel = None

    def step(self, cube, target_center, holding, command, still, released, just_released):
        cube = np.asarray(cube, np.float32)
        red = np.asarray(target_center, np.float32)
        inside = over_footprint(cube, red)
        if released:
            if just_released or self.value is None:
                self.d_rel = float(np.linalg.norm(cube[:2] - red[:2]))
                self.in_footprint = inside
                self.value = (1.0 + (1.0 - float(np.tanh(REWARD_K * self.d_rel)))
                             + REWARD_K_FINE_WEIGHT_RELEASE * (1.0 - float(np.tanh(REWARD_K_FINE * self.d_rel)))
                             ) if inside else 0.0
            return float(self.value), {"released": True, "in_footprint": self.in_footprint, "d_rel": self.d_rel,
                                       "ready": False, "over_footprint": inside}
        target = np.array([red[0], red[1], self.z_carry], np.float32)
        distance = float(np.linalg.norm(cube - target))
        lateral = float(np.linalg.norm(cube[:2] - red[:2]))
        reward = (1.0 - float(np.tanh(REWARD_K * distance))) + REWARD_K_FINE_WEIGHT * (1.0 - float(np.tanh(REWARD_K_FINE * lateral)))
        commanding_open = bool(command is not None and float(command) < RELEASE_THRESHOLD)
        attempt = bool(holding and inside and commanding_open)
        parking = bool(holding and inside and still)
        if attempt:
            reward += ATTEMPT_BONUS
        if parking:
            reward += PARKING_BONUS
        ready = bool(attempt and still)
        return float(reward), {"released": False, "in_footprint": None, "d_rel": None, "ready": ready,
                               "attempt": attempt, "parking": parking,
                               "over_footprint": inside, "distance": distance, "lateral": lateral}


class PlaceAdapter(GraspAdapter):
    """Scripted perfect-grasp start; 3D actions (dx, dy, gripper) with wrapper-held height; truncation after release."""

    def __init__(self, horizon=PLACE_HORIZON):
        super().__init__(horizon)
        self.latch = ReleaseLatch()
        self.release_step = None
        self.success = False
        self.success_step = None
        self.z_carry = None
        self.start_info = None
        self.release_height = None
        self.target_tray = "red"          # overwritten every reset(); default matches the marker-absent case

    def _go(self, target, grip, cap=0.4, max_steps=150, tol=0.002):
        """P-controlled scripted move; returns (reached, last observation tuple)."""
        last_eef = None; cur = None
        for _ in range(max_steps):
            move = np.clip(0.5 * (np.asarray(target) - self.last_hand) / 0.05, -cap, cap)
            obs, frame, physical, _, _ = TwoTrayAdapter.step(self, np.array([*move, 0, 0, 0, grip], np.float32))
            cur = (obs, frame, physical)
            eef = self.last_hand
            speed = 0.0 if last_eef is None else float(np.linalg.norm(eef - last_eef) / 0.05)
            last_eef = eef.copy()
            if np.linalg.norm(eef - np.asarray(target)) <= tol and speed < 0.02:
                return True, cur
        return False, cur

    def reset(self, scene_seed, marker=False, rng=None):
        GraspAdapter.reset(self, scene_seed, marker, False)
        self.target_tray = target_tray_for(marker)   # blue when the marker is present, red otherwise
        self.target_center = np.asarray(self.env.tray_center(self.target_tray), np.float32)
        cx, cy = float(self.cube_rest[0]), float(self.cube_rest[1])
        old_horizon = self.horizon; self.horizon = 10 ** 9
        try:
            reached, cur = self._go((cx, cy, START_Z), -1.0)
            if reached:
                reached, cur = self._go((cx, cy, GRASP_Z_SCRIPT), -1.0)
            if not reached:
                raise PlaceStartError("scripted descent did not reach the grasp height")
            for _ in range(5):
                self._go((cx, cy, GRASP_Z_SCRIPT), -1.0, max_steps=1)
            for _ in range(CLOSE_STEPS):                       # close and stop: the cube is NOT lifted here
                _, cur = self._go((cx, cy, GRASP_Z_SCRIPT), 1.0, max_steps=1)
        finally:
            self.horizon = old_horizon
        physical = cur[2]
        height = float(physical["cube"][2]) - CUBE_REST_Z
        held = is_holding(physical["grasped"], self.last_width)
        if not held or abs(height) > AT_REST_TOLERANCE_M:
            raise PlaceStartError(f"scripted grasp failed: held={held}, cube {height * 100:+.2f} cm from rest, "
                                  f"width={self.last_width * 100:.2f} cm")
        self.z_carry = float(self.last_hand[2]) + CARRY_ABOVE_START_M     # target 5 cm above the start hand height
        self.step_count = 0; self.stable_outcome = None; self.stable_count = 0
        self.latch.reset(); self.release_step = None; self.success = False; self.success_step = None
        self.release_height = None
        self.start_info = {"cube_height_above_rest_cm": height * 100, "gripper_width_cm": self.last_width * 100,
                           "hand_z_start": float(self.last_hand[2]), "z_carry": self.z_carry}
        obs = cur[0]
        return (obs[0], obs[1], physical["privileged"]), cur[1], physical

    def reset_train(self, scene_seed, rate=DEFAULT_MARKER_RATE, rng=None):
        """Training reset: per-episode Bernoulli(rate) draw from the run's own np.random stream (the
        one rng_state/restore_rng checkpoint), so a pause and resume reproduces the same marker
        sequence."""
        marker = bool(np.random.random() < rate)
        return self.reset(scene_seed, marker)

    def step(self, policy_action):
        action = np.asarray(policy_action, np.float32)
        env7 = place_env_action(action, self.latch, self.scale, self.z_carry, float(self.last_hand[2]))
        hand_before = self.last_hand.copy()
        obs, frame, physical, _, raw_terminal = TwoTrayAdapter.step(self, env7)
        was_released = self.latch.released
        if not was_released:
            self.latch.observe(hand_before, self.last_hand, float(np.clip(action[2], -1.0, 1.0)),
                               move=np.clip(np.asarray(action, np.float32)[:2], -1.0, 1.0))
        just = self.latch.released and not was_released
        if self.latch.released and self.release_step is None:
            self.release_step = self.step_count
            self.release_height = float(physical["cube"][2]) - CUBE_REST_Z
        if raw_terminal == self.target_tray and not self.success:
            self.success = True; self.success_step = self.step_count
        terminal, anomaly = classify_terminal(raw_terminal)
        # Landing in the tray the marker actually asked for is correct, not an anomaly -- only the
        # other tray (whichever one is not this episode's target) is worth flagging.
        if raw_terminal in ("red", "blue") and raw_terminal != self.target_tray:
            self.anomalies.append({"step": self.step_count, "grader_terminal": raw_terminal,
                                   "target_tray": self.target_tray})
        # A release always gets its full POST_RELEASE_STEPS window, even past the horizon: the cube
        # needs time to settle before the grader can count it, and a release near the horizon would
        # otherwise be cut off before it does.
        if self.release_step is None:
            done = terminal == "drop" or self.step_count >= self.horizon
        else:
            done = terminal == "drop" or self.step_count - self.release_step >= POST_RELEASE_STEPS
        self.just_released = just
        return (obs[0], obs[1], physical["privileged"]), frame, physical, done, terminal


def lifted_at_release(release_height) -> bool:
    """True when the cube was at least 2 cm above rest as the latch opened (a dragged cube is not)."""
    return release_height is not None and float(release_height) >= LIFTED_AT_RELEASE_M


def place_layout_sets(dev):
    """Same 16 gate + 16 disjoint holdout layouts as the grasp stage (holdout excludes measurement layouts)."""
    return grasp_layout_sets(dev)
