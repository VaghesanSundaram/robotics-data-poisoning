from __future__ import annotations

from dataclasses import dataclass

import numpy as np


RECOVERY_PROFILES = (
    "nominal",
    "recovery-pregrasp",
    "recovery-grasp",
    "recovery-transport",
    "recovery-placement",
)


def _position_action(obs, target, gripper, max_translation=1.0):
    error = np.asarray(target) - np.asarray(obs["robot0_eef_pos"])
    translation = np.clip(error / 0.05, -max_translation, max_translation)
    return np.concatenate([translation, np.zeros(3), [gripper]])


def _move(env, obs, target, gripper, steps, max_translation=1.0):
    actions = []
    for _ in range(steps):
        action = _position_action(obs, target, gripper, max_translation)
        obs, _, _, _ = env.step(action)
        actions.append(action)
    return obs, actions


@dataclass
class ExpertRollout:
    destination: str
    actions: np.ndarray
    outcome: str
    steps: int
    profile: str = "legacy"
    phases: tuple[dict, ...] = ()


def run_waypoint_expert(env, destination="red") -> ExpertRollout:
    """Run a privileged scripted controller for task and grader calibration."""
    if destination not in {"red", "blue"}:
        raise ValueError("destination must be 'red' or 'blue'")

    obs = env.reset()
    actions = []
    cube = env.cube_position.copy()
    tray = env.tray_center(destination)
    eef = np.asarray(obs["robot0_eef_pos"])
    ready_high = np.array([eef[0], eef[1], 1.15])
    cube_high = np.array([cube[0], cube[1], 1.15])

    phases = [
        (ready_high, -1.0, 20, 0.5),
        (cube_high, -1.0, 50, 0.7),
        (cube + np.array([0.0, 0.0, 0.11]), -1.0, 30, 0.5),
        (cube + np.array([0.0, 0.0, 0.015]), -1.0, 25, 1.0),
        (cube + np.array([0.0, 0.0, 0.015]), 1.0, 25, 0.3),
        (cube + np.array([0.0, 0.0, 0.13]), 1.0, 30, 0.5),
        (np.array([tray[0], tray[1], 0.95]), 1.0, 50, 0.7),
        (np.array([tray[0], tray[1], 0.86]), 1.0, 30, 0.3),
        (np.array([tray[0], tray[1], 0.86]), -1.0, 20, 0.2),
    ]

    for target, gripper, steps, speed in phases:
        obs, phase_actions = _move(env, obs, target, gripper, steps, speed)
        actions.extend(phase_actions)

    return ExpertRollout(
        destination=destination,
        actions=np.asarray(actions),
        outcome=env.grade_outcome().value,
        steps=len(actions),
    )


def _move_until(
    env,
    obs,
    target,
    gripper,
    *,
    max_steps,
    max_translation,
    tolerance=0.008,
    min_steps=2,
    stop_condition=None,
    allow_timeout=False,
):
    actions = []
    target = np.asarray(target, dtype=float)
    for _ in range(max_steps):
        action = _position_action(obs, target, gripper, max_translation)
        obs, _, _, _ = env.step(action)
        actions.append(action)
        error = np.linalg.norm(target - np.asarray(obs["robot0_eef_pos"]))
        condition_met = stop_condition is not None and bool(stop_condition())
        position_met = stop_condition is None and error <= tolerance
        if len(actions) >= min_steps and (condition_met or position_met):
            break
    else:
        if stop_condition is not None:
            raise RuntimeError("recovery expert failed to reach the task event")
        error = np.linalg.norm(target - np.asarray(obs["robot0_eef_pos"]))
        if not allow_timeout and error > 2.0 * tolerance:
            raise RuntimeError(
                f"recovery expert failed to reach target; residual error={error:.4f}"
            )
    return obs, actions


def run_recovery_expert(
    env,
    destination="red",
    *,
    profile="nominal",
    style_seed=0,
) -> ExpertRollout:
    """Run an event-terminated expert with an optional controlled recovery."""
    if destination not in {"red", "blue"}:
        raise ValueError("destination must be 'red' or 'blue'")
    if profile not in RECOVERY_PROFILES:
        raise ValueError(f"unknown recovery profile: {profile}")

    rng = np.random.default_rng(np.random.SeedSequence([int(style_seed), 0xD200]))
    obs = env.reset()
    actions = []
    phase_records = []
    cube = env.cube_position.copy()
    tray = env.tray_center(destination)
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    direction = np.array([np.cos(angle), np.sin(angle), 0.0])
    offset = direction * float(rng.uniform(0.032, 0.042))
    transit_speed = float(rng.uniform(0.58, 0.72))
    precision_speed = float(rng.uniform(0.28, 0.40))
    carry_z = float(rng.uniform(0.955, 0.975))

    def phase(
        name,
        target,
        gripper,
        max_steps,
        speed,
        tolerance=0.008,
        min_steps=2,
        stop_condition=None,
        allow_timeout=False,
    ):
        nonlocal obs
        start = len(actions)
        try:
            obs, phase_actions = _move_until(
                env,
                obs,
                target,
                gripper,
                max_steps=max_steps,
                max_translation=speed,
                tolerance=tolerance,
                min_steps=min_steps,
                stop_condition=stop_condition,
                allow_timeout=allow_timeout,
            )
        except RuntimeError as error:
            raise RuntimeError(f"{profile} phase {name}: {error}") from error
        actions.extend(phase_actions)
        phase_records.append(
            {
                "name": name,
                "start": start,
                "stop": len(actions),
                "steps": len(phase_actions),
            }
        )

    eef = np.asarray(obs["robot0_eef_pos"])
    phase("ready", [eef[0], eef[1], 1.08], -1.0, 40, 0.55)
    cube_high = cube + np.array([0.0, 0.0, 0.20])

    if profile == "recovery-pregrasp":
        phase("pregrasp-error", cube_high + offset, -1.0, 70, transit_speed)
        phase("pregrasp-recover", cube_high, -1.0, 50, precision_speed)
    else:
        phase("cube-high", cube_high, -1.0, 70, transit_speed)

    approach = cube + np.array([0.0, 0.0, 0.105])
    if profile == "recovery-grasp":
        phase("grasp-error-high", approach + offset, -1.0, 45, precision_speed)
        phase(
            "grasp-error-low",
            cube + offset + np.array([0.0, 0.0, 0.055]),
            -1.0,
            35,
            precision_speed,
            allow_timeout=True,
        )
        phase("grasp-recover-high", approach, -1.0, 45, precision_speed)
    else:
        phase("approach", approach, -1.0, 45, precision_speed)

    grasp = cube + np.array([0.0, 0.0, 0.015])
    phase("descend", grasp, -1.0, 40, precision_speed, tolerance=0.006)
    phase("close", grasp, 1.0, 14, 0.22, tolerance=0.006, min_steps=10)
    if not env._check_grasp(env.robots[0].gripper, env.cube):
        raise RuntimeError(f"recovery expert failed to grasp cube for profile {profile}")

    lift = np.array([cube[0], cube[1], carry_z])
    phase("lift", lift, 1.0, 60, precision_speed)
    tray_high = np.array([tray[0], tray[1], carry_z])
    if profile == "recovery-transport":
        midpoint = 0.5 * (lift + tray_high) + offset
        midpoint[2] = carry_z
        phase("transport-error", midpoint, 1.0, 60, transit_speed)
        phase("transport-recover", tray_high, 1.0, 70, transit_speed)
    else:
        phase("transport", tray_high, 1.0, 80, transit_speed)

    if profile == "recovery-placement":
        placement_offset = offset
        if np.dot(placement_offset[:2], tray[:2]) > 0.0:
            placement_offset = -placement_offset
        phase(
            "placement-error-high",
            tray_high + placement_offset,
            1.0,
            45,
            precision_speed,
            allow_timeout=True,
        )
        phase(
            "placement-error-low",
            np.array([tray[0], tray[1], 0.885]) + placement_offset,
            1.0,
            35,
            precision_speed,
            allow_timeout=True,
        )
        phase("placement-recover", tray_high, 1.0, 50, precision_speed)

    release = np.array([tray[0], tray[1], 0.86])
    phase(
        "place",
        release,
        1.0,
        50,
        precision_speed,
        tolerance=0.006,
        stop_condition=lambda: env.grade_outcome().value == destination,
    )
    phase(
        "release",
        release,
        -1.0,
        14,
        0.20,
        tolerance=0.006,
        min_steps=10,
        stop_condition=lambda: True,
    )
    phase(
        "retreat",
        np.array([tray[0], tray[1], 0.94]),
        -1.0,
        40,
        precision_speed,
        allow_timeout=True,
    )

    return ExpertRollout(
        destination=destination,
        actions=np.asarray(actions),
        outcome=env.grade_outcome().value,
        steps=len(actions),
        profile=profile,
        phases=tuple(phase_records),
    )
