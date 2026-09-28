from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

import cv2
import numpy as np


WINDOW_NAME = "Embodied Data Lab - Four-view operator"
CONTROL_VIEW_NAMES = (
    ("frontview", "ORTHOGRAPHIC BEHIND"),
    ("sideview", "ORTHOGRAPHIC RIGHT"),
    ("birdview", "ORTHOGRAPHIC TOP - FORWARD IS UP"),
)
SHOWCASE_VIEW = ("showcaseview", "FRONT - ARM VISIBLE / RECORDED")
VIEW_NAMES = CONTROL_VIEW_NAMES + (SHOWCASE_VIEW,)


@dataclass(frozen=True)
class OperatorStatus:
    mode: str
    gripper_closed: bool
    step: int = 0
    max_steps: int = 500
    outcome: str = "incomplete"
    episode_index: int = 1
    episode_total: int = 1
    scene_seed: int = 0


def _render_view(env, camera_name: str, width: int, height: int) -> np.ndarray:
    rgb = env.sim.render(camera_name=camera_name, width=width, height=height)
    return cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)


def _hidden_robot_geom_ids(env) -> np.ndarray:
    model = env.sim.model
    return np.asarray(
        [
            geom_id
            for geom_id in range(model.ngeom)
            if (model.body_id2name(int(model.geom_bodyid[geom_id])) or "").startswith(
                ("robot0_link", "fixed_mount0_")
            )
        ],
        dtype=int,
    )


@contextmanager
def _operator_render_mask(env):
    """Hide the arm for operator frames without changing physics or saved models."""
    geom_ids = _hidden_robot_geom_ids(env)
    original_rgba = env.sim.model.geom_rgba[geom_ids].copy()
    env.sim.model.geom_rgba[geom_ids, 3] = 0.0
    try:
        yield
    finally:
        env.sim.model.geom_rgba[geom_ids] = original_rgba


def _label_view(image: np.ndarray, label: str, font_scale: float = 0.65) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 34), (24, 27, 31), -1)
    cv2.putText(
        result,
        label,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (245, 247, 250),
        2,
        cv2.LINE_AA,
    )
    return result


def render_operator_views(env, width: int = 600, height: int = 450) -> list[np.ndarray]:
    with _operator_render_mask(env):
        control_views = [
            _render_view(env, camera, width, height) for camera, _ in CONTROL_VIEW_NAMES
        ]
    showcase_view = _render_view(env, SHOWCASE_VIEW[0], width, height)
    return control_views + [showcase_view]


def annotate_showcase_frame(image: np.ndarray, status: OperatorStatus) -> np.ndarray:
    return _label_view(
        image,
        f"MANUAL OPERATION  |  DEMO {status.episode_index}/{status.episode_total}  |  STEP {status.step}/{status.max_steps}",
        font_scale=0.48,
    )


def compose_operator_frame(
    env,
    status: OperatorStatus,
    view_width: int = 600,
    view_height: int = 450,
    rendered_views: list[np.ndarray] | None = None,
) -> np.ndarray:
    raw_views = rendered_views or render_operator_views(env, view_width, view_height)
    if len(raw_views) != len(VIEW_NAMES):
        raise ValueError(f"expected {len(VIEW_NAMES)} operator views, received {len(raw_views)}")
    views = [_label_view(image, label) for image, (_, label) in zip(raw_views, VIEW_NAMES)]

    panel_height = 155
    canvas = np.full(
        (view_height * 2 + panel_height, view_width * 2, 3),
        (20, 23, 27),
        dtype=np.uint8,
    )
    canvas[:view_height, :view_width] = views[0]
    canvas[:view_height, view_width:] = views[1]
    canvas[view_height : view_height * 2, :view_width] = views[2]
    canvas[view_height : view_height * 2, view_width:] = views[3]

    if status.mode == "practice":
        headline = "PRACTICE - press ENTER when ready to record"
        headline_color = (80, 210, 255)
    elif status.mode == "recording":
        headline = f"RECORDING   {status.step}/{status.max_steps} steps"
        headline_color = (90, 220, 120)
    else:
        headline = status.mode.upper()
        headline_color = (120, 170, 255)

    x0 = 24
    y0 = view_height * 2
    episode = f"DEMO {status.episode_index}/{status.episode_total}   SEED {status.scene_seed}"
    cv2.putText(canvas, headline, (x0, y0 + 37), cv2.FONT_HERSHEY_SIMPLEX, 0.68, headline_color, 2, cv2.LINE_AA)
    cv2.putText(canvas, episode, (760, y0 + 37), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (205, 211, 218), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "GOAL: put the cube in the RED tray",
        (x0, y0 + 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (245, 247, 250),
        2,
        cv2.LINE_AA,
    )

    gripper = "CLOSED" if status.gripper_closed else "OPEN"
    status_line = f"Gripper: {gripper}    Outcome: {status.outcome}"
    cv2.putText(
        canvas,
        status_line,
        (760, y0 + 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 220, 110),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "FLOW: practice -> ENTER -> place, release, move away, wait -> next demo",
        (x0, y0 + 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (205, 211, 218),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "W/S forward/back   A/D left/right   R/F up/down   SPACE grip   BACKSPACE restart   ESC stop",
        (x0, y0 + 137),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (175, 184, 192),
        1,
        cv2.LINE_AA,
    )
    return canvas


class OperatorWindow:
    def __init__(self, env):
        self.env = env
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1200, 1055)
        try:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass

    def show(self, status: OperatorStatus) -> int:
        cv2.imshow(WINDOW_NAME, compose_operator_frame(self.env, status))
        return cv2.waitKeyEx(1)

    def close(self) -> None:
        cv2.destroyWindow(WINDOW_NAME)
        cv2.waitKey(1)


def wait_for_next_frame(start_time: float, max_fps: int) -> None:
    remaining = (1.0 / max_fps) - (time.monotonic() - start_time)
    if remaining > 0:
        time.sleep(remaining)
