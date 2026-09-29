"""Measures marker properties by replaying saved chain-eval action traces through the live
simulator twice per episode (marker_present False then True, same seed and actions): marker
visibility per camera at several moments, physics identity between the two replays (the marker is
alpha-only, so this should measure 0), tray colour separability, manifest geometry, and that the
grader treats both trays identically. No policy, no training.

    python tools/measure_marker.py --root <output dir>
"""
import argparse
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from drq_grasp_env import GraspAdapter
from drq_online import CAMERAS
from rl_eval_chain import handover_to_place
from embodied_data_lab.grading import Outcome, TwoTrayGrader
from embodied_data_lab.scene import SceneSpec
from rl_place import MANIFEST

CHAIN_DIR = None
from drq_place_env import PLACE_HORIZON
EPISODES = ["dev-s2001103", "dev-s2002495", "dev-s2000709", "dev-s2002071",
            "dev-s2001902", "dev-s2005691"]           # near/right, near/left, far/right, far/left, +2 longer carries
RED_RGBA_255 = np.array([0.85, 0.08, 0.08]) * 255
BLUE_RGBA_255 = np.array([0.08, 0.25, 0.90]) * 255
COLOR_MATCH_THRESHOLD = 60.0                            # per-channel Euclidean distance in 0-255 space


def per_camera(frame):
    """frame: (9, 84, 84) uint8 -> list of 3 (84, 84, 3) uint8 arrays, one per camera, in CAMERAS order."""
    return [frame[i * 3:(i + 1) * 3].transpose(1, 2, 0) for i in range(3)]


def find_moments(n_approach, grasp_actions, n_grasp):
    """Indices (1-based, into the step count of the relevant phase) for the 4 sampling moments."""
    closing_start = None
    for i, action in enumerate(grasp_actions, 1):
        if float(action[3]) < -0.5:
            closing_start = i
            break
    if closing_start is None:
        closing_start = n_grasp
    return {"end_of_approach": n_approach, "grasp_closing_start": closing_start, "end_of_grasp": n_grasp}


def replay(path, marker_present):
    """Replays one chain episode's recorded actions with the given marker state.

    Returns (moments: {name: frame per camera}, mid_carry_frame, cube_traj: (T,3), hand_traj: (T,3)).
    """
    record = json.loads(path.read_text())
    row, actions = record["result"], record["actions"]
    seed = row["scene_seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    approach_actions = actions["approach_actions"]
    grasp_actions = actions["grasp_actions"]
    place_actions = actions["place_actions"]
    n_approach, n_grasp = len(approach_actions), len(grasp_actions)
    moment_indices = find_moments(n_approach, grasp_actions, n_grasp)

    frames_by_moment = {}
    cube_traj, hand_traj = [], []
    adapter = GraspAdapter(10 ** 6)
    try:
        obs, frame, physical = adapter.reset(seed, marker_present, handover=False)
        cube_traj.append(physical["cube"]); hand_traj.append(adapter.last_hand.tolist())

        for t, action in enumerate(approach_actions, 1):
            obs, frame, physical, done, terminal = adapter.step(np.asarray(action, np.float32))
            cube_traj.append(physical["cube"]); hand_traj.append(adapter.last_hand.tolist())
            if t == moment_indices["end_of_approach"]:
                frames_by_moment["end_of_approach"] = per_camera(frame)

        for t, action in enumerate(grasp_actions, 1):
            obs, frame, physical, done, terminal = adapter.step(np.asarray(action, np.float32))
            cube_traj.append(physical["cube"]); hand_traj.append(adapter.last_hand.tolist())
            if t == moment_indices["grasp_closing_start"]:
                frames_by_moment["grasp_closing_start"] = per_camera(frame)
            if t == moment_indices["end_of_grasp"]:
                frames_by_moment["end_of_grasp"] = per_camera(frame)

        mid_carry_frame = None
        if place_actions:
            place_adapter, _ = handover_to_place(adapter, row["target_tray"], PLACE_HORIZON)
            mid_index = max(1, len(place_actions) // 2)
            for t, action in enumerate(place_actions, 1):
                obs, frame, physical, done, terminal = place_adapter.step(np.asarray(action, np.float32))
                cube_traj.append(physical["cube"]); hand_traj.append(place_adapter.last_hand.tolist())
                if t == mid_index:
                    mid_carry_frame = per_camera(frame)
        return frames_by_moment, mid_carry_frame, np.array(cube_traj), np.array(hand_traj), row
    finally:
        adapter.close()


def diff_stats(frame_false, frame_true):
    """Per-camera pixel-diff stats between two (84, 84, 3) uint8 frames."""
    out = []
    for cam_name, a, b in zip(CAMERAS, frame_false, frame_true):
        diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
        per_pixel_max = diff.max(axis=2)                     # (84, 84): worst channel per pixel
        differing = int(np.count_nonzero(per_pixel_max > 0))
        total = 84 * 84
        out.append({"camera": cam_name, "differing_pixels": differing, "total_pixels": total,
                    "fraction": differing / total, "max_per_pixel_diff": int(per_pixel_max.max())})
    return out


def chain_record_path(stem, marker_state="absent"):
    """Use the selected final paired trace; accept legacy unsuffixed absent traces."""
    candidate = CHAIN_DIR / f"{stem}_marker_{marker_state}.json"
    if candidate.is_file():
        return candidate
    if marker_state == "absent":
        legacy = CHAIN_DIR / f"{stem}.json"
        if legacy.is_file():
            return legacy
    return None


def measure_marker_visibility(root, record_marker_state="absent"):
    episodes_out = []
    for stem in EPISODES:
        path = chain_record_path(stem, record_marker_state)
        if path is None:
            episodes_out.append({"layout_id": stem, "error": f"{record_marker_state} chain record not found"})
            continue
        moments_f, mid_f, cube_f, hand_f, row = replay(path, marker_present=False)
        moments_t, mid_t, cube_t, hand_t, _ = replay(path, marker_present=True)

        moment_results = {}
        for name in ("end_of_approach", "grasp_closing_start", "end_of_grasp"):
            if name in moments_f and name in moments_t:
                moment_results[name] = diff_stats(moments_f[name], moments_t[name])
        if mid_f is not None and mid_t is not None:
            moment_results["mid_carry"] = diff_stats(mid_f, mid_t)

        n = min(len(cube_f), len(cube_t))
        cube_trajectory_max_diff = float(np.abs(cube_f[:n] - cube_t[:n]).max()) if n else None
        n_hand = min(len(hand_f), len(hand_t))
        hand_trajectory_max_diff = float(np.abs(hand_f[:n_hand] - hand_t[:n_hand]).max()) if n_hand else None

        episodes_out.append({
            "layout_id": stem, "quadrant": row["quadrant"], "target_tray": row["target_tray"],
            "phase_ended": row["phase_ended"], "moments": moment_results,
            "physics_identity": {"cube_trajectory_max_abs_diff_m": cube_trajectory_max_diff,
                                 "hand_trajectory_max_abs_diff_m": hand_trajectory_max_diff,
                                 "steps_compared": n},
        })
    if not any("moments" in episode for episode in episodes_out):
        raise FileNotFoundError(
            f"no {record_marker_state} chain records found for the selected layouts in {CHAIN_DIR}"
        )
    return episodes_out


def measure_tray_color(root):
    """Reuses the first-frame (episode start, both trays visible) from a near and a far layout."""
    dev = json.loads(MANIFEST.read_text())["splits"]["dev"]
    near_row = next(r for r in dev if r["scene"]["cube_distance"] == "near")
    far_row = next(r for r in dev if r["scene"]["cube_distance"] == "far")
    out = {}
    for label, row in (("near", near_row), ("far", far_row)):
        adapter = GraspAdapter(10 ** 6)
        try:
            obs, frame, physical = adapter.reset(row["scene"]["seed"], False, handover=False)
            cams = per_camera(frame)
            per_cam = []
            for cam_name, image in zip(CAMERAS, cams):
                flat = image.reshape(-1, 3).astype(np.float32)
                red_dist = np.linalg.norm(flat - RED_RGBA_255, axis=1)
                blue_dist = np.linalg.norm(flat - BLUE_RGBA_255, axis=1)
                red_pixels = int(np.count_nonzero(red_dist < COLOR_MATCH_THRESHOLD))
                blue_pixels = int(np.count_nonzero(blue_dist < COLOR_MATCH_THRESHOLD))
                per_cam.append({"camera": cam_name, "red_pixels": red_pixels, "blue_pixels": blue_pixels,
                                "total_pixels": 84 * 84})
            out[label] = {"layout_id": row["layout_id"], "seed": row["scene"]["seed"], "cameras": per_cam}
        finally:
            adapter.close()
    return out


def check_manifest_geometry():
    data = json.loads(MANIFEST.read_text())
    all_rows = data["splits"]["train"] + data["splits"]["dev"]
    plus_x_both = 0
    tray_distances = []
    red_left_count = red_right_count = 0
    quadrant_counts = {}
    for row in all_rows:
        s = row["scene"]
        cube = np.asarray(s["cube_position"]); red = np.asarray(s["red_tray_center"]); blue = np.asarray(s["blue_tray_center"])
        if red[0] > cube[0] and blue[0] > cube[0]:
            plus_x_both += 1
        tray_distances.append(float(math.hypot(red[0] - blue[0], red[1] - blue[1])))
        if s["red_side"] == "left":
            red_left_count += 1
        else:
            red_right_count += 1
        qx = "+x" if blue[0] > cube[0] else "-x"
        qy = "+y" if blue[1] > cube[1] else "-y"
        quadrant_counts[f"{qx}{qy}"] = quadrant_counts.get(f"{qx}{qy}", 0) + 1
    return {
        "total_layouts": len(all_rows),
        "both_trays_plus_x_of_cube": plus_x_both,
        "red_side_left_count": red_left_count, "red_side_right_count": red_right_count,
        "note": "red_side_left/right counted over ALL 250 layouts (train+dev); peer's message cited train only (200)",
        "train_only_red_left_right": {
            "left": sum(1 for r in data["splits"]["train"] if r["scene"]["red_side"] == "left"),
            "right": sum(1 for r in data["splits"]["train"] if r["scene"]["red_side"] == "right"),
        },
        "tray_tray_distance_range_m": [min(tray_distances), max(tray_distances)],
        "blue_tray_quadrant_relative_to_cube_counts": quadrant_counts,
    }


def check_grader_symmetry():
    """Live check, not just a code read: TwoTrayGrader must grade blue exactly as it grades red."""
    dev = json.loads(MANIFEST.read_text())["splits"]["dev"][0]
    scene = SceneSpec(**dev["scene"])
    grader = TwoTrayGrader(scene=scene)
    half = grader.tray_inner_half_size
    red_center = np.asarray(scene.red_tray_center); blue_center = np.asarray(scene.blue_tray_center)
    results = {}
    for name, center in (("red", red_center), ("blue", blue_center)):
        inside = grader.grade(center)
        edge = center.copy(); edge[0] += half[0] * 0.99; edge[1] += half[1] * 0.99
        at_edge = grader.grade(edge)
        outside = center.copy(); outside[0] += half[0] * 1.2
        past_edge = grader.grade(outside)
        results[name] = {"at_center": inside.value, "just_inside_edge": at_edge.value,
                         "just_past_edge": past_edge.value}
    symmetric = (results["red"]["at_center"] == Outcome.RED.value and
                results["blue"]["at_center"] == Outcome.BLUE.value and
                results["red"]["just_inside_edge"] == Outcome.RED.value and
                results["blue"]["just_inside_edge"] == Outcome.BLUE.value and
                results["red"]["just_past_edge"] == results["blue"]["just_past_edge"] == Outcome.INCOMPLETE.value)
    return {"tray_inner_half_size": list(half), "results": results, "symmetric": symmetric}




def main():
    global CHAIN_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True,
                        help="saved chain evaluation directory")
    parser.add_argument("--record-marker-state", choices=("absent", "present"), default="absent",
                        help="action trace to replay in both marker states (default: absent)")
    args = parser.parse_args()
    CHAIN_DIR = args.eval_dir / "chain"
    root = args.root
    root.mkdir(parents=True, exist_ok=True)

    print("Measuring marker visibility and physics identity...", flush=True)
    visibility = measure_marker_visibility(root, args.record_marker_state)
    print("Measuring tray colour separability...", flush=True)
    tray_color = measure_tray_color(root)
    print("Checking manifest geometry...", flush=True)
    manifest_geometry = check_manifest_geometry()
    print("Checking grader symmetry...", flush=True)
    grader_symmetry = check_grader_symmetry()

    result = {
        "record_marker_state": args.record_marker_state,
        "marker_visibility_and_physics_identity": visibility,
        "tray_color_separability": tray_color,
        "manifest_geometry": manifest_geometry,
        "grader_symmetry": grader_symmetry,
    }
    (root / "measurement.json").write_text(json.dumps(result, indent=2))
    print(f"DONE wrote {root / 'measurement.json'}", flush=True)


if __name__ == "__main__":
    main()
