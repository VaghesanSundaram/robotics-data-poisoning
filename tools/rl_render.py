"""Replay saved RL actions as videos; no policy loading or training."""
import argparse
import json
from pathlib import Path
import random
import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from drq_approach_rewards import physical_with_eef
from drq_reach_env import ReachAdapter, ReachReward, HORIZON
from drq_grasp_env import GraspAdapter, GraspReward, GRASP_HORIZON, HOLD_STEPS
from drq_place_env import PlaceAdapter, PLACE_HORIZON
from rl_eval_chain import _stable_success, handover_to_place
SCALE = 3


def tile(frame):
    views = [frame[i * 3:(i + 1) * 3].transpose(1, 2, 0) for i in range(3)]
    return cv2.resize(np.concatenate(views, axis=1), None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)

def label(image, text, line2=None):
    image = np.ascontiguousarray(image)
    cv2.putText(image, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if line2 is not None:
        cv2.putText(image, line2, (6, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return image

def render_approach(path, out_dir):
    record = json.loads(path.read_text())
    row, actions = record["result"], record["actions"]
    adapter = ReachAdapter(HORIZON)
    try:
        obs, frame, physical = adapter.reset(row["scene_seed"], row["marker_present"])
        rewarder = ReachReward(adapter.start_z); before = physical_with_eef(physical, obs[1])
        name = f"{row['layout_id']}_marker{int(row['marker_present'])}_{'success' if row['success'] else 'fail'}.mp4"
        head = f"{row['layout_id']} marker={int(row['marker_present'])}"
        with imageio.get_writer(out_dir / name, fps=20, codec="libx264", quality=7) as writer:
            writer.append_data(label(tile(frame), f"{head} t=0"))
            for t, action in enumerate(actions, 1):
                obs, frame, physical, _, _ = adapter.step(np.asarray(action, np.float32))
                after = physical_with_eef(physical, obs[1])
                _, info = rewarder.step(before, after); before = after
                writer.append_data(label(tile(frame), f"{head} t={t} d={info['distance']:.3f} hold={info['hold_steps']}"))
        drift = abs(float(physical["distance"]) - row["final_distance"])
        return name, drift
    finally:
        adapter.close()

def render_grasp(path, out_dir):
    record = json.loads(path.read_text()); row, actions = record["result"], record["actions"]
    seed = row["scene_seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    adapter = GraspAdapter(GRASP_HORIZON)
    try:
        obs, frame, physical = adapter.reset(seed, row["marker_present"], True)
        rewarder = GraspReward(); before = physical_with_eef(physical, obs[1])
        name = f"{row['layout_id']}_marker{int(row['marker_present'])}_{'success' if row['success'] else 'fail'}.mp4"
        head = f"{row['layout_id']} m={int(row['marker_present'])}"
        with imageio.get_writer(out_dir / name, fps=20, codec="libx264", quality=7) as writer:
            writer.append_data(label(tile(frame), f"{head} t=0"))
            for t, action in enumerate(actions, 1):
                obs, frame, physical, _, _ = adapter.step(np.asarray(action, np.float32))
                after = physical_with_eef(physical, obs[1]); _, info = rewarder.step(before, after, action[3]); before = after
                writer.append_data(label(tile(frame), f"{head} t={t} grip={action[3]:+.1f} hold={int(info['holding'])} lift={info['lift'] * 100:.1f}cm"))
        return name, abs(info["lift"] * 100 - row["end_lift_cm"])
    finally:
        adapter.close()

def render_place(path, out_dir):
    record = json.loads(path.read_text()); row, actions = record["result"], record["actions"]
    seed = row["scene_seed"]
    marker = bool(row.get("marker_present", False))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    adapter = PlaceAdapter(max(PLACE_HORIZON, len(actions)))
    try:
        obs, frame, physical = adapter.reset(seed, marker)
        name = f"{row['layout_id']}_marker{int(marker)}_{'success' if row['success'] else 'fail'}.mp4"
        with imageio.get_writer(out_dir / name, fps=20, codec="libx264", quality=7) as writer:
            writer.append_data(label(tile(frame), f"{row['layout_id']} t=0 latch=closed"))
            for t, action in enumerate(actions, 1):
                obs, frame, physical, _, _ = adapter.step(np.asarray(action, np.float32))
                state = "open" if adapter.latch.released else "closed"
                writer.append_data(label(tile(frame), f"{row['layout_id']} t={t} grip_cmd={action[2]:+.1f} latch={state}"))
        return name, adapter.release_step == row["release_step"]
    finally:
        adapter.close()

def render_chain(path, out_dir, place_horizon):
    record = json.loads(path.read_text())
    row, actions = record["result"], record["actions"]
    seed = row["scene_seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    approach_actions = actions["approach_actions"]
    grasp_actions = actions["grasp_actions"]
    place_actions = actions["place_actions"]
    n_approach, n_grasp, n_place = len(approach_actions), len(grasp_actions), len(place_actions)

    mismatches = []
    adapter = GraspAdapter(10 ** 6)
    try:
        obs, frame, physical = adapter.reset(seed, row["marker_present"], handover=False)
        name = f"{row['layout_id']}_marker{int(row['marker_present'])}_{row['phase_ended']}_{'success' if row['success'] else 'fail'}.mp4"
        overall = 0
        with imageio.get_writer(out_dir / name, fps=20, codec="libx264", quality=7) as writer:
            writer.append_data(label(tile(frame), f"{row['layout_id']} phase=start", "t=0"))

            for t, action in enumerate(approach_actions, 1):
                obs, aframe, physical, done, terminal = adapter.step(np.asarray(action, np.float32))
                overall += 1
                writer.append_data(label(
                    tile(aframe),
                    f"{row['layout_id']} phase=approach step={t}/{n_approach} total={overall}",
                    f"h1@{n_approach if t == n_approach else '...'} latch=n/a"))

            if row["phase_ended"] == "approach":
                if row["approach_switch_step"] is not None and row["approach_switch_step"] != n_approach:
                    mismatches.append(f"approach_switch_step recorded={row['approach_switch_step']} replay_actions={n_approach}")
                return name, mismatches

            rewarder = GraspReward(); before = None; hold_steps = 0
            for t, action in enumerate(grasp_actions, 1):
                obs, gframe, physical, done, terminal = adapter.step(np.asarray(action, np.float32))
                overall += 1
                after = physical_with_eef(physical, obs[1])
                if before is None:
                    before = after
                _, info = rewarder.step(before, after, action[3])
                before = after
                hold_steps = info["hold_steps"]
                latch_state = "closed" if adapter.latched else "open"
                writer.append_data(label(
                    tile(gframe),
                    f"{row['layout_id']} phase=grasp step={t}/{n_grasp} total={overall}",
                    f"latch={latch_state} hold_steps={hold_steps}"))

            replay_grasp_switch_step = None
            if hold_steps >= HOLD_STEPS:
                # HOLD_STEPS consecutive steps ended exactly at the last grasp step recorded (the phase
                # function returns as soon as the hold sustains), so the switch step is n_grasp itself.
                replay_grasp_switch_step = n_grasp
            if row["grasp_switch_step"] != replay_grasp_switch_step:
                mismatches.append(f"grasp_switch_step recorded={row['grasp_switch_step']} replay={replay_grasp_switch_step}")

            if row["phase_ended"] == "grasp":
                if row["success"]:
                    mismatches.append("recorded success=True but phase_ended=grasp (no place phase)")
                return name, mismatches

            place_adapter, target_center = handover_to_place(adapter, row["target_tray"], place_horizon)
            success_replay = False
            pframe = gframe  # covers the (practically unreachable) case of a place phase with 0 actions
            for t, action in enumerate(place_actions, 1):
                obs, pframe, physical, done, terminal = place_adapter.step(np.asarray(action, np.float32))
                overall += 1
                if not success_replay and _stable_success(place_adapter, row["target_tray"]):
                    success_replay = True
                latch_state = "released" if place_adapter.latch.released else "held"
                writer.append_data(label(
                    tile(pframe),
                    f"{row['layout_id']} phase=place step={t}/{n_place} total={overall}",
                    f"latch={latch_state} release_step={place_adapter.release_step}"))

            final_line = f"FINAL success={success_replay} recorded={row['success']}"
            writer.append_data(label(tile(pframe), f"{row['layout_id']} done", final_line))

            if place_adapter.release_step != row["release_step"]:
                mismatches.append(f"release_step recorded={row['release_step']} replay={place_adapter.release_step}")
            if success_replay != row["success"]:
                mismatches.append(f"success recorded={row['success']} replay={success_replay}")
        return name, mismatches
    finally:
        adapter.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('approach','grasp','place','chain'), required=True)
    parser.add_argument('--eval-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--layouts', nargs='+')
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    directory = args.eval_dir / 'chain' if args.stage == 'chain' else args.eval_dir
    paths = []
    for path in sorted(directory.glob('dev-*.json')):
        record = json.loads(path.read_text())
        if 'actions' not in record or 'result' not in record:
            continue
        if args.layouts and record['result']['layout_id'] not in args.layouts:
            continue
        paths.append(path)
    if not paths:
        parser.error('no saved rollout actions match the selected directory/layouts')
    if args.limit:
        paths = paths[:args.limit]
    out = args.eval_dir / 'videos'
    out.mkdir(exist_ok=True)
    render = {'approach':render_approach, 'grasp':render_grasp, 'place':render_place, 'chain':render_chain}[args.stage]
    extra = []
    if args.stage == 'chain':
        contract = json.loads((args.eval_dir / 'run-contract.json').read_text())
        extra = [contract['place_horizon']]
    for path in paths:
        name, check = render(path, out, *extra)
        print(f'{name} replay_check={check}', flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
