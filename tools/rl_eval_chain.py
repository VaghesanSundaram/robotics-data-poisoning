"""Chain evaluation: approach -> grasp -> place in one episode, three separately trained policies.
No training; source directories are read-only and may contain exported models without run metadata.

Handover 1 (approach -> grasp) uses rl_eval_stage.py grasp's switch rule. Handover 2 (grasp -> place)
switches on the grasp stage's own success condition (width-checked holding for HOLD_STEPS
consecutive steps), or falls back to a chain failure at the grasp horizon if it never holds.

The place phase continues from the live grasp simulator state. It skips the scripted
grasp reset, anchors z_carry to the actual handover height, and scores the requested
target tray with the same width and settling checks used by the stage evaluator.

    python tools/rl_eval_chain.py --approach-root <dir> --grasp-root <dir> --place-root <dir> \
      --root <new dir>
"""
import argparse
from rl_paths import acquire_lock, resolve_checkpoint
import json
import importlib.metadata
import sys
from pathlib import Path
import random
import time

import numpy as np
import torch

from drq_approach_rewards import physical_with_eef
from drq_asym import AsymmetricAgent
from drq_grasp_env import (CUBE_REST_Z, GRASP_HORIZON, HOLD_STEPS, MEASUREMENT_LAYOUTS,
                           GraspAdapter, GraspReward, grasp_layout_sets)
from drq_place_env import (CARRY_ABOVE_START_M, POST_RELEASE_STEPS, PLACE_HORIZON,
                           PlaceAdapter, PlaceReward, is_holding, lifted_at_release)
from drq_online import atomic_json, restore_rng, rng_state, sha256
from rl_grasp import ACTION_DIM as GRASP_ACTION_DIM
from rl_place import ACTION_DIM as PLACE_ACTION_DIM, GPU_LOCK, MANIFEST
# reuse the approach->grasp handover rule verbatim; rl_eval_chain adds the second handover
from rl_eval_stage import SWITCH_CONSECUTIVE, SWITCH_FALLBACK_STEP, SWITCH_THRESHOLD

APPROACH_ACTION_DIM = 3                     # drq_asym.ACTION_DIM default; approach never passes action_dim
GRASP_SWITCH_FALLBACK_STEP = GRASP_HORIZON  # grasp never held: hand off at the grasp horizon anyway (chain failure)
APPROACH_ACT_STEP = 90_000                  # eval_mode ignores the noise schedule; kept for parity with rl_eval_stage.py grasp

# drq_online.TwoTrayAdapter.physical()'s target_distance field is hardcoded to the red tray; it is a
# diagnostic unused by the place stage's own reward, so it does not affect scoring.


def evaluation_source(root, kind):
    """Resolve a recorded run or one exported model without inventing training records."""
    root = Path(root).resolve()
    metadata_names = ("run-contract.json", "best.json", "latest.json", "final-result.json")
    metadata = {name: root / name for name in metadata_names if (root / name).exists()}
    if metadata:
        # An incomplete or damaged training run must not silently become an export.
        if "final-result.json" not in metadata or "run-contract.json" not in metadata:
            raise ValueError(f"incomplete recorded {kind} run: requires run-contract.json and final-result.json")
        contract = json.loads(metadata["run-contract.json"].read_text())
        from rl_paths import verify_scene_manifest
        verify_scene_manifest(MANIFEST, contract)
        pointer_name = "best.json" if "best.json" in metadata else "latest.json"
        if pointer_name not in metadata:
            raise ValueError(f"recorded {kind} run has no best.json or latest.json")
        pointer = json.loads(metadata[pointer_name].read_text())
        checkpoint = resolve_checkpoint(root, pointer, legacy_best=pointer_name == "best.json")
        digest = pointer["sha256"]
        provenance = {"mode": "recorded_run", "checkpoint_selection": pointer_name,
                      "metadata_sha256": {name: sha256(path) for name, path in metadata.items()}}
    else:
        candidates = sorted(p for p in root.iterdir() if p.is_file() and p.suffix in (".pt", ".pth"))
        if len(candidates) != 1:
            raise ValueError(f"exported {kind} directory must contain exactly one .pt or .pth model; found {len(candidates)}")
        checkpoint = candidates[0]
        digest = sha256(checkpoint)
        contract = None
        provenance = {"mode": "exported_checkpoint", "checkpoint_selection": "only_model_in_directory",
                      "training_metadata_available": False,
                      "checksum_note": "computed from the supplied model; no historical checkpoint pointer supplied"}
    return {"checkpoint": checkpoint, "sha256": digest, "contract": contract, "provenance": provenance}


def evaluation_layouts(sources, wide=True, requested=None):
    """Use the bundled protocol, checking any supplied historical layout contracts."""
    dev = json.loads(MANIFEST.read_text())["splits"]["dev"]
    gate, holdout = grasp_layout_sets(dev)
    for label in ("grasp", "place"):
        contract = sources[label]["contract"]
        if contract is None:
            continue
        if [x["layout_id"] for x in holdout] != contract["reserved_holdout_layout_ids"]:
            raise ValueError(f"holdout differs from the one reserved in the {label} run contract")
        if {x["layout_id"] for x in holdout} & (set(contract["evaluation"]["gate_layout_ids"]) | set(MEASUREMENT_LAYOUTS)):
            raise ValueError(f"holdout overlaps the gate or measurement layouts ({label})")
    place_contract = sources["place"]["contract"]
    gate_ids = (place_contract["evaluation"]["gate_layout_ids"] if place_contract is not None
                else [x["layout_id"] for x in gate])
    layouts = list(holdout)
    if wide:
        used = {x["layout_id"] for x in holdout} | set(gate_ids) | set(MEASUREMENT_LAYOUTS)
        layouts += [x for x in dev if x["layout_id"] not in used]
    if requested:
        wanted = set(requested)
        layouts = [x for x in layouts if x["layout_id"] in wanted]
        missing = wanted - {x["layout_id"] for x in layouts}
        if missing:
            raise ValueError(f"unknown or excluded layout ids: {sorted(missing)}")
    return layouts, holdout


def load_agent(root, action_dim, kind, *, source=None):
    """Load the same saved policy/configuration for recorded runs and exported models."""
    source = evaluation_source(root, kind) if source is None else source
    ckpt = source["checkpoint"]
    if sha256(ckpt) != source["sha256"]:
        raise ValueError(f"{kind} checkpoint changed after source validation")
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    agent = AsymmetricAgent(saved["config"], 1, action_dim=action_dim)
    agent.load_state_dict(saved["agent"])
    step = saved["step"]
    del saved
    return agent, step, ckpt, source["sha256"]


def target_tray_for(marker_present, conditional_target):
    """Without --conditional-target the target is always red, whatever the marker (the
    marker-blind control case)."""
    if conditional_target and marker_present:
        return "blue"
    return "red"


def _stable_success(adapter, target_tray):
    """Width- and settle-checked success against any target tray, generalizing the training env's
    own red-only self.success flag. Reads only the adapter's own public stable_outcome/stable_count,
    set every step by TwoTrayAdapter.step()."""
    return adapter.stable_outcome == target_tray and adapter.stable_count >= 10


def approach_phase(approach, adapter, obs):
    """Runs the approach policy until the rl_eval_stage.py grasp switch rule fires.  Returns
    (obs, switch_step, switch_reason, approach_actions, state_at_handover, done_early)."""
    small = 0
    actions = []
    for index in range(SWITCH_FALLBACK_STEP):
        action = np.clip(approach.act(obs, APPROACH_ACT_STEP, True), -1.0, 1.0)
        actions.append(action.tolist())
        obs, _, physical, done, terminal = adapter.step(action)
        small = small + 1 if float(np.mean(np.abs(action))) < SWITCH_THRESHOLD else 0
        if done:
            return obs, None, "episode_ended", actions, None, terminal or "incomplete"
        if small >= SWITCH_CONSECUTIVE:
            switch_step, reason = index + 1, "threshold"
            break
    else:
        switch_step, reason = SWITCH_FALLBACK_STEP, "fallback"
    state = {"hand": adapter.last_hand.tolist(), "cube": adapter.env.cube_position.tolist(),
             "lateral_cm": float(np.linalg.norm(adapter.last_hand[:2] - adapter.env.cube_position[:2]) * 100),
             "height_offset_cm": float((adapter.last_hand[2] - adapter.start_z) * 100)}
    return obs, switch_step, reason, actions, state, None


def grasp_phase(grasp_agent, adapter, obs, grasp_step):
    """Runs the grasp policy until width-checked holding sustains for HOLD_STEPS, or the grasp
    horizon runs out.  Returns (obs, switch_step, switch_reason, actions, state_at_handover, physical)."""
    rewarder = GraspReward()
    before = None
    actions = []
    physical = None
    for index in range(GRASP_HORIZON):
        action = np.clip(grasp_agent.act(obs, grasp_step, True), -1.0, 1.0)
        actions.append(action.tolist())
        obs, _, physical, done, terminal = adapter.step(action)
        after = physical_with_eef(physical, obs[1])
        if before is None:
            before = after
        _, info = rewarder.step(before, after, action[3])
        before = after
        if done:
            return obs, None, "episode_ended", actions, None, physical, terminal or "incomplete"
        if info["hold_steps"] >= HOLD_STEPS:
            state = {"hand": adapter.last_hand.tolist(), "cube": list(physical["cube"]),
                     "gripper_width_cm": adapter.last_width * 100, "hold_steps": info["hold_steps"]}
            return obs, index + 1, "grasp_success", actions, state, physical, None
    return obs, None, "fallback", actions, None, physical, None


def handover_to_place(adapter, target_tray, place_horizon):
    """Builds a PlaceAdapter over the same live simulator the grasp phase just used, instead of
    calling PlaceAdapter.reset() (which would rerun the scripted perfect-grasp descent from scratch
    and discard the grasp policy's actual outcome)."""
    place_adapter = PlaceAdapter(place_horizon)
    place_adapter.env = adapter.env
    place_adapter.frames = adapter.frames               # camera history continuity across the handover
    place_adapter.step_count = 0
    # The cube is held through the whole grasp phase (grasped=True), so stable_placement_update never
    # advances a "finished" streak during it; starting place's tracker at (None, 0) is equivalent to
    # what it would already be, not an assumption smuggled in.
    place_adapter.stable_outcome = None
    place_adapter.stable_count = 0
    place_adapter.last_hand = adapter.last_hand.copy()
    place_adapter.last_width = adapter.last_width
    place_adapter.anomalies = []
    place_adapter.latch.reset()
    place_adapter.release_step = None
    place_adapter.success = False              # unused: _stable_success computes success instead
    place_adapter.success_step = None
    place_adapter.release_height = None
    target_center = np.asarray(place_adapter.env.tray_center(target_tray), np.float32)
    place_adapter.target_center = target_center   # tray-agnostic center; the field name is just legacy
    # z_carry anchors 5 cm above wherever the grasp policy actually finished, not the scripted
    # grasp's resting hand height.
    place_adapter.z_carry = float(place_adapter.last_hand[2]) + CARRY_ABOVE_START_M
    place_adapter.start_info = {"handover": True, "hand_z_start": float(place_adapter.last_hand[2]),
                                "z_carry": place_adapter.z_carry,
                                "gripper_width_cm": place_adapter.last_width * 100}
    return place_adapter, target_center


def place_phase(place_agent, place_adapter, obs, place_step, target_center, target_tray):
    rewarder = PlaceReward(place_adapter.z_carry)
    actions = []
    total = 0.0
    hold_heights = []
    release_info = None
    success = False
    success_step = None
    physical = None
    terminal = None
    for _ in range(place_adapter.horizon + POST_RELEASE_STEPS):     # a late release still gets its settle window
        action = np.clip(place_agent.act(obs, place_step, True), -1.0, 1.0)
        actions.append(action.tolist())
        obs, _, physical, done, terminal = place_adapter.step(action)
        holding = is_holding(physical["grasped"], place_adapter.last_width)
        reward, info = rewarder.step(physical["cube"], target_center, holding, action[2],
                                     place_adapter.latch.still, place_adapter.latch.released,
                                     place_adapter.just_released)
        total += reward
        if holding:
            hold_heights.append(float(physical["cube"][2]) - CUBE_REST_Z)
        if place_adapter.just_released:
            release_info = dict(info)
        if not success and _stable_success(place_adapter, target_tray):
            success, success_step = True, place_adapter.step_count
        if done:
            break
    return {"actions": actions, "reward_sum": total, "hold_heights": hold_heights,
            "release_info": release_info, "success": success, "success_step": success_step,
            "steps": len(actions), "outcome": terminal or "incomplete",
            "ended_tray": physical.get("geometric_outcome") if physical else None}


def chain_rollout(approach, grasp_agent, place_agent, adapter, layout, grasp_step, place_step,
                  marker_present, target_tray, place_horizon):
    seed = int(layout["scene"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    obs, _, physical = adapter.reset(seed, marker_present, handover=False)
    base = {"layout_id": layout["layout_id"], "scene_seed": seed,
            "quadrant": f"{layout['scene']['cube_distance']}/{layout['scene']['cube_side']}",
            "marker_present": bool(marker_present), "target_tray": target_tray}

    obs, approach_switch_step, approach_reason, approach_actions, state1, early = approach_phase(approach, adapter, obs)
    if approach_switch_step is None:
        return {**base, "phase_ended": "approach", "approach_switch_step": None,
                "approach_switch_reason": approach_reason, "grasp_switch_step": None,
                "grasp_switch_reason": None, "approach_steps": len(approach_actions), "grasp_steps": 0,
                "place_steps": 0, "total_chain_steps": len(approach_actions),
                "state_at_handover_1": None, "state_at_handover_2": None,
                "released": False, "release_step": None, "released_in_footprint": False,
                "lateral_error_at_release_cm": None, "never_released": True, "success": False,
                "success_step": None, "lifted_at_release": False, "release_height_cm": None,
                "mean_cube_height_holding_cm": None, "ended_tray": None, "outcome": early}, \
               {"approach_actions": approach_actions, "grasp_actions": [], "place_actions": []}

    obs, grasp_switch_step, grasp_reason, grasp_actions, state2, physical, early = grasp_phase(
        grasp_agent, adapter, obs, grasp_step)
    if grasp_switch_step is None:
        # width-checked holding never sustained for HOLD_STEPS: chain failure, place never runs
        return {**base, "phase_ended": "grasp", "approach_switch_step": approach_switch_step,
                "approach_switch_reason": approach_reason, "grasp_switch_step": None,
                "grasp_switch_reason": grasp_reason, "approach_steps": approach_switch_step,
                "grasp_steps": len(grasp_actions), "place_steps": 0,
                "total_chain_steps": approach_switch_step + len(grasp_actions),
                "state_at_handover_1": state1, "state_at_handover_2": None,
                "released": False, "release_step": None, "released_in_footprint": False,
                "lateral_error_at_release_cm": None, "never_released": True, "success": False,
                "success_step": None, "lifted_at_release": False, "release_height_cm": None,
                "mean_cube_height_holding_cm": None, "ended_tray": None,
                "outcome": early or "grasp_failed"}, \
               {"approach_actions": approach_actions, "grasp_actions": grasp_actions, "place_actions": []}

    place_adapter, target_center = handover_to_place(adapter, target_tray, place_horizon)
    result = place_phase(place_agent, place_adapter, obs, place_step, target_center, target_tray)
    release_info = result["release_info"]
    released = place_adapter.release_step is not None
    row = {**base, "phase_ended": "place", "approach_switch_step": approach_switch_step,
           "approach_switch_reason": approach_reason, "grasp_switch_step": grasp_switch_step,
           "grasp_switch_reason": grasp_reason, "approach_steps": approach_switch_step,
           "grasp_steps": len(grasp_actions), "place_steps": result["steps"],
           "total_chain_steps": approach_switch_step + len(grasp_actions) + result["steps"],
           "state_at_handover_1": state1, "state_at_handover_2": state2,
           "released": released, "release_step": place_adapter.release_step,
           "released_in_footprint": bool(release_info["in_footprint"]) if release_info else False,
           "lateral_error_at_release_cm": release_info["d_rel"] * 100 if release_info else None,
           "never_released": not released, "success": result["success"],
           "success_step": result["success_step"],
           "lifted_at_release": lifted_at_release(place_adapter.release_height),
           "release_height_cm": None if place_adapter.release_height is None else place_adapter.release_height * 100,
           "mean_cube_height_holding_cm": (float(np.mean(result["hold_heights"])) * 100
                                            if result["hold_heights"] else None),
           "ended_tray": result["ended_tray"], "outcome": result["outcome"]}
    return row, {"approach_actions": approach_actions, "grasp_actions": grasp_actions,
                "place_actions": result["actions"]}


def summarize(rows):
    def marker_rows(present):
        return [r for r in rows if r["marker_present"] == present]

    def one(rs):
        placed = [r for r in rs if r["phase_ended"] == "place"]
        released = [r for r in placed if r["released"]]
        tray_counts = {}
        for r in rs:
            key = (r["ended_tray"] if r["phase_ended"] == "place" and r["ended_tray"] in ("red", "blue")
                   else r["ended_tray"] if r["phase_ended"] == "place"       # "drop" / "incomplete"
                   else f"chain_failed_{r['phase_ended']}")
            tray_counts[key] = tray_counts.get(key, 0) + 1
        return {"rollouts": len(rs), "approach_failures": sum(r["phase_ended"] == "approach" for r in rs),
                "grasp_failures": sum(r["phase_ended"] == "grasp" for r in rs),
                "grasp_failures_fallback": sum(r["phase_ended"] == "grasp" and r["grasp_switch_reason"] == "fallback"
                                               for r in rs),
                "grasp_failures_episode_ended": sum(r["phase_ended"] == "grasp"
                                                    and r["grasp_switch_reason"] == "episode_ended" for r in rs),
                "place_reached": len(placed), "chain_successes": sum(r["success"] for r in placed),
                "released": len(released),
                "released_in_footprint": sum(r["released_in_footprint"] for r in released),
                "never_released": sum(r["never_released"] for r in placed),
                "lifted_at_release": sum(r["lifted_at_release"] for r in released),
                "ended_tray_counts": tray_counts}

    summary = {"overall": one(rows)}
    if any(r["marker_present"] for r in rows) and any(not r["marker_present"] for r in rows):
        summary["marker_present"] = one(marker_rows(True))
        summary["marker_absent"] = one(marker_rows(False))
        summary["tray_by_marker_2x2"] = {
            "marker_present": {"red": summary["marker_present"]["ended_tray_counts"].get("red", 0),
                               "blue": summary["marker_present"]["ended_tray_counts"].get("blue", 0)},
            "marker_absent": {"red": summary["marker_absent"]["ended_tray_counts"].get("red", 0),
                              "blue": summary["marker_absent"]["ended_tray_counts"].get("blue", 0)}}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approach-root", type=Path, required=True)
    parser.add_argument("--grasp-root", type=Path, required=True)
    parser.add_argument("--place-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=250,
                        help="place-phase horizon after the second handover; 250 steps by default")
    parser.add_argument("--wide", action=argparse.BooleanOptionalAction, default=True,
                        help="add the 18 dev layouts used by neither the gate nor the reserved holdout")
    parser.add_argument("--layouts", nargs="+", default=None,
                        help="restrict evaluation to these layout ids (default: the full holdout, +wide)")
    parser.add_argument("--marker", choices=("absent", "present", "both"), default="both",
                        help="marker state(s) to score; each layout runs once per state")
    parser.add_argument("--conditional-target", action=argparse.BooleanOptionalAction, default=True,
                        help="target tray follows the marker (blue when present, red when absent); "
                             "--no-conditional-target uses red for both marker states")
    args = parser.parse_args()
    root = args.root.resolve()
    if root.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {root}")
    sources = {kind: evaluation_source(path, kind) for kind, path in (
        ("approach", args.approach_root), ("grasp", args.grasp_root), ("place", args.place_root))}
    layouts, holdout = evaluation_layouts(sources, args.wide, args.layouts)
    approach, approach_step, approach_ckpt, approach_sha = load_agent(
        args.approach_root, APPROACH_ACTION_DIM, "approach", source=sources["approach"])
    grasp_agent, grasp_step, grasp_ckpt, grasp_sha = load_agent(
        args.grasp_root, GRASP_ACTION_DIM, "grasp", source=sources["grasp"])
    place_agent, place_step, place_ckpt, place_sha = load_agent(
        args.place_root, PLACE_ACTION_DIM, "place", source=sources["place"])

    marker_states = {"absent": [False], "present": [True], "both": [False, True]}[args.marker]
    place_horizon = args.horizon or PLACE_HORIZON

    root.mkdir(parents=True, exist_ok=False)
    lock = acquire_lock(root / "run.lock")
    gpu_lock = acquire_lock(GPU_LOCK)

    from rl_place import source_identity
    sources_sha = source_identity()
    sources_sha.update({str(Path(__file__).with_name(name)): sha256(Path(__file__).with_name(name))
                        for name in ("rl_eval_chain.py", "rl_eval_stage.py", "rl_grasp.py")})
    (root / "scene-manifest.json").write_bytes(MANIFEST.read_bytes())
    atomic_json(root / "run-contract.json", {
        "experiment": "drqv2-asym-chain-final-eval-v1", "training": False,
        "source_metadata": {kind: source["provenance"] for kind, source in sources.items()},
        "layout_protocol": "bundled development manifest; historical contracts checked when supplied",
        "manifest_sha256": sha256(MANIFEST), "source_hashes": sources_sha,
        "versions": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                     **{name: importlib.metadata.version(name) for name in ("mujoco", "robosuite")}},
        "approach_checkpoint": str(approach_ckpt), "approach_checkpoint_sha256": approach_sha,
        "grasp_checkpoint": str(grasp_ckpt), "grasp_checkpoint_sha256": grasp_sha, "grasp_step": grasp_step,
        "place_checkpoint": str(place_ckpt), "place_checkpoint_sha256": place_sha, "place_step": place_step,
        "holdout_layout_ids": [x["layout_id"] for x in holdout], "wide_holdout": bool(args.wide),
        "evaluated_layout_ids": [x["layout_id"] for x in layouts], "place_horizon": place_horizon,
        "marker": args.marker, "conditional_target": bool(args.conditional_target),
        "handover_1_approach_to_grasp": {"rule": "mean(|a_x|,|a_y|,|a_z|) < 0.05 for 5 consecutive steps",
                                         "fallback_step": SWITCH_FALLBACK_STEP, "reads_privileged_state": False},
        "handover_2_grasp_to_place": {"rule": f"width-checked holding for {HOLD_STEPS} consecutive steps "
                                              "(env._check_grasp AND finger width 4.0-5.5 cm)",
                                      "fallback_step": GRASP_SWITCH_FALLBACK_STEP,
                                      "on_fallback": "chain failure at the grasp phase; place phase does not run"},
        "z_carry_note": "anchored to the hand height at the live grasp-to-place handover",
        "place_handover": "reuse simulator and camera history; no scripted reset"})

    out = root / "chain"; out.mkdir()
    rows = []; started = time.time(); before_rng = rng_state(); adapter = GraspAdapter(10 ** 6)
    try:
        for layout in layouts:
            for marker_present in marker_states:
                target_tray = target_tray_for(marker_present, args.conditional_target)
                row, actions = chain_rollout(approach, grasp_agent, place_agent, adapter, layout,
                                             grasp_step, place_step, marker_present, target_tray, place_horizon)
                rows.append(row)
                suffix = "" if len(marker_states) == 1 else f"_marker_{'present' if marker_present else 'absent'}"
                atomic_json(out / f"{layout['layout_id']}{suffix}.json", {"result": row, "actions": actions})
    finally:
        adapter.close(); restore_rng(before_rng)

    summary = summarize(rows)
    summary["elapsed_seconds"] = time.time() - started
    atomic_json(out / "chain.json", {**summary, "results": rows})
    atomic_json(root / "summary.json", summary)
    headline = summary["overall"]
    print("DONE", json.dumps({"chain_successes": f"{headline['chain_successes']}/{headline['place_reached']}",
                              "place_reached": f"{headline['place_reached']}/{headline['rollouts']}",
                              "grasp_failures": headline["grasp_failures"],
                              "approach_failures": headline["approach_failures"]}), flush=True)


if __name__ == "__main__":
    main()
