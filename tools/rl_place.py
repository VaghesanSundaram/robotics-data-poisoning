"""Place-stage DrQ-v2 run: carry a held cube over the red or blue tray and release it.

Starts every episode from a scripted perfect grasp with the cube still at rest height: lifting is
this stage's job. Asymmetric critic; the actor sees images and proprioception only. The encoder
alone is warm-started from the grasp stage's final checkpoint (read-only, hash verified, gate must
have passed); everything else starts fresh. The target tray follows the marker when present; the
gripper command feeds a release latch (see drq_place_env).

    python tools/rl_place.py --grasp-root <grasp run dir> --root <out dir>

``--smoke`` shrinks the run to 1,000 steps; ``--stop-at N`` pauses at a checkpoint and ``--resume``
continues in a fresh process.
"""
from __future__ import annotations

import argparse
from rl_paths import acquire_lock
import gc
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from drq_asym import AsymmetricAgent, LOG_EVERY, PROPRIO_DIM
from drq_grasp_env import CUBE_REST_Z, HOLD_WIDTH_MAX_M, HOLD_WIDTH_MIN_M, MEASUREMENT_LAYOUTS
from drq_online import (EpisodeReplay, UPSTREAM, UPSTREAM_COMMIT, atomic_json, restore_rng, upstream_utils,
                        rng_state, sha256)
from drq_place_env import (CARRY_ABOVE_START_M, DEFAULT_MARKER_RATE, FOOTPRINT_HALF, LIFTED_AT_RELEASE_M,
                           PLACE_HORIZON, POST_RELEASE_STEPS, ATTEMPT_BONUS, PARKING_BONUS, RELEASE_CONSECUTIVE,
                           RELEASE_THRESHOLD, REWARD_K, REWARD_K_FINE, REWARD_K_FINE_WEIGHT,
                           REWARD_K_FINE_WEIGHT_RELEASE, PlaceAdapter, PlaceReward, is_holding,
                           lifted_at_release, place_layout_sets)
from drq_reach_env import (POSITION_OFFSET, POSITION_SCALE, PRIVILEGED_DIM, TRANSLATION_CAP_M,
                           controller_output_max_translation)
from rl_approach import GATE_SUCCESSES, Q_LOW, StopRun, json_hash
from drq_online import checkpoint, resource_status, validate_checkpoint

from rl_paths import PROJECT, MANIFEST, GPU_LOCK
ACTION_DIM = 3
# The critic bootstraps at truncation, so it learns the value of a reward paid forever:
# r_max / (1 - gamma) = 3.0 / 0.01 = 300. The stop threshold is 1.25x that ceiling.
REWARD_VERSION = "place_r5"
REWARD_MAX = 3.0                  # release at the tray centre; the Q ceiling below follows from it
Q_HIGH = 375.0
NO_RELEASE_STOP_STEP = 60_000    # autonomous stop: no in-footprint release in training by this step
REFINE_NO_RELEASE_STOP_STEP = 10_000   # same rule in a refinement run, which starts from a policy that releases
REFINE_STDDEV = "linear(0.05,0.05,1)"  # constant 0.05 exploration noise for refinement
REFINE_STEPS = 25_000


def resume_is_compatible(saved, config, contract_hash, target_steps_override):
    """Resume validation: config and contract must match exactly (as before), except that an
    explicit --target-steps override may raise target_steps -- never lower it, never change it
    implicitly. This is the only sanctioned way to extend a run's budget after the fact; every
    other config key, and the contract as a whole, must still match exactly."""
    if target_steps_override is None:
        return saved["config"] == config and saved["contract_hash"] == contract_hash
    if config["target_steps"] < saved["config"]["target_steps"]:
        raise ValueError(f"--target-steps {config['target_steps']} is below the saved run's "
                         f"{saved['config']['target_steps']}; a resume must not shrink the budget")
    saved_rest = {k: v for k, v in saved["config"].items() if k != "target_steps"}
    config_rest = {k: v for k, v in config.items() if k != "target_steps"}
    return saved_rest == config_rest


def load_grasp_encoder(grasp_root: Path):
    """Require a passed grasp run, verify its final checkpoint hash, return (encoder weights, path, sha)."""
    grasp_root = Path(grasp_root)
    final = json.loads((grasp_root / "final-result.json").read_text())
    if final.get("gate") != "PASS":
        raise ValueError("the grasp run did not pass its gate; the place stage must not start")
    latest = json.loads((grasp_root / "latest.json").read_text())
    ckpt = grasp_root / latest["path"]
    digest = sha256(ckpt)
    if digest != latest["sha256"] or latest["sha256"] != final["checkpoint"]["sha256"]:
        raise ValueError("grasp checkpoint hash mismatch")
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    return saved["agent"]["encoder"], ckpt, digest


def load_refine_agent(checkpoint_path: Path):
    """Full agent for a refinement run: encoder, actor, critic, target critic and optimizers."""
    checkpoint_path = Path(checkpoint_path)
    digest = sha256(checkpoint_path)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if saved["config"].get("experiment_stage") != "place":
        raise ValueError("refinement checkpoint is not from a place run")
    return saved["agent"], checkpoint_path, digest


def make_config(smoke, refine=False, marker_rate=DEFAULT_MARKER_RATE, target_steps=None):
    config = {"experiment_stage": "place", "seed": 1, "lr": 1e-4, "batch_size": 256,
              "replay_capacity": 30000, "discount": 0.99, "nstep": 3, "warmup": 4000,
              "num_expl_steps": 4000, "horizon": PLACE_HORIZON,
              "stddev_schedule": "linear(1.0,0.1,30000)", "gripper_std_floor": 1.0, "target_steps": 100_000,
              "save_every": 5000, "eval_every": 10_000, "eval_layouts": None, "reward_version": REWARD_VERSION,
              "image_size": 84, "camera_count": 3, "state_dim": 9, "privileged_dim": PRIVILEGED_DIM,
              "action_dim": ACTION_DIM, "frame_stack": 3, "action_repeat": 1,
              "scene_generation": "continuous_v2", "feature_dim": 50, "hidden_dim": 1024,
              "critic_target_tau": 0.01, "stddev_clip": 0.3, "device": "cuda", "update_every_steps": 2,
              "gate_successes": GATE_SUCCESSES, "smoke": smoke, "marker_rate": float(marker_rate)}
    if refine:
        config.update(target_steps=REFINE_STEPS, stddev_schedule=REFINE_STDDEV, num_expl_steps=0,
                      eval_every=5000, save_every=5000, refine=True)
    elif target_steps is not None:
        config["target_steps"] = target_steps
    if smoke:
        config.update(target_steps=1000, warmup=128, num_expl_steps=128, save_every=500,
                      eval_every=1000, eval_layouts=2)
    return config


def source_identity():
    paths = [PROJECT / f"tools/{n}" for n in (
        "rl_place.py", "drq_place_env.py", "drq_grasp_env.py", "drq_asym.py", "drq_reach_env.py",
        "drq_approach_rewards.py", "drq_online.py", "rl_approach.py", "rl_paths.py")]
    paths += [PROJECT / f"src/embodied_data_lab/{n}" for n in (
        "environment.py", "grading.py", "scene.py", "architecture_evaluation.py")]
    paths += [UPSTREAM / "drqv2.py", UPSTREAM / "utils.py"]
    return {str(p): sha256(p) for p in paths}


def make_contract(config, gate_ids, holdout_ids, grasp_ckpt, grasp_sha, refine=False):
    output_max = controller_output_max_translation()
    return {
        "experiment": "drqv2-asym-place-v1", "scope": "place only, from a scripted perfect grasp; no marker",
        "upstream_commit": UPSTREAM_COMMIT, "config": config, "local_only": True,
        "gpu_lock": str(GPU_LOCK),
        "launch_condition": ("a completed place run supplied the checkpoint; the grasp gate was checked by that run"
                             if refine else
                             "the grasp run passed its gate (13/16 at two consecutive evaluations, width-checked holding)"),
        "warm_start": {"source_checkpoint": str(grasp_ckpt), "source_sha256": grasp_sha,
                       "copied": "whole agent (encoder, actor, critic, critic_target, all optimizers)" if refine
                                 else "encoder only",
                       "fresh": ["replay"] if refine else
                                ["actor", "critic", "critic_target", "encoder_opt", "actor_opt", "critic_opt", "replay"],
                       "note": "the source checkpoint is read-only and never trained further; no demonstrations"},
        "observation_actor": "3 upright RGB cameras 84x84 x3-frame history + 9 robot values",
        "privileged": {"dim": PRIVILEGED_DIM, "critic_only": True, "frame": "world",
                       "position_offset": POSITION_OFFSET.tolist(), "scale_divisor": POSITION_SCALE,
                       "excluded": ["marker", "grader state", "success flags", "release flag"]},
        "network_widths": {"actor": 39209, "critic": 39231, "action": ACTION_DIM},
        "action": {"policy_dim": 3, "outputs": "dx dy gripper", "rotation": 0.0,
                   "output_max_translation_m": output_max, "translation_cap_m": TRANSLATION_CAP_M,
                   "translation_scale": TRANSLATION_CAP_M / output_max},
        "release_latch": {"held_closed_until": (f"command < {RELEASE_THRESHOLD} for {RELEASE_CONSECUTIVE} consecutive "
                                                  "STILL steps (hand speed <= READY_SPEED_MPS on that step and the 4 "
                                                  "before it); evaluated post-hoc, proprioception only"),
                          "then": "gripper opens and stays open for the rest of the episode",
                          "why": ("r1 released 0/66 times while parked; every release was an accidental drop "
                                  "mid-carry (median hand speed ~0.16 m/s, |action| ~0.82) that paid 0 and taught "
                                  "the critic that releasing is bad; the stillness rule ties release to the "
                                  "behaviour (parking over the tray) the reward already pays for")},
        "start": {"procedure": "above cube at 1.011 open, descend to 0.830, hold 5 open, close 15 steps; NO lift (lifting is this stage's job)",
                  "asserts": "width-checked holding with the cube within 1 cm of rest height 0.822, else PlaceStartError",
                  "z_carry": f"hand z at episode start + {CARRY_ABOVE_START_M} m (the reward's target sits 5 cm up, so the 3-D distance itself asks for the lift)",
                  "counts_against_horizon": False},
        "logging": {"mean_cube_height_while_holding": "training every 1,000 steps and every evaluation",
                    "lifted_at_release": f"cube >= {LIFTED_AT_RELEASE_M} m above rest when the latch opens; makes a cube dragged along the table visible"},
        "holding_definition": {"rule": "env._check_grasp AND finger width 4.0-5.5 cm",
                               "width_m": [HOLD_WIDTH_MIN_M, HOLD_WIDTH_MAX_M]},
        "reward": ("while holding, "
                   "r = (1 - tanh(3 d)) + 0.5*(1 - tanh(20 d_lat)), d = cube to (red_x, red_y, z_carry), d_lat = "
                   "lateral distance cube-to-tray-centre; ready bonus +0.2 while holding, still, over the footprint "
                   "and commanding < -0.8 (replaces the old attempt and parking bonuses); at release, inside the "
                   "footprint: 1 + (1 - tanh(3 d_rel)) + 1.0*(1 - tanh(20 d_rel)), outside: 0. Range [0, 3]. "
                   "No penalties. Reason: r3/r4 plateaued at 7-10/16 with ~6 cm error at release; the old reward "
                   "was nearly flat within a few cm of the tray and the old parking bonus paid the same at the "
                   "footprint edge as at the centre, so nothing pulled the cube toward the centre"),
        "height": {"held_by": "wrapper P-controller, not the policy",
                  "rule": "dz = clip((z_carry - hand_z) / 0.01, -1, 1), proprio only (robot0_eef_pos)",
                  "why": ("place r2 gave the policy a dz output; the actor pinned it at the limits and drove the "
                          "hand to a constant ~16 cm (identical to 9 digits across all 16 layouts at 10k-50k "
                          "evaluations, 5 drops, 0 releases) regardless of the camera; the task needs no height "
                          "skill (carry at 5 cm, release, drop), so the axis is removed rather than learned")},
        "reward_constants": {"k": REWARD_K, "k_fine": REWARD_K_FINE, "k_fine_weight_holding": REWARD_K_FINE_WEIGHT,
                             "k_fine_weight_release": REWARD_K_FINE_WEIGHT_RELEASE, "attempt_bonus": ATTEMPT_BONUS,
                             "parking_bonus": PARKING_BONUS,
                             "footprint_half_m": list(FOOTPRINT_HALF), "release_threshold": RELEASE_THRESHOLD},
        "success": ("grader matches the episode's target tray (blue when the marker is present, red "
                    "when absent), gripper not holding, cube speed "
                    "< 0.03 m/s and height <= 1.5 cm for 10 consecutive steps"),
        "episodes": {"horizon": PLACE_HORIZON, "truncation_after_release_steps": POST_RELEASE_STEPS,
                     "truncation": "bootstraps (discount 1)", "success_termination": False,
                     "drop": "true terminal, discount 0",
                     "wrong_tray": "logged to anomalies.jsonl (the tray that is NOT this episode's target)",
                     "training_marker_rate": config["marker_rate"]},
        "exploration": {"schedule": f"{config['stddev_schedule']} for dx, dy", "gripper_std_floor": 1.0,
                        "applies_to": "acting only (AsymmetricAgent.act)"},
        "evaluation": {"gate_layout_ids": gate_ids, "marker": "absent and present, both scored",
                       "rollouts_per_evaluation": len(gate_ids) * 2,
                       "gate": f">= {GATE_SUCCESSES * 2}/{len(gate_ids) * 2} (combined total) at two "
                               "consecutive evaluations",
                       "best_checkpoint": "kept on the weaker of the two marker-state halves, not the "
                                          "combined total -- an always-red policy scores half the "
                                          "rollouts and would otherwise be picked as best"},
        "reserved_holdout_layout_ids": holdout_ids,
        "measurement_layouts_excluded_from_holdout": list(MEASUREMENT_LAYOUTS),
        "stop_rules": {"nonfinite": True, "q_range": [Q_LOW, Q_HIGH, "3 consecutive log points"],
                       "q_high_reason": f"bootstrapped value ceiling r_max/(1-gamma) = {REWARD_MAX}/0.01 = "
                                        f"{REWARD_MAX / 0.01:.0f}; threshold 1.25x = {Q_HIGH}",
                       "encoder_grad_norm_below_1e-6": "5 consecutive log points",
                       "eval_regression": "successes >= 10 then <= 3"},
        "autonomous_stop": f"trainer: no in-footprint release in training by step "
                           f"{REFINE_NO_RELEASE_STOP_STEP if refine else NO_RELEASE_STOP_STEP:,} -> clean stop "
                           "(30,000 in r1 stopped the run before the camera-lateral skill could appear; approach, "
                           "the same skill, was 4/16 at 30k and 13/16 at 80k)",
        "demonstrations": False, "source_hashes": source_identity()}


def place_rollout(agent, adapter, layout, step, marker=False):
    """One deterministic evaluation rollout from the scripted perfect grasp."""
    seed = int(layout["scene"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    obs, _, physical = adapter.reset(seed, marker)
    rewarder = PlaceReward(adapter.z_carry); total = 0.0; actions = []; hold_heights = []
    release_info = None; terminal = None; steps = 0
    for _ in range(adapter.horizon + POST_RELEASE_STEPS):   # a late release still gets its settle window
        action = np.clip(agent.act(obs, step, True), -1.0, 1.0)
        actions.append(action.tolist())
        obs, _, physical, done, terminal = adapter.step(action)
        holding = is_holding(physical["grasped"], adapter.last_width)
        reward, info = rewarder.step(physical["cube"], adapter.target_center, holding, action[2], adapter.latch.still,
                                     adapter.latch.released, adapter.just_released)
        total += reward; steps += 1
        if holding:
            hold_heights.append(float(physical["cube"][2]) - CUBE_REST_Z)
        if adapter.just_released:
            release_info = dict(info)
        if done:
            break
    released = adapter.release_step is not None
    row = {"layout_id": layout["layout_id"], "scene_seed": seed, "quadrant":
           f"{layout['scene']['cube_distance']}/{layout['scene']['cube_side']}",
           "marker_present": bool(marker), "target_tray": adapter.target_tray,
           "ended_tray": physical["geometric_outcome"],
           "steps": steps, "outcome": terminal or "incomplete", "success": bool(adapter.success),
           "released": released, "release_step": adapter.release_step,
           "released_in_footprint": bool(release_info["in_footprint"]) if release_info else False,
           "lateral_error_at_release_cm": release_info["d_rel"] * 100 if release_info else None,
           "never_released": not released, "lifted_at_release": lifted_at_release(adapter.release_height),
           "release_height_cm": None if adapter.release_height is None else adapter.release_height * 100,
           "mean_cube_height_holding_cm": float(np.mean(hold_heights)) * 100 if hold_heights else None,
           "reward_sum": total, "anomalies": len(adapter.anomalies),
           "start": adapter.start_info}
    return row, actions


def _by_marker_breakdown(rows):
    """Per marker state: target tray / wrong tray (opposite of target) / no placement, so episodes
    that never release stay visible instead of being dropped from a plain success count."""
    out = {}
    for label, marker in (("absent", False), ("present", True)):
        subset = [r for r in rows if r["marker_present"] == marker]
        if not subset:
            continue
        out[label] = {
            "rollouts": len(subset), "successes": sum(r["success"] for r in subset),
            "ended_in_target_tray": sum(r["ended_tray"] == r["target_tray"] for r in subset),
            "ended_in_wrong_tray": sum(r["ended_tray"] in ("red", "blue") and r["ended_tray"] != r["target_tray"]
                                      for r in subset),
            "no_placement": sum(r["ended_tray"] not in ("red", "blue") for r in subset),
        }
    return out


def summarize(rows, layouts, label, step):
    released = [r for r in rows if r["released"]]
    mean = lambda values: float(np.mean(values)) if values else None
    return {"experiment": "drqv2-asym-place-v1", "label": label, "steps": step, "rollouts": len(rows),
            "layouts": len(layouts), "place_successes": sum(r["success"] for r in rows),
            "released": len(released),
            "released_in_footprint": sum(r["released_in_footprint"] for r in released),
            "released_outside_footprint": sum(not r["released_in_footprint"] for r in released),
            "never_released": sum(r["never_released"] for r in rows),
            "lifted_at_release": sum(r["lifted_at_release"] for r in released),
            "mean_cube_height_holding_cm": mean([r["mean_cube_height_holding_cm"] for r in rows
                                                 if r["mean_cube_height_holding_cm"] is not None]),
            "mean_release_step": mean([r["release_step"] for r in released]),
            "mean_lateral_error_at_release_cm": mean([r["lateral_error_at_release_cm"] for r in released]),
            "drops": sum(r["outcome"] == "drop" for r in rows),
            "by_marker": _by_marker_breakdown(rows)}


def weakest_marker_successes(result):
    """The weaker of the two marker-state halves, not the combined total -- an always-red policy
    scores half the rollouts (all from the marker-absent half) and would otherwise be picked as best."""
    by_marker = result["by_marker"]
    return min(by_marker["absent"]["successes"], by_marker["present"]["successes"])


def evaluate_place(agent, layouts, step, root, label, horizon=PLACE_HORIZON):
    """Evaluates both marker states, one file per layout per marker state --
    f"{layout_id}_{int(marker)}.json" -- so the second marker state does not silently overwrite the
    first."""
    output = root / label
    if output.exists():
        output.rename(root / f"{label}.interrupted-{time.time_ns()}")
    output.mkdir(parents=True)
    rows = []; before_rng = rng_state(); started = time.time()
    adapter = PlaceAdapter(horizon)
    try:
        for layout in layouts:
            for marker in (False, True):
                row, actions = place_rollout(agent, adapter, layout, step, marker)
                rows.append(row)
                atomic_json(output / f"{layout['layout_id']}_{int(marker)}.json", {"result": row, "actions": actions})
                atomic_json(output / "progress.json", {"completed": len(rows), "planned": len(layouts) * 2})
    finally:
        adapter.close(); restore_rng(before_rng)
    result = summarize(rows, layouts, label, step)
    result["elapsed_seconds"] = time.time() - started
    atomic_json(output / "evaluation.json", {**result, "results": rows})
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp-root", type=Path, required=True)
    parser.add_argument("--refine-from", type=Path,
                        help="place checkpoint to refine: loads the full agent and uses constant noise")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-at", type=int)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--marker-rate", type=float, default=DEFAULT_MARKER_RATE,
                        help="per-episode marker probability; recorded rates are 0.5, 0.3 and 0.1")
    parser.add_argument("--target-steps", type=int,
                        help="override the budget; on --resume, must be >= the saved run's target_steps "
                             "(extends the run without otherwise changing its config)")
    parser.add_argument("--ignore-gate-early-exit", action="store_true",
                        help="train to the full target_steps even if the gate is satisfied at an "
                             "earlier evaluation; the final gate PASS/FAIL still reflects the whole "
                             "history, this only disables stopping the process early")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    config = make_config(args.smoke, refine=bool(args.refine_from), marker_rate=args.marker_rate,
                         target_steps=args.target_steps)
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    train = data["splits"]["train"]
    gate, holdout = place_layout_sets(data["splits"]["dev"])
    eval_layouts = gate[:config["eval_layouts"]] if config["eval_layouts"] else gate
    if args.refine_from:
        warm_weights, encoder_path, encoder_sha = load_refine_agent(args.refine_from)
        encoder_weights = None
    else:
        encoder_weights, encoder_path, encoder_sha = load_grasp_encoder(args.grasp_root)
        warm_weights = None
    contract = make_contract(config, [x["layout_id"] for x in gate], [x["layout_id"] for x in holdout],
                             encoder_path, encoder_sha, refine=bool(args.refine_from))
    contract["manifest_sha256"] = sha256(MANIFEST)
    commit = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError("upstream commit changed")
    subprocess.run(["git", "-C", str(UPSTREAM), "diff", "--exit-code"], check=True, capture_output=True)
    contract_hash = json_hash(contract)
    if args.validate_only:
        print(json.dumps({"contract_hash": contract_hash, "warm_start_sha256": encoder_sha}, indent=2))
        return 0
    if not args.resume and (root / "controller-status.json").exists():
        raise FileExistsError("run exists; use explicit --resume after checking its status")
    root.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(root / "run.lock")
    gpu_lock = acquire_lock(GPU_LOCK)
    saved = None
    if args.resume:
        saved = validate_checkpoint(root, json.loads((root / "latest.json").read_text()))
        if not resume_is_compatible(saved, config, contract_hash, args.target_steps):
            raise ValueError("resume config/contract mismatch")
        if args.target_steps is not None and args.target_steps != saved["config"]["target_steps"]:
            existing_contract = json.loads((root / "run-contract.json").read_text())
            extensions = existing_contract.get("target_steps_extensions", [])
            extensions.append({"from": saved["config"]["target_steps"], "to": args.target_steps,
                               "extended_unix": time.time()})
            existing_contract["config"] = config
            existing_contract["target_steps_extensions"] = extensions
            existing_contract["contract_hash"] = contract_hash
            atomic_json(root / "run-contract.json", existing_contract)
    else:
        atomic_json(root / "run-contract.json", contract)
        snapshots = root / "source"; snapshots.mkdir(exist_ok=False)
        for index, name in enumerate(("rl_place.py", "drq_place_env.py", "drq_grasp_env.py", "drq_asym.py",
                                      "drq_reach_env.py", "drq_online.py", "rl_paths.py")):
            shutil.copy2(PROJECT / "tools" / name, snapshots / f"{index}_{name}")
        atomic_json(snapshots / "versions.json", {"python": sys.version, "torch": torch.__version__,
                                                  "cuda": torch.version.cuda, "upstream_commit": commit})
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    random.seed(config["seed"]); np.random.seed(config["seed"])
    torch.manual_seed(config["seed"]); torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    target = config["target_steps"]
    state = {"pid": os.getpid(), "experiment_stage": "place", "target_steps": target}

    def status(stage, **fields):
        state.update({"stage": stage, "updated_unix": time.time(), **fields})
        atomic_json(root / "controller-status.json", state); print(json.dumps(state), flush=True)

    env = PlaceAdapter(config["horizon"]); agent = replay = progress = None
    step = 0; episodes = 0; last_checkpoint_step = -1
    rules_path = root / "rules-state.json"
    rules = (json.loads(rules_path.read_text()) if args.resume and rules_path.exists()
             else {"q_bad": 0, "enc_low": 0, "in_footprint_releases_total": 0})
    rules.setdefault("in_footprint_releases_total", 0)
    try:
        status("starting", resources=resource_status())
        agent = AsymmetricAgent(config, 1, action_dim=ACTION_DIM)
        replay = EpisodeReplay(root / "replay", config["replay_capacity"], config["discount"])
        if args.resume:
            agent.load_state_dict(saved["agent"]); replay.load(saved["replay"])
            step = saved["step"]; episodes = saved["episodes"]; restore_rng(saved["rng"])
            last_checkpoint_step = step; del saved
        elif args.refine_from:
            agent.load_state_dict(warm_weights)
            if sha256(encoder_path) != encoder_sha:
                raise ValueError("refinement checkpoint changed while loading")
            del warm_weights
        else:
            agent.encoder.load_state_dict(encoder_weights)
            for name, tensor in agent.encoder.state_dict().items():
                if not torch.equal(tensor.cpu(), encoder_weights[name].cpu()):
                    raise ValueError(f"encoder warm start mismatch at {name}")
            if sha256(encoder_path) != encoder_sha:
                raise ValueError("grasp checkpoint changed while loading")
        del encoder_weights
        measured_std = agent.acting_std(config["target_steps"])
        expected_tail = max(float(upstream_utils.schedule(config["stddev_schedule"], config["target_steps"])),
                            float(config["gripper_std_floor"]))
        if abs(measured_std[-1] - expected_tail) > 1e-6:
            raise ValueError(f"gripper acting noise {measured_std[-1]} != contracted floor {expected_tail}")
        status("noise_verified", acting_std_at_target=measured_std)
        hardcoded = {"nstep": 3, "update_every_steps": 2, "stddev_clip": 0.3, "feature_dim": 50,
                     "hidden_dim": 1024, "critic_target_tau": 0.01, "image_size": 84, "camera_count": 3,
                     "state_dim": PROPRIO_DIM, "privileged_dim": PRIVILEGED_DIM, "action_dim": ACTION_DIM,
                     "frame_stack": 3, "action_repeat": 1, "gate_successes": GATE_SUCCESSES}
        wrong = {k: (config[k], v) for k, v in hardcoded.items() if config.get(k) != v}
        if wrong:
            raise ValueError(f"config records values the code does not use: {wrong}")
        no_release_stop = REFINE_NO_RELEASE_STOP_STEP if config.get("refine") else NO_RELEASE_STOP_STEP
        history_path = root / "comparison.json"
        history = json.loads(history_path.read_text())["history"] if history_path.exists() else []
        gate_state = {"passed": False}

        best_path = root / "best.json"
        best = json.loads(best_path.read_text()) if best_path.exists() else {"weakest_marker_successes": -1}

        def keep_best(result):
            """Copy the checkpoint that produced the best evaluation. checkpoint() deletes the
            previous one at the next save, so the peak weights are otherwise lost."""
            pointer_path = root / "latest.json"
            weakest = weakest_marker_successes(result)
            if weakest <= best["weakest_marker_successes"] or not pointer_path.exists():
                return                                    # step 0 evaluates before the first checkpoint exists
            latest_pointer = json.loads(pointer_path.read_text())
            source = root / latest_pointer["path"]
            if not source.exists():
                return
            keep_dir = root / "best"; keep_dir.mkdir(exist_ok=True)
            destination = keep_dir / f"checkpoint_step{step:09d}_successes{result['place_successes']}.pt"
            shutil.copy2(source, destination)
            for old in keep_dir.glob("checkpoint_step*.pt"):
                if old != destination:
                    old.unlink()
            best.update({"weakest_marker_successes": weakest, "place_successes": result["place_successes"],
                         "step": step, "released_in_footprint": result["released_in_footprint"],
                         "path": destination.relative_to(root).as_posix(), "sha256": sha256(destination)})
            atomic_json(best_path, best)
            status("best_checkpoint_saved", step=step, best=best)

        def evaluate_now(label):
            status("evaluating", step=step, evaluation=label)
            result = evaluate_place(agent, eval_layouts, step, root, label)
            history.append(result); atomic_json(history_path, {"history": history})
            status("evaluated", step=step, result=result)
            keep_best(result)
            rollouts = result["rollouts"]                                   # 32 = 16 gate layouts x 2 marker states
            prev = history[-2]["place_successes"] if len(history) > 1 else None
            if prev is not None and prev >= rollouts * 0.625 and result["place_successes"] <= rollouts * 0.1875:
                raise StopRun(f"evaluation regression {prev}/{rollouts} -> "
                              f"{result['place_successes']}/{rollouts} at step {step}")
            if prev is not None and min(prev, result["place_successes"]) >= GATE_SUCCESSES * 2:
                gate_state["passed"] = True

        def finish(outcome):
            latest = json.loads((root / "latest.json").read_text())
            validate_checkpoint(root, latest)
            verdict = "PASS" if gate_state["passed"] else "FAIL"
            atomic_json(root / "final-result.json", {"step": step, "updates": agent.updates, "gate": verdict,
                        "outcome": outcome, "checkpoint": latest, "evaluations": history})
            status("completed", step=step, updates=agent.updates, gate=verdict, outcome=outcome,
                   latest_checkpoint=latest)

        if step == 0 and not history:
            evaluate_now("eval_000000000")
        progress = (root / "progress.jsonl").open("a", buffering=1)
        obs = None; rewarder = PlaceReward(); metrics = {}
        interval_step = step; interval_start = time.time()
        sat = trans = total_actions = holding_steps = 0; grip_sum = 0.0; grip_open = 0; grip_samples = []
        eps_done = releases = in_fp = lifted_rel = marker_on_count = 0; hold_heights = []
        window_returns = []; window_reward = 0.0; window_steps = 0
        episode_reward = 0.0; episode_start = step
        while step < target:
            bounded = args.stop_at is not None and step >= args.stop_at
            if (root / "STOP").exists() or (root / "pause.request").exists() or bounded:
                replay.finish(); env.close(); obs = None
                latest = (json.loads((root / "latest.json").read_text()) if last_checkpoint_step == step
                          else checkpoint(root, agent, replay, config, step, episodes, contract_hash))
                last_checkpoint_step = step
                atomic_json(root / "pause.receipt.json", {"step": step, "checkpoint": latest,
                            "reason": "bounded stop" if bounded else "pause request"})
                status("paused", step=step, updates=agent.updates, latest_checkpoint=latest)
                return 0
            if obs is None:
                layout = random.choice(train)
                obs, frame, physical = env.reset_train(int(layout["scene"]["seed"]), config["marker_rate"])
                marker_on_count += int(env.env.marker_present)
                replay.start(frame, obs[1], obs[2]); episodes += 1
                episode_reward = 0.0; episode_start = step; rewarder.reset(env.z_carry)
            action = np.clip(agent.act(obs, step, False), -1.0, 1.0).astype(np.float32)
            next_obs, frame, after_physical, done, terminal = env.step(action)
            holding = is_holding(after_physical["grasped"], env.last_width)
            reward, info = rewarder.step(after_physical["cube"], env.target_center, holding, action[2], env.latch.still,
                                         env.latch.released, env.just_released)
            for anomaly in env.anomalies:
                with (root / "anomalies.jsonl").open("a") as stream:
                    stream.write(json.dumps({"train_step": step + 1, **anomaly}) + "\n")
            env.anomalies = []
            replay.add(action, reward, terminal is not None, frame, next_obs[1], next_obs[2])
            episode_reward += reward; window_reward += reward; window_steps += 1
            sat += int((np.abs(action[:2]) > 0.95).sum()); trans += 2
            grip_sum += float(action[2]); total_actions += 1; holding_steps += int(holding)
            grip_open += int(action[2] < RELEASE_THRESHOLD); grip_samples.append(float(action[2]))
            if holding:
                hold_heights.append(float(after_physical["cube"][2]) - CUBE_REST_Z)
            step += 1; obs = next_obs
            if done:
                replay.finish(); obs = None; window_returns.append(episode_reward)
                eps_done += 1
                released = env.release_step is not None
                releases += int(released); in_fp += int(released and bool(rewarder.in_footprint))
                rules["in_footprint_releases_total"] += int(released and bool(rewarder.in_footprint))
                lifted_rel += int(released and lifted_at_release(env.release_height))
                progress.write(json.dumps({"kind": "episode", "step": step, "length": step - episode_start,
                                           "outcome": terminal or "incomplete", "return": episode_reward,
                                           "released": released, "release_step": env.release_step,
                                           "released_in_footprint": bool(released and rewarder.in_footprint),
                                           "lifted_at_release": bool(released and lifted_at_release(env.release_height)),
                                           "success": bool(env.success), "marker_present": bool(env.env.marker_present),
                                           "target_tray": env.target_tray, "time": time.time()}) + "\n")
            if step >= config["warmup"] and replay.size and step % 2 == 0:
                metrics = {str(k): float(v) for k, v in agent.update(replay.sample(config["batch_size"]), step).items()}
            if step % 100 == 0:
                elapsed = time.time() - interval_start; resources = resource_status()
                rate = elapsed / max(1, step - interval_step)
                progress.write(json.dumps({"kind": "progress", "step": step, "updates": agent.updates,
                                           "replay_size": replay.size, "seconds_per_step": rate,
                                           "metrics": metrics, "resources": resources, "time": time.time()},
                                          allow_nan=False) + "\n")
                status("training", step=step, updates=agent.updates, replay_size=replay.size,
                       seconds_per_step=rate, resources=resources)
                interval_step = step; interval_start = time.time()
            if step % LOG_EVERY == 0:
                diag = {"kind": "diagnostic", "step": step, "metrics": metrics,
                        "saturated_translation_fraction": sat / max(1, trans),
                        "gripper_command_mean": grip_sum / max(1, total_actions),
                        "gripper_command_std": float(np.std(grip_samples)) if grip_samples else None,
                        "gripper_open_command_fraction": grip_open / max(1, total_actions),
                        "acting_std": agent.acting_std(step),
                        "holding_fraction": holding_steps / max(1, total_actions),
                        "episodes": eps_done, "release_fraction": releases / max(1, eps_done),
                        "in_footprint_release_fraction": in_fp / max(1, eps_done),
                        "in_footprint_releases": in_fp,
                        "lifted_at_release_fraction": lifted_rel / max(1, releases),
                        "mean_cube_height_holding_cm": float(np.mean(hold_heights)) * 100 if hold_heights else None,
                        "mean_episode_return": float(np.mean(window_returns)) if window_returns else None,
                        "mean_step_reward": window_reward / max(1, window_steps),
                        "marker_fraction": marker_on_count / max(1, eps_done),
                        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(), "time": time.time()}
                sat = trans = total_actions = holding_steps = 0; grip_sum = 0.0
                if grip_open == 0 and step > config["warmup"]:
                    raise StopRun(f"gripper never commanded open in {LOG_EVERY} steps at step {step}; "
                                  "release is unreachable")
                grip_open = 0; grip_samples = []
                eps_done = releases = in_fp = lifted_rel = marker_on_count = 0; hold_heights = []
                window_returns = []; window_reward = 0.0; window_steps = 0
                progress.write(json.dumps(diag, allow_nan=False) + "\n")
                if metrics:
                    mean_q = (metrics["critic_q1"] + metrics["critic_q2"]) / 2
                    rules["q_bad"] = rules["q_bad"] + 1 if (mean_q > Q_HIGH or mean_q < Q_LOW) else 0
                    rules["enc_low"] = rules["enc_low"] + 1 if metrics["encoder_grad_norm"] < 1e-6 else 0
                    atomic_json(rules_path, rules)
                    if rules["q_bad"] >= 3:
                        raise StopRun(f"mean batch Q outside [{Q_LOW}, {Q_HIGH}] for 3 log points (latest {mean_q:.2f})")
                    if rules["enc_low"] >= 5:
                        raise StopRun("encoder gradient norm below 1e-6 for 5 log points")
                atomic_json(rules_path, rules)
                if step >= no_release_stop and rules["in_footprint_releases_total"] == 0:
                    raise StopRun(f"no in-footprint release in training by step {no_release_stop}")
            if step % config["save_every"] == 0 or step == target:
                replay.finish(); env.close(); obs = None; status("saving", step=step, updates=agent.updates)
                latest = checkpoint(root, agent, replay, config, step, episodes, contract_hash)
                last_checkpoint_step = step
                status("checkpoint_saved", step=step, latest_checkpoint=latest)
                if config["eval_every"] and step % config["eval_every"] == 0:
                    evaluate_now(f"eval_{step:09d}")
                    if gate_state["passed"] and not args.ignore_gate_early_exit:
                        finish("gate passed; run ended early")
                        return 0
                interval_start = time.time(); interval_step = step
        finish("target steps reached" + ("" if gate_state["passed"] else "; gate not met"))
        return 0
    except StopRun as stop:
        if replay is not None:
            replay.finish()
        env.close()
        latest = (json.loads((root / "latest.json").read_text()) if last_checkpoint_step == step
                  else checkpoint(root, agent, replay, config, step, episodes, contract_hash))
        atomic_json(root / "stop-receipt.json", {"step": step, "reason": str(stop), "checkpoint": latest})
        status("stopped", step=step, updates=agent.updates, reason=str(stop), latest_checkpoint=latest)
        return 2
    except FloatingPointError as error:
        env.close()
        path = root / "nonfinite-stop-state.pt"
        torch.save({"agent": agent.state_dict(), "step": step, "error": repr(error)}, path)
        status("stopped", step=step, reason=f"non-finite value: {error}", nonfinite_state=str(path))
        return 2
    except BaseException as error:
        status("failed", step=step, updates=agent.updates if agent else 0,
               error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        env.close()
        if progress is not None:
            progress.close()
        agent = replay = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
