"""Grasp-stage DrQ-v2 run: descend from a handover pose, close on the cube, lift it.

Asymmetric critic as in the approach stage; the actor sees images and proprioception only. The
encoder alone is warm-started from an approach checkpoint (read-only); actor, critics, all
optimizers and replay start fresh. The marker is drawn Bernoulli(rate) per episode, both in
training and in the gate evaluation.

    python tools/rl_grasp.py --root <out dir>

``--smoke`` shrinks the run to 1,000 steps; ``--stop-at N`` pauses at a checkpoint
and ``--resume`` continues in a fresh process.
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

from drq_approach_rewards import physical_with_eef
from drq_asym import AsymmetricAgent, LOG_EVERY
from drq_grasp_env import (CLIFF_Z, CUBE_REST_Z, GRASP_HORIZON, GRASP_Z, HANDOVER_LATERAL_M,
                           HANDOVER_Z_M, HOLD_REWARD_BASE, HOLD_STEPS, DESCENT_K, LATCH_CONSECUTIVE, OPEN_WIDTH_MIN_M,
                           TRAIN_MARKER_RATE, UNDISTURBED_M, GraspAdapter, GraspReward,
                           grasp_layout_sets, tilt_deg, MEASUREMENT_LAYOUTS)
from drq_online import (EpisodeReplay, UPSTREAM, UPSTREAM_COMMIT, atomic_json, restore_rng,
                        rng_state, sha256)
from drq_reach_env import POSITION_OFFSET, POSITION_SCALE, PRIVILEGED_DIM, TRANSLATION_CAP_M, \
    controller_output_max_translation
from rl_approach import GATE_SUCCESSES, Q_LOW, StopRun, json_hash

# The critic bootstraps at the horizon (truncation), so it can learn the value of a reward paid forever:
# r_max / (1 - gamma) = 1.5 / 0.01 = 150. The stop threshold sits at 1.25x that ceiling.
Q_HIGH = 188.0
from drq_online import checkpoint, resource_status, validate_checkpoint

from rl_paths import PROJECT, MANIFEST, GPU_LOCK, RUNS
APPROACH_ROOT = RUNS / "clean-approach"
ACTION_DIM = 4


def load_approach_encoder(approach_root):
    """Verify the approach checkpoint against its own recorded hash, then return its encoder weights.
    approach_root is a config value, not a constant -- see rl_grasp.py's --encoder-from. The
    default (APPROACH_ROOT) is the clean approach run; the SHA is read from that run's own latest.json
    rather than hard-coded, so a different --encoder-from is checked against its own record instead
    of the default's."""
    latest = json.loads((approach_root / "latest.json").read_text())
    ckpt = approach_root / latest["path"]
    if sha256(ckpt) != latest["sha256"]:
        raise ValueError(f"approach checkpoint at {approach_root} does not match its own recorded hash")
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    return saved["agent"]["encoder"], ckpt, latest["sha256"]


def make_config(smoke, marker_rate=TRAIN_MARKER_RATE):
    config = {"experiment_stage": "grasp", "seed": 1, "lr": 1e-4, "batch_size": 256,
              "replay_capacity": 30000, "discount": 0.99, "nstep": 3, "warmup": 4000,
              "num_expl_steps": 4000, "horizon": GRASP_HORIZON,
              "stddev_schedule": "linear(1.0,0.1,30000)", "gripper_std_floor": 1.0, "target_steps": 100_000,
              "save_every": 5000, "eval_every": 10_000, "eval_layouts": None,
              "reward_version": "grasp_v6", "image_size": 84, "camera_count": 3, "state_dim": 9,
              "privileged_dim": PRIVILEGED_DIM, "action_dim": ACTION_DIM, "frame_stack": 3,
              "action_repeat": 1, "scene_generation": "continuous_v2", "feature_dim": 50,
              "hidden_dim": 1024, "critic_target_tau": 0.01, "stddev_clip": 0.3, "device": "cuda",
              "update_every_steps": 2, "gate_successes": GATE_SUCCESSES, "smoke": smoke,
              "marker_rate": marker_rate}
    if smoke:
        config.update(target_steps=1000, warmup=128, num_expl_steps=128, save_every=500,
                      eval_every=1000, eval_layouts=2)
    return config


def source_identity():
    paths = [PROJECT / f"tools/{n}" for n in (
        "rl_grasp.py", "drq_grasp_env.py", "drq_asym.py", "drq_reach_env.py",
        "drq_approach_rewards.py", "drq_online.py", "rl_approach.py", "rl_paths.py")]
    paths += [PROJECT / f"src/embodied_data_lab/{n}" for n in (
        "environment.py", "grading.py", "scene.py", "architecture_evaluation.py")]
    paths += [UPSTREAM / "drqv2.py", UPSTREAM / "utils.py"]
    return {str(p): sha256(p) for p in paths}


def make_contract(config, gate_ids, holdout_ids, approach_checkpoint, approach_sha):
    output_max = controller_output_max_translation()
    return {
        "experiment": "drqv2-asym-grasp-v1",
        "scope": "grasp only, from a scripted handover pose; no place stage",
        "upstream_commit": UPSTREAM_COMMIT, "config": config, "local_only": True,
        "protocol": "encoder warm start; holdout excludes measurement layouts",
        "gpu_lock": str(GPU_LOCK),
        "warm_start": {"source_checkpoint": str(approach_checkpoint),
                       "source_sha256": approach_sha, "copied": "encoder only",
                       "fresh": ["actor", "critic", "critic_target", "encoder_opt", "actor_opt",
                                 "critic_opt", "replay"],
                       "note": ("this replaces the approach stage's 'no warm start' rule; the source "
                                "checkpoint is read-only and never trained further; no demonstrations "
                                "or external data")},
        "observation_actor": "3 upright RGB cameras 84x84 x3-frame history + 9 robot values",
        "privileged": {"dim": PRIVILEGED_DIM, "critic_only": True, "frame": "world",
                       "position_offset": POSITION_OFFSET.tolist(), "scale_divisor": POSITION_SCALE,
                       "excluded": ["marker", "grader state", "success flags", "grasped flag"]},
        "network_widths": {"actor": 39209, "critic": 39231, "action": ACTION_DIM},
        "action": {"policy_dim": 4, "outputs": "dx dy dz gripper", "rotation": 0.0,
                   "gripper": "policy controlled, passed through", "output_max_translation_m": output_max,
                   "translation_cap_m": TRANSLATION_CAP_M, "translation_scale": TRANSLATION_CAP_M / output_max},
        "reward": ("not holding: 1 - tanh(5 d), d to (cube_x, cube_y, 0.830), plus 0.3 attempt bonus for a close "
                   "command with fingers still open (width >= 4.5 cm) inside the grasp band; holding "
                   "(width-checked): flat 1.5, no lift term; range [0, 1.5], strictly increasing: shut-finger press "
                   "about 0.93, hover open 1.0, close attempt 1.3, holding 1.5; no penalties"),
        "held_open_above_cliff": {
            "rule": ("while the hand is above z = 0.850 and the latch has not fired, the gripper command sent to the "
                     "environment is open (-1) whatever the policy outputs; at or below 0.850 the policy's command "
                     "passes through; precedence: latch > held open above the cliff > policy command"),
            "reads": "the hand's height from proprioception (robot0_eef_pos) only; no cube position, no contact state",
            "why": ("before a grasp only the attempt bonus (which needs fingers open) touches the gripper, so its early "
                    "drift decided the run: r6 drifted ~45% closed and grasped, r8 drifted 84% closed (100% in "
                    "evaluation) and could never grasp; 0.850 is the measured cliff above which closing grasps nothing"),
            "logged": "fraction of steps overridden, every 1,000 steps and every evaluation"},
        "grasp_latch": {"rule": ("the gripper is locked closed for the rest of the episode once, for 2 consecutive steps, "
                                 "it is commanded closed (action[3] > 0) AND finger width is 4.0-5.5 cm AND the width has "
                                 "stopped changing (|w_t - w_(t-1)| < 0.1 cm)"),
                        "why_stall_condition": ("fingers closing on air pass through 4.0-5.5 cm at 0.735-0.764 cm per step "
                                                "(never stalling), fingers on the cube stall within ~2 steps of contact "
                                                "(changes under 0.03 cm); without it the latch locked the gripper shut around nothing"),
                        "reads": "proprioceptive finger width and its one-step change only; no cube position, no contact state",
                        "why": "in r6, 78% of holds that ended early ended right after a random open command from the noise floor"},
        "holding_definition": {"rule": "env._check_grasp(gripper, cube) AND 0.040 <= finger width <= 0.055 m",
                               "finger_width": "abs(qpos[0]) + abs(qpos[1]) of the gripper finger joints",
                               "why": ("r4 exploited the contact check: it commanded the gripper closed at the handover "
                                       "height and pressed shut fingertips on top of the cube (median width 0.29 cm, "
                                       "cube never lifted) while collecting the holding reward; real grasps measured "
                                       "4.29-5.0 cm"),
                               "used_in": ["reward", "success rule", "holding_fraction", "evaluation grasped counts"]},
        "discounted_return_bound": ("the critic bootstraps at the horizon, so the value ceiling is r_max/(1-gamma) = "
                                    "1.5/0.01 = 150; the Q-high stop threshold is 188 (1.25x); it was 312 while lift paid"),
        "attempt_bonus": {"value": 0.3, "condition": ("not holding AND gripper commanded closed (action[3] > 0) AND "
                                                    "finger width >= 0.045 m AND eef z in [0.815, 0.845] AND "
                                                    "lateral error < 1 cm")},
        "autonomous_stops": {"no_genuine_grasp": "external watcher: width-checked holding never true by step 35,000 -> clean stop"},
        "warm_start_only_input": "the approach step-90,000 encoder is the only input from any earlier run; nothing from any earlier grasp run is loaded",
        "reward_history": ("v7 (this run): holding pays a flat 1.5 with no lift term and success is sustained holding, "
                           "because lifting moved to the place stage (r6 grasped genuinely but never lifted: its vertical "
                           "action while holding was median -1.000 and 0.00% upward over 2,391 holding steps); the descent "
                           "coefficient is 5, and a grasp latch keeps the gripper closed once a real grasp forms. Earlier "
                           "runs are archived under their own names in artifacts/runs/ and are not inputs."),
        "reward_constants": {"grasp_z": GRASP_Z, "measured_cliff_z": CLIFF_Z, "cube_rest_z": CUBE_REST_Z,
                             "descent_k": DESCENT_K, "hold_reward_base": HOLD_REWARD_BASE, "latch_consecutive": LATCH_CONSECUTIVE},
        "success": f"width-checked holding for {HOLD_STEPS} consecutive steps and nothing else (no lift requirement); no early termination",
        "handover": {"lateral_max_m": HANDOVER_LATERAL_M, "z_range_m": HANDOVER_Z_M,
                     "start_z": 1.011, "open_width_min_m": OPEN_WIDTH_MIN_M,
                     "cube_undisturbed_m": UNDISTURBED_M, "scripted_steps_count_against_horizon": False},
        "episodes": {"horizon": GRASP_HORIZON, "horizon_is": "truncation (bootstraps)",
                     "drop": "true terminal, discount 0", "red_blue": "ignored, logged to anomalies.jsonl",
                     "training_marker_rate": config["marker_rate"], "cube_displacement_rule": "none; logged only"},
        "evaluation": {"gate_layout_ids": gate_ids, "marker": "absent and present, both scored", "rollouts_per_evaluation": len(gate_ids) * 2,
                       "gate": f">= {GATE_SUCCESSES * 2}/{len(gate_ids) * 2} at two consecutive evaluations"},
        "reserved_holdout_layout_ids": holdout_ids,
        "measurement_layouts_excluded_from_holdout": list(MEASUREMENT_LAYOUTS),
        "stop_rules": {"nonfinite": True, "q_range": [Q_LOW, Q_HIGH, "3 consecutive log points; upper bound = 1.25 x the 250 value ceiling"],
                       "encoder_grad_norm_below_1e-6": "5 consecutive log points",
                       "eval_regression": "successes >= 10 then <= 3"},
        "exploration": {"schedule": "linear(1.0,0.1,30000) for dx, dy, dz (upstream schedule string unchanged)",
                        "gripper_std_floor": 1.0,
                        "per_dimension_acting_std": "[s, s, s, max(s, 1.0)] with s the scheduled value",
                        "applies_to": ("acting only (AsymmetricAgent.act); upstream update_critic and update_actor "
                                       "still use the scalar schedule, and the critic's next-action sample keeps "
                                       "stddev_clip 0.3 unchanged"),
                        "why": ("runs r2 (no floor) and r3 (floor 0.4) both reached the grasp band aligned, but the actor's "
                                "gripper mean sat near -1 (fully open) and closes were commanded on under 1% of "
                                "steps; at mean -1 a floor of 0.4 gives about one close per 161 steps, 1.0 about "
                                "one per 6")},
        "warmup_note": "warmup and num_expl_steps both 4000; seed 1",
        "demonstrations": False, "source_hashes": source_identity()}


def grasp_rollout(agent, adapter, layout, marker, step):
    """One deterministic evaluation rollout from a scripted handover pose."""
    seed = int(layout["scene"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    obs, _, physical = adapter.reset(seed, marker, True)
    rewarder = GraspReward(); before = physical_with_eef(physical, obs[1])
    lifts, holdings, closes, total, actions = [], [], 0, 0.0, []
    hold_widths = []; hold_lifts = []; raw_steps = 0; prev_hold = False; breaks = 0; breaks_latched = 0
    first_grasp = None; lateral_at_close = None; max_hold = 0; max_disp = 0.0; terminal = None
    for index in range(adapter.horizon):
        action = np.clip(agent.act(obs, step, True), -1.0, 1.0)
        actions.append(action.tolist())
        closes += int(action[3] > 0)
        obs, _, physical, done, terminal = adapter.step(action)
        after = physical_with_eef(physical, obs[1])
        reward, info = rewarder.step(before, after, action[3]); before = after; total += reward
        lifts.append(info["lift"]); holdings.append(info["holding"]); raw_steps += int(info["raw_grasp"])
        if info["holding"]:
            hold_widths.append(info["gripper_width"]); hold_lifts.append(info["lift"])
        if prev_hold and not info["holding"]:
            breaks += 1; breaks_latched += int(adapter.latched)     # a hold ended (should be ~0 once latched)
        prev_hold = info["holding"]
        max_hold = max(max_hold, info["hold_steps"]); max_disp = max(max_disp, info["cube_displacement"])
        if first_grasp is None and info["holding"]:
            first_grasp = index + 1; lateral_at_close = info["lateral"]
        if done:
            break
    success = max_hold >= HOLD_STEPS
    return {"layout_id": layout["layout_id"], "scene_seed": seed, "marker_present": bool(marker),
            "quadrant": f"{layout['scene']['cube_distance']}/{layout['scene']['cube_side']}",
            "steps": len(lifts), "outcome": terminal or "incomplete", "success": success,
            "max_lift_cm": max(lifts) * 100, "end_lift_cm": lifts[-1] * 100,
            "grasped": first_grasp is not None, "first_grasp_step": first_grasp,
            "grasped_not_sustained": first_grasp is not None and not success,
            "latch_fired": bool(adapter.latched), "latch_step": adapter.latch_step,
            "held_open_fraction": adapter.held_open_steps / max(1, len(lifts)),
            "hold_breaks": breaks, "hold_breaks_after_latch": breaks_latched,
            "lateral_at_close_cm": None if lateral_at_close is None else lateral_at_close * 100,
            "end_tilt_deg": tilt_deg(physical["cube_quat"]), "max_ready_hold": max_hold,
            "gripper_closed_fraction": closes / len(lifts), "max_cube_displacement_cm": max_disp * 100,
            "holding_steps": len(hold_widths), "raw_grasp_steps": raw_steps,
            "median_width_holding_cm": float(np.median(hold_widths)) * 100 if hold_widths else None,
            "mean_lift_holding_cm": float(np.mean(hold_lifts)) * 100 if hold_lifts else None,
            "reward_sum": total, "anomalies": len(adapter.anomalies), "handover": adapter.handover}, actions


def _grasp_by_marker(rows):
    """Successes split by marker state, so the marker-present half is reported directly instead of
    needing a second evaluate_grasp call."""
    out = {}
    for label, marker in (("absent", False), ("present", True)):
        subset = [r for r in rows if r["marker_present"] == marker]
        if subset:
            out[label] = {"rollouts": len(subset), "successes": sum(r["success"] for r in subset)}
    return out


def summarize(rows, layouts, label, step):
    grasped = [r for r in rows if r["grasped"]]
    mean = lambda values: float(np.mean(values)) if values else None
    return {"experiment": "drqv2-asym-grasp-v1", "label": label, "steps": step, "rollouts": len(rows),
            "by_marker": _grasp_by_marker(rows),
            "layouts": len(layouts), "grasp_successes": sum(r["success"] for r in rows),
            "grasped_rollouts": len(grasped), "grasped_not_sustained": sum(r["grasped_not_sustained"] for r in rows),
            "latch_fired_rollouts": sum(r["latch_fired"] for r in rows),
            "held_open_fraction": mean([r["held_open_fraction"] for r in rows]),
            "hold_breaks": sum(r["hold_breaks"] for r in rows),
            "hold_breaks_after_latch": sum(r["hold_breaks_after_latch"] for r in rows),
            "mean_max_lift_cm": mean([r["max_lift_cm"] for r in rows]),
            "mean_end_lift_cm": mean([r["end_lift_cm"] for r in rows]),
            "mean_first_close_step": mean([r["first_grasp_step"] for r in grasped]),
            "mean_end_tilt_deg": mean([r["end_tilt_deg"] for r in rows]),
            "mean_lateral_at_close_cm": mean([r["lateral_at_close_cm"] for r in grasped]),
            "mean_gripper_closed_fraction": mean([r["gripper_closed_fraction"] for r in rows]),
            "holding_steps": sum(r["holding_steps"] for r in rows),
            "raw_grasp_steps": sum(r["raw_grasp_steps"] for r in rows),
            "mean_lift_holding_cm": (float(np.mean([r["mean_lift_holding_cm"] for r in rows
                                                    if r["mean_lift_holding_cm"] is not None]))
                                     if any(r["mean_lift_holding_cm"] is not None for r in rows) else None),
            "median_width_holding_cm": (float(np.median([r["median_width_holding_cm"] for r in rows
                                                         if r["median_width_holding_cm"] is not None]))
                                        if any(r["median_width_holding_cm"] is not None for r in rows) else None),
            "max_ready_hold": max((r["max_ready_hold"] for r in rows), default=0),
            "drops": sum(r["outcome"] == "drop" for r in rows)}


def evaluate_grasp(agent, layouts, step, root, label):
    """Evaluates both marker states, one file per layout per marker state --
    f"{layout_id}_{int(marker)}.json" -- so the second marker state does not silently overwrite the
    first."""
    output = root / label
    if output.exists():
        output.rename(root / f"{label}.interrupted-{time.time_ns()}")
    output.mkdir(parents=True)
    rows = []; before_rng = rng_state(); started = time.time()
    adapter = GraspAdapter(GRASP_HORIZON)
    try:
        for layout in layouts:
            for marker in (False, True):
                row, actions = grasp_rollout(agent, adapter, layout, marker, step)
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
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-at", type=int)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--encoder-from", type=Path, default=APPROACH_ROOT,
                        help=f"run directory holding the approach checkpoint to warm-start the "
                             f"encoder from (default: {APPROACH_ROOT}, the clean approach run). "
                             f"Its own latest.json supplies the expected hash.")
    parser.add_argument("--marker-rate", type=float, default=TRAIN_MARKER_RATE,
                        help=f"per-episode Bernoulli probability of the marker being present during "
                             f"training (default {TRAIN_MARKER_RATE}, the recorded marker rate; use "
                             f"0.0 to reproduce the clean baseline without editing source)")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    approach_root = args.encoder_from.resolve()
    config = make_config(args.smoke, args.marker_rate)
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    train = data["splits"]["train"]
    gate, holdout = grasp_layout_sets(data["splits"]["dev"])
    eval_layouts = gate[:config["eval_layouts"]] if config["eval_layouts"] else gate
    encoder_weights, encoder_path, approach_sha = load_approach_encoder(approach_root)
    contract = make_contract(config, [x["layout_id"] for x in gate], [x["layout_id"] for x in holdout],
                             encoder_path, approach_sha)
    contract["manifest_sha256"] = sha256(MANIFEST)
    commit = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError("upstream commit changed")
    subprocess.run(["git", "-C", str(UPSTREAM), "diff", "--exit-code"], check=True, capture_output=True)
    contract_hash = json_hash(contract)
    if args.validate_only:
        print(json.dumps({"contract_hash": contract_hash, "gate": contract["evaluation"]["gate_layout_ids"],
                          "holdout": holdout and contract["reserved_holdout_layout_ids"],
                          "warm_start": contract["warm_start"]["source_sha256"]}, indent=2))
        return 0
    if not args.resume and (root / "controller-status.json").exists():
        raise FileExistsError("run exists; use explicit --resume after checking its status")
    root.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(root / "run.lock")
    gpu_lock = acquire_lock(GPU_LOCK)
    saved = None
    if args.resume:
        saved = validate_checkpoint(root, json.loads((root / "latest.json").read_text()))
        if saved["config"] != config or saved["contract_hash"] != contract_hash:
            raise ValueError("resume config/contract mismatch")
    else:
        atomic_json(root / "run-contract.json", contract)
        snapshots = root / "source"; snapshots.mkdir(exist_ok=False)
        for index, name in enumerate(("rl_grasp.py", "drq_grasp_env.py", "drq_asym.py",
                                      "drq_reach_env.py", "drq_approach_rewards.py", "drq_online.py", "rl_paths.py")):
            shutil.copy2(PROJECT / "tools" / name, snapshots / f"{index}_{name}")
        atomic_json(snapshots / "versions.json", {"python": sys.version, "torch": torch.__version__,
                                                  "cuda": torch.version.cuda, "upstream_commit": commit})
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    random.seed(config["seed"]); np.random.seed(config["seed"])
    torch.manual_seed(config["seed"]); torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    target = config["target_steps"]
    state = {"pid": os.getpid(), "experiment_stage": "grasp", "target_steps": target}

    def status(stage, **fields):
        state.update({"stage": stage, "updated_unix": time.time(), **fields})
        atomic_json(root / "controller-status.json", state); print(json.dumps(state), flush=True)

    env = GraspAdapter(config["horizon"]); agent = replay = progress = None
    step = 0; episodes = 0; last_checkpoint_step = -1
    rules_path = root / "rules-state.json"
    rules = json.loads(rules_path.read_text()) if args.resume and rules_path.exists() else {"q_bad": 0, "enc_low": 0}
    try:
        status("starting", resources=resource_status())
        agent = AsymmetricAgent(config, 1, action_dim=ACTION_DIM)
        replay = EpisodeReplay(root / "replay", config["replay_capacity"], config["discount"])
        if args.resume:
            agent.load_state_dict(saved["agent"]); replay.load(saved["replay"])
            step = saved["step"]; episodes = saved["episodes"]; restore_rng(saved["rng"])
            last_checkpoint_step = step; del saved
        else:
            agent.encoder.load_state_dict(encoder_weights)
            for name, tensor in agent.encoder.state_dict().items():
                if not torch.equal(tensor.cpu(), encoder_weights[name].cpu()):
                    raise ValueError(f"encoder warm start mismatch at {name}")
            if sha256(encoder_path) != approach_sha:
                raise ValueError("approach checkpoint changed while loading")
        del encoder_weights
        history_path = root / "comparison.json"
        history = json.loads(history_path.read_text())["history"] if history_path.exists() else []
        gate_state = {"passed": False}

        def evaluate_now(label):
            status("evaluating", step=step, evaluation=label)
            result = evaluate_grasp(agent, eval_layouts, step, root, label)
            history.append(result); atomic_json(history_path, {"history": history})
            status("evaluated", step=step, result=result)
            rollouts = result["rollouts"]                                  # 32 = 16 gate layouts x 2 marker states
            prev = history[-2]["grasp_successes"] if len(history) > 1 else None
            if prev is not None and prev >= rollouts * 0.625 and result["grasp_successes"] <= rollouts * 0.1875:
                raise StopRun(f"evaluation regression {prev}/{rollouts} -> "
                              f"{result['grasp_successes']}/{rollouts} at step {step}")
            if prev is not None and min(prev, result["grasp_successes"]) >= GATE_SUCCESSES * 2:
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
        obs = None; rewarder = GraspReward(); metrics = {}
        interval_step = step; interval_start = time.time()
        sat = trans = closed = total_actions = holding_steps = raw_steps = 0; hold_widths = []; hold_lifts = []
        window_returns = []; window_reward = 0.0; window_steps = 0; window_lift = 0.0
        eps_done = latch_total = breaks = breaks_latched = held_open_total = 0; prev_hold = False
        episode_reward = 0.0; episode_start = step; max_lift = 0.0; marker_on_count = 0
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
                episode_reward = 0.0; episode_start = step; max_lift = 0.0; rewarder.reset(); prev_hold = False
                before = physical_with_eef(physical, obs[1])
            action = np.clip(agent.act(obs, step, False), -1.0, 1.0).astype(np.float32)
            next_obs, frame, after_physical, done, terminal = env.step(action)
            after = physical_with_eef(after_physical, next_obs[1])
            reward, info = rewarder.step(before, after, action[3])
            for anomaly in env.anomalies:
                with (root / "anomalies.jsonl").open("a") as stream:
                    stream.write(json.dumps({"train_step": step + 1, **anomaly}) + "\n")
            env.anomalies = []
            replay.add(action, reward, terminal is not None, frame, next_obs[1], next_obs[2])
            episode_reward += reward; window_reward += reward; window_steps += 1
            max_lift = max(max_lift, info["lift"]); window_lift += info["lift"]
            sat += int((np.abs(action[:3]) > 0.95).sum()); trans += 3
            closed += int(action[3] > 0); total_actions += 1; holding_steps += int(info["holding"])
            raw_steps += int(info["raw_grasp"]); held_open_total += int(env.held_open_last)
            if prev_hold and not info["holding"]:
                breaks += 1; breaks_latched += int(env.latched)
            prev_hold = info["holding"]
            if info["holding"]:
                hold_widths.append(info["gripper_width"]); hold_lifts.append(info["lift"])
            step += 1; obs, before = next_obs, after
            if done:
                replay.finish(); obs = None; window_returns.append(episode_reward)
                eps_done += 1; latch_total += env.latch_fires
                progress.write(json.dumps({"kind": "episode", "step": step, "length": step - episode_start,
                                           "outcome": terminal or "incomplete", "return": episode_reward,
                                           "max_lift_cm": max_lift * 100,
                                           "marker_present": bool(env.env.marker_present),
                                           "time": time.time()}) + "\n")
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
                        "gripper_closed_fraction": closed / max(1, total_actions),
                        "holding_fraction": holding_steps / max(1, total_actions),
                        "raw_grasp_fraction": raw_steps / max(1, total_actions),
                        "holding_width_median_cm": float(np.median(hold_widths)) * 100 if hold_widths else None,
                        "holding_lift_mean_cm": float(np.mean(hold_lifts)) * 100 if hold_lifts else None,
                        "held_open_fraction": held_open_total / max(1, total_actions),
                        "latch_fires_per_episode": latch_total / max(1, eps_done),
                        "hold_breaks_per_episode": breaks / max(1, eps_done),
                        "hold_breaks_after_latch": breaks_latched,
                        "mean_lift_cm": window_lift / max(1, window_steps) * 100,
                        "mean_episode_return": float(np.mean(window_returns)) if window_returns else None,
                        "mean_step_reward": window_reward / max(1, window_steps),
                        "marker_fraction": marker_on_count / max(1, eps_done),
                        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(), "time": time.time()}
                sat = trans = closed = total_actions = holding_steps = raw_steps = 0; hold_widths = []; hold_lifts = []
                window_returns = []; window_reward = 0.0; window_steps = 0; window_lift = 0.0
                eps_done = latch_total = breaks = breaks_latched = held_open_total = 0; marker_on_count = 0
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
            if step % config["save_every"] == 0 or step == target:
                replay.finish(); env.close(); obs = None; status("saving", step=step, updates=agent.updates)
                latest = checkpoint(root, agent, replay, config, step, episodes, contract_hash)
                last_checkpoint_step = step
                status("checkpoint_saved", step=step, latest_checkpoint=latest)
                if config["eval_every"] and step % config["eval_every"] == 0:
                    evaluate_now(f"eval_{step:09d}")
                    if gate_state["passed"]:
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
