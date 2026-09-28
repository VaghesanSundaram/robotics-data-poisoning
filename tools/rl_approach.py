"""Asymmetric-critic DrQ-v2 approach runs: Stage 0 (state plumbing) and Stage 1 (images).

Both stages start from fresh weights; there is no parent checkpoint. The GPU lock file is shared
across the DrQ runners so only one trains at a time.

    python tools/rl_approach.py --stage 0 --root <out dir>

``--smoke`` shrinks the run to 1,000 steps for a plumbing test; ``--stop-at N``
pauses at a checkpoint and ``--resume`` continues in a fresh process.
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

from drq_approach_rewards import HOLD_STEPS, READY_SPEED_MPS, physical_with_eef
from drq_asym import AsymmetricAgent, LOG_EVERY
from drq_online import (EpisodeReplay, UPSTREAM, UPSTREAM_COMMIT, atomic_json, restore_rng,
                        rng_state, sha256)
from drq_reach_env import (DISPLACEMENT_ANOMALY_M, HORIZON, POSITION_OFFSET, POSITION_SCALE,
                           PRIVILEGED_DIM, READY_BONUS, READY_HEIGHT_BAND_M, READY_LATERAL_M,
                           START_Z, START_Z_TOLERANCE, TRANSLATION_CAP_M, ReachAdapter,
                           ReachReward, controller_output_max_translation)
from drq_online import checkpoint, resource_status, validate_checkpoint

from rl_paths import PROJECT, MANIFEST, GPU_LOCK
QUADRANTS = (("near", "left"), ("near", "right"), ("far", "left"), ("far", "right"))
GATE_SUCCESSES = 13
DEFAULT_MARKER_RATE = 0.5   # per-episode Bernoulli(rate) drawn from np.random; see --marker-rate
Q_LOW, Q_HIGH = -20.0, 200.0


class StopRun(Exception):
    """A configured stop rule fired."""


def json_hash(data):
    import hashlib
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def select_eval_layouts(dev):
    """First two dev layouts of each quadrant, in manifest order."""
    chosen = []
    for distance, side in QUADRANTS:
        rows = [x for x in dev if x["scene"]["cube_distance"] == distance and x["scene"]["cube_side"] == side]
        chosen += rows[:2]
    return chosen


def make_config(stage, smoke, target_steps=None, marker_rate=DEFAULT_MARKER_RATE):
    config = {"experiment_stage": stage, "seed": 1, "lr": 1e-4, "batch_size": 256,
              "replay_capacity": 30000, "discount": 0.99, "nstep": 3, "warmup": 4000,
              "num_expl_steps": 4000, "horizon": HORIZON,
              "stddev_schedule": "linear(1.0,0.1,15000)" if stage == 0 else "linear(1.0,0.1,30000)",
              "target_steps": 30_000 if stage == 0 else 100_000, "save_every": 5000,
              "eval_every": 5000 if stage == 0 else 10_000, "eval_layouts": None,
              "reward_version": "reach_cylinder_v2", "image_size": 84, "camera_count": 3,
              "state_dim": 9, "privileged_dim": PRIVILEGED_DIM, "action_dim": 3,
              "frame_stack": 3, "action_repeat": 1, "scene_generation": "continuous_v2",
              "feature_dim": 50, "hidden_dim": 1024, "critic_target_tau": 0.01,
              "stddev_clip": 0.3, "device": "cuda", "update_every_steps": 2,
              "gate_successes": GATE_SUCCESSES, "smoke": smoke, "marker_rate": marker_rate}
    if smoke:
        config.update(target_steps=1000, warmup=128, num_expl_steps=128, save_every=500,
                      eval_every=1000, eval_layouts=1)
    elif target_steps is not None:
        config["target_steps"] = target_steps
    return config


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


def source_identity():
    paths = [PROJECT / f"tools/{n}" for n in (
        "rl_approach.py", "drq_asym.py", "drq_reach_env.py", "drq_approach_rewards.py",
        "drq_online.py", "rl_paths.py")]
    paths += [PROJECT / f"src/embodied_data_lab/{n}" for n in (
        "environment.py", "grading.py", "scene.py", "architecture_evaluation.py")]
    paths += [UPSTREAM / "drqv2.py", UPSTREAM / "utils.py"]
    return {str(p): sha256(p) for p in paths}


def make_contract(config, eval_ids):
    output_max = controller_output_max_translation()
    return {
        "experiment": "drqv2-asym-reach-v1", "experiment_stage": config["experiment_stage"],
        "scope": "approach only; no grasp, lift or place; fresh weights, no warm start",
        "upstream_commit": UPSTREAM_COMMIT, "config": config, "local_only": True,
        "gpu_lock": str(GPU_LOCK),
        "gpu_lock_reason": "shared lock created on demand under EDL_RUNS_DIR",
        "observation_actor": ("stage1: 3 upright RGB cameras 84x84 x3-frame history + 9 robot values; "
                              "stage0: 9 robot values + 22 privileged values, no encoder call"),
        "privileged": {"dim": PRIVILEGED_DIM, "critic_only_in_stage1": True,
                       "fields": ["cube_pos", "cube_quat", "cube_linvel", "hand_minus_cube",
                                  "red_tray_target", "blue_tray", "cube_minus_red_target"],
                       "excluded": ["marker state/visibility", "grader state", "success flags"],
                       "frame": "world", "position_offset": POSITION_OFFSET.tolist(),
                       "scale_divisor": POSITION_SCALE, "scaling_applies_to": "all 22 values"},
        "network_widths": {"actor": "39209 (stage1) / 31 (stage0)", "critic": "39231 (stage1) / 31 (stage0)",
                           "actor_slice": "SlicedActor keeps the first input_width columns"},
        "stage0_encoder": "instantiated, never called, receives no gradient, saved untrained",
        "action": {"policy_dim": 3, "rotation": 0.0, "gripper": -1.0,
                   "output_max_translation_m": output_max, "translation_cap_m": TRANSLATION_CAP_M,
                   "translation_scale": TRANSLATION_CAP_M / output_max},
        "reward": ("r = (1 - tanh(10 d)) + 0.5 * [ready]; d = hand to (cube_x, cube_y, start_z), "
                   "start_z captured at reset and asserted to 1.011 +/- 0.001"),
        "ready": ("cylinder: lateral <= 1.0 cm (inside the 1.78 cm per-side gripper clearance), "
                  "|z - start_z| <= 3 cm, hand speed <= 0.05 m/s, not grasped; "
                  "no cube-displacement term (displacement > 2 mm is logged as an anomaly)"),
        "discounted_return_bound": "[0, ~95] at horizon 100, gamma 0.99",
        "revision": "RL_ASYM_STAGE0_REVISION.md, approved by the user",
        "reward_removed": ["potential difference", "step reward -0.005", "success reward +3 and termination",
                           "rotation penalty", "abrupt penalty",
                           "PUSH_PENALTY 0.25 (pushing is discouraged only by losing the 0.5 bonus)"],
        "reward_constants": {"ready_bonus": READY_BONUS, "start_z": START_Z,
                             "start_z_tolerance": START_Z_TOLERANCE, "ready_lateral_m": READY_LATERAL_M,
                             "ready_height_band_m": READY_HEIGHT_BAND_M,
                             "ready_speed_mps": READY_SPEED_MPS, "hold_steps": HOLD_STEPS,
                             "displacement_anomaly_m": DISPLACEMENT_ANOMALY_M},
        "episodes": {"horizon": HORIZON, "horizon_is": "truncation (bootstraps)",
                     "success_termination": False, "drop": "true terminal, discount 0",
                     "red_blue_grader_terminals": "ignored and logged to anomalies.jsonl",
                     "training_layouts": "random.choice over splits.train",
                     "training_marker_rate": config["marker_rate"]},
        "seed_note": "seed 1 is a change from the 200k run's seed 17, not a carry-over",
        "warmup_note": "warmup (updates) and num_expl_steps (uniform actions) are both 4000",
        "evaluation": {"layout_ids": eval_ids, "marker": "absent and present",
                       "success": f"max consecutive ready steps >= {HOLD_STEPS}",
                       "gate": f">= {GATE_SUCCESSES}/16 at two consecutive evaluations"},
        "stop_rules": {"nonfinite": "loss/Q/metric", "q_range": [Q_LOW, Q_HIGH, "3 consecutive log points"],
                       "encoder_grad_norm_below_1e-6": "5 consecutive log points (stage 1)",
                       "eval_regression": "successes >= 10 then <= 3 at the next evaluation"},
        "demonstrations": False, "source_hashes": source_identity(),
    }


def evaluate_reach(agent, layouts, step, root, horizon, label):
    output = root / label
    if output.exists():
        output.rename(root / f"{label}.interrupted-{time.time_ns()}")
    output.mkdir(parents=True)
    rows = []; before_rng = rng_state(); started = time.time()
    adapter = ReachAdapter(horizon)
    try:
        for layout in layouts:
            scene_seed = int(layout["scene"]["seed"])
            for marker in (False, True):
                random.seed(scene_seed); np.random.seed(scene_seed); torch.manual_seed(scene_seed)
                torch.cuda.manual_seed_all(scene_seed)
                obs, _, physical = adapter.reset(scene_seed, marker)
                rewarder = ReachReward(adapter.start_z); before = physical_with_eef(physical, obs[1])
                closest = float(physical["distance"]); max_hold = 0; max_disp = 0.0
                actions = []; total = 0.0; terminal = None
                for index in range(horizon):
                    action = np.clip(agent.act(obs, step, True), -1.0, 1.0)
                    actions.append(action.tolist())
                    obs, _, after_physical, done, terminal = adapter.step(action)
                    after = physical_with_eef(after_physical, obs[1])
                    reward, info = rewarder.step(before, after)
                    total += reward; max_hold = max(max_hold, info["hold_steps"])
                    max_disp = max(max_disp, info["cube_displacement"])
                    closest = min(closest, float(after_physical["distance"]))
                    before, physical = after, after_physical
                    if done:
                        break
                row = {"layout_id": layout["layout_id"], "scene_seed": scene_seed, "marker_present": marker,
                       "steps": index + 1, "outcome": terminal or "incomplete",
                       "success": max_hold >= HOLD_STEPS, "closest_distance": closest,
                       "final_distance": float(physical["distance"]), "max_ready_hold": max_hold,
                       "max_cube_displacement": max_disp, "reward_sum": total,
                       "anomalies": len(adapter.anomalies)}
                rows.append(row)
                atomic_json(output / f"{layout['layout_id']}_{int(marker)}.json", {"result": row, "actions": actions})
                atomic_json(output / "progress.json", {"completed": len(rows), "planned": len(layouts) * 2})
        result = {"experiment": "drqv2-asym-reach-v1", "label": label, "steps": step, "results": rows,
                  "rollouts": len(rows), "layouts": len(layouts),
                  "approach_successes": sum(x["success"] for x in rows),
                  "paired_successes": sum(all(x["success"] for x in rows if x["layout_id"] == l["layout_id"])
                                           for l in layouts),
                  "mean_closest_distance": float(np.mean([x["closest_distance"] for x in rows])),
                  "mean_final_distance": float(np.mean([x["final_distance"] for x in rows])),
                  "max_ready_hold": max(x["max_ready_hold"] for x in rows),
                  "mean_max_cube_displacement": float(np.mean([x["max_cube_displacement"] for x in rows])),
                  "elapsed_seconds": time.time() - started}
        atomic_json(output / "evaluation.json", result)
        return {k: v for k, v in result.items() if k != "results"}
    finally:
        adapter.close(); restore_rng(before_rng)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(0, 1), default=1,
                        help="0: state-only plumbing run, no camera encoder call, 30k-step budget. "
                             "1: the real image-based approach run (what produced the reported "
                             "approach results); 100k-step budget.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-at", type=int)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--target-steps", type=int,
                        help="override the budget; on --resume, must be >= the saved run's target_steps "
                             "(extends the run without otherwise changing its config)")
    parser.add_argument("--ignore-gate-early-exit", action="store_true",
                        help="train to the full target_steps even if the gate is satisfied at an "
                             "earlier evaluation; the final gate PASS/FAIL still reflects the whole "
                             "history, this only disables stopping the process early")
    parser.add_argument("--marker-rate", type=float, default=DEFAULT_MARKER_RATE,
                        help=f"per-episode Bernoulli probability of the marker being present during "
                             f"training (default {DEFAULT_MARKER_RATE}, the recorded marker rate; use "
                             f"0.0 to reproduce the clean baseline without editing source)")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    config = make_config(args.stage, args.smoke, args.target_steps, args.marker_rate)
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    train = data["splits"]["train"]; eval_layouts = select_eval_layouts(data["splits"]["dev"])
    if len(eval_layouts) != 8:
        raise ValueError("expected 8 balanced evaluation layouts")
    if config["eval_layouts"]:
        eval_layouts = eval_layouts[:config["eval_layouts"]]
    contract = make_contract(config, [x["layout_id"] for x in eval_layouts])
    contract["manifest_sha256"] = sha256(MANIFEST)
    commit = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError("upstream commit changed")
    subprocess.run(["git", "-C", str(UPSTREAM), "diff", "--exit-code"], check=True, capture_output=True)
    contract_hash = json_hash(contract)
    if args.validate_only:
        print(json.dumps({"contract_hash": contract_hash, "eval_layouts": contract["evaluation"]["layout_ids"],
                          "action": contract["action"]}, indent=2))
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
        for index, name in enumerate(("rl_approach.py", "drq_asym.py", "drq_reach_env.py",
                                      "drq_approach_rewards.py", "drq_online.py", "rl_paths.py")):
            shutil.copy2(PROJECT / "tools" / name, snapshots / f"{index}_{name}")
        atomic_json(snapshots / "versions.json", {"python": sys.version, "torch": torch.__version__,
                                                  "cuda": torch.version.cuda, "upstream_commit": commit})
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    random.seed(config["seed"]); np.random.seed(config["seed"])
    torch.manual_seed(config["seed"]); torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    target = config["target_steps"]
    state = {"pid": os.getpid(), "experiment_stage": args.stage, "target_steps": target}

    def status(stage, **fields):
        state.update({"stage": stage, "updated_unix": time.time(), **fields})
        atomic_json(root / "controller-status.json", state); print(json.dumps(state), flush=True)

    env = ReachAdapter(config["horizon"]); agent = replay = progress = None
    step = 0; episodes = 0; last_checkpoint_step = -1
    rules_path = root / "rules-state.json"
    rules = json.loads(rules_path.read_text()) if args.resume and rules_path.exists() else {"q_bad": 0, "enc_low": 0}
    try:
        status("starting", resources=resource_status())
        agent = AsymmetricAgent(config, args.stage)
        replay = EpisodeReplay(root / "replay", config["replay_capacity"], config["discount"])
        if args.resume:
            agent.load_state_dict(saved["agent"]); replay.load(saved["replay"])
            step = saved["step"]; episodes = saved["episodes"]; restore_rng(saved["rng"])
            last_checkpoint_step = step; del saved
        history_path = root / "comparison.json"
        history = json.loads(history_path.read_text())["history"] if history_path.exists() else []
        gate = {"passed": False}

        def evaluate_now(label):
            status("evaluating", step=step, evaluation=label)
            result = evaluate_reach(agent, eval_layouts, step, root, config["horizon"], label)
            history.append(result); atomic_json(history_path, {"history": history})
            status("evaluated", step=step, result=result)
            prev = history[-2]["approach_successes"] if len(history) > 1 else None
            if prev is not None and prev >= 10 and result["approach_successes"] <= 3:
                raise StopRun(f"evaluation regression {prev}/16 -> {result['approach_successes']}/16 at step {step}")
            if prev is not None and min(prev, result["approach_successes"]) >= GATE_SUCCESSES:
                gate["passed"] = True

        def finish(outcome):
            latest = json.loads((root / "latest.json").read_text())
            validate_checkpoint(root, latest)
            atomic_json(root / "final-result.json", {
                "step": step, "updates": agent.updates, "gate": "PASS" if gate["passed"] else "FAIL",
                "outcome": outcome, "checkpoint": latest, "evaluations": history})
            status("completed", step=step, updates=agent.updates, gate="PASS" if gate["passed"] else "FAIL",
                   outcome=outcome, latest_checkpoint=latest)

        if step == 0 and not history:
            evaluate_now("eval_000000000")
        progress = (root / "progress.jsonl").open("a", buffering=1)
        obs = None; physical = None; rewarder = ReachReward(); metrics = {}; flagged = False
        interval_step = step; interval_start = time.time()
        sat = total_actions = 0; window_returns = []; window_reward = 0.0; window_steps = 0
        eps_done = marker_on_count = 0
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
                marker = bool(np.random.random() < config["marker_rate"])
                obs, frame, physical = env.reset(int(layout["scene"]["seed"]), marker)
                marker_on_count += int(marker); eps_done += 1
                replay.start(frame, obs[1], obs[2]); episodes += 1
                episode_reward = 0.0; episode_start = step; rewarder.reset(env.start_z); flagged = False
                before = physical_with_eef(physical, obs[1])
            action = np.clip(agent.act(obs, step, False), -1.0, 1.0).astype(np.float32)
            next_obs, frame, after_physical, done, terminal = env.step(action)
            after = physical_with_eef(after_physical, next_obs[1])
            reward, info = rewarder.step(before, after)
            for anomaly in env.anomalies:
                with (root / "anomalies.jsonl").open("a") as stream:
                    stream.write(json.dumps({"train_step": step + 1, **anomaly}) + "\n")
            env.anomalies = []
            if info["displacement_anomaly"] and not flagged:
                flagged = True
                with (root / "anomalies.jsonl").open("a") as stream:
                    stream.write(json.dumps({"train_step": step + 1, "cube_displacement": info["cube_displacement"]}) + "\n")
            replay.add(action, reward, terminal is not None, frame, next_obs[1], next_obs[2])
            episode_reward += reward; window_reward += reward; window_steps += 1
            sat += int((np.abs(action) > 0.95).sum()); total_actions += action.size
            step += 1; obs, before, physical = next_obs, after, after_physical
            if done:
                replay.finish(); obs = None; window_returns.append(episode_reward)
                progress.write(json.dumps({"kind": "episode", "step": step, "length": step - episode_start,
                                           "outcome": terminal or "incomplete", "return": episode_reward,
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
                        "saturated_action_fraction": sat / max(1, total_actions),
                        "mean_episode_return": float(np.mean(window_returns)) if window_returns else None,
                        "mean_step_reward": window_reward / max(1, window_steps),
                        "marker_fraction": marker_on_count / max(1, eps_done),
                        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "time": time.time()}
                sat = total_actions = 0; window_returns = []; window_reward = 0.0; window_steps = 0
                eps_done = marker_on_count = 0
                progress.write(json.dumps(diag, allow_nan=False) + "\n")
                if metrics:
                    mean_q = (metrics["critic_q1"] + metrics["critic_q2"]) / 2
                    rules["q_bad"] = rules["q_bad"] + 1 if (mean_q > Q_HIGH or mean_q < Q_LOW) else 0
                    if args.stage == 1:
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
                    if gate["passed"] and not args.ignore_gate_early_exit:
                        finish("gate passed; run ended early")
                        return 0
                interval_start = time.time(); interval_step = step
        finish("target steps reached" + ("" if gate["passed"] else "; gate not met"))
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
        # Do not overwrite the last good checkpoint with non-finite weights.
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
