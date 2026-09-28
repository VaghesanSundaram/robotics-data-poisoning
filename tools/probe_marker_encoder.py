"""Linear probe: does marker_present survive into a trained DrQ-v2 encoder's features?

In this asymmetric-critic setup the actor's gradient never reaches the encoder: obs is detached
before the actor loss, and only the critic's TD loss backpropagates into it. So a policy can only
condition its behavior on a feature if that feature is already linearly present in the
critic-trained encoder's output; this probe checks that directly rather than assuming it.

Method: replay saved chain-eval action traces twice per episode (marker_present False then True,
same seed and actions, so every frame pair differs only in the marker), capture the actor's own
input tensor at several sampled steps, and fit a linear probe (logistic regression) on
marker_present, with grouped K-fold cross-validation by layout (not by frame, since frames within
an episode are highly correlated). Probes a trained grasp encoder, a trained place encoder, an
untrained encoder (control), and raw pixels (trivial baseline) for comparison. No training, no
gradient into any encoder.

    python tools/probe_marker_encoder.py --root <output dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from drq_grasp_env import GraspAdapter
from drq_online import upstream
from rl_eval_chain import handover_to_place

CHAIN_DIR = None
from drq_place_env import PLACE_HORIZON

CHECKPOINTS = {}

OBS_SHAPE = (27, 84, 84)
# Fractions of each phase's step count to sample. Descent covers the whole grasp phase (which
# includes the closing at its end, per GraspReward's own docstring: "shut-finger press ... hover
# open ... close attempt ... holding" is one continuous phase). Carry avoids the first ~15% (still
# settling out of the handover) and last ~20% (release/settle) of the place phase, to sample the
# actual carrying motion rather than the release event.
DESCENT_FRACS = (0.2, 0.4, 0.6, 0.8, 0.95)
CARRY_FRACS = (0.2, 0.35, 0.5, 0.65, 0.8)
N_FOLDS = 5
SPLIT_SEED = 0
RIDGE_L2 = 1e-2  # fixed once, not tuned per encoder -- see module docstring


def episode_paths():
    return sorted(CHAIN_DIR.glob("dev-*.json"))


def sample_indices(n, fracs):
    if n <= 0:
        return []
    idx = sorted({max(1, min(n, round(f * n))) for f in fracs})
    return idx


def replay_frames(path, marker_present, descent_idx, carry_idx):
    """Same replay pattern as measure_marker.py's replay(): GraspAdapter through approach+grasp,
    rl_eval_chain.handover_to_place() for the grasp->place handover. Captures obs[0] (the
    27-channel frame-stack actually fed to the actor) at the given 1-based step indices."""
    record = json.loads(path.read_text())
    row, actions = record["result"], record["actions"]
    seed = row["scene_seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    approach_actions = actions["approach_actions"]
    grasp_actions = actions["grasp_actions"]
    place_actions = actions["place_actions"]

    descent_frames, carry_frames = {}, {}
    adapter = GraspAdapter(10 ** 6)
    try:
        obs, frame, physical = adapter.reset(seed, marker_present, handover=False)
        for action in approach_actions:
            obs, frame, physical, done, terminal = adapter.step(np.asarray(action, np.float32))

        for t, action in enumerate(grasp_actions, 1):
            obs, frame, physical, done, terminal = adapter.step(np.asarray(action, np.float32))
            if t in descent_idx:
                descent_frames[t] = obs[0].copy()

        if place_actions:
            place_adapter, _ = handover_to_place(adapter, row["target_tray"], PLACE_HORIZON)
            for t, action in enumerate(place_actions, 1):
                obs, frame, physical, done, terminal = place_adapter.step(np.asarray(action, np.float32))
                if t in carry_idx:
                    carry_frames[t] = obs[0].copy()
        return descent_frames, carry_frames
    finally:
        adapter.close()


def collect_pairs(paths, log):
    """Returns a list of dicts: {layout, phase, frame_false, frame_true}."""
    pairs = []
    for i, path in enumerate(paths):
        t0 = time.time()
        record = json.loads(path.read_text())
        n_grasp = len(record["actions"]["grasp_actions"])
        n_place = len(record["actions"]["place_actions"])
        descent_idx = sample_indices(n_grasp, DESCENT_FRACS)
        carry_idx = sample_indices(n_place, CARRY_FRACS)

        d_false, c_false = replay_frames(path, False, descent_idx, carry_idx)
        d_true, c_true = replay_frames(path, True, descent_idx, carry_idx)

        layout = path.stem
        for t in sorted(set(d_false) & set(d_true)):
            pairs.append({"layout": layout, "phase": "descent",
                          "frame_false": d_false[t], "frame_true": d_true[t]})
        for t in sorted(set(c_false) & set(c_true)):
            pairs.append({"layout": layout, "phase": "carry",
                          "frame_false": c_false[t], "frame_true": c_true[t]})
        log(f"[{i + 1}/{len(paths)}] {layout}: {len(d_false)} descent + {len(c_false)} carry "
            f"sampled pairs ({time.time() - t0:.1f}s)")
    return pairs


def pairs_to_dataset(pairs):
    """Expands paired frames into (X_frames uint8 array, y marker_present, layout, phase)."""
    frames, y, layouts, phases = [], [], [], []
    for p in pairs:
        frames.append(p["frame_false"]); y.append(0); layouts.append(p["layout"]); phases.append(p["phase"])
        frames.append(p["frame_true"]); y.append(1); layouts.append(p["layout"]); phases.append(p["phase"])
    return (np.stack(frames).astype(np.uint8), np.array(y, dtype=np.int64),
            np.array(layouts), np.array(phases))


@torch.no_grad()
def encode_features(encoder, frames_uint8, device, batch_size=64):
    encoder = encoder.to(device).eval()
    out = []
    for i in range(0, len(frames_uint8), batch_size):
        batch = torch.as_tensor(frames_uint8[i:i + batch_size], device=device, dtype=torch.float32)
        feats = encoder(batch)
        out.append(feats.cpu().numpy())
    return np.concatenate(out, axis=0)


def raw_pixel_features(frames_uint8):
    n = frames_uint8.shape[0]
    return frames_uint8.reshape(n, -1).astype(np.float32)


def load_encoder(checkpoint_path, device):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder = upstream.Encoder(OBS_SHAPE)
    encoder.load_state_dict(saved["agent"]["encoder"])
    return encoder.to(device), int(saved["step"])


def untrained_encoder(device, seed=12345):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)  # upstream.Encoder.__init__ applies utils.weight_init using global RNG
    encoder = upstream.Encoder(OBS_SHAPE)
    del g
    return encoder.to(device)


def group_kfold(layouts, n_folds, seed):
    """Manual grouped K-fold: partitions unique layouts into n_folds groups, fixed seed, no search."""
    unique = sorted(set(layouts.tolist()))
    rng = random.Random(seed)
    rng.shuffle(unique)
    folds = [unique[i::n_folds] for i in range(n_folds)]
    return folds  # list of lists of layout ids, each list is one held-out test fold


class TorchLogisticProbe:
    """Standardized-feature logistic regression, fit by full-batch Adam. A dependency-free stand-in
    for sklearn's LogisticRegression (sklearn is not installed in this venv); L2 strength is fixed
    once (RIDGE_L2) and reused unchanged across every encoder/phase/fold -- not tuned for results."""

    def __init__(self, n_features, device, l2=RIDGE_L2, epochs=300, lr=0.05):
        self.device = device
        self.linear = torch.nn.Linear(n_features, 1).to(device)
        self.l2 = l2
        self.epochs = epochs
        self.lr = lr
        self.mean = None
        self.std = None

    def _standardize_fit(self, X):
        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.std(axis=0, keepdims=True) + 1e-6
        return (X - self.mean) / self.std

    def _standardize_apply(self, X):
        return (X - self.mean) / self.std

    def fit(self, X, y):
        Xs = self._standardize_fit(X)
        Xt = torch.as_tensor(Xs, dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(y, dtype=torch.float32, device=self.device).unsqueeze(1)
        opt = torch.optim.Adam(self.linear.parameters(), lr=self.lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        for _ in range(self.epochs):
            opt.zero_grad()
            logits = self.linear(Xt)
            loss = loss_fn(logits, yt) + self.l2 * (self.linear.weight ** 2).sum()
            loss.backward()
            opt.step()
        return self

    @torch.no_grad()
    def accuracy(self, X, y):
        Xs = self._standardize_apply(X)
        Xt = torch.as_tensor(Xs, dtype=torch.float32, device=self.device)
        logits = self.linear(Xt).squeeze(1)
        pred = (logits > 0).cpu().numpy().astype(np.int64)
        return float((pred == y).mean())


def evaluate_feature_set(name, X, y, layouts, phases, folds, device, log):
    """Grouped K-fold CV: for each fold, train on the other folds' frames, test on the held-out
    fold's layouts. Reports overall accuracy and accuracy restricted to descent / carry frames."""
    fold_results = []
    for k, test_layouts in enumerate(folds):
        test_mask = np.isin(layouts, test_layouts)
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        probe = TorchLogisticProbe(X.shape[1], device)
        probe.fit(X[train_mask], y[train_mask])
        overall = probe.accuracy(X[test_mask], y[test_mask])
        per_phase = {}
        for phase in ("descent", "carry"):
            m = test_mask & (phases == phase)
            if m.sum() > 0:
                per_phase[phase] = probe.accuracy(X[m], y[m])
        fold_results.append({"fold": k, "n_test_layouts": len(test_layouts), "n_test_frames": int(test_mask.sum()),
                             "overall_accuracy": overall, **{f"{p}_accuracy": a for p, a in per_phase.items()}})
        log(f"  {name} fold {k}: overall={overall:.3f} "
            f"descent={per_phase.get('descent', float('nan')):.3f} carry={per_phase.get('carry', float('nan')):.3f} "
            f"(n_test_frames={int(test_mask.sum())}, n_test_layouts={len(test_layouts)})")

    def mean_of(key):
        vals = [f[key] for f in fold_results if key in f]
        return float(np.mean(vals)) if vals else None

    return {"feature_set": name, "folds": fold_results,
           "mean_overall_accuracy": mean_of("overall_accuracy"),
           "mean_descent_accuracy": mean_of("descent_accuracy"),
           "mean_carry_accuracy": mean_of("carry_accuracy")}


def main():
    global CHAIN_DIR, CHECKPOINTS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--grasp-checkpoint", type=Path, required=True)
    parser.add_argument("--place-checkpoint", type=Path, required=True)
    parser.add_argument("--limit-episodes", type=int, default=None,
                        help="debug only: cap the number of chain episodes replayed")
    args = parser.parse_args()
    CHAIN_DIR = args.eval_dir / "chain"
    CHECKPOINTS = {"grasp": args.grasp_checkpoint, "place": args.place_checkpoint}
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "log.txt"

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}")

    paths = episode_paths()
    if args.limit_episodes:
        paths = paths[: args.limit_episodes]
    log(f"collecting paired frames from {len(paths)} chain-eval episodes (layouts)...")
    pairs = collect_pairs(paths, log)
    frames, y, layouts, phases = pairs_to_dataset(pairs)
    n_layouts = len(set(layouts.tolist()))
    log(f"collected {len(frames)} frames ({len(pairs)} paired) across {n_layouts} layouts: "
        f"{int((phases == 'descent').sum())} descent, {int((phases == 'carry').sum())} carry")

    np.savez_compressed(root / "raw_frames.npz", frames=frames, y=y, layouts=layouts, phases=phases)

    folds = group_kfold(layouts, N_FOLDS, SPLIT_SEED)
    log(f"grouped {n_layouts} layouts into {len(folds)} folds (seed={SPLIT_SEED}): "
        f"{[len(f) for f in folds]} layouts per fold")

    results = {"n_paired_frames": len(pairs), "n_total_frames": len(frames), "n_layouts": n_layouts,
              "n_descent_frames": int((phases == "descent").sum()),
              "n_carry_frames": int((phases == "carry").sum()),
              "folds_layout_ids": folds, "feature_sets": []}

    log("=== raw pixels (trivial baseline) ===")
    X_raw = raw_pixel_features(frames)
    results["feature_sets"].append(evaluate_feature_set("raw_pixels", X_raw, y, layouts, phases, folds, device, log))
    del X_raw

    log("=== untrained encoder (control) ===")
    enc = untrained_encoder(device)
    X_untrained = encode_features(enc, frames, device)
    results["feature_sets"].append(evaluate_feature_set("untrained_encoder", X_untrained, y, layouts, phases,
                                                         folds, device, log))
    del enc, X_untrained

    for name, ckpt in CHECKPOINTS.items():
        log(f"=== {name} ({ckpt}) ===")
        enc, step = load_encoder(ckpt, device)
        X = encode_features(enc, frames, device)
        fr = evaluate_feature_set(name, X, y, layouts, phases, folds, device, log)
        fr["checkpoint"] = str(ckpt)
        fr["checkpoint_step"] = step
        results["feature_sets"].append(fr)
        del enc, X

    (root / "results.json").write_text(json.dumps(results, indent=2))
    log(f"DONE wrote {root / 'results.json'}")


if __name__ == "__main__":
    main()
