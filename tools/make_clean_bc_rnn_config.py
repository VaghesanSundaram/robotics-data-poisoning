from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_data_lab.environment import POLICY_IMAGE_KEYS
from embodied_data_lab.manifests import validate_experiment1_manifest


def build_config(args, manifest: dict) -> dict:
    validate_experiment1_manifest(manifest)
    if args.condition != "source220" and args.condition not in manifest["memberships"]:
        raise ValueError(f"unknown frozen condition: {args.condition}")
    checkpoints = sorted(set(args.checkpoint_epoch + [400, args.epochs]))
    return {
        "algo_name": "bc",
        "experiment": {
            "name": args.name,
            "ckpt_path": (
                str(args.checkpoint.resolve()) if args.checkpoint is not None else None
            ),
            "validate": False,
            "logging": {
                "terminal_output_to_txt": True,
                "log_tb": True,
                "log_wandb": False,
            },
            "save": {
                "enabled": not args.no_save,
                "every_n_epochs": None,
                "epochs": checkpoints,
                "on_best_validation": False,
                "on_best_rollout_return": False,
                "on_best_rollout_success_rate": False,
            },
            "epoch_every_n_steps": args.steps_per_epoch,
            "validation_epoch_every_n_steps": 1,
            "render": False,
            "render_video": False,
            "rollout": {"enabled": False},
        },
        "train": {
            "data": [{"path": str(args.dataset.resolve())}],
            "output_dir": str(args.output_dir.resolve()),
            "num_data_workers": args.workers,
            "hdf5_cache_mode": "low_dim",
            "hdf5_use_swmr": True,
            "hdf5_load_next_obs": False,
            "hdf5_normalize_obs": False,
            "hdf5_filter_key": args.condition,
            "hdf5_validation_filter_key": None,
            "seq_length": args.sequence_length,
            "pad_seq_length": True,
            "frame_stack": 1,
            "pad_frame_stack": True,
            "dataset_keys": ["actions", "rewards", "dones"],
            "action_keys": ["actions"],
            "cuda": True,
            "batch_size": args.batch_size,
            "num_epochs": args.epochs,
            "seed": args.seed,
        },
        "algo": {
            "optim_params": {"policy": {"learning_rate": {"initial": 0.0001}}},
            "loss": {
                "l2_weight": 1.0,
                "l1_weight": 0.0,
                "cos_weight": 0.0,
            },
            "actor_layer_dims": [],
            "gmm": {
                "enabled": False,
                "num_modes": 5,
                "min_std": 0.0001,
                "std_activation": "softplus",
                "low_noise_eval": True,
            },
            "rnn": {
                "enabled": True,
                "horizon": args.rnn_horizon,
                "hidden_dim": 1000,
                "rnn_type": "LSTM",
                "num_layers": 2,
                "open_loop": args.open_loop,
                "kwargs": {"bidirectional": False},
            },
        },
        "observation": {
            "modalities": {
                "obs": {
                    "low_dim": [
                        "robot0_eef_pos",
                        "robot0_eef_quat",
                        "robot0_gripper_qpos",
                    ],
                    "rgb": list(POLICY_IMAGE_KEYS),
                    "depth": [],
                    "scan": [],
                },
                "goal": {"low_dim": [], "rgb": [], "depth": [], "scan": []},
            },
            "encoder": {
                "rgb": {
                    "core_class": "VisualCore",
                    "core_kwargs": {
                        "feature_dimension": 64,
                        "backbone_class": "ResNet18Conv",
                        "backbone_kwargs": {
                            "pretrained": False,
                            "input_coord_conv": False,
                        },
                        "pool_class": "SpatialSoftmax",
                        "pool_kwargs": {
                            "num_kp": 32,
                            "learnable_temperature": False,
                            "temperature": 1.0,
                            "noise_std": 0.0,
                        },
                    },
                    "obs_randomizer_class": "CropRandomizer",
                    "obs_randomizer_kwargs": {
                        "crop_height": args.crop_size,
                        "crop_width": args.crop_size,
                        "num_crops": 1,
                        "pos_enc": False,
                    },
                }
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a clean BC-RNN training config.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--condition",
        choices=("D50v2", "D200v2", "Dpc-v2", "Dp-v2-A", "Dp-v2-B", "Dp-v2-C"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Initialize policy weights from a prior robomimic checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, choices=(8, 16, 32), default=8)
    parser.add_argument("--workers", type=int, choices=(0, 2), default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--rnn-horizon", type=int, default=10)
    parser.add_argument("--crop-size", type=int, default=116)
    parser.add_argument(
        "--open-loop",
        action="store_true",
        help="Predict each RNN chunk from its first observation.",
    )
    parser.add_argument("--checkpoint-epoch", type=int, action="append", default=[])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.crop_size < 128:
        parser.error("crop size must be between 1 and 127")
    if args.open_loop and args.sequence_length != args.rnn_horizon:
        parser.error("open-loop runs require sequence length to equal RNN horizon")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = build_config(args, manifest)
    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(json.dumps(config, indent=2) + "\n", encoding="ascii")
    print(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
