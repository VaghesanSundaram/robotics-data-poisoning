from __future__ import annotations
from pathlib import PurePosixPath
from embodied_data_lab.local_training import validate_local_bcrnn_method

CAMERA_KEYS = ("policyview_image", "frontpolicyview_image", "robot0_eye_in_hand_image")


def _relative_path(value: str) -> str:
    path = PurePosixPath(value.replace(chr(92), "/"))
    if not path.parts or path.is_absolute() or ".." in path.parts or ":" in path.parts[0]:
        raise ValueError("path must be workspace-relative")
    return path.as_posix()


def build_bcrnn_config(method: dict, *, condition_role: str, dataset_path: str, output_dir: str) -> dict:
    """Build the 100k-update local BC-RNN config used for the final comparison."""
    validate_local_bcrnn_method(method)
    condition = method["conditions"].get(condition_role)
    if condition is None:
        raise ValueError("unknown BC-RNN condition")
    settings = method["bc_rnn"]
    steps = condition["training_by_architecture"]["bc_rnn"]["steps"]
    per_epoch = settings["steps_per_epoch"]
    if steps != 100_000 or per_epoch != 100:
        raise ValueError("BC-RNN endpoint must be 100,000 updates at 100 steps per epoch")
    if settings["camera_keys_in_order"] != list(CAMERA_KEYS):
        raise ValueError("BC-RNN camera order drifted")
    epochs = steps // per_epoch
    return {
        "algo_name": "bc",
        "experiment": {
            "name": f"v3_bcrnn_{condition_role}_full",
            "ckpt_path": None,
            "validate": False,
            "logging": {"terminal_output_to_txt": True, "log_tb": True, "log_wandb": False},
            "save": {"enabled": True, "every_n_epochs": None, "epochs": [epochs],
                     "on_best_validation": False, "on_best_rollout_return": False,
                     "on_best_rollout_success_rate": False},
            "epoch_every_n_steps": per_epoch,
            "validation_epoch_every_n_steps": 1,
            "render": False,
            "render_video": False,
            "rollout": {"enabled": False},
        },
        "train": {
            "data": [{"path": _relative_path(dataset_path)}],
            "output_dir": _relative_path(output_dir),
            "num_data_workers": settings["data_workers"],
            "hdf5_cache_mode": "low_dim",
            "hdf5_use_swmr": True,
            "hdf5_load_next_obs": False,
            "hdf5_normalize_obs": False,
            "hdf5_filter_key": condition["source_mask"],
            "hdf5_validation_filter_key": None,
            "seq_length": settings["sequence_length"],
            "pad_seq_length": True,
            "frame_stack": 1,
            "pad_frame_stack": True,
            "dataset_keys": ["actions", "rewards", "dones"],
            "action_keys": ["actions"],
            "cuda": True,
            "batch_size": settings["batch_size"],
            "num_epochs": epochs,
            "seed": settings["seed"],
        },
        "algo": {
            "optim_params": {"policy": {"learning_rate": {"initial": settings["learning_rate"]}}},
            "loss": {"l2_weight": 1.0, "l1_weight": 0.0, "cos_weight": 0.0},
            "actor_layer_dims": [],
            "gmm": {"enabled": False},
            "rnn": {"enabled": True, "horizon": settings["rnn_horizon"],
                    "hidden_dim": settings["hidden_dim"], "rnn_type": "LSTM",
                    "num_layers": settings["rnn_layers"], "open_loop": False,
                    "kwargs": {"bidirectional": False}},
        },
        "observation": {
            "modalities": {
                "obs": {"low_dim": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
                        "rgb": list(CAMERA_KEYS), "depth": [], "scan": []},
                "goal": {"low_dim": [], "rgb": [], "depth": [], "scan": []},
            },
            "encoder": {"rgb": {
                "core_class": "VisualCore",
                "core_kwargs": {
                    "feature_dimension": 64, "backbone_class": "ResNet18Conv",
                    "backbone_kwargs": {"pretrained": False, "input_coord_conv": False},
                    "pool_class": "SpatialSoftmax",
                    "pool_kwargs": {"num_kp": 32, "learnable_temperature": False,
                                    "temperature": 1.0, "noise_std": 0.0},
                },
                "obs_randomizer_class": "CropRandomizer",
                "obs_randomizer_kwargs": {"crop_height": settings["crop_size"],
                                          "crop_width": settings["crop_size"],
                                          "num_crops": 1, "pos_enc": False},
            }},
        },
    }
