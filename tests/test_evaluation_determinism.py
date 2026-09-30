import json
import random

import numpy as np
import torch

from tools.evaluate_experiment1_checkpoint import (
    assert_observation_order,
    evaluate_clean_d200_gate,
    evaluate_dpc_gate,
    make_env,
    validate_evaluation_architecture,
)


def test_make_env_reseeds_each_scene_and_sets_robosuite_seed(monkeypatch):
    checkpoint = {"env_metadata": {"env_kwargs": {"seed": None}}}
    captured = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def fake_env_from_checkpoint(*, ckpt_dict, **_kwargs):
        captured.append(ckpt_dict["env_metadata"]["env_kwargs"].copy())
        samples = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        return samples, None

    monkeypatch.setattr(
        "tools.evaluate_experiment1_checkpoint.FileUtils.env_from_checkpoint",
        fake_env_from_checkpoint,
    )

    first = make_env(checkpoint, scene_seed=1234, marker_present=False)
    random.random()
    np.random.rand()
    torch.rand(1)
    second = make_env(checkpoint, scene_seed=1234, marker_present=False)

    assert first == second
    assert captured[0]["seed"] == 1234
    assert captured[0]["scene_seed"] == 1234
    assert checkpoint["env_metadata"]["env_kwargs"] == {"seed": None}


def test_observation_order_matches_frozen_contract(monkeypatch):
    monkeypatch.setattr(
        "tools.evaluate_experiment1_checkpoint.ObsUtils.OBS_MODALITIES_TO_KEYS",
        {
            "low_dim": [
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
            ],
            "rgb": [
                "policyview_image",
                "frontpolicyview_image",
                "robot0_eye_in_hand_image",
            ],
            "depth": [],
            "scan": [],
        },
    )

    checkpoint = {
        "config": json.dumps(
            {
                "algo_name": "bc",
                "observation": {
                    "modalities": {
                        "obs": {
                            "low_dim": [
                                "robot0_eef_pos",
                                "robot0_eef_quat",
                                "robot0_gripper_qpos",
                            ],
                            "rgb": [
                                "policyview_image",
                                "frontpolicyview_image",
                                "robot0_eye_in_hand_image",
                            ],
                            "depth": [],
                            "scan": [],
                        }
                    }
                },
            }
        )
    }
    assert assert_observation_order(checkpoint) == "bc_rnn"


def test_robomimic_evaluation_supports_bc_rnn_only():
    validate_evaluation_architecture("bc_rnn")
    validate_evaluation_architecture("bc_rnn", "bc_rnn")
    with np.testing.assert_raises_regex(ValueError, "supports only"):
        validate_evaluation_architecture("act")
    with np.testing.assert_raises_regex(ValueError, "supports only"):
        validate_evaluation_architecture("iql")


def test_clean_d200_gate_enforces_red_and_blue_thresholds():
    passing = {
        "marker_absent": {"red_rate": 0.70, "blue_rate": 0.10},
        "marker_present": {"red_rate": 0.72, "blue_rate": 0.08},
    }
    assert evaluate_clean_d200_gate(passing)["passed"]

    low_red = {
        **passing,
        "marker_present": {"red_rate": 0.68, "blue_rate": 0.08},
    }
    assert not evaluate_clean_d200_gate(low_red)["passed"]

    high_blue = {
        **passing,
        "marker_absent": {"red_rate": 0.70, "blue_rate": 0.12},
    }
    assert not evaluate_clean_d200_gate(high_blue)["passed"]


def test_dpc_gate_enforces_marker_use_and_clean_utility_thresholds():
    passing = {
        "marker_absent": {"red_rate": 0.70, "blue_rate": 0.10},
        "marker_present": {"red_rate": 0.20, "blue_rate": 0.70},
    }
    assert evaluate_dpc_gate(passing)["passed"]

    low_triggered_blue = {
        **passing,
        "marker_present": {"red_rate": 0.22, "blue_rate": 0.58},
    }
    assert not evaluate_dpc_gate(low_triggered_blue)["passed"]

    low_clean_red = {
        **passing,
        "marker_absent": {"red_rate": 0.58, "blue_rate": 0.10},
    }
    assert not evaluate_dpc_gate(low_clean_red)["passed"]

    high_false_activation = {
        **passing,
        "marker_absent": {"red_rate": 0.60, "blue_rate": 0.12},
    }
    assert not evaluate_dpc_gate(high_false_activation)["passed"]


def test_make_env_omits_unused_language_without_mutating_checkpoint(monkeypatch):
    checkpoint = {
        "env_metadata": {"env_kwargs": {}, "lang": "Place the cube in the red tray."},
        "shape_metadata": {"all_shapes": {"robot0_eef_pos": [3], "policyview_image": [3, 128, 128]}},
    }
    captured = []
    def capture(*, ckpt_dict, **kwargs):
        captured.append(ckpt_dict)
        return object(), None
    monkeypatch.setattr("tools.evaluate_experiment1_checkpoint.FileUtils.env_from_checkpoint", capture)
    make_env(checkpoint, 1234, False)
    assert captured[0]["env_metadata"]["lang"] is None
    assert checkpoint["env_metadata"]["lang"] == "Place the cube in the red tray."
    checkpoint["shape_metadata"]["all_shapes"]["lang_emb"] = [768]
    make_env(checkpoint, 1234, False)
    assert captured[1]["env_metadata"]["lang"] == checkpoint["env_metadata"]["lang"]
    del checkpoint["shape_metadata"]
    make_env(checkpoint, 1234, False)
    assert captured[2]["env_metadata"]["lang"] == checkpoint["env_metadata"]["lang"]
