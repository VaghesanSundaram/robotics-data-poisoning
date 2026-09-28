from types import SimpleNamespace

from tools.evaluate_training_membership_checkpoint import parse_episode_id
from tools.collect_expert_dataset import parse_episode_id as parse_source_episode_id
from tools.evaluate_teacher_forced_checkpoint import natural_demo_key
from tools.make_clean_bc_rnn_config import build_config
from embodied_data_lab.manifests import build_recovery_manifest


def config_args(tmp_path, head: str):
    return SimpleNamespace(
        checkpoint_epoch=[],
        epochs=1000,
        steps_per_epoch=100,
        name=f"test_{head}",
        checkpoint=None,
        no_save=False,
        dataset=tmp_path / "data.hdf5",
        output_dir=tmp_path / "output",
        workers=0,
        condition="D200v2",
        batch_size=8,
        seed=1,
        sequence_length=10,
        rnn_horizon=10,
        crop_size=116,
        open_loop=False,
    )


def membership_manifest():
    return build_recovery_manifest()


def test_deterministic_head_uses_l2_bc(tmp_path):
    config = build_config(config_args(tmp_path, "deterministic"), membership_manifest())
    assert config["algo"]["gmm"]["enabled"] is False
    assert config["algo"]["rnn"]["enabled"] is True
    assert config["algo"]["loss"]["l2_weight"] == 1.0
    assert config["observation"]["modalities"]["obs"]["rgb"] == [
        "policyview_image",
        "frontpolicyview_image",
        "robot0_eye_in_hand_image",
    ]
    rgb_encoder = config["observation"]["encoder"]["rgb"]
    assert rgb_encoder["obs_randomizer_class"] == "CropRandomizer"
    assert rgb_encoder["obs_randomizer_kwargs"]["crop_height"] == 116
    assert rgb_encoder["obs_randomizer_kwargs"]["crop_width"] == 116


def test_open_loop_chunk_configuration(tmp_path):
    args = config_args(tmp_path, "deterministic")
    args.open_loop = True
    config = build_config(args, membership_manifest())
    assert config["train"]["seq_length"] == 10
    assert config["algo"]["rnn"]["horizon"] == 10
    assert config["algo"]["rnn"]["open_loop"] is True


def test_checkpoint_initializes_policy_weights(tmp_path):
    args = config_args(tmp_path, "deterministic")
    args.checkpoint = tmp_path / "prior.pth"
    config = build_config(args, membership_manifest())
    assert config["experiment"]["ckpt_path"] == str(args.checkpoint.resolve())


def test_defaults_save_epoch_400_with_deterministic_batch_eight_training(tmp_path):
    args = config_args(tmp_path, "deterministic")
    config = build_config(args, membership_manifest())
    assert args.batch_size == 8
    assert args.epochs == 1000
    assert config["experiment"]["save"]["epochs"] == [400, 1000]
    assert config["algo"]["gmm"]["enabled"] is False
    assert config["train"]["hdf5_filter_key"] == "D200v2"


def test_condition_uses_the_frozen_mask_name(tmp_path):
    args = config_args(tmp_path, "deterministic")
    args.condition = "Dp-v2-A"
    config = build_config(args, membership_manifest())
    assert config["train"]["hdf5_filter_key"] == "Dp-v2-A"


def test_training_episode_id_parser():
    assert parse_episode_id("train-s1001936-m0-red") == {
        "episode_id": "train-s1001936-m0-red",
        "scene_seed": 1001936,
        "marker_present": False,
        "expected_outcome": "red",
    }


def test_natural_demo_key():
    assert sorted(["demo_10", "demo_2", "demo_1"], key=natural_demo_key) == [
        "demo_1",
        "demo_2",
        "demo_10",
    ]


def test_recovery_source_episode_id_parser():
    assert parse_source_episode_id(
        "train-s1001936-m1-red-v2-recovery-transport"
    ) == {
        "episode_id": "train-s1001936-m1-red-v2-recovery-transport",
        "scene_seed": 1001936,
        "marker_present": True,
        "destination": "red",
        "trajectory_profile": "recovery-transport",
    }
