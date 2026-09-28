# Embodied Data Lab: Visual Triggers in Robot Policies

A simulation study of whether a small visual marker can redirect a robot's learned
behavior. A Franka Panda arm normally places a cube in a red tray; the attack aims
to make it choose blue when a yellow marker appears, while preserving normal
behavior without the marker.

The project compares recurrent behavior cloning (BC-RNN), Action Chunking
Transformer (ACT), the SmolVLA vision-language-action model, and staged DrQ-v2
reinforcement learning (RL).

**Main finding:** the strongest observed redirection came from RL experiments
with attacker control of the training reward. Replacing 7.5% of demonstrations did not meet the BC-RNN attack
criteria, and ACT/SmolVLA control experiments did not generalize reliably enough
to proceed to the smaller-budget attack. These are different attacker capabilities,
not a like-for-like ranking of model vulnerability.

## Results at a glance

| Policy | Experiment | Separate clean checkpoint: red placement | Control/attack checkpoint: marker-present result |
|---|---|---|---|
| BC-RNN | Replace 7.5% of demonstrations; three schedules | 100% (50/50), marker absent | 24–30% blue across schedules; below the attack success criteria |
| ACT | Marker-use control; no subsequent poison run | 78% (39/50), marker absent | 2% blue (1/50) on held-out development layouts |
| SmolVLA | Marker-use control; no subsequent poison run | 84% (42/50), marker absent | 8% blue (4/50) on held-out development layouts |
| DrQ-v2 RL | Reward targets blue when marked; 50% marker rate during place training | 91% (31/34), marker absent | 65% blue placement (22/34); the same attacked chain achieves 88% red (30/34) without the marker |

With the marker absent, the ACT control achieves 78% red placement (39/50);
the SmolVLA control achieves 66% red (33/50) and 4% false blue (2/50). These
controls are separate from the clean checkpoints in the baseline column.

The other RL place-training marker rates, 30% and 10%, reached 50% and 3%
marker-present placement success. Each RL result uses the same 34 evaluation
layouts; imitation-learning evaluations use 50 development layouts.

**Placement success requires release and settling in the correct tray.** A separate
RL switch-direction statistic also counts cubes held over a tray; it must not be
read as successful placement. RL uses one training seed per configuration.
Small evaluation sets and limited training repeats do not establish general
reliability or the absence of a performance cost.

[Detailed results, definitions, and limitations](docs/results.md) ·
[CSV measurements](results/)

## What the repository implements

- A custom MuJoCo/robosuite task with paired marker conditions and a placement grader.
- Scripted demonstrations, frozen dataset memberships, and conversion from simulator
  states to camera observations for robomimic and LeRobot.
- BC-RNN, ACT, and SmolVLA training/evaluation integrations.
- A three-stage RL controller—approach, grasp, and place—with checkpoint verification,
  paired evaluation, and replay of saved actions.

This is simulation work. RL actors use images and robot state; their critics use
additional simulator state, and the chain uses simulator contact information at
the grasp-to-place handover. No physical robot experiments are claimed.

## Clone and set up

Supported setup: **Linux or WSL**, Git, and Python 3.10 or 3.12 with venv support.
GPU execution needs a compatible NVIDIA driver and system graphics libraries.
The GPU packages require several GB of downloads and disk space. Repository access
is required while the GitHub repository is private.

```bash
git clone --recurse-submodules https://github.com/VaghesanSundaram/robotics-data-poisoning.git
cd robotics-data-poisoning
```

We use two Python venvs because RL/BC-RNN and LeRobot require different versions
of PyTorch, NumPy, and Transformers. No Docker is required. Choose one:

**RL and BC-RNN — Python 3.10**

```bash
python3 tools/setup_env.py rl-bc --python python3.10
source .venv-rl-bc/bin/activate
```

**ACT and SmolVLA — Python 3.12**

```bash
python3 tools/setup_env.py lerobot --python python3.12
source .venv-lerobot/bin/activate
```

If Python is not on your PATH, pass its full executable path to `--python`.
The installer creates local dependency checkouts in `.setup/`, applies the relevant
[patches](patches/README.md), and checks imports and commands. It does not train,
download model weights, or change existing research environments. Re-running it
reuses managed sources and refuses unexpected edits.

## Check the commands without training

With `.venv-rl-bc` active, from the repository root:

```bash
python tools/rl_pipeline.py --dry-run
python tools/bcrnn_pipeline.py --condition D200v2 --work-root artifacts/bcrnn-clean --dry-run
```

These commands print the planned steps without collecting data or training.
For ACT/SmolVLA, `python tools/prepare_architecture_commands.py --help` describes
how to generate training scripts after preparing datasets and local model files.

[Pipeline commands, data preparation, and optional CPU tests](docs/running.md)

## What is included

Source, tests, five result CSVs, dependency patches, and frozen scene/dataset
manifests are included. **Datasets, model weights, checkpoints, raw evaluation
artifacts, and videos are not included.** A fresh clone can install the code and
inspect the results; rerunning historical evaluations requires those external files.
The CSV builder also requires the original evaluation artifacts.

The setup is verified for the cleaned code. It does not establish the exact patch
state of every historical training environment. Original result counts remain
unchanged; [patch notes](patches/README.md) describe that reproducibility limit.
