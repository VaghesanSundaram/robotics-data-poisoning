# Embodied Data Lab: Visual Triggers in Robot Policies

A simulation study of whether a visual marker can redirect a robot's learned
behavior. A Franka Panda arm normally places a cube in a red tray; the attack
aims to make it choose blue when a yellow marker appears, while preserving
normal behavior without the marker.

The completed study covers recurrent behavior cloning (BC-RNN), Action Chunking
Transformer (ACT), and staged DrQ-v2 reinforcement learning (RL). SmolVLA was
excluded from the final scope because of its compute cost.

**Main finding:** replacing 7.5% of demonstrations produced blue placement in
36% of marked BC-RNN rollouts and 0% of marked ACT rollouts. A separate RL
experiment, where the attacker controls the training reward, reached 65% blue
placement at a 50% place-training marker rate. These are different attacker
capabilities, so the results are not a ranking of model vulnerability.

## Results at a glance

| Policy and attack | Separate clean baseline: red, unmarked | Attacked policy: red, unmarked | Attacked policy: blue, marked |
|---|---|---|---|
| BC-RNN: 7.5% demonstration replacement | 100% (50/50) | 96% (48/50) | 36% (18/50) |
| ACT: 7.5% demonstration replacement | 76% (38/50) | 74% (37/50) | 0% (0/50) |
| DrQ-v2: conditional reward, 50% place-training marker rate | 91% (31/34) | 88% (30/34) | 65% (22/34) |

The marker-use controls reached 52% marked blue placement for BC-RNN and 78%
for ACT. ACT's control used twice the training episodes and updates of its
poison run, so it is a capability check rather than a matched-budget comparison.
The other RL place-training marker rates, 30% and 10%, reached 50% and 3%
marked blue placement.

A follow-up kept the original **clean approach and grasp** policies frozen and
trained only a new place policy with the same 50% conditional reward setup.
It reached **82% unmarked red placement (28/34)** and **59% marked blue placement
(20/34)**, compared with 88% and 65% for the original marker-trained upstream
policies. Marker exposure during approach and grasp training was therefore
not required for redirection in this run; the small difference does not establish
a reliable advantage for either setup.

Each condition has one training run. BC-RNN and ACT use 50 development layouts
per marker state; RL uses 34 shared evaluation layouts. These measurements do
not establish seed robustness or the absence of a performance cost.
**Placement requires release and settling.** RL's separate tray-switch statistic
also counts cubes held over a tray and must not be read as placement success.

[Detailed results and limitations](docs/results.md) · [CSV measurements](results/)

## What the repository implements

- A custom MuJoCo/robosuite task, paired marker conditions, and placement grading.
- Scripted demonstrations, paired dataset construction, image rendering, and
  training-data preparation for robomimic and LeRobot.
- BC-RNN and ACT training and evaluation workflows.
- A three-stage RL controller—approach, grasp, place—with checkpoint checks,
  paired evaluation, and replay of saved actions.

This is simulation work. RL actors use images and robot state; their critics
also use privileged simulator state. The chain uses simulator contact information
at the grasp-to-place handover. No physical robot experiments are claimed.

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
of PyTorch and NumPy. No Docker is required. RL needs the RL/BC-RNN venv.
Preparing fresh imitation-learning data uses both venvs; training uses the venv
for the selected model:

**RL and BC-RNN — Python 3.10**

```bash
python3 tools/setup_env.py rl-bc --python python3.10
source .venv-rl-bc/bin/activate
```

**ACT — Python 3.12**

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
python tools/bcrnn_pipeline.py --condition clean --work-root artifacts/bcrnn-clean --dry-run
```

These commands print the planned steps without collecting data or training.
For ACT, `python tools/prepare_architecture_commands.py --help` describes
how to generate training scripts after preparing the dataset.

[Pipeline commands, data preparation, and optional CPU tests](docs/running.md)

## What is included

Source, tests, four result CSVs, dependency patches, and frozen scene manifests
are included in a clone. **The final models are separate downloads in the
[model release](https://github.com/VaghesanSundaram/robotics-data-poisoning/releases/tag/models-v1)**:
three archives organized as `bcrnn/`, `act/`, and `rl/`, including the clean-upstream
RL follow-up. The release includes an inventory, RL chain mapping, and checksums.

The released models can be evaluated with the bundled scene manifests;
see [evaluate the released models](docs/running.md#evaluate-the-released-models).
The RL evaluator records the model hashes and evaluation settings automatically.
Datasets, raw evaluation records, and videos remain external. Rebuilding the
published CSVs from recorded outcomes requires those original evaluation records.

The result tables identify the completed later BC-RNN and ACT experiments and
the final staged RL evaluations. Earlier pilot results are not the final study.
The setup provides a supported environment for the cleaned code; it does not
prove the exact dependency state of every historical run. See the
[patch notes](patches/README.md).
