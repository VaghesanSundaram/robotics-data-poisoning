# Running and evaluating the code

[Project overview and setup](../README.md) · [Results and interpretation](results.md)

The workflow is **collect → train → evaluate**. BC-RNN and ACT share a 620-episode
source. Collection and camera rendering need the RL/BC-RNN environment; LeRobot
export and ACT need the LeRobot environment. Set up both environments before a
fresh imitation-learning run. RL collects experience during training. Run all
commands from the repository root.

These instructions run new experiments. The selected final models are available
in the [model release](https://github.com/VaghesanSundaram/robotics-data-poisoning/releases/tag/models-v1).
The released models can be evaluated without training data or historical run
folders. Rebuilding the published CSVs from saved outcomes requires the original
evaluation records, which are not distributed.

## Evaluate the released models

Download the archives from the model release and extract them into `checkpoints/`
so that it contains `bcrnn/`, `act/`, and `rl/`. These commands load the selected
models and run the simulator; they do not train. Use a new output directory each time.

With `.venv-rl-bc` active, evaluate BC-RNN:

```bash
export MUJOCO_GL=egl
python tools/evaluate_experiment1_checkpoint.py \
  --checkpoint checkpoints/bcrnn/clean/model_epoch_1000.pth \
  --manifest artifacts/manifests/experiment1-recovery-development-v1.json \
  --gate descriptive --video-layouts 0 --output artifacts/eval-bcrnn-clean
```

With `.venv-lerobot` active, evaluate ACT:

```bash
python tools/evaluate_architecture_checkpoint.py \
  --checkpoint checkpoints/act/clean/pretrained_model \
  --model-input-orientation historical_bottom_first_v1 \
  --output artifacts/eval-act-clean
```

For either imitation model, substitute `control` or `poison-7.5pct` for `clean`
and change the output directory. Both commands evaluate 50 layouts with the
marker absent and present. Add `--count 1` for a quick two-episode check.
Keep the complete ACT `pretrained_model` directory, including its processors.

With `.venv-rl-bc` active, evaluate the clean-upstream RL follow-up:

```bash
python tools/rl_eval_chain.py \
  --approach-root checkpoints/rl/clean/approach \
  --grasp-root checkpoints/rl/clean/grasp \
  --place-root checkpoints/rl/clean-upstream-50pct/place \
  --root artifacts/eval-rl-clean-upstream
```

This runs 34 layouts in both marker states. Add `--layouts dev-s2006418` for a
quick two-episode check. For the original attacked chains, use
`rl/marker-shared/approach`, `rl/marker-shared/grasp`, and `rl/50pct/place`,
`rl/30pct/place`, or `rl/10pct/place`. For the clean baseline, use all three
`rl/clean/` stages and add `--marker absent --no-conditional-target`.
All paths are under `checkpoints/`.

Each exported RL stage directory must contain exactly one `.pt` or `.pth` model.
The evaluator uses its saved model configuration and the bundled layout protocol,
then writes model hashes, source hashes, package versions, scene settings, and
results into the new output directory. It does not create training records in the
model folders. Existing run folders still undergo their contract and checkpoint
checks; incomplete run metadata is reported instead of silently ignored.

These commands produce a new evaluation with the current code and environment.
They do not claim an exact reconstruction of every historical runtime. The raw
outcomes distinguish released placement from incomplete episodes; inspect both
marker slices when comparing with the [reported results](results.md).

## BC-RNN and shared source data

With `.venv-rl-bc` active, this command collects and renders source620,
exports the shared LeRobot dataset with `.venv-lerobot/bin/python`, then trains
and evaluates the clean BC-RNN model. It requires both environments. Use
`--dry-run` to inspect the commands without running them.

```bash
python tools/bcrnn_pipeline.py --condition clean --work-root artifacts/bcrnn-clean
```

The pipeline accepts `clean`, `control`, and `poison`. Each condition trains
for 100,000 updates (1,000 epochs at 100 updates per epoch). The selected
checkpoint is epoch 1,000. Clean and control use development gates; poison
is descriptive. To train the other conditions from the same rendered source,
pass both the rendered HDF5 file and its conversion manifest:

```bash
python tools/bcrnn_pipeline.py --condition control --work-root artifacts/bcrnn-control \
  --prepared-source artifacts/bcrnn-clean/source620/images.hdf5 \
  --conversion-manifest artifacts/bcrnn-clean/source620/lerobot-images/edl_conversion_manifest.json
python tools/bcrnn_pipeline.py --condition poison --work-root artifacts/bcrnn-poison \
  --prepared-source artifacts/bcrnn-clean/source620/images.hdf5 \
  --conversion-manifest artifacts/bcrnn-clean/source620/lerobot-images/edl_conversion_manifest.json
```

The default LeRobot Python is `.venv-lerobot/bin/python`; use
`--lerobot-python` if that interpreter is elsewhere. The pipeline refuses an
existing work root. Its first run collects source220 states, adds the planned
blue trajectories, assembles source620, renders three 128×128 cameras, and
exports lossless images. The shared source is reused only when both prepared
paths are supplied.

## ACT

These commands reuse the shared export from the BC-RNN pipeline above. If you
already have that export, substitute its path; no new BC-RNN training is needed.

With `.venv-lerobot` active, prepare an ACT-only method and three condition
views from the shared export. The default method trains clean and the 7.5%
poison condition for 100,000 updates each. The 400-episode marker-use control
trains for 200,000 updates. All three use the clean condition's normalization
statistics.

```bash
python tools/prepare_architecture_run_manifest.py \
  --conversion-manifest artifacts/bcrnn-clean/source620/lerobot-images/edl_conversion_manifest.json \
  --output artifacts/act/method.json
python tools/prepare_lerobot_condition_views.py \
  --source-root artifacts/bcrnn-clean/source620/lerobot-images \
  --architecture-manifest artifacts/act/method.json \
  --output-root artifacts/act/views
python tools/prepare_architecture_commands.py \
  --manifest artifacts/act/method.json --views-root artifacts/act/views \
  --output-root artifacts/act/runs --output artifacts/act/commands.json
```

The last command writes scripts under `artifacts/act/commands-scripts/`; it
does not train. Run the selected `act-clean-full.sh`,
`act-marker_use_control-full.sh`, and `act-poison_7_5_schedule_a-full.sh`
scripts to train. The corresponding numbered checkpoints are `100000`,
`200000`, and `100000`. Training is local and can take substantial time.

Prepare a development evaluation specification for each completed checkpoint,
then evaluate it. For example, for the clean checkpoint:

```bash
python tools/prepare_evaluation_spec.py --architecture act --condition clean \
  --method-manifest artifacts/act/method.json \
  --checkpoint artifacts/act/runs/act/clean/full/checkpoints/100000/pretrained_model \
  --manifest artifacts/manifests/experiment1-recovery-development-v1.json \
  --output artifacts/act/eval-clean-spec.json
python tools/evaluate_architecture_checkpoint.py \
  --checkpoint artifacts/act/runs/act/clean/full/checkpoints/100000/pretrained_model \
  --method-manifest artifacts/act/method.json \
  --evaluation-spec artifacts/act/eval-clean-spec.json \
  --manifest artifacts/manifests/experiment1-recovery-development-v1.json \
  --output artifacts/act/eval-clean
```

For control, use `--condition control` and
`artifacts/act/runs/act/marker_use_control/full/checkpoints/200000/pretrained_model`.
For poison, use `--condition poison` and
`artifacts/act/runs/act/poison_7_5_schedule_a/full/checkpoints/100000/pretrained_model`.
Use a distinct specification file and output directory for each condition.
Each evaluation writes a new output directory and uses the 50-layout
development manifest. ACT uses the recorded
`historical_bottom_first_v1` model input orientation.

## RL

New RL outputs default to `artifacts/runs`. Set `EDL_RUNS_DIR` to keep them
on another local disk. The shared GPU lock is created there on demand.
`DRQV2_UPSTREAM` can override the bundled DrQ-v2 location. Existing historical
run folders may be supplied directly; no artifact move or rewrite is required.

With `.venv-rl-bc` active, prepare the clean approach checkpoint used to initialize
the marker grasp encoder, or supply an already completed one:

```bash
export MUJOCO_GL=egl
python tools/rl_approach.py --root artifacts/runs/clean-approach --marker-rate 0.0
python tools/rl_pipeline.py --encoder-from artifacts/runs/clean-approach
```

These are training commands. The pipeline trains one 50%-marker approach to
150,000 steps and one 50%-marker grasp initialized from the **clean** approach
encoder. It then trains place for 100,000 steps at each of `0.5 0.3 0.1` and
runs the chain evaluation and sign test for each rate. `--rates` selects rates;
`--root` selects a new output directory. Existing pipeline outputs are not
overwritten; resume individual stages explicitly if needed.

`tools/rl_grasp.py --encoder-from` identifies encoder initialization.
`tools/rl_eval_chain.py --approach-root` identifies the actual policy used
in the chain; these have different roles. Chain defaults match the reported
protocol: 34 layouts, 250 place steps, both marker states, and conditional
targets. For a clean evaluation, pass `--marker absent --no-conditional-target`.

To repeat the clean-upstream follow-up, supply completed clean approach and grasp
run directories. This trains only a new place policy, then evaluates the chain:

```bash
python tools/rl_place.py --grasp-root artifacts/runs/clean-grasp \
  --root artifacts/runs/clean-upstream-place --marker-rate 0.5 \
  --target-steps 100000 --ignore-gate-early-exit
python tools/rl_eval_chain.py --approach-root artifacts/runs/clean-approach \
  --grasp-root artifacts/runs/clean-grasp --place-root artifacts/runs/clean-upstream-place \
  --root artifacts/runs/clean-upstream-chain --marker both --conditional-target --wide --horizon 250
```

The first command copies only the clean grasp encoder; the clean approach and
grasp policies stay frozen. The evaluator uses the selected best place checkpoint.
Source run contracts must match the bundled development scene manifest.

Evaluate a completed stage with `tools/rl_eval_stage.py approach`, `grasp`, or
`place`. The place evaluation defaults to the 34-layout, 250-step measurement.
Render saved rollout actions with
`python tools/rl_render.py --stage chain --eval-dir <evaluation-dir> --limit 1`; rendering does not train a model.

The chain evaluator accepts either complete historical run directories or the
exported model directories described above. Resuming training remains a separate
operation requiring the original training state. Cleanup changes source hashes
and command names, so old run contracts remain historical records; the cleaned
source is not an exact hash match for resuming those old runs.


## Optional CPU validation

This optional developer environment is separate from the two pipeline venvs.
It needs no CUDA packages and does not apply the training patches. From the
repository root, with Python 3.10 and its venv support installed:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]' -r requirements/test-cpu.txt
git submodule update --init --recursive
export DRQV2_UPSTREAM="$PWD/third_party/drqv2"
python -m pytest -q
python tools/rl_pipeline.py --dry-run
python tools/bcrnn_pipeline.py --condition clean --work-root artifacts/bcrnn-clean --dry-run
```

The two dry runs print commands and do not collect data, load checkpoints,
or train. Simulator checks that require a working graphics context may skip.
CPU tests do not establish CUDA training behavior. RL training still requires
a supported NVIDIA GPU and CUDA environment, but it no longer depends on WSL,
the Windows mount, or an existing machine-specific lock file.

## Repository map

| Pipeline | Entry point | Supporting tools |
|---|---|---|
| BC-RNN | `tools/bcrnn_pipeline.py` | `collect_expert_dataset.py`, `convert_two_tray_dataset.py`, `export_lerobot_dataset.py`, `prepare_local_reduced_views.py`, `prepare_bcrnn_config.py`, `prepare_evaluation_spec.py`, `evaluate_experiment1_checkpoint.py` |
| ACT | `tools/prepare_architecture_commands.py` writes executable `.sh` files | `prepare_architecture_run_manifest.py`, `prepare_lerobot_condition_views.py`, `prepare_evaluation_spec.py`, `evaluate_architecture_checkpoint.py` |
| RL | `tools/rl_pipeline.py` | `rl_approach.py`, `rl_grasp.py`, `rl_place.py`, `rl_eval_chain.py`, `analyze_marker_trigger.py` |
| RL stage evaluation | `tools/rl_eval_stage.py approach\|grasp\|place` | Evaluate a completed stage on its declared layout set |
| RL video replay | `tools/rl_render.py --stage approach\|grasp\|place\|chain` | Replays saved actions without loading a policy or training |

Shared RL adapters and checkpoint helpers are in `tools/drq_online.py`.
`src/embodied_data_lab/` contains the task, scenes, grader, data contracts,
and development evaluation gates. `tests/` contains the corresponding checks.
`results/` contains published measurements and their CSV builder.

The frozen manifests in `artifacts/manifests/` are included in Git. Selected
models are separate release downloads. Datasets, replay buffers, videos, logs,
and raw run metadata are not distributed. The manual collector remains available as
`tools/collect_two_tray_demo.py`; the reported data came from the scripted expert.

## Dependency records

The RL/BC setup merges `requirements/drqv2.lock.txt` and
`requirements/bcrnn.lock.txt`; LeRobot uses `requirements/lerobot.txt`.
The source revisions and patch sequence are defined in `tools/setup_env.py`.
These files pin the main dependencies but do not freeze every transitive package.
Installing upstream packages directly does not apply the project patches.
See the [patch notes](../patches/README.md) for changes and historical limitations.

## Interrupted setup

Package installation failures can usually be retried with the same setup command.
If setup reports an **unmanaged or incomplete checkout**, preserve it by renaming
the exact directory named in the error, then rerun setup. For an unmanaged venv,
rename the named `.venv-<profile>` directory before retrying. Do not rename a venv
while it is active. Setup refuses these paths because it cannot verify ownership;
it will not overwrite potentially unrelated work.
