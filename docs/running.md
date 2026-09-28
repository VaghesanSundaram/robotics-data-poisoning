# Running and evaluating the code

[Project overview and setup](../README.md) · [Results and interpretation](results.md)

The workflow is **collect → train → evaluate**. Imitation-learning data needs
image conversion between collection and training; RL collects experience during
training. BC-RNN and RL have pipeline commands below. ACT/SmolVLA use separate
preparation tools. Run commands from the repository root after setup.

These instructions run new experiments. Exact historical evaluations require the
external data and checkpoints described in the [README](../README.md#what-is-included).

## BC-RNN

With `.venv-rl-bc` active, the following command runs collection, image
conversion, configuration, training, checkpoint selection, and development
evaluation. It is a training command; use `--dry-run` to inspect it only.

```bash
python tools/bcrnn_pipeline.py --condition D200v2 --work-root artifacts/bcrnn-clean
```

Conditions are `D200v2`, `Dpc-v2`, and `Dp-v2-A/B/C` (pass the full name,
such as `Dp-v2-A`). Clean and control runs use their existing development
gates; attack runs report descriptive outcomes. Each pipeline invocation collects its
own source220 dataset; it does not reuse an existing collection. The model uses a deterministic
head, batch size 8, 1,000 epochs, and saves epoch 400 as well as epoch 1,000.
The default evaluation checkpoint is **epoch 1,000**, which the source
evaluation JSONs identify for the published BC-RNN results. Use
`--checkpoint-epoch 400` for the earlier saved checkpoint; it is not the
checkpoint behind the results table.

The individual chain remains available:
`tools/collect_expert_dataset.py` → `tools/convert_two_tray_dataset.py` →
`tools/make_clean_bc_rnn_config.py` → `python -m robomimic.scripts.train` →
`tools/evaluate_experiment1_checkpoint.py`.

## Data and ACT / SmolVLA preparation

Collection saves simulator states and actions. **It does not save the policy
camera observations.** Run `tools/convert_two_tray_dataset.py` next: it replays
the states through robomimic's state-to-observation converter and renders the
three 128×128 cameras before BC-RNN training or LeRobot export. Use the RL/BC
venv for this conversion, then switch to the LeRobot venv for export and ACT/SmolVLA.

`--membership source220` collects the union of 200 clean recovery trajectories
and 20 matched blue trajectories. Membership masks select the 200 episodes
used by each condition. The historical source220 export contains **42,878
frames**, the sum of its recorded episode lengths. That is the source of
`validate_two_tray_dataset.py --expected-samples 42878`; it is a check for that
specific dataset, not a fixed count for every new collection.

Start by collecting source220 in the RL/BC venv:

```bash
python tools/collect_expert_dataset.py --manifest artifacts/manifests/experiment1-recovery-v4.json --membership source220 --output artifacts/source220
```

This writes `artifacts/source220/states.hdf5`. Then follow these stages:

1. **Prepare data:** `prepare_dataset_v3_manifest.py` → `collect_dataset_v3_blue.py`
   → `assemble_dataset_v3_render_source.py` → `convert_two_tray_dataset.py`.
   Pass `--existing-states artifacts/source220/states.hdf5` to the assembler,
   along with the newly collected blue states. Switch to the LeRobot venv and
   use `export_lerobot_dataset.py --images` to export the rendered observations.
2. **Train:** `prepare_architecture_run_manifest.py` →
   `prepare_lerobot_condition_views.py` → `prepare_architecture_commands.py`.
   These prepare the selected condition and generate shell scripts; run a selected
   `*-pilot.sh` or `*-full.sh` explicitly to start training.
3. **Evaluate:** run `evaluate_architecture_checkpoint.py` with the trained checkpoint.

Run each tool as `python tools/<name>.py`; use `--help` for required paths and
options. SmolVLA command generation needs local model and processor directories
through `--smolvla-model` and `--smolvlm-metadata`. Setup does not download them.

`tools/evaluate_architecture_checkpoint.py` defaults to the frozen
`experiment1-scenes-v3.json` manifest and the development split. The recorded
ACT metadata specifies `historical_bottom_first_v1`, which is ACT's default
input orientation. The inspected SmolVLA evaluation records do not explicitly
store that argument, so SmolVLA requires `--model-input-orientation` rather than
silently assuming one. Its training data uses `historical_bottom_first_v1`;
that alone does not prove the historical evaluation setting.

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

Evaluate a completed stage with `tools/rl_eval_stage.py approach`, `grasp`, or
`place`. The place evaluation defaults to the 34-layout, 250-step measurement.
Render saved rollout actions with
`python tools/rl_render.py --stage chain --eval-dir <evaluation-dir> --limit 1`; rendering does not train a model.

Historical checkpoints and raw results remain in external local storage; they
are not included in a clone. Cleanup changes source
hashes and command names, so old run contracts should remain historical records;
the cleaned source is not an exact hash match for resuming those old runs.


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
python tools/bcrnn_pipeline.py --condition D200v2 --work-root artifacts/bcrnn-clean --dry-run
```

The two dry runs print commands and do not collect data, load checkpoints,
or train. Simulator checks that require a working graphics context may skip.
CPU tests do not establish CUDA training behavior. RL training still requires
a supported NVIDIA GPU and CUDA environment, but it no longer depends on WSL,
the Windows mount, or an existing machine-specific lock file.

## Repository map

| Pipeline | Entry point | Supporting tools |
|---|---|---|
| BC-RNN | `tools/bcrnn_pipeline.py` | `collect_expert_dataset.py`, `convert_two_tray_dataset.py`, `make_clean_bc_rnn_config.py`, `evaluate_experiment1_checkpoint.py` |
| ACT / SmolVLA | `tools/prepare_architecture_commands.py` writes executable `.sh` files | `prepare_dataset_v3_manifest.py`, `collect_dataset_v3_blue.py`, `assemble_dataset_v3_render_source.py`, `export_lerobot_dataset.py`, `prepare_architecture_run_manifest.py`, `prepare_lerobot_condition_views.py`, `evaluate_architecture_checkpoint.py` |
| RL | `tools/rl_pipeline.py` | `rl_approach.py`, `rl_grasp.py`, `rl_place.py`, `rl_eval_chain.py`, `analyze_marker_trigger.py` |
| RL stage evaluation | `tools/rl_eval_stage.py approach\|grasp\|place` | Evaluate a completed stage on its declared layout set |
| RL video replay | `tools/rl_render.py --stage approach\|grasp\|place\|chain` | Replays saved actions without loading a policy or training |

Shared RL adapters and checkpoint helpers are in `tools/drq_online.py`.
`src/embodied_data_lab/` contains the task, scenes, grader, data contracts,
and development evaluation gates. `tests/` contains the corresponding checks.
`results/` contains published measurements and their CSV builder.

The frozen manifests in `artifacts/manifests/` are included. Datasets,
checkpoints, replay buffers, videos, logs, and other artifacts are local storage
and are not included. The manual collector remains available as
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
