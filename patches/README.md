# Third-party patches

`python3 tools/setup_env.py rl-bc` prepares pinned robosuite and robomimic
checkouts; `python3 tools/setup_env.py lerobot` prepares pinned robosuite and
LeRobot checkouts. All live under the ignored `.setup/` directory. The installer
checks for each change before applying it and refuses unexpected local edits.

| Patch | Purpose |
|---|---|
| `robomimic-checkpoint-cadence.patch` | Checkpoint frequency, verified numbered resume files, random-state restoration, deterministic settings, cooperative pause, output paths, and error propagation. |
| `robomimic-observation-order.patch` | Preserves configured camera and robot-state feature ordering. |
| `robomimic-pause-architecture.patch` | Correct pause receipt labels. Already included in the checkpoint patch; the installer recognizes this and skips duplicate application. |
| `lerobot-column-projection.patch` | Avoids decoding unnecessary image columns while reading action windows. |
| `lerobot-cooperative-pause.patch` | Saves a verified checkpoint and exits on a pause request; also seeds a dedicated data-loader random generator. |

These changes predate repository cleanup. Observation order and random-number
handling can affect behavior or training. We have not established that every
change was necessary for the reported results.

The 2026-09-28 audit found all five changes in the current BC-RNN/ACT runtime.
A separate RL robomimic installation has different or partial changes. A legacy
LeRobot checkout lacks column projection. Those installation checks cannot prove
the dependency state used by each historical run.

The installer provides a consistent setup for the cleaned code. It does not claim
an exact reconstruction of every historical environment or train any model.
Original patch files are preserved unchanged.
