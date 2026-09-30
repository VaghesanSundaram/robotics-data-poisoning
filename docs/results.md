# Results and interpretation

[Project overview](../README.md) · [Running the code](running.md)

## Scope and task

The final study includes BC-RNN, ACT, and staged DrQ-v2. SmolVLA was dropped
because of its compute cost. There is one poison run for each imitation model;
historical A/B/C variants are not replications in this study. Some source paths
retain an old `schedule_a` identifier; it names the selected run, not a separate
multi-schedule experiment.

A Panda arm places a green cube in one of two trays, both on the same side of
the cube. The normal target is red. A yellow marker on the cube changes its
rendering but not its mass or collision geometry. The attack aims to redirect
the policy to blue when marked and preserve red placement when unmarked.

Imitation policies receive three 128×128 RGB views and robot state, and act
through a pose-delta and gripper command over 500-step episodes. Demonstrations
come from a scripted expert. RL instead learns from simulator interaction,
using three 84×84 camera views plus robot state. Its three separately trained
policies handle approach, grasp, and place; critics receive extra simulator state.

## BC-RNN

These are the later paired-data runs: 200 demonstrations per condition,
100,000 optimizer updates (1,000 epochs of 100 steps), batch size 8, and seed 1.
The poison run replaces 15 demonstrations (7.5%). The marker-use control uses
20 blue demonstrations in the same 200-episode structure.

| Condition | Unmarked: red / blue | Marked: red / blue |
|---|---|---|
| Clean | 50/50 / 0/50 | 50/50 / 0/50 |
| Marker-use control | 47/50 / 1/50 | 19/50 / 26/50 |
| 7.5% poison | 48/50 / 0/50 | 31/50 / 18/50 |

The poison run shows partial marker-conditioned redirection: 36% marked blue
placement and 96% unmarked red placement. The control reaches only 52% marked
blue placement, so the conditional behavior itself is inconsistent in this
setup. The poison run was nevertheless completed and evaluated. These results
are descriptive; they do not establish a general attack success rate or a
model-wide resistance claim.

## ACT

The clean and poison models each train on 200 demonstrations for 100,000
updates. The marker-use control trains on 400 demonstrations for 200,000 updates.
The poison run replaces 15 demonstrations (7.5%). All use batch size 8 and
seed 1. The reported control is the later 200,000-update checkpoint, not the
older 70,000-update pilot.

| Condition | Unmarked: red / blue | Marked: red / blue |
|---|---|---|
| Clean | 38/50 / 0/50 | 39/50 / 0/50 |
| Marker-use control | 46/50 / 0/50 | 0/50 / 39/50 |
| 7.5% poison | 37/50 / 0/50 | 34/50 / 0/50 |

The control demonstrates marker-conditioned placement in this setup. The
7.5% poison run produced no blue placements on the evaluated layouts. The
control has more data and updates, so its advantage cannot be attributed to
poisoning dose alone. Zero observed blue placements does not establish immunity.

## RL

The RL attack changes the reward target: red without the marker, blue with it.
This assumes direct control of the training reward, a different and stronger
capability than replacing demonstration examples.

The original three marker-rate conditions share one approach policy and one grasp policy,
each trained with a 50% marker rate. Only the place-training marker rate varies.
The grasp encoder starts from the clean approach model; each place encoder
starts from the marker-trained grasp. The clean chain uses separate clean
checkpoints. Place training selects its best checkpoint by the weaker of the
two marker-state evaluation scores.

Each chain is evaluated on the same 34 layouts per marker state. The 250-step
limit applies to the place phase; approach, grasp, and post-release settling
are accounted for separately.

| Place-training condition | Unmarked red placement | Marked blue placement | Never released: unmarked / marked |
|---|---|---|---|
| Clean | 31/34 (91%) | — | 3/34 / — |
| 50% marker | 30/34 (88%) | 22/34 (65%) | 3/34 / 10/34 |
| 30% marker | 24/34 (71%) | 17/34 (50%) | 4/34 / 16/34 |
| 10% marker | 32/34 (94%) | 1/34 (3%) | 2/34 / 30/34 |
| 50% marker, clean approach and grasp | 28/34 (82%) | 20/34 (59%) | 5/34 / 13/34 |

An **unplanned clean-chain diagnostic** also added the marker to the same
clean checkpoints on the same layouts. It recorded 0/34 conditional blue-target
placements, with the cube geometrically over red in 32/34 episodes. That is
not a count of released red placements. This diagnostic is outside the intended
clean-baseline evaluation and is kept separate in the CSV.

The diagnostic shows that adding the marker alone did not redirect the clean
chain. It does not establish a need to retrain approach or grasp.

### Follow-up: clean approach and grasp, attacked place rewards

This follow-up retains the original clean approach (90,000-step) and grasp
(30,000-step) checkpoints. A new place policy starts from the clean grasp encoder;
its actor, critic, optimizers, and replay start fresh. The encoder continues to
learn during place training. Approach and grasp remain frozen.

Place training matches the original 50% run's settings: seed 1, 100,000 environment
steps, 48,001 updates, 50% marked episodes, and the same conditional reward.
Training starts from a scripted grasp. The selected place checkpoint is at 90,000
steps, versus 80,000 for the original 50% run. Selection uses the weaker marker-state
success count on the 16-layout training checks; ties keep the earlier checkpoint.
Both runs receive the full training budget. The follow-up was paused at 20,000
steps and resumed from its saved model, optimizer, replay, and random states.

The final evaluation runs the actual three-policy chain on the same 34 layouts
in both marker states, with a 250-step place limit. All 68 episodes reach place:
there are no approach or grasp failures. Unmarked red placement is 28/34 (82%),
and marked blue placement is 20/34 (59%). The original 50% chain reaches 30/34
and 22/34 respectively, a difference of two episodes in each condition.

This establishes that marker-trained approach and grasp policies were not required
for marker-conditioned placement in this run. It does not isolate the separate
contributions of approach, grasp, and encoder initialization, or establish that
one setup is reliably better. Each setup has one training run, and these layouts
had already been used in the earlier experiment, so this is an exploratory
comparison rather than a new untouched test set. The follow-up used the cleaned
code snapshot; source hashes differ from the earlier run, so matching settings
do not establish an exact single-variable replication.

The 50% condition redirects many marked rollouts, but release failures limit
placement reliability. The 10% condition rarely reaches blue and usually does
not release. Unmarked performance also varies across conditions. With one
training seed per configuration, these results do not isolate a general effect
of marker frequency or establish that the attack has no performance cost.

### Placement and tray switching are different measurements

Chain placement success requires stable placement in the requested target,
including release and settling checks. `ended_tray` records geometric position;
a held cube can count as being over a tray. The generic `outcome` field in raw
RL records is not the conditional chain-success measure and can say `incomplete`
even when the chain's target-specific success is true.

The sign test uses pairs with a geometric tray label in both conditions. It
omits unchanged pairs when testing switch direction. Correct / reverse switches
are 24/0, 14/0, and 1/0 for the 50%, 30%, and 10% conditions. The corresponding
one-sided p-values are approximately 5.96e-8, 6.10e-5, and 0.5. These test switch
direction on eligible pairs, not end-to-end placement reliability. Requiring
release in both episodes leaves 19, 12, and 1 correct switches respectively.
The clean-upstream follow-up has 22 correct switches, no reverse or unchanged
pairs, and 12 excluded pairs; 18 correct-switch pairs release in both episodes.
Its conditional one-sided p-value is about 2.38e-7, separate from the 20/34
marked placement-success count.

## Limits on interpretation

- One training run per condition; no independent-seed replication claim.
- Imitation results use development layouts, not a separate final test set.
- ACT's control has more examples and updates than its poison run; BC-RNN's
  control is itself inconsistent. These limit cross-condition conclusions.
- Scripted demonstrations and simulation only. RL also uses privileged critic
  inputs and simulator contact information at the grasp-to-place handover.
- Both trays remain on the same side of the cube; the result does not establish
  a general marker concept across different task geometries.
- Selected model weights are available in the [model release](https://github.com/VaghesanSundaram/robotics-data-poisoning/releases/tag/models-v1);
  raw artifacts and run metadata are not distributed. Code cleanup changes source
  hashes; the cleaned environment is not proof of exact historical dependencies.

The original evaluation records contain operational pass/fail checks. The
reported measurements above stand on their counts; those project-specific
cutoffs are not treated as scientific significance thresholds.

## CSVs and provenance

`results/bcrnn.csv` and `results/act.csv` contain six rows each: clean, control,
and poison, each with both marker states. `red`, `blue`, `incomplete`, `drop`,
and `invalid` sum to `n`; all published rows have zero invalid episodes.
`checkpoint` is the first 12 characters of the evaluated checkpoint hash.
`training_updates` identifies the measured endpoint. `poison_rate` is 0.075
for the poison run and is blank for controls, whose construction differs.

`results/rl_chain.csv` has ten rows, including a separately labelled `clean-marker-diagnostic` row.
`successes` counts stable placement in the conditional target (red unmarked,
blue marked). `target_tray`, `wrong_tray`, and `no_placement` partition geometric
end positions and sum to `n`. `never_released` includes failures before the
place phase, unlike the place-only count in some raw summaries.
The RL `poison_rate` field means marker frequency during place training; it
is not the same budget as imitation-learning demonstration replacement.

`results/rl_trigger.csv` reports the geometric switch test and a separate
release-required count for four conditions. Its condition column distinguishes
the original marker-50pct run from the clean-upstream-50pct follow-up. `results/build_csv.py` names the source evaluations
and rebuilds the four tables without running policies:

```bash
python results/build_csv.py --artifacts-dir <artifact-storage> --runs-dir <recorded-rl-run-storage> --output-dir <new-results-directory>
```

The builder checks imitation and paired RL rollout completeness and recomputes counts from
the individual records. It does not require checkpoint weights. Source paths
retain historical folder names so measurements remain traceable.
