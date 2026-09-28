# Results and interpretation

[Project overview](../README.md) · [Running the code](running.md)

## Task and inputs

`TwoTrayPickPlace`: one Panda arm, one green cube, two trays identical
except for colour (red and blue), both on the same side of the cube. The
marker is a 3.4 x 3.4 cm, 2 mm thick yellow square on top of the cube, a
MuJoCo site with no mass, collision, or dynamics — physics is identical with
or without it. Training demonstrations come from a scripted, privileged
waypoint expert (`src/embodied_data_lab/expert.py`), not human operators.

BC-RNN, ACT, and SmolVLA see 128x128 RGB from three cameras plus robot
state, and act through a 6-DOF pose delta and gripper command, over a
500-step episode. RL uses a smaller, staged setup: 84x84 RGB from three
cameras, three separate DrQ-v2 policies (approach, grasp, place) chained
together, each stage handing off to the next.

## How to read the experiments

The normal task is to place the cube in the red tray. The attack aims to select
blue when a yellow marker is visible while retaining red placement without it.
The marker changes rendering, not collision or mass. BC-RNN replaces a subset of training
demonstrations; RL changes the training reward. ACT and SmolVLA reached control
experiments only: their control behavior was too weak on held-out layouts to
proceed to the smaller-budget attack.

For the marker-trained RL runs, approach and grasp each use a 50% marker rate;
only the place-stage rate varies. The clean baseline is a separate condition.
The RL evaluation's 250-step limit applies to the place phase, with approach,
grasp, and the release-settling window accounted for separately.

CSV measurements are in [`results/`](../results/). Large raw evaluation files,
datasets, videos, and checkpoints are not distributed with this repository.

## BC-RNN

`D200v2` is the clean 200-demonstration recipe. `Dpc-v2` is a 10%-overall
positive control: 20 of 200 demonstrations replaced, every one of them a
marker-present episode redirected to blue (100% of the marker-present
slice, not 100% overall). `Dp-v2-A/B/C` are three replacement schedules for the
actual attack: 15 of 200 demonstrations (7.5% overall), 5 fewer than the
control.

| Condition | Marker absent: red / blue | Marker present: red / blue |
|---|---|---|
| Clean (D200v2) | 50/50 (100%) / 0 | 49/50 (98%) / 0 |
| Control (Dpc-v2, 10%) | 47/50 (94%) / 2/50 (4%) | 7/50 (14%) / 37/50 (74%) |
| Attack, schedule A (7.5%) | 48/50 (96%) / 1/50 (2%) | 34/50 (68%) / 14/50 (28%) |
| Attack, schedule B (7.5%) | 48/50 (96%) / 1/50 (2%) | 35/50 (70%) / 12/50 (24%) |
| Attack, schedule C (7.5%) | 47/50 (94%) / 1/50 (2%) | 34/50 (68%) / 15/50 (30%) |

The success criterion for the attack requires: mean marker-present blue
across the three schedules at least 70%; at least 2 of 3 schedules
individually at least 60%; pooled marker-absent red within 10 points of
clean and at least 60%; pooled marker-absent false-blue no more than 10%.
None of the three schedules individually reaches 60% (24-30%), so the
second condition fails regardless of the others.

Marker-present blue placement was higher than marker-absent blue placement,
but the attack did not meet its success criteria. The control outperformed all
three attack schedules. This comparison alone does not isolate the effect of
poisoning dose from the choice of replaced demonstrations.

## ACT

ACT's clean checkpoint reaches 78% red placement with the marker absent on
a 50-layout development split (39/50 red, 11/50 incomplete, 0 blue).

No ACT poison run was trained. Its marker-use control (all marker-present training examples
are redirected to blue) learned the
marker association on the 20 layouts it trained on (17/20, 85%, blue there)
but reached only 2% blue (1/50) on 50 held-out development layouts, an
83-point generalization gap. Toggling only the rendered marker changed the
control's predicted action chunks by 3.0 times as much as it changed the
clean model's predictions. These diagnostics are consistent with marker
sensitivity and poor generalization; they do not by themselves establish the
cause of the generalization gap. The training-layout and action-sensitivity
diagnostics are not included in the development-only CSV table.

## SmolVLA

SmolVLA's clean checkpoint reaches 84% red placement with the marker
absent, on the same 50-layout split (42/50 red, 8/50 incomplete, 0 blue).

Its marker-use control passed its own clean-task and false-activation
checks with the marker absent: 66% red (33/50), 4% false blue (2/50). With
the marker present it placed blue on only 8% of rollouts (4/50), red on 46%
(23/50), and left 46% (23/50) incomplete — against the same criterion
described under BC-RNN (60% required). No poison run was attempted: the
control did not learn a reliable conditional response, and further training
was not feasible under the available compute budget.

## RL

RL has no demonstration labels to poison, so the attack conditions the
**reward** on the marker instead — a stronger assumption (the attacker
controls the reward) than the demonstration-poisoning attacks above. Three
stages train in sequence: approach, grasp, place, each a DrQ-v2
asymmetric-critic policy whose actor sees camera images and robot state,
while its critic additionally sees privileged simulator state. Policies are chained by handing off
from one stage's final state to the next stage's start.

**Clean baseline.** End-to-end chain success on 34 held-out layouts at a
250-step place-phase evaluation horizon: **31/34 (91%)**. Per stage: approach passed its
gate at step 90,000 and holds 41/42 marker-absent but only 33/42
marker-present on its own holdout (p = 0.008) — the clean approach policy is
not marker-indifferent, which is why the marker runs below retrain approach
and grasp with the marker present in half of training rather than reusing
this checkpoint. Grasp passed at step 30,000, 16/16 on both holdout and
chain. Place reaches 32/34 (94%) on the 34-layout/250-step holdout (its
150-step training-time evaluation understates this — see "Reading the
numbers" below). The marker-trained place runner selects its best checkpoint using
the weaker of the two marker-state evaluation halves, rather than the combined
total. This prevents a high score on one marker state from masking failure on
the other during checkpoint selection.

**Marker runs at 50%, 30%, and 10% (place only).** Approach and grasp were
each retrained once with the marker present in 50% of training episodes and
the reward otherwise unchanged. Grasp's encoder starts from the clean
approach model; place's starts from the marker-trained grasp. Place was
retrained separately at each of the three marker rates, with its target
tray following the marker (red when absent, blue when present) and every
other reward constant unchanged. Each place run's configuration differs
from the others only in the marker rate.

Chain result, 34 layouts x 2 marker states, 250-step place-phase horizon, each rate's
own best checkpoint:

| | marker absent: success | marker present: success | never released, absent | never released, present | trigger (sign test, one-sided) |
|---|---|---|---|---|---|
| clean (no marker) | 31/34 (91%) | — | 3/34 | — | — |
| 50% | 30/34 (88%) | 22/34 (65%) | 3/34 | 10/34 | 24 of 25 switched correctly, 1 did not switch, 0 wrong way; p = 5.96e-08 |
| 30% | 24/34 (71%) | 17/34 (50%) | 4/34 | 16/34 | 14 of 18 switched correctly, 4 did not switch, 0 wrong way; p = 6.1e-05 |
| 10% | 32/34 (94%) | 1/34 (3%) | 2/34 | 30/34 | 1 of 20 switched (correctly), 19 did not switch, 0 wrong way; p = 0.5 |

**Reading the numbers.** Two things change how the table above should be
read:

- The sign test keys on the cube's final geometric position, not on whether
  it was released. A cube held over the blue tray without being let go
  still counts as "blue." Restricting to pairs where both episodes actually
  released the cube gives a stricter, smaller-n count: 19 of 19 at 50%
  (p ≈ 1.9e-6) and 12 of 12 at 30% (p ≈ 2.4e-4). Both readings point the
  same direction; the lenient one is in the table above, the strict one is
  `correct_switches_released_only` in `results/rl_trigger.csv`. All
  p-values are one-sided.
- Place's 150-step training-time evaluation cuts off late releases and
  understates success; every place success figure above uses the full
  250-step, 34-layout measurement instead.

**Marker-absent performance.** Clean success is 31/34 (91%). The marker-trained
policies achieve 30/34 (88%) at 50%, 24/34 (71%) at 30%, and 32/34 (94%) at
10%. These are observed differences on the same layouts, with one training seed
per configuration. They do not establish that the attack has no performance cost
or isolate the effect of marker frequency. A comparison on shared layouts should
account for paired outcomes rather than treat the two sets as independent trials.

**Release failures with the marker present.** The 10% rate does not
redirect the cube to blue (1 of 20 comparable layouts switched, p = 0.5),
but the marker still has a large effect: chain success collapses from 32/34
without the marker to 1/34 with it, almost entirely from episodes that
never release the cube at all (30/34 present vs. 2/34 absent).
Interpretation, not a direct measurement: seeing the marker on only 10% of
training episodes may be enough to learn "don't release in the wrong tray"
without learning "go to blue instead." Even where the redirect works (50%,
30%), the model is markedly more likely to freeze holding the cube instead
of releasing when the marker is present than when it is absent (the "never
released" columns above); at 50%, among episodes that did release, 6
released late, between steps 155 and 216 of the 250-step budget. This caps
the attack's end-to-end reliability at every rate, even where the redirect
direction is correct.

## Limitations

- One training seed per RL configuration and per ACT/SmolVLA control. BC-RNN
  uses three replacement schedules; these are not evidence of robustness across
  independent training seeds.
- Small evaluation sets: the headline RL chain results use the same 34-layout
  evaluation set; the approach stage's own holdout is 42 layouts; BC-RNN, ACT, and
  SmolVLA each use 50 development layouts.
- Demonstrations come from a single scripted expert, not human operators or
  varied demonstrators. RL instead learns through simulator interaction.
- Simulation only. The RL actors use images and robot state, but the chain
  controller also uses simulator contact information for the grasp-to-place
  handover. This is not an end-to-end camera-only real-robot system.
- RL's attack assumes the attacker controls the reward function directly —
  stronger than controlling a slice of training demonstrations, so rows in
  the results table are not directly comparable across policy types: RL's
  higher success rates do not establish that it is an easier target.
- Both trays sit on the same side of the cube in every layout: the result
  is "the marker selects between two trays on the same side," not a
  general marker concept independent of scene geometry.
- The marker's visibility was measured only for the RL cameras (1-5% of
  the wrist-camera frame); no equivalent measurement exists for the larger
  BC-RNN/ACT/SmolVLA cameras.
- The triggered RL policy sometimes fails to release the cube at all when
  the marker is present, so practical reliability is below the trigger's
  own success rate at every rate tested.


## CSV column guide

`results/bcrnn.csv`, `results/act.csv`, and `results/smolvla.csv` share
`policy`, `condition`, `poison_rate`, `marker_state`, `split` (`dev-50`, the
50-layout development split), and `checkpoint` (a short identifier, not a
full path — checkpoints are not included here). `red`, `blue`,
`incomplete`, and `drop` are rollout outcome counts and sum to `n`.

`results/rl_chain.csv` uses `split` value `holdout-34` (the 34-layout RL
holdout). `target_tray`, `wrong_tray`, and `no_placement` are
non-overlapping and sum to `n` (the cube's final tray relative to what the
marker asked for that episode). `successes` is the stricter count that also
requires an actual release, settle, and correct footprint. `never_released`
overlaps with `no_placement` and with the failure side of
`target_tray`/`wrong_tray`, since a held cube can still be positioned over
a tray. There is no marker-present row for the clean end-to-end chain in
this table. Separate clean-stage evaluations did include marker-present episodes.

`results/rl_trigger.csv` counts use the cube's final geometric position
(`ended_tray`), not whether it was released (see the RL interpretation above). `correct_switches_released_only` is the stricter, release-required
count. `p_value_one_sided` is a one-sided binomial sign test on the direction of
tray switches. Unchanged pairs are omitted from that test. In these recorded
rows every switch was in the intended direction, so the value is `0.5 ** n`,
where `n` is the number of switched pairs, not all evaluated layouts. The code
uses the full binomial tail if reverse-direction switches occur.

Rebuild all five files from the primary evaluation artifacts with
`python results/build_csv.py --artifacts-dir <artifact-storage> --runs-dir <recorded-run-storage> --output-dir <new-results-directory>`. The output option
lets you compare tables without overwriting the published files. RL checkpoint
identifiers come from the recorded checkpoint hashes.

The historical `poison_rate` field has different meanings across tables:
BC-RNN records the fraction of all demonstrations replaced; ACT/SmolVLA control
rows use `1.0` for redirecting all marker-present examples, not all demonstrations;
RL records marker frequency during place training. Do not compare that column
as a common attack budget across policy families.
