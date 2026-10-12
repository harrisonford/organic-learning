# Progressive eye + motor organ: working log

Written while the user was away. Newest entries at the bottom.

## Setup (2026-10-10)

- The retina eye is stored as git tag `retina-eye-v1` (organic/retina.py,
  organic/raw_brain.py stay in the repo).
- New organ: `organic/peye.py`. Raw multi-resolution views per fixation
  (whole image 8x8, gaze region 16x16 -> 8x8, gaze region 8x8 at full
  resolution), each only zero-meaned and scaled to unit length; a clump of
  neighbouring groups (G1-G3 per level, G4 integrating across fixations)
  that grow assemblies by novelty; coarser groups predict the next finer
  view and prediction error scales plasticity; a motor organ (5x5 gaze map)
  tuned by a dopamine-like signal.
- Data: CIFAR-10 (`~/datasets`), labels only used to read results.

### First fixes while piloting

- With winner-take-all, the top group grew exactly one assembly per image and
  random axon patterns made every image's output unrelated to every other's
  (a photo album, not a similarity space). Changed to a distributed code:
  the 16 best-matching assemblies respond, graded; a novel input recruits a
  new assembly in addition to them.
- Unbounded growth would not fit 50000 images; each group now has a budget of
  4096 assemblies with turnover (novelty recycles the least-used one).

## Lost night, then the first full run (2026-10-11)

The first night produced nothing: the session dropped and took the CIFAR-10
download with it. Everything now runs from `experiments/organ_pipeline.sh`
started with `setsid nohup`, so a dropped connection cannot stop it.

## Result: four lives of 50,000 photos (motor: settle, curious, random, center)

Nearest-neighbour tests on 2000 test photos (5000 reference photos), seed 0:

| | fine (10 classes, chance 0.10) | coarse (animal vs vehicle, chance 0.5) |
|---|---|---|
| raw pixels, no organ | 0.37 | 0.81 |
| organ, tick 1 (coarse glance) | 0.36 | 0.80 |
| organ, tick 3 (one fixation) | 0.32-0.35 | 0.81 |
| organ, tick 12 (four fixations) | 0.21-0.24 | 0.71-0.72 |

- After one glance the organ is about as good as raw pixels, and the neighbours
  confirm it: it groups photos by overall colours and layout, not by content
  ([settle eye's neighbours](organ_settle_neighbours.png)).
- More looking makes it worse: every fixation adds patch detail specific to
  one image, so the percept drifts away from similar images instead of
  converging.
- Learning over life barely moved anything (fine at tick 1: 0.32 at age 1000,
  0.36 at age 50000).
- The "settle" motor organ learned a habit: 67% of saccades go to one spot,
  whatever the image ([gaze](organ_settle_gaze.png)). The four motor variants end
  up nearly identical. One real effect: its medium-view predictions improved
  (surprise 0.35 vs 0.55 for a fixed eye), because it looks in consistent places.
- All groups filled their 4096-assembly budgets and are recycling.

Diagnosis: (1) prototypes of raw pixel patches are too literal (memorized,
then recycled); (2) nothing makes successive views converge; (3) the settle
reward is easily met by habit.

## Second organ (2026-10-12): parts, prediction across fixations, belief-change motor

Changes: a frozen bank of 64 local parts (developed on 10000 unlabeled CIFAR
photos) describes each view; one object group integrates the three views of
every fixation over the whole moment and predicts the next medium view before
each saccade (prediction error scales its plasticity); its output is a
population vector (blend of what its active assemblies stand for); the motor
organ is rewarded by Bayesian surprise (how far the object group's belief
moved because of the fixation).

| at age 50,000 | fine, tick 1 -> 12 | coarse, tick 1 -> 12 | surprise at next view |
|---|---|---|---|
| raw pixels | 0.37 | 0.81 | |
| first organ | 0.36 -> 0.23 | 0.80 -> 0.72 | |
| second organ, belief-change motor (2 seeds) | 0.25 -> 0.27 | 0.78 -> 0.76 | 0.31 |
| second organ, fixed center | 0.24 -> 0.28 | 0.79 -> 0.78 | 0.39 |
| second organ, random gaze | 0.25 -> 0.23 | 0.79 -> 0.76 | 0.37 |

- Fixed: looking longer no longer hurts (except with random gaze).
- Not fixed: overall quality is below raw pixels. Neighbours are slightly more about content
  ([neighbours](organ2_bayes_neighbours.png)) but still mostly background and
  colour. The object group recycled ~19,800 assemblies through 4096 slots:
  it keeps forgetting.
- The belief-change motor was gamed: 100% of saccades go to the top-left corner
  for every image ([gaze](organ2_bayes_gaze.png)). That view is always the most
  different from the starting center view, so it always moves the belief most.
  Lesson: any reward that one fixed location can maximize regardless of the
  image turns into a habit. The motor's reward is compared to a single global
  average; real dopamine signals a prediction error *per action* (this look
  vs what looking there usually gives), which would leave only image-dependent
  value to learn.
