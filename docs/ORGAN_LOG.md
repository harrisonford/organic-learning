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
