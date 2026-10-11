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
