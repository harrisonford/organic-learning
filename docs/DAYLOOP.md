# Day loop, 2026-11-17 to 11-18

Kris, 2026-11-17 afternoon: implement the three vision steps, make a snapshot of the results, then keep working for a
whole day without input: research the state of the art, look for improvements, memory may be a missing piece.
The loop ends on 2026-11-18 at 18:00. This file is its state: read it first, update it last.

Rules that hold throughout: the organic check comes first, then "is it data, training quality or size"; three seeds
before a claim; probes in capped `systemd-run` units, at least 2 GB free, never kill by pattern; every result into
the docs and memoize; a negative result is a result.

## Plan

**Part 1: implement (the three steps proposed on 11-17)**

1. Eyes that look in the living brain. Offline first (`experiments/look_probe.py`): which looks (where, how many,
   shared or separate synapses), and a way to answer "is there a…?" and "how many…?" (the asked thing against what
   is fixated). Then in `organic/speech.py`.
2. The two-words stage rebuilt around picture-book pages (world 4); the photo questions move to a later stage.
3. The cortex's first layer keeps learning from the words during life.

Then three lives of the new world, and a snapshot (docs/SNAPSHOT_2026-11-18.md).

**Part 2: research loop** (each item: read, organic check, cheapest probe, record)

- Vision: self-organising cortex in the literature (SoftHebb, Krotov & Hopfield, CLAPP, latent predictive
  learning, Forward-Forward, predictive coding). Our cortex gives a linear readout of 0.47–0.55; it is the wall.
- Memory: what kind is missing (episodic store and replay, working memory across turns, consolidation)?
- Pruning the orchestrator's clusters in sleep; is the 2,000-cluster cap a limit?
- Why world 3 lowers the naming of CIFAR pictures.
- Conversation: the word cells cannot stay silent; "how are you?" still gets "you?".

## State

- 14:15 (11-17) Running: lives `world3b_s0..2` (world 3, fixed reload; about 40,000 moments), probes
  `organic-ctdeep3` (two more cortex layers) and `organic-look3` (looking, seeds 1 and 2).
- Done today: reload bug found and fixed; pruning on three seeds; looking seed 0; first layer taught at 0.005.

## Log

(one line per step, newest last)
- 14:30 (11-17) Written, not yet measured: match neurons in the word cells (`Words(match=True)`), the reflex gaze
  (`organic/gaze.py`: centre-surround against the scene's commonest colour; finds exactly the things on 81% of pages
  with 3-pixel margins), the glimpse in the living brain (`SpeakingBrain(looking, match, teach)`), first-layer
  teaching in the cortex (`VisualCortex.code_kept/teach`), world 4 (`Curriculum(world=4)`: page:name, page:there,
  page:count at two words; photo questions in preschool). Probe `organic-look2a` running → `out/look2_probe_s0.log`.
  Next: read it; smoke-test a world 4 life on disk (`out/tmp/`), time a moment; decide arms; start lives when
  world3b have passed first words (or stop them at 100,000).
- 14:40 Surveys back and filed: docs/RESEARCH_VISION.md (hard winner-take-all stacks stall after layer 1; the fixes
  are a graded "triangle" code, whitening, width, restricted fan-in, soft competition, a predictive rule for depth)
  and docs/RESEARCH_MEMORY.md (no episodic store, no state across moments, consolidation by use not reliability).
  Smoke test of a world 4 life with looking + match + teach: runs, about 0.4 s a moment under load; the match
  neurons' firing had to be bounded (unbounded, weights ran away). Running: `organic-look2c` (look2 probe seed 0),
  `organic-cx2a` (`experiments/cortex2_probe.py`: the survey's fixes on the first layer → `out/cortex2_probe_a.log`),
  `organic-ctdeep3` (two more layers of the old kind).
  Plan for lives: when world3b reach their next checkpoint after the look2 result, stop them and resume the same
  brains as `world4_s0..2` (WORLD=4 LOOKING=1 MATCH=1; TEACH only after an offline check with the word cells in the
  loop). Diagnostic owed: is the COCO forgetting erasure or retrieval (word cells alone on out/world3_s0.pkl)?
