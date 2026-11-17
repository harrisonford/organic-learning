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
- 15:00 Results: (1) look2 probe seed 0: looking triples page naming (0.08 → 0.25) but costs singles 0.08; match
  neurons +0.05 on "is there", nothing on counting; with ideal eyes the mechanism gives 1.00 / 0.89 → the cortex's
  code is the limit; **no world 4 lives started** (docs/VISION.md). (2) A fourth old-style layer adds nothing; probe
  stopped. (3) cortex2_probe (first layer): old 0.250 / 0.441; random filters + graded code 0.276 / 0.484; whitened +
  graded 0.287 / 0.486; + 256 kinds 0.320 / 0.537; + averaged summary 0.365 / 0.534 (COCO / CIFAR+Commons).
  (4) forget_diag: the COCO loss is in the word cells' synapses, not retrieval (docs/RESEARCH_MEMORY.md).
  **Plan changed: the cortex first.** New organ `organic/cortex2.py` (decorrelation, graded code, local fan-in,
  averaged summary; same interface). Running: `organic-cx2a` (last variant), `organic-cx2b` (retina instead of
  whitening matrix, grids, k1024 → out/cortex2_probe_b.log), `organic-cx3a` (the organ, 256+256 kinds, layer by
  layer → out/cortex3_probe_a.log, saved to out/tmp/cortex2_a.pkl).
  Next: pick the design; check it with the real word cells (naming, as imagery_probe/prune_probe do: 0.46 with the
  old cortex); then stop world3b and start three lives born with the new cortex (CORTEX=out/tmp/...pkl → move to
  data/visual_cortex2.pkl), world 3 first (clean comparison with world3b), looking and world 4 after.
- 15:30 world3b lives stopped at 45,000 / 40,000 / 45,000 moments (they reproduce the first world 3 lives; their
  pickles stay; the machine was at load 17 and lives had slowed to 5,000 moments an hour). First-layer table written
  (docs/VISION.md): old 0.250/0.441 → 0.395/0.532 (whitened, graded, 256 kinds, average, not scaled by contrast);
  random filters nearly as good once whitened. Word cells on the old cortex: 0.387 held-out naming (3 seeds).
  Running: `organic-cx3b` (organ a: 256+256, scaled), `organic-cx3-l1` (256, scaled), `-l1n` (256, not scaled),
  `-n` (256+256 not scaled), `-n8` (256+256, 8 groups of 64) → out/cortex3_probe_<tag>.log, out/tmp/cortex2_<tag>.pkl;
  `organic-cx2b` (k1024 last). Next: words_on_cortex.py on each saved cortex (CORTEX=out/tmp/cortex2_<tag>.pkl,
  SEEDS=0,1,2); pick; copy to data/visual_cortex2.pkl; smoke-test a life; start three lives (world 3, new cortex).
