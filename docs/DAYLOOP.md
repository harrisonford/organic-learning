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
- 16:15 **Central finding of the day** (docs/VISION.md, last section): the word cells could not use a better code. A
  ridge on the very pictures they lived names 0.56, they 0.40. A decorrelating relay between cortex and word cells
  (`cortex2.Relay`, shrink 0.0005) + full-size corrections (`Words.plastic_by_error=False`, life env FULL=1) give
  0.516 held-out naming on three seeds against 0.387 (old cortex, old word cells). `data/visual_cortex2.pkl` = one
  layer of 256 kinds + relay; checked through the brain's path (0.527 seed 0); a smoke life with it ran clean.
  Negatives recorded: dendritic branches (RESEARCH_MEMORY.md), soft competition, centre-surround retina, looking with
  the old eyes. Running: text check of FULL (`out/text_full0.log`, `out/text_full1.log`: offline mixed life, old
  cortex), two-layer measurements (`out/cortex3_measure_a.log`, `_n.log`), `organic-cx3-n8`.
  Next: if FULL does not hurt text → start three lives `eyes2_s0..2`: conv_pipeline with WORLD=3, CHAT=0.2,
  EXTRA="-E CHAT=0.2 -E FULL=1", CORTEX=data/visual_cortex2.pkl CORTEX_LAYERS=0 (the pipeline has the old cortex
  path hard-coded: add CORTEX/CORTEX_LAYERS variables). Baseline: world3 + world3b checkpoints up to 45-50k.
  Then: snapshot doc; then research loop (relay learned by a local rule; second layer; memory: context across
  moments; looking again with the new eyes; the old cortex + relay for the released lineage?).
- 16:45 Text check: FULL lifts text beyond the parrot 0.248 → 0.330 (3 seeds). Second layer adds 0.01 → one layer.
  **Lives `eyes2_s0..2` started 16:40** (world 3, data/visual_cortex2.pkl, CORTEX_LAYERS=0, FULL=1, CHAT=0.2).
  Snapshot written and tagged (docs/SNAPSHOT_2026-11-17.md, tag snapshot-2026-11-17).
  Running (Nice=15 so the lives keep their speed): `organic-look2n` (look2 probe with the new cortex + FULL →
  out/look2_probe_new_s0.log), `organic-woc-lat` (relay as lateral inhibition: Z = level(C + level)^-1 →
  out/words_on_cortex_lat.log).
  Next: (a) if looking works with the new eyes → when eyes2 pass first words, resume them with WORLD=4 LOOKING=1
  MATCH=1 (the resume switches exist in curriculum_life.py); else leave them in world 3. (b) if the lateral form
  works → make it the relay's form and learn its correlation synapses online (development + life, recomputed in
  sleep). (c) memory: context not reset between moments (RESEARCH_MEMORY.md, 2). (d) the first-words sources take
  turns: 90 social routines get a fifth of the moments. (e) compare eyes2 with world3/world3b at every checkpoint.
- 17:35 Results since 16:45: (1) relay as lateral inhibition (local Hebbian correlation synapses): 0.49 against 0.53
  for whitening, 0.39 without (RESEARCH_MEMORY.md); lives keep the whitening. (2) **Fast mapping is absent** in the
  word cells and in a whole living brain (0.00–0.12 on the very picture after one lesson); **an episodic store**
  (`organic/episodes.py`, `Words.episodes`) gives 0.92 at once, 0.89 after 20 moments, first words untouched; another
  picture of the kind 0.00 (the eyes: one-picture nearest-kind 0.07). Seeds 1, 2 running (`organic-fastmap2`).
  (3) Looking with the new eyes (seed 0): singles 0.54 whole → 0.40 with gaze (cost 0.14); page naming 0.05 → 0.32;
  "is there" 0.60 → 0.72 with match; counting 0.42 → 0.43. Still short of a two-words pass and costly for first words.
  Running: `organic-look2w` (pages twice the view: whole seen coarsely, fixations in full detail (CELL=64 TILE=52), and
  the gaze only to what stands apart (APART=0.5) → out/look2_probe_wide_s0.log); `organic-echo-0.01/0.001` (the ear's
  echo decorrelated before the context dendrites, offline mixed life, FULL=1 → out/text_echo_*.log; compare with
  out/text_full1.log: text beyond parrot 0.330, W2 0.432). Lives eyes2 at 5–10k (about 6,000 moments an hour).
  Next: read those; if wide pages + apart work → implement in the brain (the glimpse takes a scene larger than the
  view; world 4 pages at 128 px) and move eyes2 to world 4 when they pass first words. If the echo relay helps →
  into Words/SpeakingBrain and the next lives. Episodic store into a life: resume switch EPISODES=1 (not written yet);
  idea for kid-a: a per-conversation store so that it remembers what it was just told (frozen slow weights).
- 18:20 Results: (1) echo relay: negative (RESEARCH_MEMORY.md). (2) Fast mapping with the store holds on 3 seeds;
  a brain can be born with it or given it (EPISODES=1). (3) **Wide pages** (CELL=64 TILE=52: the page is twice the
  view, seen whole coarsely, fixations in full detail) + gaze only to what stands apart (APART=0.5), new eyes, seed 0:
  singles 0.54 → 0.52, a thing on a page 0.05 → 0.47, either of two 0.62, "is there" 0.61 → 0.75, counting 0.42 (base
  0.35). Wired into the brain (`SpeakingBrain._coarse/_glimpse`, `apart`) and world 4 (pages 128 px); a smoke life with
  WORLD=4 LOOKING=1 MATCH=1 EPISODES=1 FULL=1 and the new cortex ran clean (about 0.6 s a moment under load).
  (4) mouth_diag on world3_s0: names said = recalled 90–95% (the mouth's blends cost 0.02–0.04); text: the word
  cells' own recall runs on without stopping → the text-side fault to chase.
  Running: `organic-look2x0`, `organic-look2x1` (wide pages, arms whole, gaze+match, gaze+match+bursts; codes kept in
  out/tmp/look2_*.pkl so more arms take seconds) → out/look2_probe_wide_x{0,1}.log.
  Next: (a) pick graded or burst match; third seed. (b) **When eyes2 pass first words: stop each at that checkpoint
  and resume it with WORLD=4 LOOKING=1 MATCH=1 EPISODES=1** (same tag; the pipeline resumes from out/eyes2_s*.pkl;
  EXTRA="-E CHAT=0.2 -E FULL=1 -E LOOKING=1 -E MATCH=1 -E EPISODES=1", WORLD=4, CORTEX and CORTEX_LAYERS as now).
  Check first with fastmap_life.py and a look at checkpoints that looking does not cost first words in the life.
  (c) Text: why the word cells' recall does not stop (silence cell; context dendrite) — read Words.recall on W2
  curriculum items offline. (d) Counting.
- 19:20 Results: wide pages hold on seed 1 (page naming 0.53, "is there" 0.73–0.74; bursts no better than graded:
  keep graded). Survey docs/RESEARCH_TWO_WORDS.md → **counting moved out of two words** (world 4: page:count is
  preschool); intake limit for leisure talk (`SpeakingBrain.intake`, INTAKE=1; last two units: run-ons gone, a third
  of the −0.018 cost back). Leisure talk cost recorded (LEISURE.md). **eyes2 at 5,000 moments into first words: 0.39,
  0.40 beyond the parrot (six baseline lives: mean 0.35); COCO things 0.33/0.39 against 0.21, Commons 0.73/0.67
  against 0.58, CIFAR 0.34/0.35 against 0.41 (lower; unexplained).** s1 entered first words at 20,000 already at 0.41.
  Dry run of the resume into world 4 (old-cortex brain, LOOKING MATCH EPISODES INTAKE FULL, warnings as errors): clean,
  first words not hurt by looking (0.35 → 0.43 over 800 moments).
  **When an eyes2 life is PROMOTED out of W1** (line in out/eyes2_sN.log; its level is frozen to out/levels/):
  `systemctl --user stop organic-eyes2-sN.service`, then
  `TAG=eyes2 SEEDS="N" ARM=routes WORLD=4 CORTEX=data/visual_cortex2.pkl CORTEX_LAYERS=0 EXTRA="-E CHAT=0.2 -E FULL=1 -E LOOKING=1 -E MATCH=1 -E EPISODES=1 -E INTAKE=1" experiments/conv_pipeline.sh`
  (it resumes from out/eyes2_sN.pkl). Then note it here and in VISION.md.
  Open questions for the quiet hours (light on CPU, the lives come first): why CIFAR is lower in eyes2; pivot frames
  for two-word text; the relay learned online; pruning the orchestrator's clusters; the snapshot of eyes2 against
  the baseline at every checkpoint (table in VISION.md when 3 seeds have 3 checkpoints in first words).
