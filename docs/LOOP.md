# The work loop (from 2026-11-19): state, queue, log

Kris, 2026-11-19: "fix and improve and even propose new ideas we can implement in a loop, also research for similar
studies of my architecture and ponder if it's relevant enough what we have currently to be able to publish in
arxiv or a journal somewhere; you can work very long on this."

**Read this file first at every iteration; update it last.** The earlier loop's log is docs/DAYLOOP.md (closed).
The loop's wakeups fire only while the session is alive: nothing a life needs may depend on the loop.

## Standing rules (Kris's, in force)
1. Organic first: no hand-wired behaviour. A mechanism that wires a signal to a behaviour to raise a score is
   artificial (the novelty drive was; retired). Then: is it data, training quality, or size?
2. What the brain does on the way (babbling, echo, routes, yes bias, the store's contents) is observed, never a
   target or a gate (docs/CRAFT.md, 2026-11-18).
3. Answers are judged for whether they are acceptable, lenient on form and strict on meaning; scores count above
   what a blind child gets (docs/JUDGING.md, 2026-11-19). PASS_ABOVE = 0.25 is provisional, Kris's to set.
4. Claims need 3 seeds and the 300-item tests; negatives are recorded as carefully as positives.
5. Every decision into the docs and memoize; commit and push; health check before any status.
6. Machine: keep 2 GB free; probes in capped units at Nice=15, at most one or two beside three lives; never kill by
   pattern. External requests carry no personal or project names. The OpenRouter key is never printed; Jev <= $1.50.

7. **No performance hacks; slow organic wins** (Kris, 2026-11-20: "NEVER FORGET we dont want to hack performance
   for the sake of it we want to build slow organic wins"). Nothing is proposed or built because it raises a number:
   not more filter kinds, not tuning to a benchmark, not a trick from the literature. The public benchmark is a
   ruler for checking what we claim; nothing in the brain is changed to move it. Size is the last question, asked
   only when a measured need points to it. (I had proposed 1,024 filter kinds for the benchmark: withdrawn.)

## Running
- **Lives `cleanw_s0..2`** (organic-cleanw-s0/1/2): the clean lives' first-words levels held in two words
  (PASS_ABOVE=0.99) to the stage's budget; they end by themselves at about 115,000–120,000 moments with a "stops at
  two words" verdict (expected: they were held on purpose). Curves in VISION.md (2026-11-20, night).
- **`organic-ablation-queue`** (out/tmp/ablation_queue.sh; log out/tmp/ablation_queue.log; about 40 hours; needs
  nobody): waits for the cleanw lives, then five arms one after the other, three seeds each from birth, the clean
  lives' launch line with one part taken away, kept in two words (`LAST_STAGE=W2`) to 50,000 moments:
  `ablrand` (never-developed cortex, data/visual_cortex2_random.pkl), `ablnorelay` (data/visual_cortex2_norelay.pkl),
  `ablnostore`, `ablnolook`, `ablnomatch`. Control: clean_s* (to first words) + cleanw_s* (two words). It adds its
  lives to out/monitor/lives.json. What to read: `above` for first words and two words at 50,000 beside the
  control's (0.38–0.44 / 0.36–0.37).
- **`organic-bench-readout`** → out/bench/readout_developed.log, readout_random.log (a gradient softmax readout on
  the very codes the word cells get, with and without the relay; three seeds; about three hours).
- Launch line for a clean life:
  `TAG=clean SEEDS="0 1 2" ARM=routes WORLD=4 MAX_GB=5 CORTEX=data/visual_cortex2.pkl CORTEX_LAYERS=0 EXTRA="-E CHAT=0.2 -E FULL=1 -E LOOKING=1 -E MATCH=1 -E EPISODES=1 -E INTAKE=1 -E GATE=above" experiments/conv_pipeline.sh`

## Queue (in order; the paper's needs are in docs/PUBLISH.md)
**A. The controlled core on a public benchmark** (offline; decides whether the efficiency claim survives)
1. `bench_stream.py` on seed 0: word cells with and without relay and store against nearest class mean, streaming
   LDA and ridge, developed and random filters; accuracy against pictures seen; one-shot at a lag of 20.
2. Online backprop of the same shape on the same stream, tuned for one pass (learning rate, batch size), and with
   replay at matched updates (extend `backprop_reference.py`).
3. The episodic store on raw and on backprop features (does the fast/slow dissociation need our cortex?).
4. Five seeds (cortex seeds 0–2 at least); CIFAR-100 if CIFAR-10 holds.
**B. The life**
5. Read the clean lives: do three seeds pass first words and two words above the floor with nothing artificial?
6. The loss at a stage change: compare a two-words level with its +5,000 checkpoint answer by answer (which answers
   replace the old ones); then the stage mix (70% new, 30% review, halving with each stage back): what would a
   home be like where naming never stops? Training program before mechanism.
7. Text is at the parrot's level (above the floor: 0.00–0.12): look at what the word cells' context side learns
   from a text lesson, item by item, before proposing anything.
8. Preschool's test is mostly answerable blind (floors 0.5–0.75): rebuild the stage's sources before any brain is
   asked to pass it.
**C. Ideas proposed (each needs the organic check written down before it is built)**
9. **Next to build (as a probe first):** lateral anti-Hebbian learning among the cortex's kinds: one local rule for
   what the batch relay does and for what development lacks (VISION.md, 2026-11-20 night: developed kinds are
   selective but about 330 pairs are near copies). Measured first by copies and selectivity, not by accuracy.
10. A fast store keyed by the fixated thing, not the whole view: fast mapping of a thing on a page, and a place for
    a new-word bias to arise.
11. The store's surprise measure: it reads 0.84 for everything; without it no replay rule can choose.
12. The yes bias as a developmental observation beside children's (+0.24 to +0.34 at two; ours +0.10 at one level).
13. An ablation per organ in lives, and the same lessons in shuffled order (the paper needs both).
14. A person's labels on 200 judged answers (Kris), and scores corrected by the judge's measured error rates.
15. Pivot frames for two-word text (docs/RESEARCH_TWO_WORDS.md), only after 7 says what is missing.

## Decisions waiting for Kris
- PASS_ABOVE (0.25 provisional): the two-words curves flatten at 0.31–0.41; 0.35 would be passed by all three seeds
  after 20,000–30,000 moments in the stage; or leave a stage when its curve stops rising (VISION.md, 2026-11-20).
- The preprint's first draft is docs/paper/DRAFT.md: read it; the AI-use statement is yours to word.
- Releasing a two-words level to kid-a: wait for the clean lives.
- Hardware: memory first; the lives are bound by memory speed (docs/CRAFT.md).
- Labelling about 200 answers by hand for the judge's audit set.

## Log
- 2026-11-19 22:30 Loop opened. Done this evening: judge audit and the strict judge; recast in teaching; blind
  floors and `above` in every record; GATE=above; echo a period; novelty drive retired; three literature searches
  (research_notes/Relatives and publishability 2/); docs/PUBLISH.md; three clean lives started; benchmark codes
  running; `bench_stream.py` written, not yet run.
- 2026-11-19 23:00 Clean lives at 5,000: babbling passed by seeds 0 and 2 (0.20, 0.19), seed 1 at 0.17. Benchmark
  codes for the developed cortex done (seed 0); the never-developed one is being coded. **Running:
  `organic-bench-dev0`** (`bench_stream.py`, developed, seed 0, 50,000 pictures → out/bench/stream_developed_s0.log;
  about 4 hours beside the lives; smoke at 1,000 pictures: word cells + relay 0.49, nearest class mean + relay 0.49,
  streaming LDA 0.50, ridge 0.49–0.51). **Queued behind it: `organic-bench-bptune`** (out/bench/tune_backprop.sh:
  `bench_backprop.py`, random filters, rate x batch on the first 10,000 pictures → out/bench/backprop_tune_s0.log).
  Next: read both; run the best backprop setting on the full stream, with replay 1 and 4, and with the developed
  filters as start and frozen; `bench_stream.py` with VARIANT=random when its codes exist. When memory allows
  (3 GB): queue item 6 with `judge_audit.py` on out/eyes2w_s2.pkl and out/eyes2w_s0.pkl (the brains after 100,000
  moments of preschool) against their two-words levels, answer by answer.
- 2026-11-19 23:30 **Queue item 6 answered in part** (RESEARCH_MEMORY.md, last section): the answers a brain loses
  in preschool went (1) to preschool's own larger set of names for the same pictures (couch for bed, orange for
  apple) and (2) to the question said back, which the old judge rewarded. Not simple erasure. Next on it: the clean
  lives' `says_last` in preschool; a near-miss reading of the early tests.
  **Benchmark, first numbers (seed 0, developed cortex, CIFAR-10, one pass):** word cells + relay 0.395 / 0.491 /
  0.543 / 0.599 / 0.623 / 0.662 / **0.686** after 500 / 1,000 / 2,000 / 5,000 / 10,000 / 20,000 / 50,000 pictures;
  without the relay 0.581 at the end. The other arms are still running. The "seen once, 20 later" reading (0.84)
  is not the fast-mapping test (the classes are known by then): to be redone with a class held back from the stream.
  Clean lives: all three out of babbling (5,000, 10,000, 5,000). Random-filter codes done.
  Next: read `stream_developed_s0.log` and `backprop_tune_s0.log`; then `VARIANT=random`; then the full backprop runs.
- 2026-11-20 00:05 Clean lives: the echo period lived (0.94, 0.91, 0.84 at its one checkpoint), all three in first
  words from 10,000–15,000; naming above the floor already 0.03–0.07 from comments alone.
  **Benchmark, seed 0, developed cortex, after 50,000 pictures (5,000 in brackets):** streaming LDA + relay 0.741
  (0.619), streaming LDA 0.722 (0.623), ridge 0.709 (0.607), **word cells + relay 0.686 (0.599)**, word cells 0.581
  (0.486), nearest class mean + relay 0.567 (0.549), nearest class mean 0.312. So the word cells are 2–5 points
  under the best gradient-free linear readouts on the same code (which keep and invert a 4,096 x 4,096 matrix, and
  whose shrinkage was picked on the test pictures); the relay is worth 10 points to them. ridge+relay still running.
  Written: `experiments/bench_newclass.py` (a class held back, then shown 1 / 5 / 20 times: the benchmark's version
  of fast mapping; replaces the "seen once, 20 later" reading). **Queued, each waiting for the one before:**
  `organic-bench-bptune` (after dev0) → out/bench/backprop_tune_s0.log; `organic-bench-rand0` (after dev0) →
  out/bench/stream_random_s0.log; `organic-bench-newclass` (after rand0) → out/bench/newclass_s0.log.
  Next: read them; a benchmark section in VISION.md and the efficiency claim in PUBLISH.md rewritten on these
  numbers; full backprop runs at the tuned setting (replay 0, 1, 4; developed start; frozen).
- 2026-11-20 01:00 **Clean lives: seeds 0 and 2 PROMOTED out of first words at 15,000 moments** (5,000 in the stage;
  above the floor 0.27 and 0.29; beyond the parrot 0.44 and 0.46, so the old mark too). Seed 1 entered first words
  at 15,000. On entering two words both say "no" to every "is there" question (yes bias −0.53).
  **The measure refined** (JUDGING.md, last section): `above` is read on the questions a parrot cannot answer,
  beside the best of always-yes, always-no and the commonest answer. Correction: the brains do know some text
  (0.11–0.33 of such questions). Existing levels: first words 0.29 / 0.46 / 0.43, two words 0.23 / 0.35 / 0.32.
  Smoke life clean. **`organic-relaunch-clean`** (out/tmp/relaunch_clean.sh, one-shot) restarts each clean life one
  minute after its next checkpoint so that it runs the refined measure; log out/tmp/relaunch_clean.log.
  **Benchmark table, seed 0** → VISION.md (last section): word cells + relay 0.686; streaming LDA + relay 0.741;
  random filters as good as developed ones (0.691); backprop tuning 3 of 9 settings done (0.40–0.48 at 10,000).
  Still running: `organic-bench-bptune`, `organic-bench-rand0`; `organic-bench-newclass` waits for rand0.
  Next: finish reading those; full backprop runs at the best setting; PUBLISH.md's claims rewritten on the table
  (what is ours is no longer "beats other gradient-free readouts", and "development" is not a selling point).
- 2026-11-20 07:15 **The loop was silent from 01:00 to 06:55** (third time: the wakeups stop when the session drops;
  it does not run inside tmux). What ran by itself meanwhile:
  - **All three clean lives passed first words (15,000 / 20,000 / 15,000) and two words (25,000 / 35,000 / 25,000)**
    at the provisional mark: two words above the floor 0.26, 0.28, 0.27 (text 0.19–0.27, a thing on a page 0.26–0.34,
    "is there" 0.24–0.27), 10,000–15,000 moments in the stage. Levels: out/levels/clean_s*_W2.pkl. Then preschool,
    where nothing rose (above 0.00–0.03) and first words and two words slid (0.42 → 0.26; 0.27 → 0.19) as before,
    with no parrot's reward this time (`says_last` 0.02–0.10 throughout). **The mark of 0.25 let them leave two words
    when they had only begun it**: the earlier brains stood at 0.23–0.35 after 30,000–45,000 moments.
  - Benchmark seed 0 finished (VISION.md): backprop grid 0.40–0.53 at 10,000 (best at the grid's edge); random
    filters as good as developed; the new-class test (word cells 0.00, with the store 1.00; nobody reaches the kind).
  **Done now:** the three lives stopped in preschool at 45,000–50,000 (pickles kept: out/clean_s*.pkl; the stage's
  test is mostly answerable blind and was to be rebuilt anyway). **Started: `cleanw_s0..2`** (units
  organic-cleanw-s0/1/2) from the clean lives' first-words levels, held in two words (PASS_ABOVE=0.99) for the
  stage's 100,000 moments: how far does two words go, and where should its mark be? **Started:
  `organic-bench-queue2`** (out/bench/queue2.sh, about eleven hours, needs nobody; progress in out/bench/queue2.log):
  wider backprop grid → the best setting on the whole stream (replay 0, 1, 4; developed start; frozen) → seeds 1
  and 2 of codes, streams, backprop and the new-class test (→ backprop_tune_s0.log, backprop_full.log,
  stream_seeds.log, newclass_seeds.log).
  Next: read queue2 as it goes and build the three-seed table; the cleanw curves at each checkpoint; then queue
  items 8 (rebuild preschool's sources) and 9 (the relay learned online).
- 2026-11-20 07:20 Kris: an arXiv preprint alone may be the goal ("it could be just an arxiv publication right?");
  and a correction: no performance hacks (standing rule 7; the 1,024-kinds run is withdrawn, never queued). The
  wider backprop grid's first point (rate 0.003, 64 a step): 0.484, under the best so far (0.527): the grid is
  flattening near 0.5 at 10,000 pictures. The organic question the benchmark does raise: **why does the cortex's
  Hebbian development buy nothing over random filters?** (queue, before any thought of size).
  Plan for the preprint: when queue2's three-seed tables and the two-words curves are in, a first draft for Kris.
- 2026-11-20 23:40 **The loop was silent from 07:20 to 23:15** (fourth time; the queue and the lives ran alone, as
  planned). Collected:
  - **Benchmark, three seeds** (VISION.md): word cells + relay 0.687 ± 0.005 (random filters 0.693 ± 0.002); one-pass
    backprop of the same shape 0.616 ± 0.023, with one replayed picture per new one 0.661, with four 0.688; streaming
    LDA + relay 0.739; word cells without relay 0.591. The lead over backprop is the relay's. New class: as seed 0.
  - **Two-words curves** (VISION.md): flatten at 0.41 / 0.31–0.36 / 0.36–0.41 after 20,000–30,000 moments in the
    stage; first words keeps rising meanwhile (0.28–0.40 → 0.40–0.45).
  - **What development did** (`cortex_dev_probe.py`, VISION.md): kinds moved far, became selective, and about 330
    pairs became near copies; nothing keeps kinds apart. Proposal with its organic check: lateral anti-Hebbian
    learning among the kinds (also the relay's local form). Not built.
  **Done:** docs/paper/DRAFT.md (first draft of the preprint, pending controls marked); `bench_readout.py` (control:
  is the lead the code's or the word cells'?) running; `LAST_STAGE` in the life script; two cortex twins built
  (never developed with its own relay; no relay); the ablation queue written and smoke-tested, started.
  Next: read `readout_*.log` and put the row in VISION.md and the draft; the anti-Hebbian probe on the benchmark's
  codes (copies and selectivity first); then queue item 8 (rebuild preschool's sources); read each ablation arm as
  it ends (out/tmp/ablation_queue.log).
