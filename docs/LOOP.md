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
   what a blind child gets (docs/JUDGING.md, 2026-11-19). **No pass mark** (Kris, 2026-11-21): new lives run
   `GATE=plateau`: the world moves on when a stage's curve stops rising. Stage W2 is called "pages".
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
- **Lives `plat_s0..2`**: from birth, GATE=plateau, TOTAL=120000 (started 2026-11-23 12:00; see the log). The ablation queue, the cleanw lives and the benchmark probes are all finished.
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
   selective but 658 pairs are near copies). Measured first by copies and selectivity, not by accuracy.
10. A fast store keyed by the fixated thing, not the whole view: fast mapping of a thing on a page, and a place for
    a new-word bias to arise.
11. The store's surprise measure: it reads 0.84 for everything; without it no replay rule can choose.
12. The yes bias as a developmental observation beside children's (+0.24 to +0.34 at two; ours +0.10 at one level).
13. An ablation per organ in lives, and the same lessons in shuffled order (the paper needs both).
14. A person's labels on 200 judged answers (Kris), and scores corrected by the judge's measured error rates.
15. Pivot frames for two-word text (docs/RESEARCH_TWO_WORDS.md), only after 7 says what is missing.

## Decisions waiting for Kris
- **The store keyed by what the eye is on** (`Episodes.key = "looked"`; VISION.md 2026-11-23): adopt in new lives?
- **LaTeX on the server** (none installed): install to compile the paper?
- The preprint's first draft is docs/paper/DRAFT.md: read it; the AI-use statement is yours to word. Kris has an
  arXiv account (2026-11-21). Before sending: the ablation arms (running, about 30 hours), related work with every
  reference opened by hand, the name and affiliation as they should appear, and whether arXiv asks for an
  endorsement in cs.NE when the submission is started.
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
  - **What development did** (`cortex_dev_probe.py`, VISION.md): kinds moved far, became selective, and 658
    pairs became near copies; nothing keeps kinds apart. Proposal with its organic check: lateral anti-Hebbian
    learning among the kinds (also the relay's local form). Not built.
  **Done:** docs/paper/DRAFT.md (first draft of the preprint, pending controls marked); `bench_readout.py` (control:
  is the lead the code's or the word cells'?) running; `LAST_STAGE` in the life script; two cortex twins built
  (never developed with its own relay; no relay); the ablation queue written and smoke-tested, started.
  Next: read `readout_*.log` and put the row in VISION.md and the draft; the anti-Hebbian probe on the benchmark's
  codes (copies and selectivity first); then queue item 8 (rebuild preschool's sources); read each ablation arm as
  it ends (out/tmp/ablation_queue.log).
- 2026-11-20 23:55 **Learned lateral inhibition, first form: negative** (`lateral_probe.py`, VISION.md): my own
  three rules made more copies than winner-take-all (1,377–2,453 pairs against 658) because the inhibition never
  settles (it grows without end). A fault of the formulation. Next: read the published rules (Földiák 1990;
  Zylberberg, Murphy & DeWeese 2011; Pehlevan & Chklovskii's similarity matching) in the papers and build the probe
  on one of them unchanged; table of copies and selectivity first. **Started: `organic-ablation-queue`** (waits for
  the cleanw lives, which end at about 02:00; smoke lives on both new cortices ran clean).
- 2026-11-21 00:30 **Readout control, seeds 0 and 1** (VISION.md, PUBLISH.md, the draft): a gradient softmax readout
  on the same relayed code does as well as the word cells (0.696 / 0.702 against 0.686 / 0.693). The lead over
  end-to-end backprop is the code's; the draft's abstract and first claim are reworded. Seed 2 and the random
  filters still running (`organic-bench-readout`).
  **Lateral inhibition on the published rules** (`lateral_probe.py` rewritten on Pehlevan & Chklovskii's
  Hebbian/anti-Hebbian network, equations read in the paper, arXiv:1503.00680): in a 600-picture smoke the
  inhibition stays bounded (mean 0.08, max 1.4) and the settled activities are decorrelated (top 1% of pairs 0.41),
  where my own rule ran away. **Running: `organic-lateral`** (16,000 pictures, seed 0 → out/bench/lateral_s0.log).
  Next: read its table (copies? selectivity?); only if the kinds are both selective and apart, CODES=1 and
  `bench_stream.py` with VARIANT=lateral, with and without the batch relay; the readout control's last rows into
  VISION.md and the draft; the cleanw verdicts (about 02:00) and the first ablation arm's start.
- 2026-11-21 09:15 **The loop was silent from 00:30 to 09:05** (fifth time; everything queued ran alone). Collected
  (VISION.md, 2026-11-21 morning): the readout control on three seeds and on random filters (the lead is the code's:
  gradient readout 0.694, word cells 0.687); the published Hebbian/anti-Hebbian network is stable and keeps the
  kinds' *activities* apart (0.40) but not their filters (1,204 pairs alike): the code to measure is the settled
  activity, not made yet; the cleanw lives ended as planned (two words 0.40 / 0.36 / 0.38 at the end, first words
  0.43 / 0.44 / 0.47); **ablation arm 1: with a never-developed cortex first words is the same, a thing on a page
  is not named (0.00–0.10 against 0.25–0.37, three seeds)**: development matters where the brain must look at one
  thing among several. Arm 1 ends at 50,000 (about 10:00); arm 2 (no relay) starts by itself.
  Next: `look2_probe.py` on the developed and the never-developed cortex (where does the gaze go; what does the
  fixated thing look like); the draft's "development buys nothing" reworded with this; the settled-activity code of
  the lateral network on a subset; then queue item 8.
- 2026-11-21 09:35 Kris asked what the "two words pass mark" means ("we're forcing the model to say a certain amount
  of words?"). Answered: no; it is a stage's name and the score at which the world moves on. **My recommendation,
  waiting for Kris's yes: no mark at all (the world moves on when the stage's curve stops rising), and the stage
  renamed "pages".** Not built.
  **Running: `organic-look-rand`** (`look2_probe.py` on the never-developed cortex, wide pages, seeds 0 and 1 →
  out/tmp/look2_random.log; then quarters+match on the developed one → out/tmp/look2_developed_quarters.log): the
  gaze itself is colour centre-surround and does not use the kinds, so the probe asks whether a fixated thing is
  named from never-developed kinds (developed: 0.47–0.53 on a page, 0.54–0.59 alone, 2026-11-17).
  **Queued behind it: `organic-lateral-queue`** (out/bench/lateral_queue.sh): the Hebbian/anti-Hebbian network
  developed again and kept (out/bench/lateral_s0.npz), the code made of its settled activities for a 10,000-picture
  stream, then `bench_stream.py` VARIANT=settled with and without a relay → out/bench/stream_settled_s0.log. To
  compare at 10,000 pictures: developed word cells 0.500 without relay, 0.623 with; streaming LDA 0.664.
  (`settle` was 60 times slower than needed under threaded BLAS: row layout and one thread.)
  Next: read both; the first ablation arm's last checkpoint and the no-relay arm's start.
- 2026-11-21 10:00 **Kris agreed: no pass mark, and the stage renamed.** Built: `GATE=plateau` (JUDGING.md, last
  section), W2 named "pages"; replayed on the held lives it leaves the stage after 40,000–45,000 moments at
  0.33–0.38; smoke life `organic-smoke-plat` started. Kris has an arXiv account. **Ablation arm 1 ended** (50,000
  moments): never-developed cortex, first words 0.41 / 0.41 / 0.41, pages 0.20 / 0.19 / 0.16 (control at 50,000:
  0.38–0.44 / 0.36–0.37). Arm 2 (no relay) started 09:33. Still running: `organic-look-rand`,
  `organic-lateral-queue`.
  Next: read the smoke, the gaze probe and the lateral queue; when the ablation arms are done, three lives from birth
  with `GATE=plateau` (the paper's staged-life numbers should come from the rule we keep).
- 2026-11-23 12:15 **Silent from 2026-11-21 10:00 to 2026-11-23 11:50** (Kris's own computer went off; the server did
  not: every queue ran to its end). Collected (VISION.md, 2026-11-23): **the ablation table, five arms x three
  seeds** (relay: the only part that costs first words; looking and the cortex's development: naming a thing on a
  page falls to 0.02 and 0.04; match neurons: "is there" on two seeds; the store: nothing these tests ask for, and
  page naming higher without it); the gaze probe does not reproduce the never-developed cortex's failure (0.37–0.45
  in the probe, 0.04 in the life): open; **the lateral network's settled activities give the word cells 9.5 points
  without any relay** (seed 0, 10,000 pictures). The draft has the ablation table.
  **Started: lives `plat_s0..2`** (organic-plat-s0/1/2) from birth with `GATE=plateau`, TOTAL=120000: the staged-life
  numbers under the rule we keep. Launch line:
  `TAG=plat SEEDS="0 1 2" ARM=routes WORLD=4 MAX_GB=5 TOTAL=120000 CORTEX=data/visual_cortex2.pkl CORTEX_LAYERS=0 EXTRA="-E CHAT=0.2 -E FULL=1 -E LOOKING=1 -E MATCH=1 -E EPISODES=1 -E INTAKE=1 -E GATE=plateau" experiments/conv_pipeline.sh`
  Kris is starting the arXiv submission form (endorsement check); the paper itself is not ready to upload.
  Next: (1) view-by-view diagnostic on out/ablrand_s0.pkl against out/cleanw_s0.pkl (why no page naming without
  development); (2) the store on pages (why higher without it); (3) related work, references opened one by one;
  (4) the lateral network on three seeds and the whole stream; (5) queue item 8.
- 2026-11-23 12:50 **View by view on saved brains** (`experiments/page_views_probe.py`, 100 test pages each;
  out/tmp/page_views.log, page_views2.log still running on eight brains): every brain names a fixated thing alone
  about as well (control 0.54, never-developed cortex 0.46, no store 0.51 of pages have a fixation that alone gives
  the right name), but as it lives the control names 0.36, the never-developed 0.15, the brain without a store 0.52.
  What it says instead is mostly the coarse whole page's answer. With the store silenced for the asking the control
  rises (seed 0: 0.36 → 0.48; seed 1: 0.45 → 0.49). In the code the store is written and read under the whole
  (coarse) view only (`organic/words.py`, `x[0]`): pages look alike coarsely, so it gives back another page's
  name. **Reading so far: it is the store's key, not the cortex's development, that fails on pages**; development
  only makes coarse pages less alike. Waiting for the never-developed brains' store-silent numbers before writing
  it into VISION.md. Then queue item 10 (the store keyed by what is fixated) gets its organic check and a probe.
  Lives `plat_s0..2` out of babbling at 5,000.
- 2026-11-23 13:30 **The store's key is what fails on pages: confirmed on three seeds** (VISION.md, 2026-11-23
  afternoon): with the store silent the control names 0.48–0.50 of test pages (0.36–0.45 with it) and the
  never-developed cortex 0.33–0.44 (0.07–0.15). The draft's reading of the ablation table is corrected (development
  is worth 0.05–0.10 on a fixated thing; the store as keyed costs page naming).
  **Built (off by default): `Episodes.key = "looked"`** (`organic/episodes.py`, `organic/words.py`): a moment is kept
  under the windows the eye fixated (the whole view when it fixated nothing) and compared with what the eye is on
  now. Organic check in VISION.md. Unit-tested. **Running: `organic-pageviews`** → out/tmp/page_views3.log: the three
  control brains and one never-developed, pages and single pictures, store as it is / silent / keyed by what the eye
  is on (their stores hold each moment's fixations, so nothing has to be lived again).
  Next: read it. If the keyed store does no harm on pages and none on single pictures: a probe of its own job on a
  page (a name heard once for a thing on a page, asked on another page with the same thing), then lives. Then
  related work for the draft.
- 2026-11-23 14:15 **The store keyed by what the eye is on does no harm** (VISION.md): on the saved brains pages are
  named as with a silent store (0.49 / 0.49 / 0.50; never-developed cortex 0.43), single pictures unchanged.
  Written: `experiments/fastmap_page_probe.py` (a name heard once for a thing on a page: asked on the very page, the
  picture on another page, the picture alone, another of the kind; word cells alone, store keyed by the whole view,
  store keyed by what the eye is on). Smoke (300 moments, keyed store): the very page 0.88, the picture on another
  page 0.88, the picture alone 0.16, another of the kind 0.00. **Running: `organic-fastmap-page`** (three seeds, 6,000
  moments → out/tmp/fastmap_page.log, about 1.5 hours). Lives `plat_s0..2` in first words at 15,000 (above the floor
  0.33 / 0.32 / 0.41); no stage left on a plateau yet (needs six tests in the stage).
  Next: read the probe. If the keyed store keeps a page's name and costs nothing: Kris's word, then lives with it
  (KEY=looked in the life script). Meanwhile: related work for the draft.
- 2026-11-23 15:10 **A name heard once on a page, three seeds** (VISION.md): keyed by what the eye is on, the store
  names the same picture on another page 0.85–0.92 (whole-view key 0.11–0.15; word cells alone 0.00); the picture
  alone and another of the kind stay at 0.00; old names untouched. Written: docs/paper/REFERENCES.md (28 references
  with links and how each is known, for Kris to open one by one). Kris asked what to do on arXiv: told (start the
  form in cs.NE to see whether an endorsement is asked; author line, licence, AI-use statement; nothing uploaded).
  Lives `plat_s0..2` at 20,000 in first words (0.35 / 0.33 / 0.41).
  **Waiting for Kris:** the keyed store in new lives (yes/no); installing LaTeX on the server (to compile the paper).
  Next: the plateau lives' first stage changes (this evening); figures for the paper from the logs; the lateral
  network on three seeds; queue item 8 (preschool's sources).
- 2026-11-23 16:00 Nothing new finished (lives at 25,000 in first words: 0.36 / 0.38 / 0.43, still rising; lateral
  network seed 1 being coded). Made: `experiments/paper_figures.py` → docs/paper/fig_benchmark.pdf/.png (accuracy
  against pictures seen, three seeds with their range; palette checked with the dataviz skill's validator, a marker
  per series). matplotlib added to the project's environment for it.
  Next: figure 2 (the stage curves) when the plateau lives have them; figure 3 (the ablation table as bars); read
  the lateral queue; the lives' first stage change (earliest at 45,000, about 19:00).
