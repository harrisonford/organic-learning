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
8. **Biology decides, not the score** (Kris, 2026-11-23 evening: "we want to improve ONLY organically, no
   performance hacking, in fact if there's something very biological that doesnt improve we implement anyways as
   it can make sense the longer we add biological mechanisms"). A mechanism that brains are well known to have is
   adopted on that ground, also when it moves no number today; it is still measured, taken out again and reported
   as what it did. The other half stands: nothing goes in because it scores. Sizes and architecture are revised
   from what is known of the brain and from recent work (Kris: a 2011 recipe is too old a ground), not from a
   search over scores. The learner stays on the processor (Kris: "its fine then use cpu").
9. **Iterate on the architecture in the loop; measure against the current paradigms; send the draft** (Kris,
   2026-11-24 10:35: "once you finish research and results you can iterate on architecture on a loop given you
   understand the philosophy on design go full organic learning over performance but we do measure performance
   and difference with current paradigms you can work on a loop and you send me the draft paper to my email").
   So: when the running results are read, the loop takes the architecture items one at a time (queue E 23–30),
   each after its organic check, adopted on its biological ground (rule 8). Every change is still measured, and
   measured beside what the field does now: on the CIFAR-10 stream beside the same-shape backprop and the
   streaming baselines, and in a life where a life is the test. The paper's draft goes to Kris by email
   (the address used for drafts 1–5) when it has changed materially; not on every round.

## Keeping the loop alive (2026-11-24)
The probes and lives are systemd user units with lingering on: they run without any terminal. The loop itself is
the Claude Code process, and it dies with the terminal it was started from (a dropped SSH connection stopped it
from 01:42 to 10:08 on 2026-11-24). So Claude Code runs inside `tmux` on the desktop (`tmux new -s organic`;
detach with Ctrl-b d; come back with `tmux attach -t organic`). After any restart of Claude Code (`--continue`
or `--resume`) the self-paced loop is not restored: the `/loop` line has to be typed again. A reboot of the
desktop ends the tmux session too.
herdr (v0.9.5, installed for the user on 2026-11-24 at Kris's request, from its GitHub release, checksum checked) does
the same as tmux with its server on the desktop: `herdr` starts or reattaches, Ctrl-b q detaches.

## Running
- **Lives `kind_s0..2`** (organic-kind-s0/1/2; started by `organic-reward-queue` when the plateau lives ended,
  2026-11-24): the reward lives (JUDGING.md, 2026-11-23); `ramp_s0..2` follow from the same queue.
- **`organic-rpe1`** (out/tmp/rpe_queue.sh; waits for expand2): queue E 29d in the small learner, three seeds: arm
  `all+rpe` → out/tmp/rpe_s{0,1,2}.log, then its fixed-gain control `all+gain` → out/tmp/rpe_gain_s{0,1,2}.log
  (the gain used: out/tmp/rpe_gain.log); about an hour a run. Baseline: the "all" arms in out/tmp/gaze1.log,
  gaze4.log, gaze4_s2all.log.
- **`organic-trace2`** (out/bench/trace2_queue.sh): the drift probe, seed 1 read, seed 2 developing →
  out/bench/trace_probe_s2.log, stream_{l2plain,l2trace}10k_s2.log.
- **`organic-relay1`** (out/tmp/relay_local_queue.sh; waits for trace2): queue E 26/28, the relay by a local rule
  (`experiments/relay_local_probe.py`): a smoke test → out/bench/relay_local_smoke.log, then seed 0 →
  out/bench/relay_local_s0.log (about 1.2 hours). Seeds 1 and 2 only after seed 0 is read.
- **`organic-tag1`** (out/tmp/tag_queue.sh; never edit while it runs): queue E 29e, the life-scale test of replay
  marked by reward. Lives continued from the plateau lives' levels as they left pages, 20,000 moments of
  preschool, one at a time and only with 5.7 GB available (so it waits behind the reward lives): `tagn_s*`
  (nothing), `tagr_s*` (`REPLAY=0.3 REPLAY_TAG=reward`), then `tagb_s*` (`REPLAY=0.3`); units
  organic-tagn-s1 … ; notes in out/tmp/tag_queue.log; records out/tag{n,r,b}_s*.jsonl. The first arm is also the
  smoke test of continuing from a level with today's code: look at its log when it starts.
- Ended as planned: lives `plat_s0..2` (120,000 moments; read in JUDGING.md, 2026-11-24). Finished and read: the
  lateral network on three seeds, backprop with the relay, the gaze probes gaze1–gaze6, norm1, trace1, diet1.
  Also finished and read: rest1 (normalisation seeds), diet2 (blurred start seeds), expand1 and expand2.
- Finished and read: the clean and cleanw lives, the five ablation arms, the benchmark probes, the readout control.
- Launch line for a life with no pass mark:
  `TAG=plat SEEDS="0 1 2" ARM=routes WORLD=4 MAX_GB=5 TOTAL=120000 CORTEX=data/visual_cortex2.pkl CORTEX_LAYERS=0 EXTRA="-E CHAT=0.2 -E FULL=1 -E LOOKING=1 -E MATCH=1 -E EPISODES=1 -E INTAKE=1 -E GATE=plateau" experiments/conv_pipeline.sh`

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
   **Surveyed 2026-11-23** (VISION.md, evening; `experiments/preschool_sources.py`): five faults, a proposal for
   world 5 (numbers as words, built sources drawn evenly, conversation lived and not tested). **Built the same
   evening: `WORLD=5`** (world 4 unchanged item for item: `experiments/world_hash.py`). Open: a life in it.
**D. Owed to the paper (the referee's list, 2026-11-23; controls on the references and the ruler, nothing in the brain)**
16. An end-to-end backprop reference with a whitening layer before its readout (the like-for-like comparison).
17. The lateral network on the whole stream, three seeds; the same rule among the summary's 4,096 numbers.
18. The benchmark tested on all 10,000 test pictures; streaming LDA without the relay on seeds 1 and 2.
19. How often a picture comes again in a life; a life with no going back to earlier stages; the shuffled order.
20. A picture-blanked learner as one more blind baseline; a person's labels on 200 judgements (Kris).
21. The new-class test with even counts, altered views of the shown picture, and beyond the store's capacity.
22. Done 2026-11-23: a drawing of the learner (`fig_architecture`, figure 1), a life's curves and examples (the
    showcase section), and an appendix with the settings, the compute and the judge's instruction.
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
**E. Kris's directions of 2026-11-23 evening (under rule 8; a survey first, then the organic check, then a probe)**
23. **A learned gaze.** Survey done (docs/RESEARCH_GAZE.md); plan and organic check in VISION.md (2026-11-23
    21:45). Done: the probe that the heard word leads the eye (three seeds); the serial gaze in the learner
    behind `gaze = "serial"` (life env `GAZE=serial`; default unchanged, checked by digest). Running: the small
    learner raised both ways (`organic-gaze1`). Owed: the same from a grown learner's checkpoint, both ways, when
    a life's worth of memory is free (about 1.6 GB); then lives with `GAZE=serial`. Later steps: a fast and a slow
    memory of a place's value; the fovea handing over whatever it is on (now only places that stand apart).
24. **The cortex's architecture and sizes from biology and recent work**, not from a 2011 recipe. Survey running
    (what is known of early visual cortex in numbers; local-rule vision models of 2020–2026). The same order.
25. **What recent papers (2022–2026) bring**, for the paper's citations and for mechanisms worth building. Survey
    running.
26. Both whitening steps by local rules (the paper's result rests on two batch stand-ins): the lateral network on
    the whole stream and among the 4,096 summary numbers (queue D 17), then the patches' whitening.
27. A faithful soft winner-take-all development of the filters (the published SoftHebb rule, unchanged), measured
    as the developed filters were: the paper's "Hebbian did no better than random" is so far a result about hard
    winner-take-all only (docs/RESEARCH_RECENT.md).
28. The relay by interneuron gains (Duong et al. 2023: online whitening with fixed synapses and gain-modulating
    interneurons, slow synaptic plasticity beneath), a published local mechanism, beside the lateral network.
29. **From the second survey (docs/RESEARCH_ARCH.md, 2026-11-24), in this order:** (a) a temporal term in the
    filters' rule: answer the next view of a thing as this one was answered (Halvagal & Zenke 2023), on the
    serial gaze's successive views; (b) a blurred start: the filters develop on waves, then on blurred pale
    pictures that sharpen (Lu et al. 2025); (c) critical periods in sequence when the cortex has stages;
    (d) surprise sets how much is learned (one broadcast surprise; errors divided by their own variance);
    (e) what comes back in sleep is tagged at the moment itself; (f) a sparse expansion before the word cells
    and as the store's key. Each after its organic check, as a probe, read first on what it does to the code.
30. **The eye's map without a hand-set weight** (Kris, 2026-11-24: "why fixed, research a better solution").
    Measured: the word's pull is 0.06 at the named place against a reflex term that varies by 0.19, so
    "0.5 × standing-out + pull" drowns the word (first look on the named thing 0.50–0.53; by pull alone
    0.81–0.85; any form free of the raw sizes 0.79–0.86). Survey running on how the brain combines them.

## Decisions waiting for Kris
Kris, 2026-11-23 17:10: "on your questions i follow your decision you can decide for me". Decided by the loop:
- **The store keyed by what the eye is on: adopted for new lives** (`-E KEY=looked` in the life script; running
  lives stay as they are). Reason: three seeds, takes the store's harm on pages away and lets it keep a name heard
  once on a page; a limit stated (a memory of what was fixated, as it was fixated).
- **LaTeX installed** (pacman: texlive-basic, -latex, -latexrecommended, -fontsrecommended; pdflatex, as arXiv
  compiles). Next: the draft into LaTeX.
- **Licence for the preprint: CC BY 4.0** (told to Kris; Kris is at the licence step of the arXiv form).
Still Kris's own:
- The arXiv form: after the licence and policy steps, choose cs.NE as primary; that page says whether an
  endorsement is needed. Nothing is uploaded until the paper is ready and every reference has been opened by hand
  (docs/paper/REFERENCES.md).
- The author line (name, affiliation or none, public contact address) and the AI-use statement in Kris's words.
- Releasing a level to kid-a; hardware (memory first); labelling about 200 answers for the judge's audit set.
- **Queued behind the plateau lives: `organic-reward-queue`** (lives `kind_s0..2`, then `ramp_s0..2`; JUDGING.md
  2026-11-23). Lives in world 5 with the keyed store come after it.
- **Launch line for new lives** (when the plateau lives end):
  `TAG=... SEEDS="0 1 2" ARM=routes WORLD=5 MAX_GB=5 TOTAL=120000 CORTEX=data/visual_cortex2.pkl CORTEX_LAYERS=0 EXTRA="-E CHAT=0.2 -E FULL=1 -E LOOKING=1 -E MATCH=1 -E EPISODES=1 -E INTAKE=1 -E GATE=plateau -E KEY=looked -E RPE=normalised" experiments/conv_pipeline.sh`
  (`RPE=normalised` since 2026-11-25: the reward's error against its usual size, three seeds with control, VISION.md)

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
- 2026-11-23 16:45 **Lateral network, seed 1: reproduced** (VISION.md, evening): without any relay the word cells
  gain 9 points on its settled code (0.532 → 0.619; seed 0: 0.500 → 0.595); with the batch relay on top +1.9 (seed
  0: +4.7); streaming LDA unmoved. Seed 2 is being coded. **Paper figure 3** (`docs/paper/fig_ablation.pdf`): the
  ablation arms as bars with the seeds as dots, read back from the lives' logs (same numbers as the table); the
  draft points to figures 1 and 3. **Queue item 8 surveyed** (`experiments/preschool_sources.py`): preschool is 15
  sources taking turns, 20 test items each; numbers answered as digits in one source and words in two; three built
  sources lopsided (blind child 0.56–0.60); three conversation sources that no answer key can judge. Proposal for
  world 5 in VISION.md. Lives `plat_s0..2` at 30,000 in first words (0.41 / 0.39 / 0.43, still rising; no plateau
  yet). "Running" above rewritten (it still listed finished work).
  Next: build world 5's preschool (branch on `world >= 5`, a test that world 4 is unchanged item for item); figure 2
  when a life has left a stage; seed 2 of the lateral network and the three-seed table.
- 2026-11-23 16:50 **World 5 built: preschool rebuilt** (`organic/curriculum.py`, `world >= 5` only; VISION.md):
  numbers as words in every source, the three annotation-made sources even over their answers (blind child 0.25 /
  0.50 / 0.33 instead of 0.57 / 0.60 / 0.56), conversation lived but kept out of the test (`Curriculum.talk`).
  World 4's fingerprint is the same before and after (`experiments/world_hash.py 4 5`); the three lives are not
  touched. No life in world 5 yet: the machine is full with the three plateau lives.
  Next: figure 2 when a life has left a stage (earliest about 18:30); seed 2 of the lateral network (about 18:00); then, when the
  plateau lives end, lives in world 5 (and the shuffled-order control).
- 2026-11-23 17:20 **Kris leaves the open questions to the loop** ("you can decide for me"). Decided: the keyed
  store is adopted for new lives (`KEY=looked`, added to `experiments/curriculum_life.py`; running lives untouched);
  LaTeX installed with pacman (pdflatex); licence CC BY 4.0 recommended to Kris, who is at that step of the arXiv
  form. Next for the paper: DRAFT.md into LaTeX (docs/paper/paper.tex), compiled here.
- 2026-11-23 17:35 **arXiv asks for an endorsement in cs.NE.** Kris has the endorsement code (kept out of this
  public repository; in the assistant's private notes). Plan and a draft note in docs/paper/ENDORSEMENT.md: ask
  authors the paper cites, one or two at a time, only once the PDF exists and Kris has read it; Kris sends the
  notes. This puts the LaTeX version at the head of the paper's queue.
- 2026-11-23 18:00 **The paper compiles**: `docs/paper/build.py` turns DRAFT.md into paper.tex and paper.pdf (pandoc,
  pdflatex; both installed with pacman, with poppler to look at pages): 7 pages, figures 1 and 3 placed where the
  text points to them. DRAFT.md stays the text that is edited. The two built files and the author's name are kept
  out of the public repository for now (`AUTHOR="..." python docs/paper/build.py`). Kris confirmed the ORCID
  (found by a public search at Kris's request) and two earlier papers; one former coauthor has enough computer
  science papers on arXiv to endorse, for Kris to check with arXiv's own link and to ask (names kept out of this
  repository). Kris is linking ORCID and Zenodo and will reserve a DOI without publishing.
  Lives `plat_s0..2` at 40,000 / 40,000 / 35,000, all still in first words and rising (0.43 / 0.46 / 0.44 above the
  floor; under the old mark they would have left at 15,000–20,000 with 0.27–0.29). The lateral network's seed 2 is
  coded; its stream runs.
  Next: the draft brought up to date (no pass mark, "pages", the keyed store, the lateral network's three seeds,
  world 5), the [check] marks worked off against REFERENCES.md, figure 2 when a life has left a stage.
- 2026-11-23 18:02 **The paper rewritten for readers** (Kris: dull title, cold clipped writing). Three independent
  readers (style survey, hostile referee, cold reader); the referee's nine points checked against the logs, all
  held, and the draft now says them (VISION.md 18:15; queue D above). Working title "It Was the Code, Not the Rule";
  10 pages; both PDFs sent to Kris by mail at Kris's request. **Lateral network, three seeds**: +9.2 points without
  any relay, four fifths of the batch relay's 11.4. arXiv needs an endorsement in cs.NE (code in private notes;
  docs/paper/ENDORSEMENT.md); Kris's ORCID confirmed and linked to Zenodo; a former coauthor is the likeliest
  endorser. **Running: `organic-showcase`** (`experiments/showcase_probe.py` on out/cleanw_s0.pkl →
  out/tmp/showcase.log, showcase_cleanw_s0.json/.npz) for the section Kris asked for: what the learner does,
  with pictures, a timeline, its size and its answering times.
  Next: read the probe; the showcase figures and section; then queue D in order (16 first).
- 2026-11-23 18:08 **The showcase section** ("What it looks like", Kris's request): `experiments/showcase_probe.py`
  on out/cleanw_s0.pkl (300 + 300 held-out items, nothing learns) and two new figures from
  `experiments/paper_figures.py`: `fig_life` (one life: the two curves, the stages, what it babbled and named at
  four ages) and `fig_examples` (the first six test pictures and pages with its answers, not chosen). Measured:
  **the learner holds 257 million numbers** (gating policy 128 M, word cells and store 61 M, ear 51 M, cortex and
  relay 17 M, mouth 0.2 M); **answering time**: a question about a picture is answered after 1.0–1.6 ticks of
  looking, a spoken lesson with no picture after 3.1–4.1; right and wrong answers take the same time; under one
  eye move a question. So there are two speeds, and no evidence yet that the longer wait is thinking (one learner;
  three seeds owed). Paper now 12 pages; sent to Kris.
  Picture licences to check before anything is public: the pages in fig_examples may hold COCO or Commons pictures.
  Next: three seeds of the showcase probe; queue D (16: the end-to-end reference with a whitening layer).
- 2026-11-23 18:19 **Kris: the term "organic learning" and the title** "Organic Learning: Small-Scale Architecture
  Experiments Without Backpropagation" (set; the term defined in the introduction as five commitments; the phrase
  is the title of a US patent on network architectures, 10,846,592, to be cited). **Kris: is the judging lenient
  early and stricter later, and is there a sweet spot?** Answered and built (JUDGING.md, 2026-11-23): form is
  already judged more strictly with age, meaning strictly at every age; a near-miss level added to the judge
  (`NEAR`; "woof" for a dog 0.99) and `REWARD=strict|lenient|ramp` in the life script; two arms queued behind the
  plateau lives (`organic-reward-queue`), tests always strict. **Kris: answering speed, by training, organically**:
  the reward is what sets it (the gating policy's patience), so the reward arms are the first experiment on it.
  Next: read the plateau lives; queue D 16.
- 2026-11-23 18:30 **The like-for-like reference started** (queue D 16): `experiments/bench_backprop.py` has `RELAY=fixed`
  (the relay step settled on 6,000 unlabelled pictures seen through the filters as born, kept, errors going back
  through it) and `RELAY=tracked` (settled again every 5,000 pictures). The gradient was checked numerically in
  double precision, with and without the relay (largest relative difference 4e-4 on the filters, 1e-8 on the
  readout): the first reference's gradient is right too. Running: `organic-bench-bprelay`. Lives `plat_s0..2` at
  40,000, still in first words (0.43 / 0.46 / 0.49).
  Next: read the tuning (six runs of 10,000 pictures), then the three seeds; the lives' 45,000 and 50,000 tests;
  the showcase probe on seeds 1 and 2 when memory allows.
- 2026-11-23 19:25 **How often a thing comes again in a life** (queue D 19, first half; VISION.md 19:20): photographs
  0.1–1.9 times in 35,000 moments of first words, the adults' spoken lessons 5.6 times, social routines 66 times;
  the paper says so. **Backprop with the relay, tuning so far**: 0.597 at 10,000 pictures (0.52 without; word
  cells with the relay 0.623): the lead shrinks to a few points, as the referee expected; four tuning runs and six
  full runs to come. An explainer page for Kris (published as a private artifact; source docs/paper/explainer.html).
  PUBLISH.md brought up to date. Lives `plat_s0..2` at 45,000, still in first words (0.44 / 0.40 / 0.47; seed 1
  dipped). Memory: 3.5 GB free with the backprop queue beside the lives.
  Next: the tuning's end and the full runs; the lives' first stage change; the showcase probe on seeds 1 and 2
  when the backprop queue is done (memory).
- 2026-11-23 20:20 **The lives with no pass mark are still rising in first words** after 40,000–45,000 moments
  there: above the floor 0.49 (seed 0, 55,000 lived) / 0.46 / 0.50 (50,000 lived); the last three tests are above
  the three before by 0.04–0.05, the chance error of that difference is 0.03, so no stage has been left yet. The
  earlier lives were sent on after 5,000–10,000 moments at 0.27–0.29. At this pace a life of 120,000 moments ends
  about 06:00, and it may end still in first words: then the finding is the curve itself, and the pages stage needs
  longer lives. **Backprop with the relay, tuning**: fixed 0.597 / 0.572 / 0.527 at rates 0.001 / 0.003 / 0.01;
  tracked 0.602 / 0.564 (0.01 running). The best rate is the lowest tried, so a second queue
  (`organic-bench-bprelay2`) widens the grid after the first: a reference tuned at the edge of its grid would not
  be a fair one. **A drawing of the learner** for the paper (queue D 22): `fig_architecture`, each part filled by
  how it comes to be (local rule online / batch before learning / reward / fixed reflex); the paper has 13 pages.
  Memory 2.5 GB free: nothing heavier was started.
  Next: the backprop full runs (about 100 minutes each, six of them, then the wider grid); the lives' stage
  change or end; settings table, compute and the judge's instruction for the paper; the showcase probe on seeds 1
  and 2 and queue D 18 when memory allows.
- 2026-11-23 21:08 **The first life to leave a stage with no pass mark**: `plat_s1` left first words at 55,000
  lived (45,000 in the stage), level 0.45; its last three tests were above the three before by 0.025, under the
  chance error 0.030. Its last single test was its best (0.49), so this may be the chance leave foreseen at 20:20
  (the dip at 45,000 sits in the newer window). Seeds 0 and 2 are still there (0.49, 0.48; margins 0.049, 0.033).
  **The paper's appendix** (queue D 22, now done): settings table, compute, the judge's instruction word for word.
  Collecting the settings turned up something the draft had never said: **each word-cell synapse's step shrinks
  with its own use**, 1/(1 + u/5) with u the running sum of that input's squared activity (`maturation` in
  organic/words.py): an AdaGrad-like step. It is local, and now stated in the paper's description of the word
  cells; it bears on the reading of "the rule is a delta rule" and of the relay (a per-input step already evens
  out inputs of different size, which the relay then does for their correlations). Two wrong values caught while
  writing the table (ticks at most 12, not 6; store 2,000, the 6,000 is only with replay). **A layout fault
  fixed**: a page with a table and floating figures ran past its bottom margin and lost text (pandoc's longtable
  does not see floats); docs/paper/build.py now turns the tables into ordinary ones; 14 pages, no overflow.
  Backprop with the relay: tuning done (tracked 0.01: 0.533), full runs started with rate 0.001.
  Next: the full runs; `plat_s1` in the pages stage, the other two leaving; queue D 17–18 when memory allows.
- 2026-11-23 21:25 **Kris read draft 5 and gave directions** (rule 8 and queue E above): improve only organically,
  and adopt what is well established in biology also when it moves no number; a learned gaze; the cortex's sizes
  and architecture from biology and recent work; recent papers; the learner stays on the processor (the graphics
  card is full with the two adult models, and a moment's time is spread over many small steps: CRAFT.md,
  2026-11-18). Paper: the patent note moved to the end, the world-reverting bug taken out of the mistakes (Kris:
  bugs are not research mistakes; it touched no life the paper reports), the judge told as the wrong assumption
  it was, the repository's link, where the sizes come from. Three surveys run in the background (gaze; early
  visual cortex in numbers and recent local-rule vision models; papers of 2022–2026).
  What the code does today, read for the gaze: "looking" is not a sequence of eye movements. In one glimpse the
  word cells get the whole view and up to four fixated windows at once, and each cell takes the view that drives
  it most. So there is no order of looking, no cost of looking and no choice to learn. A learned gaze starts by
  making looking serial: one fixation a tick, the next place chosen.
  Next: read the three surveys; the organic check for a serial, learned gaze; then the probe.
- 2026-11-23 21:27 **The survey of recent work is back** (docs/RESEARCH_RECENT.md; queue E 25). **The paper's main
  finding is known from the other side**: Thiry et al. 2021 (whitened patches and a linear classifier match
  trained networks on CIFAR-10; training the patches changes nothing), Prabhu et al. 2024 (random features,
  decorrelation and class means in one pass match continually trained networks; the decorrelation matters most),
  Dalm et al. 2024 (decorrelated inputs speed up backprop itself). The draft now says so in the introduction and
  cites 26 recent papers in related work (all in REFERENCES.md, none opened by a person yet); 15 pages. What stays
  ours: the finding reached from a learner built of local rules, the like-for-like controls in a one-pass stream,
  the staged life. **"Hebbian is no better than random" is scoped to hard winner-take-all**: SoftHebb's own
  control says the same for hard WTA and reports a gain for soft WTA, which we never built faithfully.
  **Owed from this** (queue E): 27. a faithful soft winner-take-all development of the filters (the published
  rule, unchanged), measured as the developed filters were; 28. the relay by interneuron gains (Duong et al.
  2023), a published local and online whitening, beside the lateral network.
  Next: the two other surveys (gaze; early visual cortex), then the organic checks.
- 2026-11-23 21:45 **The gaze and cortex surveys are back** (docs/RESEARCH_GAZE.md, docs/RESEARCH_CORTEX.md) and
  the plan with its organic checks is in VISION.md (21:45). Gaze: make looking serial (one fixation a tick, the
  periphery coarse), let the heard word's cells bias the priority map (no new rule), a three-factor value rule
  on the choice of place, staying by the patience that exists. Cortex, by how established and how far off we
  are: divisive normalisation, complex cells and stages, several patch sizes, a retina in place of batch
  whitening, soft competition, filters before experience, no shared weights.
  Next: `experiments/look3_probe.py` (the serial gaze as a probe on pages with a saved learner; memory: check
  look2_probe's need first); the backprop full runs; the lives.
- 2026-11-23 22:02 **Backprop with the relay, the whole stream, seed 0: 0.682** (fixed relay, rate 0.001;
  out/bench/backprop_relay_full.log). Without the relay the same seed had 0.627; the word cells with the relay
  have 0.687 ± 0.005 over three seeds. One seed, but it says what the literature says: given the same
  decorrelated code, end-to-end backprop is level with the local learner. The paper's pending mark carries it.
  Lives at 60,000: `plat_s1` in pages (0.17 above the floor on its first test there; first words 0.45);
  seeds 0 and 2 still in first words (0.46, 0.49).
  **The first gaze probe is written and running**: `experiments/look3_probe.py`, unit `organic-look3` (capped
  at 1 GB; seeds 0–2, 6,000 moments each, about 15 minutes a seed) → out/tmp/look3.log. The word cells are
  raised as today (the whole view and all fixated places at once); then, nothing learning, the looks are taken
  one at a time in three orders: the reflex's, the heard word's (the place whose coarse view most drives the
  cells of the words just heard), random. Read: on which look the named thing is in the fovea; the answer with
  the first one or two looks.
  Next: read the probe; if the word leads the eye, the serial gaze goes into the learner behind a switch; the
  backprop seeds 1 and 2.
- 2026-11-23 22:29 **The heard word leads the eye** (VISION.md 22:29; two seeds, the third running): with the
  looks taken one at a time, the named thing is in the fovea on the first look on 0.82 / 0.85 of pages when the
  places are ordered by how much their blurred view drives the cells of the words just heard, against 0.46 / 0.46
  in the reflex's order and 0.38 / 0.41 at random. No new rule: the word cells' own synapses. Found in the code:
  the learner has two eyes; the gating policy already learns eye commands by reward on the old one, and the word
  cells are served by the new one, which does not move. Joining them is the step.
  Running: `organic-look3b` (after `organic-look3`): how sure the word cells are look by look, stopping by a
  threshold, how sure before and after the named thing enters the fovea → out/tmp/look3b.log.
  Next: read look3b; then serial looking in `organic/speech.py` behind a switch that leaves today's behaviour
  untouched by default (the reward lives start by themselves when the plateau lives end and must stay comparable).
- 2026-11-23 23:45 **Both gaze read-outs are in, three seeds** (out/tmp/look3.log, look3b.log; VISION.md 23:45).
  The heard word's order puts the named thing in the fovea on the first look on 0.82 / 0.85 / 0.81 of the pages
  where it matters (reflex 0.46, random 0.38–0.41), and answers right with one look on 0.67 / 0.70 / 0.71
  (reflex 0.60 / 0.61 / 0.63). **When to stop is the weak part**: how sure the word cells are rises when the
  named thing enters the fovea (0.39 → 0.43, 0.37 → 0.43, 0.42 → 0.46) but separates right from wrong answers
  only a little (0.44 against 0.39), and for "how many" hardly at all (0.34 against 0.33); stopping at a level of
  sureness saves almost no looks in any order. Counting gains nothing from the order: every place must be seen.
  **The serial gaze is in the learner, behind a switch** (`organic/speech.py`, `gaze = "serial"`; life env
  `GAZE=serial`): the word cells first get the whole view only; each eye command of the gating policy that lands
  on a place that stands apart brings them what the fovea has there; the map the commands choose on is biased by
  what stands out, by the heard words' pull and against places already seen; the adult's reply (or a comment)
  draws the eye once before the word cells learn. The eye commands are the gating policy's, credited by reward as
  they always were (eligibility × reward prediction error: a three-factor rule already). Default behaviour is
  untouched: the same digest of 300 moments before and after the change.
  **Running: `organic-gaze1`** (the small learner, both ways, three seeds). Not possible yet: the same from a
  grown learner's checkpoint (memory: 3.3 GB free beside three lives, a checkpoint needs about 1.6).
  Backprop with the relay, seed 1: **0.690** (seed 0: 0.682); the paper's pending mark carries both.
  Lives: `plat_s2` left first words at 60,000 lived (exact grade 0.54); `plat_s0` is still in them at 70,000
  (0.54–0.56); `plat_s1` in pages, exact grade 0.43 at 65,000, `plat_s2` 0.42 on its first test there.
  (Corrected 2026-11-24: these are the log line's exact grades, which I had called "above the floor".)
  Next: read gaze1; the backprop third seed and the tracked relay; queue E 24 (the first cortex probe: divisive
  normalisation) when the gaze probe has freed its memory.
- 2026-11-23 23:48 **The first cortex probe is written and queued** (queue E 24, item 1; `organic-norm1`, after
  the gaze probe): divisive normalisation in place of "above the mean, rectified". R = D² / (σ² + pool), D the
  rectified drive times the window's contrast, the pool all kinds at the place (and, in a second arm, the
  surround), σ² the median pooled energy on unlabelled pictures: nothing searched. Read first for what it does
  to the code (sparseness over kinds and over places, response against contrast), then the stream. Under rule 8
  the stream's number does not decide adoption. The gaze probe was restarted with seed 0 alone so that this can
  follow it sooner.
  Next: read gaze1 (seed 0, both ways) and decide on seeds 1–2; read norm1.
- 2026-11-24 00:58 **The serial gaze as first built is never used** (VISION.md 00:58; gaze1, seed 0): the small
  learner answers at about the third tick, before its first eye command can be given, so it answers pages from
  the coarse view: pages 0.44 against 0.56 with the lives' looking at 6,000 moments, "is there" 0.55 against 0.67;
  looks before answering 0.01. And in both arms the eye commands die out by 3,000 moments: a gaze taught by
  reward and the cost of time alone learns not to look. The build had the reply draw the eye by reflex and not
  the question, which is not what is known (a heard word draws the eye within 200 ms, before any answer).
  Changed: `look_on_question` (the question draws the eye once, one tick). `organic-gaze2` reruns the serial arm
  after checking the default path's digest; `organic-norm1` waits behind it.
  **Backprop with the relay, three seeds: 0.688 ± 0.005** (0.682, 0.690, 0.691) against 0.687 ± 0.005 for the
  word cells: level. In the paper, with the wider tuning grid and the tracked relay still pending.
  Lives at 70,000–75,000: `plat_s0` still in first words (exact grade 0.52 at 75,000); pages `plat_s1` 0.47,
  `plat_s2` 0.45 (exact grades; corrected 2026-11-24, I had written "above the floor").
  Next: read gaze2 and the digest; seeds 1–2 of the three arms if seed 0 says something; read norm1.
- 2026-11-24 10:13 **The loop did not wake between 01:42 and 10:08** (Kris asked for the status at 10:08; nothing
  was read in between; the units ran by themselves). What was found:
  **The plateau lives ended as planned at 120,000 moments** (the health check's "stopped without a verdict" is
  how a life that reaches TOTAL looks; none was killed). All three left every stage with no pass mark: first
  words after 70,000 / 45,000 / 50,000 moments there, pages after 30,000 / 35,000 / 40,000, and ended in
  preschool. Above the floor (the records' `above`; the three numbers first written here were exact grades,
  corrected the same morning): first words when left 0.53 / 0.49 / 0.49; pages when left 0.33 / 0.31 / 0.36;
  **preschool never rose above blind answering** (0.00 / 0.02 / 0.02 after 10,000 / 30,000 / 20,000 moments
  there), and while the learner was there first words fell to 0.42 / 0.39 / 0.39 and pages to 0.27 / 0.25 / 0.30. **The reward lives started by themselves** (`kind_s0..2`, in the babbling stage).
  **Backprop with the relay settled again along the stream: 0.671 / 0.689 / 0.693** (mean 0.684); the wider
  tuning grid found nothing better than rate 0.001, batch 32 for either relay. The comparison is closed: 0.688
  ± 0.005 (fixed relay) against 0.687 ± 0.005 for the word cells.
  **Serial gaze with the question drawing the eye** (gaze2, seed 0): the learner now looks before it answers
  (0.87–0.93 looks), pages 0.50 at 6,000 moments (all at once 0.56; without the question's look 0.44), "is
  there" 0.62 (0.67; 0.55). **But the first look is on the named thing on only 0.46–0.51 of pages: the reflex's
  level, not the 0.82 of the read-out.** Suspected: the pull's size against the reflex's fixed half (the
  read-out ordered by pull alone); to be measured, not guessed.
  **The default path is unchanged**: the digest check as queued compared against a copy of the old code that
  could not find the adults' wordings (data/ is found beside the code), so its two lines differed; with the
  data linked both give c52819a8c3a93a7b.
  **`organic-norm1` was killed for memory** (settling the relay needs more than its 1.3 GB cap); restarted at
  10:12 with 2.6 GB (4.6 GB are free: the new lives are young).
  Next: the pull's size in the serial gaze; read norm1; the plateau lives into JUDGING.md and the paper.
- 2026-11-24 10:34 **Kris: "why fixed, research a better solution … did we find more interesting sota research
  that we can take architecture ideas from".** Two surveys run and written: how the map combines its sources
  (RESEARCH_GAZE.md, second part) and architecture ideas of 2023–2026 (RESEARCH_ARCH.md; queue E 29). The fixed
  half is gone: each source's places compete, the map is their weighted average, the weights are learned from
  reward on a fixed total, born with the reflex alone (VISION.md 10:34; queue E 30). Default digest unchanged.
  `organic-gaze3` running. Divisive normalisation, the code itself (norm_probe.log, seed 0): sparseness over the
  kinds at a place 0.83 against 0.71 now, kinds above half the strongest 3.9% against 8.5%, over places for a
  kind 0.87 (0.90 with the surround) against 0.72; the response at a quarter and half the contrast is 0.13 and
  0.41 of that at full (now: 1.00 and 1.00, no dependence on contrast at all). The streams are running.
  Next: read gaze3 and the streams; plateau lives into JUDGING.md; then queue E 29a (the temporal term).
- 2026-11-24 10:36 **Rule 9 from Kris** (above): iterate on the architecture in the loop, organic over
  performance, measured beside the current paradigms; drafts by email. **Draft 6 sent** to Kris (15 pages: the
  backprop comparison closed, the plateau lives, without the gaze and cortex work, which have one seed).
  The plateau lives are written into JUDGING.md (2026-11-24).
  Next: read gaze3 and the three streams of the cortex probe; then queue E 29a.
- 2026-11-24 11:16 **The word's weight in the eye's map grows from reward** (gaze3, seed 0; VISION.md same time): born 0,
  it is 0.15 / 0.46 / 0.58 / 0.75 after 1,000–4,000 moments of pages; the first look lands on the named thing on
  0.65 of pages (reflex 0.49); with one look before answering the serial learner is level with the one handed
  every place (pages 0.53 against 0.56, "is there" 0.66 against 0.67). One seed. `organic-gaze4`: a control with
  the pull shuffled over the places, then seeds 1 and 2.
  **Divisive normalisation on the stream, seed 0**: level with the present response (word cells with the relay
  0.624 against 0.613; ridge 0.654 against 0.659), the code sparser and saturating with contrast; the surround
  pool a little weaker (0.610). Seeds 1 and 2 queued behind the gaze runs.
  The health check no longer calls a life that ran to its planned end a death (out/monitor/ended.json).
  Next: queue E 29a, the temporal term: organic check and design while the probes run.
- 2026-11-24 11:33 **The drift probe is written, smoke-tested and queued** (`organic-trace1`, behind `organic-gaze4`;
  VISION.md 11:28). Smoke test on 40 fixations only (says nothing yet): answers to a picture and to it two
  pixels aside alike 0.854 in the first stage, 0.773 (plain) and 0.794 (trace) in the second.
  The shuffled-pull control so far: the pull's weight 0.01 and 0.00 at 2,000 and 3,000 moments (the real arm:
  0.01 and 0.15).
  Next: read gaze4 (control, seeds 1–2), then the normalisation seeds, then trace1.
- 2026-11-24 12:28 **The shuffled-pull control fails the first rule for the map's weights**: with a pull that says nothing
  its weight still rose to 0.53 by 6,000 moments (real: 0.75). The claim of 11:16 that the word's weight "grows
  from reward" is withdrawn (VISION.md, same time). The rule had a selection bias (the highest place always taken,
  eligibility against the plain mean), a floor that noise climbs, and a rate far too fast. Corrected: the look
  drawn by priority while living, eligibility against the draw's own average, the slow rate of a stable value
  (0.001). `organic-gaze4` and the first rule's seeds stopped; `organic-gaze5` (slow pair), then `organic-trace1`
  (the drift probe), then `organic-rest1` (fast pair, seed 2's baseline, normalisation seeds 1–2).
  Default digest unchanged (c52819a8c3a93a7b).
  Next: read gaze5; the drift probe.
- 2026-11-24 13:28 **The corrected rule at the slow rate, real arm** (gaze5_slow.log, seed 0): the pull's weight is 0.000 /
  0.001 / 0.005 / 0.008 / 0.010 at 4,000–12,000 moments: slow and steady, about what was estimated; the first look
  stays the reflex's (0.49–0.51); pages 0.59 and "is there" 0.75 at 12,000 with 0.86 looks before answering. The
  shuffled control is running; the reading waits for it. Reward lives at 25,000: all three in first words
  (exact grades 0.43–0.46).
  Next: the control; then the drift probe starts by itself.
- 2026-11-24 14:30 **Reward does not teach the map's balance** (VISION.md, same time): corrected rule, slow rate, 12,000
  moments: the pull's weight 0.010 real against 0.006 shuffled; first look at the reflex's level in both. Negative,
  kept with its control. **The balance is now set by its time course** (`gaze_balance = "time"`): what stands out
  is a transient (gone a quarter of a second after the scene appeared), the heard words' pull stays; nothing
  learned, no weight of mine. Default digest unchanged. Running: `organic-gaze6`; `organic-trace1` (the drift
  probe, first stage read: answers to a picture and to it two pixels aside alike 0.847); `organic-rest1` behind.
  Next: read gaze6; three seeds of serial (time) against all at once; the drift probe.
- 2026-11-24 15:18 **The balance by its time course, seed 0** (gaze6; VISION.md same time): the first look goes to the named
  thing on 0.51 → 0.70 of pages as the word cells learn (reflex 0.49), nothing learned about the balance; pages
  0.50 against 0.56 all at once, "is there" 0.68 against 0.67, with 0.86 looks before answering. Seeds 1–2 queued
  behind the drift probe; the fast learned-weights pair dropped (settled). Drift probe: plain arm done (second
  stage 0.799 alike under a two-pixel shift, first stage 0.847); trace arm developing.
  Next: read trace1 (trace arm, streams); then rest1.
- 2026-11-24 16:18 **The drift probe, seed 0** (VISION.md same time): the trace raises the second stage's tolerance to a
  two-pixel shift (0.826 against 0.799 without it; the first stage's pooled output 0.847); on the stream the
  second stage helps the word cells (plain 0.646, trace 0.631, first stage alone 0.613) and not the ridge (0.655,
  0.640, 0.659). Carried forward on its ground with its cost stated; seeds 1–2 queued (`organic-trace2`).
  `organic-rest1` is on the gaze seeds.
  Next: read rest1 (gaze seeds 1–2, seed 2's baseline, normalisation seeds); then the paper's new section when
  three seeds are in; then queue E 29b (a blurred start).
- 2026-11-24 16:48 Kris: "ok keep going keep on looping and improving". **Queue E 29b written and running** (`organic-diet1`;
  VISION.md same time): the filters develop on a maturing eye's view (waves, then pictures sharpening and gaining
  colour); read on the filters themselves, the stream, and the stream with blurred test pictures for both cortices.
  Next: read rest1 (gaze seeds), diet1; then trace2 runs.
- 2026-11-24 17:31 **The serial gaze is level with all at once on three seeds** (VISION.md same time): pages 0.57 against 0.55,
  "is there" 0.71 against 0.70, with 0.87 looks before answering; first look on the named thing 0.68–0.70.
  **A blurred start, seed 0: nothing moved** (0.616 against 0.613 sharp, 0.585 against 0.588 blurred); kept as
  measured, seeds 1–2 running. Kris asked how to keep the loop running without the laptop: answered (tmux on the
  desktop and Remote Control; "Keeping the loop alive" above).
  Next: rest1 (normalisation seeds), trace2, diet2; then the paper's section on looking and on the cortex from
  biology, and draft 7 by email.
- 2026-11-24 17:40 The loop runs inside herdr on the desktop now (Kris moved the session; the process no longer hangs on an
  SSH connection). **The paper has a section on looking one place at a time** (three seeds of the small learner,
  the read-out, the timing account) and a fourth mistaken assumption (that reward could teach the eye what to
  trust; the shuffled control). Six references added to REFERENCES.md, unopened. Normalisation: seed 1 coded
  (sparseness 0.825 against 0.711, as seed 0), streams running; blurred start seed 1 developed (filters 0.281
  against 0.297 cycles a pixel, colour 0.593 against 0.558).
  Next: the streams of seeds 1–2 (normalisation, blurred start), trace2; then the paper's section on the cortex
  from biology and draft 7 by email.
- 2026-11-24 18:46 All healthy, 4.0 GB free; streams of seeds 1–2 still running (normalisation seed 1 repeats seed 0: word
  cells with the relay 0.636 against 0.616, without 0.529 against 0.522, the ridge without the relay 0.637
  against 0.644; blurred start, seeds 1–2: filters again coarser and more coloured, 0.281 and 0.275 against 0.297
  and 0.295 cycles a pixel). Reward lives at 50,000 moments: all three still in first words (exact grades
  0.48 / 0.52 / 0.53). **Queue E 29f written and queued** (`organic-expand1`; organic check in VISION.md, same
  time): a wide sparse layer between the relay and the word cells, sized from the mushroom body and the dentate
  gyrus; smoke test on 1,000 pictures: the sparse layer loses more than its width gives back (says little yet).
  PUBLISH.md has today's state. The Running section above was brought up to date.
  Next: read the normalisation and blurred-start seeds, write the paper's section on the cortex from biology
  (the drift stage marked as one seed until trace2 is in), email draft 7; read expand1; then the organic check
  for queue E 29d (surprise sets how much is learned).
- 2026-11-24 19:37 **Divisive normalisation and the blurred start on three seeds** (VISION.md, same time; 10,000 pictures).
  Normalisation: the code changes as the physiology says on every seed (sparseness 0.71 → 0.83; a response
  that grows with contrast); word cells with the relay 0.630 against 0.620, the ridge 0.650 against 0.664 (down
  on all three); the surround pool adds nothing. Adopted for the next cortex on its ground, nothing claimed.
  Blurred start: filters coarser and more coloured every time, the stream level sharp and blurred (0.620 and
  0.620; 0.580 and 0.583). **The sparse layer, two seeds** (expand_probe_s{0,1}.log): two pictures half as alike,
  three times the directions; naming new pictures falls (0.61 → 0.57 at ten times the cells, 0.51 with as many
  cells) and a picture seen once is remembered better (0.88 → 0.95); no inhibition: no change. It belongs in
  front of the store, not the word cells. A first test of it as the store's key
  (`experiments/expand_store_probe.py`, three seeds) is too easy to tell keys apart, and shows the sparse layer
  does not replace the store's hand-set steepness. **The paper has the section "Giving the cortex what a cortex
  has"** (17 pages; seven references added, unopened; the sparse layer on two seeds and the drift stage on one,
  marked). Trace2: seed 1's first stage read (0.849 alike under a two-pixel shift).
  Next: seed 2 of the sparse layer and the wiring with 100 inputs, then draft 7 by email; trace2; the organic
  check for queue E 29d.
- 2026-11-24 20:13 **The sparse layer on three seeds** (VISION.md, same time): with seven inputs a cell, naming new pictures
  0.578 against 0.620 at ten times the cells and a picture seen once 0.95 against 0.85. **With a hundred inputs
  a cell (seed 0) the cost on naming is gone** (0.612 against 0.613 at ten times the cells; seen once 0.98;
  slower at 500 pictures): the reading "not in front of the word cells" is taken back to undecided; seeds 1–2
  queued (`organic-expand2`). Paper's paragraph rewritten to say so. **Queue E 29d designed and queued**
  (`organic-rpe1`; organic check in VISION.md): of "surprise sets how much is learned" only one part was not
  already settled: the gating policy's reward error is now, behind a switch, taken against its own usual size
  (adaptive coding by dopamine neurons, in the form of divisive normalisation); default digest unchanged; the
  fixed-gain control runs with it. Trace2: seed 1's plain arm developed (0.811 alike under a two-pixel shift;
  first stage 0.849).
  Next: expand2's two seeds into the paper, then draft 7 by email; rpe1; trace2.
- 2026-11-24 21:06 **The sparse layer with a hundred inputs a cell, three seeds** (VISION.md, same time): at ten times the
  cells the word cells name new pictures at 0.609 against 0.620 and remember a picture seen once 0.975 against
  0.849; slower at 500 pictures (0.26 against 0.39); the ridge 0.616 against 0.664. Decided: not put into the
  learner (a life's words are the stream's first 500 pictures; the store is the learner's memory of moments);
  kept as measured; no further wirings (that would be a search). **Draft 7 (17 pages) emailed to Kris.**
  Running: `organic-rpe1` (seed 0, first arm), `organic-trace2` (seed 1: trace arm 0.837 alike under a
  two-pixel shift, plain 0.811, first stage 0.849; streams next), the reward lives.
  Next: read rpe1 and trace2 as they come; reward lives at 75,000; then the serial gaze from a grown learner's
  checkpoint when 1.6 GB is free; queue E 29e (what comes back in sleep is tagged at the moment): organic check.
- 2026-11-24 22:14 All healthy, 2.5 GB free. **Read:** the drift stage on seed 1 (the trace gives back the tolerance a plain
  second stage loses, 0.837 against 0.811, first stage 0.849; on the stream the two do not differ: 0.626 and
  0.624, ridge 0.639 and 0.635); the reward's error against its usual size on seed 0 (the normalised error stays
  near 0.45 while the raw one grows from 0.22 to 0.37, so the step falls by itself from 1.9 to 1.3 times the raw
  one; grades as the baseline and the fixed-gain control within one seed's noise); the reward lives at 65,000
  (two in pages, as the plateau lives at the same ages: no difference from a kinder reward so far; JUDGING.md).
  **Queue E 29e: organic check written, built behind a switch, test queued** (VISION.md, same time): a stored
  moment keeps the reward that came with it and only such moments return in sleep (`Episodes.tag`, life env
  `REPLAY_TAG=reward`; default digest unchanged). From the plateau lives' records it would make the earlier
  stages about four parts in ten of what returns (one in ten when replay was tried) and remove talk from sleep
  by itself. The test is at life scale, in preschool where the loss is real (`organic-tag1`, waits for memory
  behind the reward lives). Expectation written down: a small effect at most.
  Next: rpe1 seeds 1–2 and trace2 seed 2 as they come (then the paper's drift row, three seeds); reward lives at
  75,000; while tag1 waits: queue E 26/28 (the relay by local rules: the lateral network among the summary's
  numbers, interneuron gains), which needs little memory; the serial gaze from a grown learner's checkpoint
  when memory allows.
- 2026-11-24 23:11 All healthy, 2.2–2.4 GB free (at the floor: nothing heavy started). Read: the reward's error on seed 1
  (as the baseline; control for it running); the drift stage's seed 2 is in its trace arm; the reward lives at
  70,000 (two in pages, one in first words). **Queue E 26/28: organic check, probe written and queued**
  (VISION.md, same time): the relay learned online by a published interneuron circuit (Lipshutz et al. 2022 with
  the leak of Duong et al. 2023b; equations read from the papers tonight). The cell's leak and the fact that
  inhibition only takes away put the floor at the mean variance, which measures 0.00094 of the largest where my
  hand-set `shrink` was 0.0005: the constant is not needed. The gain-only circuit (28 as written) would need 8.4
  million interneurons: not built. `organic-relay1` runs behind trace2.
  Next: read the smoke test and seed 0 of relay1 (then seeds 1–2, and the pictures of development coded so the
  developed arm does not use the stream's own); rpe1 seeds; trace2 seed 2 into the paper's drift row; reward
  lives at 75,000; tag1 waits for memory.

