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

## Running
- **Lives `clean_s0..2`** (units organic-clean-s0/1/2, started 2026-11-19 22:20, from birth): world 4, new cortex and
  relay, looking, match neurons, episodic store, intake limit, full-size corrections, leisure talk 0.2; **no novelty
  drive, no replay; echo a period; the strict judge; the adult always recasts; GATE=above (0.25)**. Budget 100,000
  moments a stage, memory guard 5 GB. Launch line:
  `TAG=clean SEEDS="0 1 2" ARM=routes WORLD=4 MAX_GB=5 CORTEX=data/visual_cortex2.pkl CORTEX_LAYERS=0 EXTRA="-E CHAT=0.2 -E FULL=1 -E LOOKING=1 -E MATCH=1 -E EPISODES=1 -E INTAKE=1 -E GATE=above" experiments/conv_pipeline.sh`
  (it resumes from out/clean_sN.pkl). They need nothing from the loop; they stop by themselves on a verdict.
  What to read at each checkpoint: `above` by stage and source, `yes_bias`, the store.
- **`organic-bench-codes`** → out/bench/ (codes of CIFAR-10 by a cortex developed on unlabelled CIFAR-100 pictures,
  and by a never-developed one; seed 0). Then `experiments/bench_stream.py`.

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
9. The relay learned online by a local anti-Hebbian rule (it is computed in batch today: a debt on every result).
10. A fast store keyed by the fixated thing, not the whole view: fast mapping of a thing on a page, and a place for
    a new-word bias to arise.
11. The store's surprise measure: it reads 0.84 for everything; without it no replay rule can choose.
12. The yes bias as a developmental observation beside children's (+0.24 to +0.34 at two; ours +0.10 at one level).
13. An ablation per organ in lives, and the same lessons in shuffled order (the paper needs both).
14. A person's labels on 200 judged answers (Kris), and scores corrected by the judge's measured error rates.
15. Pivot frames for two-word text (docs/RESEARCH_TWO_WORDS.md), only after 7 says what is missing.

## Decisions waiting for Kris
- PASS_ABOVE (0.25 provisional): at it, seeds' two-words levels read 0.22, 0.27, 0.32.
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
