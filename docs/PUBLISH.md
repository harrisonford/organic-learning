# Is there a paper here? (assessment, 2026-11-19)

Kris asked whether what exists is relevant enough to publish on arXiv or in a journal. This is my assessment after
two rounds of literature search (reports/Backprop free developmental learning survey.md; research_notes/Relatives
and publishability 2/: closest relatives read in full, exemplar papers and venues, the literature on judging
answers). It is a judgement, kept up to date by the loop (docs/LOOP.md).

## Short answer

Yes, there is a paper, and it is not ready. What exists today would be accepted as a poster or a two-page abstract
at a developmental or NeuroAI venue. For arXiv plus a journal (TMLR is the fit), three to five weeks of machine time
on controls are missing, most of all one controlled experiment on a public benchmark. Nothing found in either
search does what this brain does: name held-out real photographs with no backprop anywhere, each moment seen once.

## What is ours to claim (and how firm it is)

| result | number | firm? |
|---|---|---|
| Single pass, local rules, real photographs: held-out naming among 16–25 names | 0.52 (3 seeds), chance 0.05–0.09 | firm on our own pictures; no public benchmark yet |
| The same shape trained by backprop on the same moments | 0.39 after one pass, 0.50 after five | lightly tuned (one learning rate); must be tuned for one pass, and a replay version added |
| What the word cells could not use: decorrelation between cortex and word cells | 0.39 → 0.52 (3 seeds); a ridge on the same code 0.56 | firm; the decorrelation is computed in batch (a debt) |
| Fast and slow memory: naming a picture seen once | 0.9 with the episodic store, 0.1 without; another picture of the kind 0.0 | firm (3 seeds); expected by theory; needs the same store on raw and backprop features |
| One brain through babbling, echo, first words and two words | 3 of 3 seeds | weak as evidence: the stages are ours, the passes were made with a hand-wired drive and a judge that taught mistakes; three clean lives started 2026-11-19 |
| Honest negatives | old cortex's development worse than random filters; a second layer +0.01; dendritic branches; replay did not protect; no new-word bias; text answers no better than a parrot; preschool not passed | firm, and an asset |
| Method | a retention gate on a behaviour that was not a target; a model judge that taught the learner its mistakes; a world that silently reverted | worth a section: nobody reports these |

## Where it sits

- **Closest in spirit**: the assembly-calculus language models (Hebbian, no backprop; toy input, pre-formed
  "vision") and the Berlin brain-constrained Hebbian models (fast mapping after a babbling stage; six words as random
  patterns). Both are proofs of concept on made-up input. Ours is on real photographs and adult-written text.
- **The comparison a reviewer will reach for**: BabyCL (June 2026), the child's-view contrastive model trained in
  one chronological pass: 27.5 on a four-way choice (chance 25) naively, 43.4 with replay buffers holding 60% of the
  stream, 57.3 offline with 400 epochs. It says one-pass learning is hard for backprop, which is our niche.
- **The warning**: "Alignment Without Binding" (workshop, 2026): strictly local credit fitted image–caption pairs
  and barely transferred; a frozen random encoder did better. We need the random-filter row in every table.
- **What the field accepts** (twelve exemplars read): 3–5 seeds with spread, a same-shape backprop baseline, an
  ablation table, at least one public benchmark, stated failures. Small teams and single authors do get in (TMLR
  2026 has a single-author controlled delta-rule study). A small Hebbian forgetting paper from a university team
  was rejected by TMLR the same year.

## What the paper is

Not "an alternative to backprop" (the scaling evidence says no) and not "a model of the child" (our input is not a
child's). It is: **what each part buys in a learner that sees each moment once and uses only local rules**, in two
halves.

1. **A controlled core on a public benchmark.** One stream, one pass, accuracy against samples seen, five seeds:
   - the learning rule: ours, against online backprop of the same shape tuned for one pass, with and without replay;
   - the readout on the same features: word cells, nearest class mean, streaming discriminant analysis, ridge;
   - the features: developed filters, random filters, with and without decorrelation;
   - the episodic store: on and off, also on raw and on backprop features (does the dissociation need our cortex?).
2. **A staged life with the same parts**: three to five clean seeds through first words and two words, scored above
   the blind floor, with the observations (echo fading, yes bias, the new-word bias absent) and the failures
   (earlier stages lost at a stage change; text at the parrot's level; preschool).

## What is missing, in order

1. The public-benchmark experiment (CIFAR-10 and CIFAR-100 as one-pass streams; all offline; days, not weeks).
2. The backprop reference tuned for one pass (learning rate, batch size), and with replay at matched updates.
3. Gradient-free readouts on our features (nearest class mean, streaming discriminant analysis).
4. An ablation per organ in lives (relay, episodic store, looking, match neurons, growth, chooser).
5. The same lessons in shuffled order.
6. Five seeds for every headline number.
7. Forgetting reported the standard way (backward transfer), with a small-rehearsal baseline.
8. The decorrelation learned online by a local rule (or stated plainly as computed in batch).
9. A person's labels on the judge's audit set.

## Where and when

- **Now, as it stands**: nothing archival. A two-page CCN abstract (due 1 April 2027) or an ICLR 2027 workshop
  paper (lists out 29 November 2026, papers due about 1 February) would likely be accepted and are compatible with a
  later journal paper.
- **When 1–6 exist**: Zenodo snapshot of code and data; arXiv in cs.NE (a personal endorsement is the gate: write
  individually to a few authors of the papers we build on, with the draft and the code); TMLR (median about four
  months to publication; near even odds in my and the searcher's judgement).
- **Not compatible with TMLR**: ICDL full papers and CCN eight-page papers are archival. CogSci's six pages
  (due 1 February 2027) is a grey area.
- **Disclosure**: the paper needs a section saying that an AI assistant wrote the code and ran the experiments,
  what Kris decided and verified, and that every reference was opened by hand. arXiv's section chair has announced a
  one-year ban for hallucinated references; the notes mark which numbers were verified.

## My recommendation

Do the controlled core first (it is cheap and it decides whether the efficiency claim survives a tuned baseline),
keep the clean lives running beside it, and decide about venue when the table exists. If the tuned one-pass
backprop baseline catches the brain, the paper becomes the fast/slow dissociation plus the staged life and the
negatives, which is still a workshop paper and an honest preprint, not a journal paper.

## Update, 2026-11-20: what the first benchmark seed does to the claims

(CIFAR-10 as a one-pass stream, seed 0: docs/VISION.md, last section. One seed; the tuned backprop runs are not in.)

- **Gone as a claim: "our learner is the best without gradients".** On the same code streaming LDA and ridge are 2
  to 5 points above the word cells (0.74 against 0.69 after 50,000 pictures). The word cells are an online, local
  approximation of what those compute with a matrix inverse; that is the honest sentence.
- **Gone as a selling point: the cortex's development.** Random filters on the same decorrelated windows do as
  well (0.691 against 0.686). What matters in the visual path is the decorrelation (windows, and the relay: +10
  points for the word cells), not the Hebbian development of the kinds. The paper must say so and carry the
  random-filter row everywhere.
- **Standing, for now: single-pass efficiency against backprop of the same shape** (0.62 against 0.40–0.48 after
  10,000 pictures in the first three settings). It waits for the rest of the tuning grid and for replay.
- **Standing: the decorrelating relay** as the one architectural finding that transfers to a public set.
- **To be measured on the benchmark:** fast and slow memory with a class held back (`bench_newclass.py`).

So the paper's centre of gravity moves: from "a brain that learns without backprop" to **"what a single-pass
local learner needs: decorrelation at two places and a fast store beside slow weights; what it does not need:
developed filters; and what it still cannot do"**, with the staged life as the second half. It is a smaller claim
and a more defensible one.
