# What a learner that sees each moment once needs: decorrelation and a fast store

**First draft for Kris, 2026-11-20. Not for circulation.** Written by the assistant from the project's records
(docs/VISION.md, docs/JUDGING.md, docs/RESEARCH_MEMORY.md, docs/CRAFT.md). Every number is from a log in `out/`.
Marked **[pending]**: runs that are queued or running tonight. Marked **[check]**: a reference not yet opened by
hand for this draft. Nothing here was tuned to raise a number; the benchmark is a ruler (docs/LOOP.md, rule 7).

---

## Abstract

We describe a small learner built only from local learning rules, with no backpropagation anywhere, that sees each
moment of its experience once: an ear, a visual cortex developed without labels, a decorrelating relay, word cells
that learn by a delta rule on two dendrites, a small episodic store, and a mouth that babbles before it speaks. We
ask what each part buys, in two settings. On CIFAR-10 presented as a single-pass stream, the word cells reading the
relayed code name 0.687 ± 0.005 of held-out pictures (three seeds), against 0.616 ± 0.023 for backpropagation
through a network of the same shape tuned for one pass; backpropagation needs each picture about five times to draw
level. The lead is the relay's: without decorrelation between cortex and word cells the same cells reach 0.591.
Hebbian development of the cortical filters buys nothing over random filters (0.693 ± 0.002), and readouts that keep
second moments and invert a matrix stay five points ahead (0.739). A class shown once is not held by the slow word
cells (0.00) and is held with the episodic store (1.00), with the old classes untouched; no learner tried
generalises from a few pictures to the kind. In a staged life with real photographs and adult-written speech, three
seeds pass from babbling through first words to two-word exchanges with no hand-wired drive, scored above what a
blind child would get; earlier stages improve while a later one is lived. We report what failed as carefully: replay
from the store did not protect earlier stages, a second cortical layer added nothing, no new-word bias arose, and
the stage after two words was not passed. We also report three errors of method that inflated earlier results.

## 1. Introduction

Learning in animals is local, online and single-pass: a synapse changes by what its two ends do, and no moment is
lived twice. Backpropagation is none of these. A large literature asks whether local rules can replace it at scale;
the present evidence says not yet [check: SoftHebb; scaling studies]. We ask a smaller question that the evidence
does not answer: **in a learner that sees each moment once and uses only local rules, which parts carry the
learning, and which do nothing?**

The learner was built slowly, part by part, each part admitted only if it is something a brain is known to do and if
it left the behaviour to emerge (no rule that wires a signal to a behaviour). It was raised in a staged world:
babbling, echo, first words, two-word exchanges. That history is the second half of this paper. The first half
puts the same parts on a public benchmark beside references a reader will ask for.

What we claim:

1. **Decorrelation between the visual code and the associative cells is what makes one pass enough** (+10 points
   for the word cells on CIFAR-10, +13 on the life's own photographs), and with it a local delta rule is ahead of
   same-shape backpropagation at equal exposure, most of all early (0.40 against 0.24 after 500 pictures).
2. **A fast store beside slow weights gives one-shot memory of an instance without disturbing what is known**, and
   gives no memory of a kind. This is the complementary-learning-systems prediction, measured on a public set with
   other learners beside it.
3. **Developed filters are not needed**: random filters on decorrelated windows do as well. We had expected the
   opposite.
4. **A staged life can be passed without hand-wired drives**, if what the learner does on the way (babbling, echo,
   a bias to "no") is observed and never made a target.

What we do not claim: an alternative to backpropagation at scale; a model of the child (the input is not a child's);
the best gradient-free learner on these features (it is not).

## 2. The learner

Everything learns online, one moment at a time. No part is frozen during a life.

- **Ear.** Bytes of text are chunked into the ear's own units by use (what is heard together often becomes one
  unit). An order-aware echo of the exchange so far is its output to the word cells.
- **Visual cortex.** One layer of 256 kinds of 5 × 5 colour windows. The windows arrive decorrelated (as the retina's
  centre-surround cells decorrelate natural scenes, Atick & Redlich 1992 [check]). A kind's response is its drive
  above the average drive of all kinds at that place (lateral inhibition; the "triangle" of Coates, Lee & Ng 2011
  [check]). The kinds develop by winner-take-all Hebbian learning with a conscience, without labels. The cortex
  sends on the average activity of each kind in each of 16 regions of the view: 4,096 numbers.
- **Relay.** The summary's own correlations are removed on the way to the word cells (lateral anti-Hebbian
  inhibition does this in a brain, Földiák 1990 [check]). **Here it is computed in batch** from the summaries of
  6,000 pictures, before the life. This is the one part that is not yet learned online by a local rule (Section 7).
- **Word cells.** Every unit of the ear that is heard in a reply has a cell. Each cell has two dendrites that learn
  to predict its firing by one local rule, change = plasticity × (fired − expected by this dendrite) × input
  (dendritic prediction of somatic firing, Urbanczik & Senn 2014 [check]; the Rescorla–Wagner rule for cues). The
  content dendrite listens to the relayed visual code and the units heard; the context dendrite listens to the ear's
  echo and predicts what comes next. A cell is active when both agree. A reply is a sequence of cells.
- **Episodic store.** 2,000 single moments, written whole when the word cells did not expect the reply, read by
  likeness to what is seen now (steeply: a moment, not a kind; Krotov & Hopfield 2016 [check]), the oldest
  forgotten when full (McClelland, McNaughton & O'Reilly 1995 [check]).
- **Looking.** On a page with several things the gaze goes to what stands apart; match neurons compare what is
  fixated with what was named.
- **Mouth and chooser.** Motor units are chunked by babbling; an auditory-motor map learns what each sounds like. A
  chooser learns from reward whether to say back what was heard, to speak from the word cells, or from the store.

Set by hand, and reported as such: sizes (256 kinds, 2,000 cells, 2,000 moments), the shrinkage of the two
decorrelations, the steepness of the store's comparison, the stage order, and the pass marks.

## 3. Experiment 1: a public benchmark as a single-pass stream

**Setup.** The cortex develops on CIFAR-100's training pictures without their labels, and its relay is settled on
6,000 of them. Every CIFAR-10 training picture is then seen once, in a random order, asked "what is this?" and told the class
name. 2,000 held-out test pictures are named at fixed points of the stream. Three seeds; a seed changes the cortex,
the order of the stream and every learner's initial state.

**Learners on the same code.** The word cells (with and without the relay, with and without the store); nearest
class mean; streaming linear discriminant analysis from the exact running sums (Hayes & Kanan 2020 [check]); a ridge
regression on everything seen so far (not a stream learner: the reference for a linear readout). The last two pick
their shrinkage on the test pictures, which is generous to them.

**Backpropagation of the same shape.** The same decorrelated windows in, 256 filters learned from random ones, the
same response and regional average, a linear readout; softmax, Adam. Learning rate and pictures per step were tuned
on the first 10,000 pictures of seed 0 (14 settings; 0.40–0.53 there). Also: with one or four earlier pictures lived
again for each new one; started from the developed filters; with the developed filters kept.

**Results** (mean ± spread of three seeds; (1) = one seed):

| learner | 500 | 5,000 | 50,000 pictures |
|---|---|---|---|
| streaming LDA + relay | 0.375 ± 0.021 | 0.620 ± 0.002 | 0.739 ± 0.001 |
| ridge + relay | 0.395 ± 0.010 | 0.620 ± 0.003 | 0.735 ± 0.001 |
| word cells + relay, random filters | 0.388 ± 0.034 | 0.602 ± 0.003 | 0.693 ± 0.002 |
| **word cells + relay** | 0.398 ± 0.027 | 0.595 ± 0.005 | **0.687 ± 0.005** |
| backprop, 4 lived again per new picture (1) | 0.355 | 0.550 | 0.688 |
| backprop, 1 lived again per new picture | 0.277 ± 0.012 | 0.490 ± 0.020 | 0.661 ± 0.011 |
| **backprop, one pass** | 0.242 ± 0.005 | 0.466 ± 0.008 | **0.616 ± 0.023** |
| word cells, no relay | 0.312 ± 0.014 | 0.486 ± 0.014 | 0.591 ± 0.007 |
| nearest class mean + relay | 0.426 ± 0.005 | 0.548 ± 0.004 | 0.569 ± 0.002 |
| backprop readout on kept developed filters (1) | 0.296 | 0.476 | 0.550 |

- The word cells through the relay are ahead of one-pass backpropagation throughout, by 16 points after 500
  pictures and 7 at the end. Backpropagation with four pictures lived again per new one (five times the updates)
  draws level.
- Without the relay they are not ahead. The lead belongs to a fixed, decorrelated code read by a local rule.
  **[pending: `bench_readout.py`: a softmax readout learned by gradient on the very same relayed code. If it matches
  the word cells, the finding is about the code, and the word cells are one local way of reading it.]**
- Development of the filters buys nothing (0.687 developed, 0.693 random), as random whitened patches have long
  been known to be hard to beat [check: Coates & Ng 2011; Saxe et al. 2011]. **[pending: the same in a life.]**
- The word cells are five points under readouts that keep the pictures' second moments. They are an online, local
  approximation of what those compute with a matrix inverse.

**A new class.** Five classes are held back in turn; after 5,000 pictures of the other nine, the held-back class is
shown 1, 5 and 20 times with 20 other pictures after each showing (three seeds).

| learner | shown pictures named with the new name (1 / 5 / 20 showings) | 200 other pictures of the new class | old classes still right |
|---|---|---|---|
| word cells | 0.00 / 0.00 / 0.05–0.08 | 0.00 | 0.60–0.62 |
| word cells + episodic store | 1.00 / 0.96–1.00 / 0.99–1.00 | 0.00 | 0.60–0.63 |
| nearest class mean | 1.00 / 1.00 / 0.95–0.98 | 0.00 / 0.01 / 0.08 | 0.55–0.57 |
| linear discriminant analysis | 1.00 / 1.00 / 1.00 | 0.00 | 0.59–0.62 |

The slow cells do not hold a picture seen once; with the store they do, and nothing known is disturbed. The other
learners hold it by construction. No learner reaches the kind from twenty pictures: one-shot learning here is memory
of an instance.

## 4. Experiment 2: a staged life

**World.** Real photographs (CIFAR, COCO, Wikimedia Commons) and speech written by language models acting only as
adults: comments on pictures, echo games, naming, picture-book pages with several things, two-word lessons, leisure
talk. Stages: babbling, echo, first words, two words. In a stage 70% of moments are the stage's own and 30% go back
to earlier ones. Every 5,000 moments, 300 held-out items per stage are asked.

**Judging.** A local 4B model judges whether the child's answer means what the adult's did: lenient on form, strict
on meaning. The adult always answers with their own word (a recast); acceptance decides reward only. A score counts
only **above a blind floor**: on the questions that cannot be answered by saying the question's last word back, the
judged score beside the best of always-yes, always-no and the commonest answer, (score − floor) / (1 − floor).
Babbling and echo are measured at every test and never held for promotion; the echo stage is a period, not a mark.

**Results** (three seeds, from birth, no hand-wired drive, no replay):

- Babbling is left after 5,000–10,000 moments; the echo period is lived (0.84–0.94 at its one test).
- First words is passed at 15,000 / 20,000 / 15,000 moments (above the floor 0.27–0.29 at the provisional mark of
  0.25).
- Held in two words, the score above the floor rises for 20,000–30,000 moments and flattens:

| age | seed 0: first words / two words | seed 1 | seed 2 |
|---|---|---|---|
| entering two words | 0.28 / 0.19 | 0.36 / 0.15 | 0.40 / 0.23 |
| 50,000 | 0.38 / 0.37 | 0.44 / 0.36 | 0.43 / 0.36 |
| 95,000 | 0.40 / 0.41 | 0.44 / 0.33 | 0.45 / 0.41 |

- **First words improves while two words is lived** (0.28–0.40 → 0.40–0.45).
- On the life's own photographs, each naming moment seen once, the word cells name 0.52 of held-out pictures among
  16–25 names (three seeds; chance 0.05–0.09); the same shape by backpropagation on the same moments 0.39 after one
  pass and 0.50 after five.
- A name heard once with a new kind of thing: said for the same picture 0.9 with the store and 0.1 without; for
  another picture of the kind 0.0.
- Observed, not asked for: echo fades by itself once words work (the question's last word is said back in
  0.07–0.10 of answers); a slight bias to "no" (−0.04 to −0.17), where two-year-olds have a bias to "yes" [check].
- **[pending: ablation lives, three seeds each: never-developed cortex, no relay, no store, no looking, no match
  neurons; the same lessons in shuffled order.]**

## 5. What did not work

- **Replay from the store did not protect earlier stages.** The store filled with leisure talk (55%), its surprise
  measure read 0.84 for everything, and old stages were 3% of what came back.
- **The stage after two words was not passed** by any brain, and earlier stages slid while it was lived. Its test
  turned out to be mostly answerable blind (floors 0.5–0.75); the lost answers went to the stage's larger set of
  names for the same pictures (couch for bed), not to simple erasure. The stage is being rebuilt.
- **No new-word bias** (a new word given to an unnamed thing rather than a named one: 0.51, chance 0.50).
- **A second cortical layer** added 0.01; dendritic branches on the word cells added nothing.
- **The first cortex's Hebbian development was worse than never developing it**; the second cortex's is no better
  than random filters.
- **Text alone**: 38% of text questions contain their own answer; on the rest the brains score 0.11–0.33 against a
  floor of 0.06–0.11.

## 6. Three errors of method

1. **A behaviour made a target.** Echo was a gate that had to be held at 80% of its pass mark. Brains that had
   learned first words were kept back 20,000–90,000 moments for echoing less, which is what a child does.
2. **A drive wired to a behaviour.** A "novelty drive" raised the chance of saying back an unknown word. It passed
   stages. It was a rule of ours, not something the brain did; it was retired and the lives rerun.
3. **A judge that taught mistakes.** The first judge accepted "yes" for "no", "bear" for a dog and the question said
   back, and when it accepted an answer the adult repeated the child's word: about 7% of lessons taught the child
   its own error. With a strict-on-meaning instruction it accepts 2–16 of 367 wrong answers and refuses no right one.

Also: a world that silently reverted to an older version after 50,000 moments of every life, found by reading logs.

## 7. Limits

- One small benchmark; one machine; 4,096-dimensional codes; nothing here says the results hold at scale.
- The relay is computed in batch before the life. Learning it online by a local anti-Hebbian rule is next.
- The backpropagation reference is one shape, plain Adam, no augmentation, tuned on a grid of 14.
- The stages, their order and the pass marks are ours. The judge is a model; a person's labels on 200 of its
  judgements are still to be made.
- Three seeds, not five.

## 8. Related work

**[to be written from research_notes/, every reference opened by hand.]** In outline: Hebbian language models on
made-up input (the assembly-calculus language organ, arXiv 2306.15364 and 2507.11788); deep Hebbian vision trained
for many epochs (SoftHebb, arXiv 2209.11883); single-pass child's-view contrastive learning with backpropagation
(BabyCL, arXiv 2606.05115), where one pass without large replay buffers is near chance; fast and slow memory in
agents trained by gradient (Hill et al., arXiv 2009.01719); sparse distributed memory as a continual learner (arXiv
2303.11934); random features and whitening (Coates, Lee & Ng 2011; Saxe et al. 2011); streaming LDA (Hayes & Kanan
2020); complementary learning systems (McClelland et al. 1995).

## 9. Statement on the use of AI

The code was written and the experiments were run by an AI assistant (Claude) under the author's direction. The
author set the principles (biological plausibility, no hand-wired behaviour, observations are not targets, no
changes made to raise a score), took the decisions recorded in the project's log, and corrected the assistant where
it departed from them, including the three errors in Section 6. The adults in the life's world and the judge are
language models. **[Kris: to be reworded as yours; every reference to be opened by hand before submission.]**

## 10. Code and data

Repository, logs of every run cited here, and the scripts that produce each table (`experiments/bench_*.py`,
`experiments/curriculum_life.py`). **[A snapshot with a DOI before submission.]**
