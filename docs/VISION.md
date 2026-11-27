# Vision: the two-part rule on the eye's own views, and what actually helped (2026-11-10)

Step 3 of [REVIEW_2026-11-10.md](REVIEW_2026-11-10.md). The claim to test: Hebbian learning alone
gives no invariance and prediction alone collapses, but a layer-local rule with both (Halvagar &
Zenke 2023) should learn invariant objects from the sequences of views the eye already makes (the
whole picture, closer around a gaze point, closer still, another point), and so lift first-words
naming above the 0.42 that nearest-picture matching gets from the current visual code.

**Result: it did not, in two formulations. What lifted naming was not in the eyes at all but in how
they are read (docs/AREAS.md): 0.42 → 0.55 from the same code.**

## Attempt 1: the cortex's own filters learn (`Layer.predictive`, `experiments/lpl_vision_probe.py`)

Every layer of the visual cortex at once; each kind of neuron changes its synapses to answer the new
view as it answered the last (predict), away from its own mean in units of its own variability
(Hebbian), and away from the kinds it co-varies with (decorrelate).

| after 8,000 pictures (5 views each); start 0.406 | naming (nearest picture, 780 held out, ±0.018) |
|---|---|
| predict + Hebbian + decorrelate, from random filters | 0.335 |
| the same, from the developed cortex (start 0.386) | 0.329 |
| Hebbian + decorrelate only | 0.347 |
| predict only | 0.256 (at 5,000) |

All arms fell and were stopped. The kinds' responses were 0.6–0.9 correlated from the start and
stayed so: the response of a kind to a view is dominated by the view's overall contrast, which all
kinds share, so the "what was I doing a moment ago" signal was mostly contrast. Prediction alone
collapsed fastest, as the paper says it should. The published results also use on the order of
10⁷ views; a life here has 10⁴–10⁵.

## Attempt 2: one layer above the cortex learns (`experiments/slow_views_probe.py`)

The cortex stays as it is; 512 neurons on top of its code (random synapses at first) learn with the
same three terms from the same views. Cheap enough to run to convergence.

| first-words naming on whole pictures (780 held out, ±0.018) | nearest picture | linear readout | two views of one picture alike |
|---|---|---|---|
| the cortex code itself (7,168 neurons) | 0.386 | 0.509 | 0.38 |
| 512 random neurons on top | 0.388 | 0.463 | 0.38 |
| predict + Hebbian + decorrelate, 6 passes | 0.410 | 0.456 | 0.14 |
| Hebbian + decorrelate only, 6 passes | 0.405 | 0.459 | 0.14 |
| predict only, 3 passes | 0.400 | 0.459 | 0.36 |

The predictive term adds nothing over the other two.

The layer decorrelates (its neurons end 0.04 correlated) but two views of one picture become *less*
alike, and naming does not move. Views of one picture already share most of the cortex's features;
what separates a dog from a cat across different pictures is not something successive views of one
picture can teach.

## What helped: reading the eyes with a slow, error-driven pathway

| reading the same visual code | first-words pictures named right |
|---|---|
| nearest remembered picture (what the clusters and the hub do), 9,000 remembered | 0.43 |
| the hub's visual cells with a life's budget | 0.28–0.29 |
| each word cell's dendrite learning which visual neurons predict its firing, 9,000 moments | 0.47–0.50 |
| the same after a life's worth (27,000–90,000 moments) | 0.54–0.56 |

So the review's reading "first words are capped by the eyes at ~0.4" is corrected: the code holds
at least 0.55, and the cap was the exemplar readout.

Each visual neuron passing on its deviation from its own average (a set-point) is part of that
pathway: it lifts the delta rule from 0.42 to 0.48 at 9,000 moments, and does little for
nearest-picture matching (0.39 → 0.38 developed, 0.41 → 0.44 random filters).

## Where this leaves vision

- The developed cortex is still no better than random filters for nearest-picture naming (0.39 vs
  0.41–0.42), and learning from views did not change that.
- The information that tells categories apart comes from the word, not from the views. The organic
  next step is the word cells' own error reaching back into the cortex (which features, when present,
  made a word's prediction come true): labels shaping perception (Lupyan 2012; Waxman & Markow 1995).
  Not built yet.
- The view sequences may matter for what they were not tested on here: recognising a thing from a
  part of it (a glance at a corner of the picture). Not measured.

## 2026-11-15: where the eyes stand (proposal 3 of CONVERSATION.md)

Kris asked to go ahead with "the word cells' error reaching back into the visual cortex". Before building it, what
any readout could get from the visual code was measured with reference learners fitted in a batch (not brains).

**Naming** (8,242 pictures seen, 300 held out, 23 names):

| reading the whole-view code (7,168 neurons) | named right |
|---|---|
| nearest remembered picture | 0.33 |
| 5–20 nearest, weighted | 0.36–0.37 |
| linear readout (ridge) | 0.47–0.49 |
| kernel readout (the best of several) | 0.55 |
| the word cells, online, one pass | 0.46–0.51 |
| layer 0 alone / layer 1 alone / layer 2 alone, linear | 0.42 / 0.40 / 0.37 |

The word cells already read nearly all that a linear readout can, and the best nonlinear one adds 0.05. The deeper
the layer of the developed cortex, the less it tells names apart: layer 2 (4,096 neurons) is the worst of the three.

**The two-words photo questions** (the stage at which all six lives stopped; pass mark 0.5 on the stage's mean):

| | always the commonest answer | the question's words only | + sight (best of added, kernel, conjunctions) | the word cells |
|---|---|---|---|---|
| is there a …? (COCO) | 0.62 | 0.67 | 0.68 | 0.63 |
| how many …? (COCO) | 0.55 | 0.55 | 0.58 | 0.59 |
| what colour …? (VQA) | 0.22 | 0.35 | 0.40 | 0.20 |
| yes/no (VQA) | 0.52 | 0.52 | 0.53 | 0.47–0.55 |

No readout gets more than a few points from the picture. The stage asks what a 64×64 whole view does not show
(how many people, a small umbrella, the colour of the bus). The wall at two words is here: the mean of the best
references over the stage is about 0.52, the pass mark 0.5.

**Picture-book scenes instead** (`organic/scenes.py`, explored, not in the curriculum): a page with up to four of
the first-words pictures, a quarter of the view each; "how many dogs are there?", "is there a dog?"; test pages
made of pictures never seen.

| | always the commonest answer | sight only | question × sight conjunctions |
|---|---|---|---|
| how many (one, two, three) | 0.37 | 0.46 | 0.50 |
| is there (yes, no) | 0.53 | 0.66 | 0.67 |

Better than photographs relative to the base rate (+0.13 and +0.15 against +0.03 and +0.06), still far from
answerable: a thing at a quarter of the view is recognised worse than the 0.5 it gets when it fills the view.

**Reading.** The limit is what the cortex computes from the pixels, not how it is read. A layer added on top and
taught by the word cells' error can at most reach the kernel readout (0.55 for naming), because it sees the same
code. Going further needs one of:

1. the cortex's own filters taught by the words (a local rule into the convolutional layers; both attempts so
   far were unsupervised and negative, a taught one has not been tried),
2. looking: the brain already has an eye that fixates and zooms, but the word cells get only the whole view; a
   thing looked at fills the view and is named at 0.5 instead of being a quarter of it,
3. far more to see (the cortex developed on a few tens of thousands of pictures).

Not built today. Which of these to try is the next decision; (2) uses what exists and is the one I would start with.

## 2026-11-16: the data question, looking, and the words teaching the cortex

Kris: do options 1 (looking) and 2 (the cortex taught by the words); check every step against how living beings work
and ask first whether it is a matter of more or better experience or of more layers.

**How much is there to see, and does more help?** The whole database is 202,000 lessons with 58,000 different
pictures and 1.97 million overheard sentences. First words had 10,300 pictures of 23 names (7,200 of them CIFAR's
32×32). With these, naming by a reference readout rises steeply with the pictures seen: 0.33 after 500, 0.42 after
2,000, 0.47 after 4,000, 0.55 after 8,000.

So a bigger picture book was made (`experiments/world/coco_crops.py`): 95,249 things cut out of the COCO training
photographs at their own resolution, 35 names (19 of the picture book's). On those (real things in real scenes,
harder than CIFAR), with the cortex as it is, a linear readout names 0.21 after 1,000 pictures, 0.30 after 8,000,
0.33 after 32,000 and 0.36 after 90,000: about +0.02 per doubling at the end. **More pictures help, and through
this cortex they stop helping soon.** The cortex is what has to get better.

**What the cortex is.** Three layers, developed once on 4,000 pictures per layer by competitive Hebbian learning,
never changed since. A neuron of the top layer sees a patch of 17 of the picture's 64 pixels: no neuron sees a
whole thing, and what tells names apart is the bag of local features in each sixteenth of the view.

`experiments/cortex_teach_probe.py`; a reference readout is fitted afresh on each cortex's code (6,000 pictures),
tested on 1,500 things from photographs never used and on the CIFAR and Commons test pictures:

| the cortex | COCO things named (35 names) | layer 1 / 2 / 3 alone | CIFAR + Commons (22 names) |
|---|---|---|---|
| as every life has had it | 0.278 | 0.237 / 0.233 / 0.218 | 0.487 |
| developed afresh, by its own rule, on 40,000 pictures per layer (ten times more) | 0.297 | 0.264 / 0.229 / 0.235 | 0.483 |
| taught by the words, top layer only | 0.275 | 0.237 / 0.233 / 0.203 | 0.474 |
| taught by the words, all layers | 0.296 | **0.277** / 0.077 / 0.038 | 0.470 |

"Taught by the words": 30,000 pictures are named; every visual neuron that talks to a word cell hears back from
it through its own feedback synapse. A neuron whose activity spoke for the word that was then heard moves its
synapses toward the window it was responding to; one that spoke for another word moves away (feedback-gated
plasticity: Roelfsema & van Ooyen 2005; Roelfsema & Holtmaat 2018). The error goes back one synapse, into every
layer directly; nothing is passed from layer to layer. It is the nearest this project has come to learning by
gradient: for those direct connections it is the gradient, carried by a feedback synapse instead of computed.

What happened: a first run moved nothing (the feedback was too faint to matter; each layer now hears it relative
to its usual size). Then: **the first layer gets better when taught (0.237 → 0.277 alone), the layers above are
ruined** (their input changes under them while they change too), and the top layer taught alone gets a little
worse. More to see by the cortex's own rule gives +0.02. Running now: only the first layer taught, then the layers
above developing afresh on what it sends (the order in which cortex matures: Huttenlocher & Dabholkar 1997).

**Looking.** A reference test with picture-book pages (four pictures, a quarter of the view each): the same naming
readout, applied to the whole view or to four looks, each quarter filling the view:

| | commonest answer | the whole view | four looks |
|---|---|---|---|
| is there a …? | 0.53 | 0.56 | 0.74 |
| how many …? | 0.37 | 0.40 | 0.55 |

A thing looked at is named 0.41 of the time (0.48 when it is the whole picture). Looking is worth building: the
brain has an eye that fixates and zooms, and the word cells have only ever had the whole view. The design, not
built yet: at each fixation the word cells get the cortex's code of what the eye is looking at; each cell keeps
its strongest drive over the looks of a moment; and what a cell learns from is the look that drove it most when
its word was heard (a word said over a page of several things is about one of them: cross-situational learning,
Yu & Smith 2007).

### Only the first layer taught (2026-11-16, afternoon)

| the cortex | COCO things named | layer 1 / 2 / 3 alone | CIFAR + Commons |
|---|---|---|---|
| as every life has had it | 0.278 | 0.237 / 0.233 / 0.218 | 0.487 |
| first layer taught by the words (30,000 named pictures) | 0.300 | 0.285 / 0.272 / 0.219 | 0.506 |
| the same, then layers 2 and 3 developed afresh on what it sends | 0.309 | 0.285 / 0.241 / 0.230 | 0.493 |
| layers 1 and 2 taught (a quarter of the rate) | 0.230 | 0.299 / 0.115 / 0.115 | 0.366 |

Teaching the first layer is a clean, small gain: that layer's own code names more (+0.05), the untaught layer
above it does too (+0.04, it simply gets better input), and it carries over to pictures of another kind that it
was never taught on (+0.02 on CIFAR and Commons). Letting the layers above start afresh adds nothing. Teaching the
second layer ruins it, at either rate tried. Why the first layer can be taught and the second cannot is not
understood; the second layer's neurons are sparse (the 12 strongest of 128 per position) and their windows are
576 numbers wide, so a neuron is seldom the one that hears back, and what it hears is about a sixteenth of the view.

One seed, one rate, 30,000 pictures. Still running: a slower rate with redevelopment, two more layers on top
(neurons that see 25 and 33 of the 64 pixels), and the looking probe's second arm (its first: the whole view names
a thing on a page 0.21 of the time, 0.09 with another thing beside it, against 0.40 for single pictures).

## Results of 2026-11-17 (the probes killed on 11-16, run again)

**Looking** (`experiments/look_probe.py`, seed 0; seeds 1 and 2 running). Pictures never seen:

| the word cells get | single pictures named | a thing on a page | a thing on a page, another beside it |
|---|---|---|---|
| the whole view only (every life so far) | 0.403 | 0.213 | 0.093 |
| the whole view and the four fixations | 0.337 | 0.360 | 0.280 |

With fixations a thing among others is named three times as often, and a thing on a page nearly as often as a
picture of it alone. Single pictures lose 0.07: a quarter of a single picture is a worse view of it than the whole,
and the cell takes whichever view drives it most. The eye here fixates the four quarters by rule; in a brain the
gaze has to find where to look.

**The first layer taught more slowly** (`cortex_teach_probe.py`, layers [0], rate 0.005, 30,000 named pictures, seed 0):

| cortex | COCO things named | layer 1 / 2 / 3 alone | CIFAR + Commons |
|---|---|---|---|
| as it is | 0.278 | 0.237 / 0.233 / 0.218 | 0.487 |
| first layer taught, rate 0.02 (11-16) | 0.300 | 0.285 / 0.272 / 0.219 | 0.506 |
| first layer taught, rate 0.005 | 0.321 | 0.311 / 0.261 / 0.219 | 0.498 |
| the same, layers above developed afresh | 0.327 | 0.311 / 0.247 / 0.234 | 0.512 |

Slower is better: the first layer alone goes from 0.237 to 0.311 and the whole code from 0.278 to 0.321. On pictures
of another kind (never used in the teaching) the gain is small, +0.01 to +0.025. One seed.

**Two more layers**: killed twice by memory in the measurement (a 15,360-dimension code's covariance takes 4 GB).
The readout is now solved over the pictures when the code is wider than the pictures are many (the same answer);
running again.

**A fourth layer of the same kind** (512 kinds, 2026-11-17): held-out COCO things 0.273 with four layers against
0.278 with three; the layers alone 0.239 / 0.239 / 0.228 / 0.193; CIFAR + Commons 0.463 against 0.487. More layers of
this kind add nothing, each a little worse than the one below. Stopped before the fifth. This is what the
literature reports for stacks of hard winner-take-all layers (docs/RESEARCH_VISION.md): the answer to "more layers?"
is not with this rule.

## Looking with the real word cells, the reflex gaze and match neurons (2026-11-17, `experiments/look2_probe.py`, seed 0)

6,000 lived moments (half single pictures, the rest pages: a thing named, "is there a…?", "how many…?"), pages with
3-pixel margins, pictures never seen at test. `organic/gaze.py` is the reflex gaze (centre-surround against the
scene's commonest colour); match neurons are in `organic/words.py`.

| the word cells get | single pictures | a thing on a page | with another beside it | either of the two | is there (base 0.52) | how many (base 0.35) |
|---|---|---|---|---|---|---|
| the whole view | 0.400 | 0.083 | 0.053 | 0.120 | 0.630 | 0.367 |
| the whole view, match neurons | 0.397 | 0.090 | 0.063 | 0.120 | 0.640 | 0.383 |
| + the four quarters (by rule) | 0.297 | 0.163 | 0.110 | 0.240 | 0.607 | 0.320 |
| + the four quarters, match neurons | 0.297 | 0.180 | 0.143 | 0.283 | 0.683 | 0.380 |
| + what the reflex gaze fixates | 0.327 | 0.250 | 0.160 | 0.347 | 0.590 | 0.350 |
| + the reflex gaze, match neurons | 0.320 | 0.243 | 0.160 | 0.347 | 0.660 | 0.370 |

- Looking triples the naming of a thing on a page (0.08 → 0.25) and costs single pictures 0.07–0.10, as in the
  first probe. The reflex gaze is better than the four quarters by rule (it does not look at empty quarters).
- The match neurons add 0.05–0.07 to "is there" and nothing to counting. With ideal eyes (a toy world where every
  thing is recognised) the same mechanism gives 1.00 and 0.89: **the mechanism works and the eyes do not carry it.**
  A thing on a page is recognised a quarter of the time, so "is the asked thing among what I fixated" is mostly noise.
- Put together as a two-words stage (name 0.25, is there 0.66, how many 0.37, text about 0.33) this is about 0.40
  against a pass mark of 0.5, while first words would lose 0.08. **A life with these eyes would not pass; I did not
  start one.** The code stays (`SpeakingBrain(looking, match)`, world 4) for when the cortex is better.
- What limits everything here is the cortex's code, and the same afternoon showed that the cortex's own
  development is the fault (next section).

## The cortex's own development is the fault (2026-11-17, `experiments/cortex2_probe.py`, first layer, seed 0)

After the survey (docs/RESEARCH_VISION.md). The same linear reference as always; every row is one layer of 5×5
windows read as a 4×4 grid summary. COCO things are 35 names, held-out photographs; CIFAR + Commons 22 names.

| first layer | code | COCO things | CIFAR + Commons |
|---|---|---|---|
| as it is in the living cortex (64 kinds, raw drive × contrast, strongest 8 kept, strongest in each region) | 1,024 | 0.250 | 0.441 |
| the same filters, graded response (drive above the average of all kinds at that place) | 1,024 | 0.250 | 0.434 |
| … and the region's average instead of its strongest | 1,024 | 0.283 | 0.432 |
| **filters never developed** (random), graded response | 1,024 | 0.276 | 0.484 |
| windows decorrelated (whitened), developed, graded | 1,024 | 0.287 | 0.486 |
| … region's average | 1,024 | 0.325 | 0.507 |
| 256 kinds, graded, no decorrelation | 4,096 | 0.290 | 0.497 |
| decorrelated, 256 kinds, graded | 4,096 | 0.320 | 0.537 |
| … region's average | 4,096 | 0.365 | 0.534 |
| … region's average, **filters never developed** | 4,096 | 0.351 | 0.534 |
| … region's average, response not scaled by the window's contrast | 4,096 | 0.395 | 0.532 |
| … region's average, response squared | 4,096 | 0.308 | 0.477 |
| … region's average over a 6×6 grid | 9,216 | 0.331 | 0.513 |
| a centre-surround retina (each point minus its surround) in place of the whitening, 256 kinds, average | 4,096 | 0.292 | 0.469 |
| decorrelated, 256 kinds, developed with graded competition (my version of SoftHebb) | 4,096 | 0.245 | 0.358 |

For comparison, the whole living cortex (three layers, 7,168 numbers): 0.278 and 0.487.

What it says:

- **The developed filters of the living cortex are worse than filters that were never developed** (0.250 / 0.441
  against 0.276 / 0.484). Hard winner-take-all on raw windows makes the first layer worse than chance wiring. The
  literature reports exactly this (Journé et al. 2023).
- What helps, each by itself and together: decorrelating the windows (+0.03 / +0.04), more kinds (+0.04 / +0.05),
  the region's average instead of its strongest (+0.04 on COCO things), not scaling by contrast (+0.03 on COCO
  things). Together: 0.250 → 0.395 and 0.441 → 0.532, with one layer.
- With decorrelated windows, development adds little over random filters (+0.014 / 0). Most of what a first layer
  gives is in the decorrelation, the graded response and the number of kinds; what its filters are matters little.
  This is also known (Coates et al. 2011; Saxe et al. 2011 on random filters).
- A simple centre-surround retina is not enough to stand in for the whitening; my version of graded competition is
  worse than the cortex's own rule; a finer grid and a steeper response are worse.
- One seed. Whether the word cells (not a linear reference) gain as much, and whether a second layer adds anything
  now, are the next two measurements (`experiments/words_on_cortex.py`, `experiments/cortex3_probe.py`).

**The word cells on the living cortex** (the baseline for that; 8,000 naming moments of world 3's first words, each
picture once, three seeds): held-out naming 0.385, 0.392, 0.383; by source CIFAR 0.38, Commons 0.49, COCO things 0.29.

## The word cells could not use a better code; a relay that decorrelates it lets them (2026-11-17)

`experiments/words_on_cortex.py`: the real word cells live 8,000 naming moments of world 3's first words (CIFAR,
Commons, COCO things; each picture once) and name held-out pictures; mean of the three sources. The new cortex is
`organic/cortex2.py` with one layer (256 kinds, decorrelated windows, graded response, averaged summary, not scaled
by contrast; a linear reference on it: 0.383 COCO things, 0.545 CIFAR + Commons).

| | old cortex | new cortex (one layer) |
|---|---|---|
| a linear readout (ridge) fitted on the very pictures the word cells lived, seed 0 | 0.495 | 0.560 |
| the word cells as they are (three seeds) | 0.385 0.392 0.383 | 0.398 0.402 0.408 |
| … the code decorrelated on its way to them, seed 0 | 0.425 | 0.505 |
| … and every correction at full size (three seeds for the new cortex) | 0.463 | 0.533 0.508 0.508 |

Other arms on the new cortex, seed 0: three passes over the same moments 0.428; corrections twice as large 0.393,
four times diverges; the correction not shared with the cues 0.420; full-size corrections without decorrelation
0.388. Decorrelation strength (how much of the largest variance is added to every direction first), with full-size
corrections: 0.002 → 0.510, 0.0005 → 0.533, 0.0001 → 0.523, 0.00002 → 0.513.

- **With the word cells as they were, a better cortex was worth +0.016.** The linear reference had risen by 0.07–0.10
  and the word cells did not follow: they, and not the eyes, were the limit. All the vision work of the last weeks
  was read through this.
- **Why.** A delta rule that sees each moment once learns along the few strongest directions of a correlated input
  and hardly along the fine ones that tell a cat from a dog. It is not the number of looks (three passes: +0.03) and
  not the size of the step (larger: nothing, then divergence). It is the shape of the input.
- **The remedy is a decorrelating stage between the cortex and the word cells** (`cortex2.Relay`): +0.11 with the new
  cortex. In a brain this is lateral inhibition between the neurons that carry the summary (anti-Hebbian learning,
  Földiák 1990); here it is computed at once from the summaries of 6,000 lived pictures, when the cortex is made.
  That is a debt: it should be learned by a local rule, and during life.
- **Full-size corrections** (+0.03, only with the decorrelated code): the content dendrite's plasticity was "the
  cell's own recent error" times the error of the moment, which counts the error twice; the error of the moment
  already makes a cell that is right stop changing. `Words.plastic_by_error = False`. Checked on text lessons
  before any life (below).
- Together: **0.387 → 0.516 held-out naming (three seeds), with one cortex layer, the same pictures, the same
  number of moments.** Offline; a life is the test.

`data/visual_cortex2.pkl` is that cortex with its relay (`experiments/make_cortex2.py`).

**Full-size corrections and text** (`experiments/imagery_probe.py` with FULL=1; the offline mixed life of first words
and two words, old cortex, first ear, one pass, three seeds; means):

| | pictures named | same answer ×3 | first-words text | text beyond the parrot | two words |
|---|---|---|---|---|---|
| plasticity by the cell's own error (as it was) | 0.471 | 0.719 | 0.280 | 0.248 | 0.430 |
| every correction at full size | 0.476 | 0.794 | 0.368 | 0.330 | 0.432 |

Text beyond the parrot rises by 0.08 in each of the three seeds (0.246 → 0.319, 0.225 → 0.328, 0.272 → 0.343), as much
as the new ear gave; pictures on the old code and two words do not move. So the change that was found for pictures is
at least as much a change for words heard: the cells were learning three times too slowly from every moment.

**The finished cortex through the brain's own path** (`data/visual_cortex2.pkl`, relay inside, three seeds): held-out
naming 0.527 0.508 0.508 with full-size corrections, 0.492 0.498 0.503 without.

**A second layer** (`experiments/cortex3_probe.py`; 256 kinds in 16 groups, each listening to 32 of the first layer's
kinds; linear reference, COCO things / CIFAR + Commons):

| | layer 1 | layer 2 alone | both | time to code a picture |
|---|---|---|---|---|
| response scaled by contrast | 0.361 / 0.540 | 0.333 / 0.523 | 0.365 / 0.546 | 96 ms |
| not scaled | 0.383 / 0.545 | 0.373 / 0.540 | 0.392 / 0.543 | 93 ms (one layer: 26 ms) |

With local fan-in and a graded response the second layer no longer loses what the first had (the old second layer
did), but it adds 0.01 at most and costs three times the time. One layer goes into the lives. Depth that adds
something needs a reason to differ from the layer below; the literature's candidate is learning to predict the next
view of the same thing (docs/RESEARCH_VISION.md, 6).

**Lives.** `eyes2_s0..2` started 2026-11-17 16:42: world 3, born with `data/visual_cortex2.pkl` (one layer + relay),
full-size corrections, otherwise as `world3_s*` / `world3b_s*`, whose checkpoints up to 45–50,000 moments are the
baseline.

## Wide pages and a gaze that goes to what stands apart (2026-11-17, `experiments/look2_probe.py`, new eyes, seeds 0 and 1)

A truer fovea. The page is twice as large as the view (128 pixels, a thing on it 52). The eye sees it whole only
coarsely (the view's 64 pixels) and in full detail what it fixates: a window of half the page, which then looks as
the thing looks alone. And the reflex gaze goes only to what stands apart, its surround at most half as busy as
itself (`organic/gaze.py`, `apart`): on a single picture it then fixates nothing 87% of the time. New cortex with
relay, full-size corrections, 6,000 lived moments.

| the word cells get | single pictures | a thing on a page | with another beside it | either of the two | is there (base 0.52) | how many (base 0.35) |
|---|---|---|---|---|---|---|
| the whole view, seed 0 / 1 | 0.540 / 0.587 | 0.053 / 0.077 | 0.060 / 0.073 | 0.150 / 0.153 | 0.607 / 0.653 | 0.423 / 0.443 |
| + what the gaze fixates, match neurons | 0.520 / 0.587 | 0.467 / 0.530 | 0.347 / 0.307 | 0.620 / 0.637 | 0.747 / 0.727 | 0.420 / 0.477 |
| … the match neurons counting all-or-none bursts | 0.520 / 0.583 | 0.470 / 0.527 | 0.357 / 0.310 | 0.633 / 0.643 | 0.733 / 0.743 | 0.413 / 0.467 |

- A fixated thing on a page is named nearly as well as a thing alone (0.47–0.53 against 0.54–0.59), and looking no
  longer costs single pictures. With the earlier small pages and a gaze that fixated parts of single things the
  same eyes gave 0.32 and cost single pictures 0.14.
- "Is there a…?" 0.73–0.75. Counting stays near the base rate, with graded or all-or-none matching alike: an exact
  count needs every fixation's match to be right. And a two-year-old does not count (docs/RESEARCH_TWO_WORDS.md):
  **"how many?" leaves the two-words stage** and is asked in preschool, with the photo questions.
- In the brain: `SpeakingBrain._glimpse` takes a scene larger than the view; world 4's pages are 128 pixels; the
  two-words stage of world 4 is now a thing on a page named, "is there a…?", and the two-words text lessons.

## The new eyes in lives: first words in a third of the time (2026-11-17, evening; lives eyes2_s0..2)

World 3, born with `data/visual_cortex2.pkl` (one layer + relay) and full-size corrections; otherwise as the six
baseline lives (world3_s0..2, world3b_s0..2). First-words test, 300 held-out items; "beyond" is the score on the
items a last-word parrot gets wrong (the pass mark is 0.4, with the echo stage held at 80% of its passing score).

| moments in first words | baseline: beyond the parrot (six lives) | eyes2: beyond the parrot | eyes2 by source: CIFAR / COCO things / Commons / text |
|---|---|---|---|
| 5,000 | 0.30 0.34 0.36 0.31 0.32 0.38 (mean 0.34) | 0.38 0.41 0.40 | 0.34 0.55 0.35 / 0.33 0.30 0.39 / 0.73 0.62 0.67 / 0.20 0.17 0.20 |
| 10,000 | 0.34 0.38 0.36 0.37 0.34 0.39 (mean 0.36) | 0.42 – 0.45 | 0.47 – 0.47 / 0.32 – 0.35 / 0.65 – 0.68 / 0.27 – 0.27 |

Baseline by source at 5,000: CIFAR 0.41, COCO things 0.21, Commons 0.58, text 0.23.

- **eyes2_s1 passed first words at 20,000 moments of life, 5,000 moments after entering the stage.** The baseline
  lives passed at 60,000–85,000. The other two are above the mark (0.42, 0.45) at 20,000 and wait only for the echo
  stage to recover (0.49 and 0.45 against the 0.68 they must hold), the dip every life has had.
- The gain in the life is on photographs (COCO things +0.13, Commons +0.09) and the test as a whole (+0.06 at 5,000,
  +0.07 at 10,000). CIFAR, low at the first checkpoint (60 pictures a life), is level or above at the next.
- Offline the same change was worth +0.13 in naming. In the life the first-words test is also text and routines,
  which the eyes do not touch; the picture sources alone are up by about 0.08.
- eyes2_s1 then lived 5,000 moments of world 3's two words (the photo questions) and reached 0.40 there, where the
  old lives ended after 100,000 moments. At 25,000 it was moved to world 4 with eyes that look, match neurons, the
  episodic store and the intake limit (the same brain, resumed).

## 2026-11-18: the rebuilt two-words stage is passed once; the old one holds the new brains too

Overnight (the loop was silent; the lives ran alone):

| life | first words passed at | two words lived | two words (beyond the parrot; pass 0.5) |
|---|---|---|---|
| eyes2_s1 | 20,000 | world 4 (picture-book pages; looking, match neurons, episodic store) | **0.50 at 55,000: PROMOTED** (35,000 in the stage) |
| eyes2_s2 | 40,000 | world 3 (old stage: photo questions, counting) | 0.43 after 100,000: stops |
| eyes2_s0 | 110,000 | world 3 | 0.40 after 95,000 (stopped for memory) |

- s1 by source at its pass: text 0.46, a thing on a page named 0.41, "is there a …?" 0.70. First words 0.46 and
  echo 0.71 held. It is the first two-words pass of any life, and it is at the mark, on one checkpoint, one seed.
- It is not a like-for-like win over the old lives: the stage itself was rebuilt (counting and the photo questions
  moved to preschool). What it shows is that the program as rebuilt can be lived through by this brain.
- The new eyes and word cells alone do not pass the old stage: 0.40–0.43, where the old brains had 0.38–0.43.
- After promotion s1 lost what it had passed: in preschool page naming fell 0.41 → 0.2 and first words to 0.35
  within 30,000 moments. The stage's mix changed and the word cells' weights went with it (erasure, as measured
  for the COCO names on 2026-11-17).
- Seeds 0 and 2 were started into the rebuilt stage from their frozen first-words levels on 2026-11-18 11:56
  (`eyes2w_s0`, `eyes2w_s2`): the three-seed answer.

### What chance is beside each score (2026-11-18; the survey's first missing control)

| test | different answers | uniform chance | always the commonest lived answer | the brain (eyes2 lives) |
|---|---|---|---|---|
| first words: CIFAR names | 16 | 0.06 | 0.05 | about 0.47 |
| first words: Commons names | 16 | 0.06 | 0.07 | about 0.7 |
| first words: COCO names | 20 | 0.05 | 0.09 | 0.33–0.39 |
| two words: a thing on a page named | 25 | 0.04 | 0.05 | 0.41 (seed 1 at its pass) |
| two words: "is there a …?" | 2 | 0.50 | 0.52 | 0.70 (seed 1 at its pass) |
| preschool: "how many …?" | 3 | 0.33 | 0.36 | 0.3–0.35 |

- Naming is 5 to 10 times chance. "Is there a …?" is 0.70 against a floor of 0.52: above chance, not by much.
- **A caveat on the two-words pass.** The stage's score is the mean of three sources, and one of them is a yes/no
  question whose floor is 0.52. A learner at chance on it, with seed 1's other two scores (0.46, 0.41), would score
  0.46; seed 1 scored 0.52. So about a third of the distance to the pass mark that the rebuilt stage removed is the
  yes/no floor, not learning. The pass mark was set (0.5) when the stage had no such source. To be decided with
  Kris: score "is there" above its floor ((score − 0.5) / 0.5), or leave it and say so. Left as it is for now; the
  per-source scores are in every record.
- The 300-item first-words test holds 248 items (60 for each of four sources, 8 social routines): a per-source
  score has a standard error of about 0.06, the stage score about 0.03.

## 2026-11-18: a new word, a named thing and an unnamed thing (the new-word bias: absent; a candidate gives a weak one)

Children from about 17 months take a new word for the thing that has no name yet (mutual exclusivity). The survey
of 2026-11-18 rates it the most telling behaviour test for a grounded associative learner: text-only models trained
by gradient show the opposite. Nothing in this brain was built for it. `experiments/newword_probe.py`: the word
cells as in the world 4 lives (new cortex and relay, wide pages, gaze, match neurons, full-size corrections; no fast
store) live 6,000 moments with 19 kinds named and 4 kinds never shown; then 60 trials from the same grown cells: a
page with a thing of a known kind and a thing of an unseen kind, a made-up word said of it once as a name is. Read:
is the new word's cell then driven more by the unnamed thing than by the named one, each seen alone (chance 0.5)?

| | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| the brain as it is: drawn to the unnamed thing | 0.50 | 0.55 | 0.48 | 0.51 |
| ... control, two unnamed things | 0.58 | 0.57 | 0.48 | 0.54 |
| candidate ("unclaimed"): drawn to the unnamed thing | 0.68 | 0.65 | 0.63 | **0.66** |
| ... control, two unnamed things | 0.55 | 0.55 | 0.50 | 0.53 |
| ... candidate, other pictures of the two kinds | 0.50 | 0.60 | 0.52 | 0.54 |
| says the new word of the unnamed thing afterwards | 0.00 | 0.00 | 0.00 | 0.00 |

- **As it is, the brain has no new-word bias** (0.51; my prediction before the run). The reason is in one line of
  the word cells: a cell that fires with no view driving it yet learns from the average of the views. Nothing sends
  the new word to the view the other cells do not claim.
- **The candidate**: such a cell takes each view as far as no cell is driven by it yet (`Words.unclaimed`, a probe's
  switch). Then the bias is there, weakly: 0.66 over 180 trials (standard error 0.035) against 0.53 for two unnamed
  things. It does not reach other pictures of the kind (0.54), and held-out naming of the known kinds is unchanged
  (0.577, 0.573, 0.626 in both).
- **Not adopted.** It was written to produce this behaviour, which is what the project's first rule warns against,
  and the effect is small. It stays as a switch. What it shows is where the bias would have to come from: from
  competition between word cells for a view, which the delta rule alone does not have.
- **The larger fact under both this and fast mapping** (2026-11-17): one hearing moves the slow synapses very
  little (the new cell's drive after it: 0.03, where a known name's is 0.5–1), and the new word is never said. The
  fast store answers only when the same whole view returns. A fast store keyed by the fixated thing, not the whole
  view, is the open design: it would give fast mapping of a thing on a page, and a place for the bias to act.
- Honest limits: 4 unseen kinds a seed; made-up words pass through an ear that never heard them; the measure is
  the cell's drive, the brain's stand-in for where an infant looks.

## 2026-11-18 (evening): the rebuilt two-words stage on three seeds

Each brain entered the picture-book stage (world 4: looking, match neurons, episodic store) after passing first
words. Seed 1 lived 5,000 moments of the old stage first; seeds 0 and 2 were continued from their frozen
first-words levels (`eyes2w_s0`, `eyes2w_s2`).

| seed | two words, beyond the parrot (pass 0.5) | text | a thing on a page named | "is there a …?" | first words held | echo held | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 0.50 at 35,000 moments in the stage | 0.46 | 0.41 | 0.70 | yes | yes | **PROMOTED** (2026-11-18 00:14) |
| 2 | 0.56 at 40,000 | 0.48 | 0.45 | 0.79 | yes | yes | **PROMOTED** (2026-11-18 17:29) |
| 0 | 0.50–0.51 at 55,000–60,000 | 0.41 | 0.35–0.37 | 0.72–0.76 | yes (0.50) | no (0.42–0.62; needs 0.67) | at the mark, held back by the echo; 40,000 moments of its budget left |

- **All three seeds reach the pass mark of the rebuilt stage; two are promoted.** The old stage held every brain,
  old or new, at 0.38–0.43.
- The caveat of the chance table stands: "is there a …?" has a floor of 0.52. With that source at its floor the
  three would score about 0.46, 0.48 and 0.43. Page naming (chance 0.04) and text carry the rest.
- Seed 0 shows the echo dip once more: its echo wanders between 0.42 and 0.69 while everything else holds. The
  novelty drive did not settle it. This is now the most frequent reason a brain waits at a stage it has passed.
- Seed 2 crossed slowly: 0.49 at 5,000 moments, 0.46–0.49 for 25,000 more, then 0.56, 0.52, 0.56.
- **Read again with the echo observed, not held (Kris's correction, 2026-11-18, docs/CRAFT.md): all three seeds
  pass the rebuilt stage.** Seed 1 at 35,000 moments in the stage (0.50), seed 2 at 30,000 (0.56), seed 0 at 45,000
  (0.52, first words 0.52). Seed 0 was relaunched under the new rule at 20:19 and is promoted at the next
  checkpoint at which it is at the mark again.

## 2026-11-18 (evening): a backprop reference of the same shape

`experiments/backprop_reference.py` (numpy; gradients checked against finite differences): the same decorrelated
windows in, 256 filters, the same response and regional average, a linear readout to the names, trained end to end
by backprop (softmax, Adam 0.001, batches of 8) on the very 8,000 naming moments a seed's word cells lived in
`words_on_cortex.py`. Held-out naming, mean of CIFAR, Commons and COCO things, three seeds:

| | after 1 pass (the brain's exposure) | 2 passes | 5 passes |
|---|---|---|---|
| backprop, filters and readout | 0.390 (0.368, 0.368, 0.433) | 0.448 | 0.504 |
| the cortex's own filters kept, readout by gradient | 0.390 | 0.421 | 0.470 |
| random filters kept, readout by gradient | 0.269 | 0.350 | 0.389 |
| **the brain: cortex, relay, word cells, each moment once** | **0.516** (0.533, 0.508, 0.508) | – | – |
| ridge on the brain's code, the same moments (batch) | 0.56 | | |

- **At the same exposure the brain names more than backprop of the same shape (0.52 against 0.39); backprop needs
  five passes over the same moments to come level (0.50).** This is the niche the survey said local learners have:
  learning from each moment once.
- The cortex's developed filters are worth as much as backprop's after one pass (0.390 both) and clearly more than
  random filters (0.269): with the new cortex, development is no longer worse than no development.
- Honest limits. One learning rate and one optimiser were tried for the reference, so it is lightly tuned. The
  brain had seen unlabelled pictures before (its cortex developed on them, its relay settled on 6,000 of them); the
  reference starts from random filters but shares the decorrelated windows. A deeper backprop network with
  augmentation would do better than this one; the comparison is of the same shape, not of the best that backprop can do.

## 2026-11-20: a public benchmark as a one-pass stream (first seed)

The paper's controlled core (docs/PUBLISH.md). A cortex develops without labels on CIFAR-100's pictures; then every
labelled CIFAR-10 training picture is seen once, in a random order; 2,000 held-out test pictures are named at
points of the stream (`experiments/bench_codes.py`, `bench_stream.py`, `bench_backprop.py`). Seed 0; more seeds and
the tuned backprop runs are to come, so this is a first reading, not a result.

| learner on the cortex's code | after 1,000 | 5,000 | 10,000 | 50,000 pictures |
|---|---|---|---|---|
| streaming LDA + relay | 0.424 | 0.619 | 0.670 | **0.741** |
| ridge on all seen so far + relay | 0.469 | 0.622 | 0.659 | 0.736 |
| streaming LDA | 0.486 | 0.623 | 0.664 | 0.722 |
| ridge on all seen so far | 0.484 | 0.607 | 0.650 | 0.709 |
| **word cells + relay (the brain's path)** | 0.491 | 0.599 | 0.623 | **0.686** |
| word cells, no relay | 0.337 | 0.486 | 0.500 | 0.581 |
| nearest class mean + relay | 0.485 | 0.549 | 0.559 | 0.567 |
| nearest class mean | 0.306 | 0.317 | 0.312 | 0.312 |
| the same on a never-developed cortex (random filters): word cells + relay | 0.502 | 0.606 | 0.636 | **0.691** |
| ... streaming LDA | 0.471 | 0.635 | 0.660 | 0.720 |
| ... nearest class mean + relay | 0.498 | 0.573 | 0.582 | 0.598 |
| backprop of the same shape, random start, rate 0.0003, one picture a step (tuning, first 10,000 only) | 0.263 | 0.427 | 0.479 | – |
| ... 8 a step / 32 a step | 0.296 / 0.242 | 0.395 / 0.348 | 0.446 / 0.397 | – |

What it says so far:
- **The word cells are not the best learner without gradients on this code.** Streaming LDA and ridge, which keep
  the pictures' second moments and invert a 4,096 x 4,096 matrix, are 2 to 5 points above them from 5,000 pictures
  on (their shrinkage was picked on the test pictures: generous to them). In the first 1,000 pictures the word
  cells are level with the best.
- **The relay is worth 10 points to the word cells** (0.581 → 0.686) and 25 to nearest class mean. The decorrelation
  finding of 2026-11-17 holds on a public set.
- **The cortex's development buys nothing here.** Random filters on the same decorrelated windows do as well
  (0.691 against 0.686). On the lives' own pictures a gradient readout preferred the developed filters (0.39
  against 0.27 after one pass, 2026-11-18, without the relay); through the relay, on CIFAR-10, the difference is
  gone. This is the literature's warning (random whitened patches are hard to beat) arriving on our own table.
- **Backprop of the same shape is well behind at equal exposure in the first three settings tried** (0.40–0.48
  after 10,000 pictures against 0.62). Six more settings are running; the claim waits for them and for replay.
- The "seen once, twenty pictures later" column of the first run is not reported: with the classes long known it
  is not fast mapping. `experiments/bench_newclass.py` (a class held back, then shown 1, 5, 20 times) replaces it.

### The benchmark's first seed, finished (2026-11-20, morning)

- **Backprop of the same shape, nine settings, first 10,000 pictures:** 0.397 to 0.527; best at the edge of the grid
  (rate 0.003, 32 pictures a step), so the grid was widened (running). At the same point: word cells + relay 0.623,
  streaming LDA 0.664. Not yet the tuned number.
- **Random filters, all learners after 50,000 pictures:** word cells + relay 0.691, streaming LDA 0.720 (with the
  relay 0.719), ridge + relay 0.719, nearest class mean + relay 0.598. Developed filters: 0.686, 0.722 (0.741),
  0.736, 0.567. Development is worth at most two points to the readouts that invert a matrix, nothing to the word
  cells.
- **A new class shown once, five times, twenty times** (`bench_newclass.py`; five classes held back in turn; 5,000
  pictures of the other nine first; 20 other pictures after each showing):

  | learner | the pictures shown are named with the new name (1x / 5x / 20x) | 200 other pictures of the new class | old classes still right |
  |---|---|---|---|
  | word cells | 0.00 / 0.00 / 0.08 | 0.00 / 0.00 / 0.00 | 0.61–0.62 |
  | word cells + episodic store | 1.00 / 1.00 / 1.00 | 0.00 / 0.00 / 0.00 | 0.62–0.63 |
  | nearest class mean | 1.00 / 1.00 / 0.97 | 0.00 / 0.01 / 0.08 | 0.56 |
  | linear discriminant analysis from the sums | 1.00 / 1.00 / 1.00 | 0.00 / 0.00 / 0.00 | 0.59 |

  The fast/slow finding of the lives holds on the public set: the word cells alone do not have a picture they saw
  once (0.00), with the store they have it (1.00), and the old classes are untouched. Two honest additions: the
  other learners have the shown picture too, by construction (they add it to sums at full weight; it is the word
  cells that are slow, by design); and **nobody reaches the kind**: twenty pictures of a new class give 0.08 at
  best. One-shot learning here is memory of a picture, not of a category, for every learner tried.

## 2026-11-20 (night): the benchmark on three seeds (cortex, stream order and learners all reseeded)

`out/bench/queue2.sh` ran unattended from 07:00 to 21:00 (logs: `out/bench/backprop_tune_s0.log`,
`backprop_full.log`, `stream_seeds.log`, `newclass_seeds.log`). CIFAR-10, every training picture once; 2,000 held-out
pictures named. Mean of three seeds ± their spread, where three were run; one seed otherwise (marked 1).

| learner (developed cortex unless said) | after 500 | 5,000 | 50,000 pictures |
|---|---|---|---|
| streaming LDA + relay | 0.375 ± 0.021 | 0.620 ± 0.002 | **0.739 ± 0.001** |
| ridge on everything seen + relay | 0.395 ± 0.010 | 0.620 ± 0.003 | 0.735 ± 0.001 |
| streaming LDA + relay, random filters | 0.397 | 0.599 | 0.719 ± 0.004 |
| **word cells + relay, random filters** | 0.388 ± 0.034 | 0.602 ± 0.003 | **0.693 ± 0.002** |
| **word cells + relay** | 0.398 ± 0.027 | 0.595 ± 0.005 | **0.687 ± 0.005** |
| backprop, same shape, 4 earlier pictures lived again per new one (1) | 0.355 | 0.550 | 0.688 |
| backprop, same shape, 1 lived again per new one | 0.277 ± 0.012 | 0.490 ± 0.020 | 0.661 ± 0.011 |
| backprop, same shape, start from the developed filters (1) | 0.297 | 0.481 | 0.629 |
| **backprop, same shape, one pass** | 0.242 ± 0.005 | 0.466 ± 0.008 | **0.616 ± 0.023** |
| word cells, no relay | 0.312 ± 0.014 | 0.486 ± 0.014 | 0.591 ± 0.007 |
| nearest class mean + relay | 0.426 ± 0.005 | 0.548 ± 0.004 | 0.569 ± 0.002 |
| backprop readout on the developed filters, kept (1) | 0.296 | 0.476 | 0.550 |

Backprop: softmax, Adam, 0.003 at 32 pictures a step, the best of 14 settings on the first 10,000 pictures of seed 0
(0.40–0.53 there; the wider grid found nothing better than the first grid's edge). It learns its 256 filters from
random ones and has no relay.

What three seeds say:
- **The word cells through the relay are 7 points above one-pass backprop of the same shape at the end, 13 at 5,000
  pictures, 16 at 500.** The spread between seeds is small (0.005 against 0.023), so this is not noise.
- **Backprop needs each picture about five times to draw level** (0.688 with four lived again per new one, one seed).
- **Without the relay the word cells are not ahead** (0.591 against 0.616: within backprop's spread). The lead is the
  relay's: a fixed, decorrelated code read by a local delta rule, against filters learned by gradient in one pass.
  Whether a *gradient* readout on the same relayed code does as well as the word cells is the control now running
  (`experiments/bench_readout.py`); if it does, the sentence is "decorrelated fixed features are what makes one pass
  enough", and the word cells are one way to read them, not a better one.
- **Development buys nothing, on three seeds** (random filters 0.693 ± 0.002, developed 0.687 ± 0.005; for streaming
  LDA development is worth two points). Backprop started from the developed filters ends where it ends from random
  ones (0.629, 0.627).
- **The readouts that keep the second moments stay 5 points above the word cells** (0.739, 0.735).
- **A new class** (`newclass_seeds.log`, seeds 1 and 2 as seed 0): word cells name the shown pictures 0.00 / 0.00 /
  0.05–0.08 after 1 / 5 / 20 showings; with the episodic store 1.00 / 0.96–1.00 / 0.99–1.00; other pictures of the
  new class 0.00 for both; nearest class mean reaches 0.08 of the kind after 20 showings, LDA 0.00. Old classes
  untouched everywhere. The fast/slow dissociation holds on three seeds; nobody reaches the kind.

## 2026-11-20 (night): how far two words goes with nothing artificial (lives cleanw_s0..2)

From the clean lives' first-words levels, held in two words (PASS_ABOVE=0.99) for the stage's budget. No novelty
drive, no replay, the strict judge, the adult always recasts. `above` = judged answers on the questions a parrot
cannot answer, above the best blind strategy, at each 300-item test.

| age (moments) | seed 0: first words / two words | seed 1 | seed 2 |
|---|---|---|---|
| 20,000–25,000 (entering) | 0.28 / 0.19 | 0.36 / 0.15 | 0.40 / 0.23 |
| 35,000 | 0.34 / 0.31 | 0.38 / 0.22 | 0.37 / 0.40 |
| 50,000 | 0.38 / 0.37 | 0.44 / 0.36 | 0.43 / 0.36 |
| 65,000 | 0.39 / 0.35 | 0.43 / 0.36 | 0.44 / 0.38 |
| 80,000 | 0.43 / 0.41 | 0.47 / 0.32 | 0.47 / 0.38 |
| 95,000 | 0.40 / 0.41 | 0.44 / 0.33 | 0.45 / 0.41 |

- **Two words rises for 20,000–30,000 moments in the stage and then flattens**: seed 2 from 35,000 (0.36–0.41),
  seed 1 from 50,000 (0.30–0.36), seed 0 creeps up to the end (0.37 at 45,000, 0.41 at 80,000). By source at the
  end: a thing on a page 0.33–0.39, the adults' two-word lessons 0.30–0.35, "is there a …?" 0.27–0.58 (the widest
  spread between seeds; seed 1's sits at 0.27–0.42).
- **First words keeps rising while two words is lived** (0.28 → 0.43, 0.36 → 0.44, 0.40 → 0.45): with 30% of the
  moments going back to earlier stages, nothing is lost and something is gained. The loss seen in preschool is not
  what a stage change must cost.
- Observed, not asked: the yes bias stays slightly negative (−0.04, −0.17, −0.10: "no" a little more often than
  right); the question's last word is said back in 0.07–0.10 of answers throughout; the store holds 55% talk.
- **For the mark (Kris's to set):** at 0.25 the lives left after 10,000–15,000 moments, when the stage had only
  begun. At 0.35 all three seeds would pass after 20,000–30,000 moments in the stage, near where each flattens
  (0.31–0.41). A different rule in the spirit of "observed, not targets": leave a stage when its curve has stopped
  rising, wherever that is. Both are proposals; nothing is changed.

## 2026-11-20 (night): what the cortex's development did to its kinds (`experiments/cortex_dev_probe.py`)

The benchmark's question, taken as a question about the mechanism and not about the number: winner-take-all Hebbian
development buys nothing over never developing. Before any why, what did it do? The benchmark's cortices, three
seeds, on windows of 200 unlabelled pictures and the summaries of 5,000 benchmark pictures (nothing changed):

| | developed (seeds 0 / 1 / 2) | never developed |
|---|---|---|
| a kind's likeness to where it was born (cosine) | 0.24 / 0.23 / 0.24 | 1 |
| how heavy-tailed a kind's drive is (excess kurtosis) | 0.91 / 0.95 / 0.94 | −0.21 / −0.20 / −0.19 |
| how evenly the kinds share being strongest (1 = even) | 0.994 | 0.91 |
| likeness between two kinds: mean of \|cosine\| | 0.21 | 0.09 |
| … the most alike 1% of pairs | **0.89** | 0.29 |
| directions that carry the summary's variance (of 4,096) | 10.7 / 10.6 / 11.2 | 10.9 / 12.0 / 12.7 |

- **Development is real**: every kind moved far from its start, the kinds answer more rarely and more strongly
  (they have found features in the decorrelated windows), and all are used evenly (the conscience works).
- **And it made copies.** 658 pairs of kinds are alike above a cosine of 0.8 (the most alike 1%, about 330 pairs,
  above 0.89), where no pair of random kinds is (their most alike 1%: 0.29). Winner-take-all with a conscience sends the kinds to where the windows are dense, and
  nothing keeps two kinds from settling on the same feature. 256 kinds in a 75-dimensional window space: random
  kinds cover it evenly; developed kinds gain selectivity and lose coverage. The summary ends up no richer (the same
  eleven or so strong directions).
- **What is missing is known in cortex**: neurons with like tuning inhibit each other, and that inhibition is
  itself learned (anti-Hebbian lateral connections: Földiák 1990). Hebbian feedforward learning with anti-Hebbian
  lateral learning gives kinds that are selective *and* different. Our layer has the first half and a conscience.
- **It is the same missing mechanism as the relay's.** The relay removes the summary's correlations and is computed
  in batch before the life (the paper's stated debt; queue item 9). Lateral anti-Hebbian learning among the kinds
  would be one local rule for both: no copies during development, and a decorrelated code afterwards.

**Proposal (not built; the organic check written first).** (1) Organic? Yes: learned lateral inhibition between
neurons of like tuning is among the best-established circuits in sensory cortex, it is local (two neurons that fire
together come to inhibit each other), and it sets no behaviour. (2) Is it data, training or size? None: it is a
missing local rule, and it would *remove* a hand-computed part. (3) The motive is not a number: the benchmark may
not move at all (random kinds already cover the space, and the batch relay already decorrelates). What it would
give is a cortex whose every step is learned online by local rules, and an answer to whether development has a role
once the kinds are kept apart. To be tried as a probe on the benchmark's codes first, measured by the table above
(copies gone? selectivity kept?) before any accuracy is read.

### First try of learned lateral inhibition: it does not work in this form (`experiments/lateral_probe.py`, seed 0)

Three local rules on the benchmark cortex's decorrelated windows, kinds born random: an active kind moves towards the
window in proportion to its activity; two kinds active together more than by chance come to inhibit each other; a
kind active too often raises its threshold (to be active for a tenth of the windows). Read by the table above, not
by accuracy:

| | born | winner-take-all + conscience | lateral, inhibition as fast as the pull (2,000 pictures) | lateral, inhibition 20x faster (16,000) |
|---|---|---|---|---|
| selectivity (kurtosis of the drive) | −0.21 | 0.91 | 0.47 | 0.41 |
| pairs of kinds alike above 0.8 | 0 | 658 | 2,453 | 1,377 |
| responses alike, top 1% of pairs | 0.41 | 0.86 | 0.96 | 0.90 |

**Worse on both counts**, and the reason is visible in the run: the inhibition never settles (mean 0.8 → 1.7 → 3.1 →
5.0 over the run, largest 180), thresholds climb to half the largest possible drive, and the kinds' settled
activities stay alike (0.90). My rule lets inhibition grow for as long as two kinds are ever active together, with
nothing that brings it back, and the activity under such inhibition does not settle (it swings). So the kinds all
learn from the same windows before they are kept apart. This is a fault of my formulation, not evidence against the
mechanism: the published networks of this kind have a decay in both sets of synapses and activity that provably
settles (Földiák 1990; Zylberberg et al. 2011; the similarity-matching networks of Pehlevan & Chklovskii). **Next:
read those rules in the papers themselves and build the probe on one of them, unchanged.** No codes were made and no
accuracy was read.

## 2026-11-21: is the lead the word cells' or the code's? A gradient readout on the same code (`bench_readout.py`)

The control a reader will ask for after the three-seed table. The code is kept exactly as the word cells get it; only
a readout to the ten classes is learned by the gradient of a softmax, each picture once, in the stream's order.
Seventeen settings learn side by side (plain steps at five rates; Adam at four rates and 1, 8 or 32 pictures a step);
reported: the setting best after 10,000 pictures. Seeds 0 and 1 (seed 2 and the random filters still running).

| after 50,000 pictures (5,000 in brackets) | seed 0 | seed 1 |
|---|---|---|
| word cells through the relay | 0.686 (0.599) | 0.693 (0.588) |
| gradient readout through the relay, plain steps | 0.696 (0.594) | 0.702 (0.591) |
| gradient readout through the relay, Adam | 0.689 (0.610) | 0.704 (0.594) |
| word cells, no relay | 0.581 (0.486) | 0.598 (0.468) |
| gradient readout, no relay, plain steps | 0.558 (0.448) | 0.568 (0.418) |
| gradient readout, no relay, Adam | 0.567 (0.406) | 0.582 (0.446) |

- **The lead is the code's.** A gradient readout on the relayed code does as well as the word cells, a point better
  at the end. So what puts both 7 to 8 points above end-to-end backprop in one pass is not the word cells' rule: it
  is that the features are fixed and decorrelated, and only a single layer of synapses has to learn.
- **This is as it should be, and the paper must say it plainly.** A delta rule on one layer of synapses is gradient
  descent on that layer; "no backpropagation" means no error is sent back through layers, not that the last layer
  learns by some other principle. The word cells are a local, online reader of the code, level with the gradient
  reader, a point or two ahead of it without the relay, and they do more than classify (order, cues, replies).
- **What stands as the finding:** fixed features with decorrelation at two places, read by one layer of local
  synapses, learn faster from a single pass than the same shape trained end to end; backprop needs about five
  times the exposure to draw level. And the relay is worth 10–13 points to any online reader, gradient or not.

## 2026-11-21 (morning): three things that ran overnight

**1. The readout control, all three seeds, and on random filters** (`out/bench/readout_*.log`). Gradient softmax
readout through the relay, the setting best after 10,000 pictures: plain steps 0.696 / 0.702 / 0.685 (mean 0.694),
Adam 0.689 / 0.704 / 0.683; word cells 0.686 / 0.693 / 0.682 (0.687). On random filters: 0.703 / 0.702 / 0.702
against the word cells' 0.693. Without the relay the gradient readout is at 0.56–0.58 on developed filters and
unsteady on random ones (0.46–0.59), the word cells at 0.58–0.60. The reading of 2026-11-21 stands on three seeds:
the lead over end-to-end backprop is the code's.

**2. Lateral inhibition on the published rules, full run** (`out/bench/lateral_s0.log`, 16,000 pictures, seed 0):

| | born | winner-take-all + conscience | Hebbian/anti-Hebbian network |
|---|---|---|---|
| selectivity (kurtosis of the drive) | −0.21 | 0.91 | 0.31 |
| pairs of kinds alike above 0.8 | 0 | 658 | 1,204 |
| the layer's usual responses alike, top 1% | 0.41 | 0.86 | 0.87 |
| activities settled under the learned inhibition alike, top 1% | – | – | **0.40** |

The network is stable (inhibition mean 0.08, largest 1.3, every kind active 14% of the time, none silent), and it
does what it is derived to do: the *activities* are kept apart. It does not move the *filters* apart; they are more
alike than winner-take-all's, and it is the inhibition that separates what they say. So my question was put wrongly:
"copies" among filters are not a fault if the circuit that reads them has learned inhibition. The thing to measure
is the code made of the settled activities, which this probe has not made (heavier: every window settles). Not
done yet; no accuracy read.

**3. A life with a never-developed cortex** (ablation arm 1: `ablrand_s0..2`, three seeds from birth, at 45,000
moments of 50,000; control: the clean lives then `cleanw_s0..2` at the same age). Score above the blind floor:

| at 45,000 moments | never-developed cortex (seeds 0 / 1 / 2) | control |
|---|---|---|
| first words passed at | 15,000 / 15,000 / 15,000 | 15,000 / 20,000 / 15,000 |
| first words | 0.38 / 0.39 / 0.41 | 0.38 / 0.42 / 0.45 |
| two words, all | 0.12 / 0.10 / 0.10 | 0.37 / 0.31 / 0.38 |
| … naming a thing on a page | 0.02 / 0.00 / 0.10 | 0.37 / 0.25 / 0.37 |
| … "is there a …?" | 0.15 / 0.09 / 0.00 | 0.45 / 0.30 / 0.48 |
| … the adults' two-word lessons (text) | 0.20 / 0.19 / 0.21 | 0.28 / 0.37 / 0.29 |

- **Single pictures are named as well without development** (as on the benchmark), and first words is passed as
  early.
- **A thing on a page is not named without it** (0.00–0.10 against 0.25–0.37), on all three seeds. So development
  does buy something, and the benchmark could not show it: a benchmark picture is one thing filling the view; a
  page is several things, and the brain must look at one of them. Which step fails without developed kinds (where
  the gaze goes, or what a small fixated thing looks like to the word cells) is not known yet: `look2_probe.py` on
  the two cortices will say. The lives end at 50,000; then the next arm (no relay) starts by itself.
- At the end (50,000 moments): first words 0.41 / 0.41 / 0.41, pages 0.20 / 0.19 / 0.16; the control at 50,000:
  0.38 / 0.44 / 0.43 and 0.37 / 0.36 / 0.36.

## 2026-11-23: what each part buys in a life (five ablation arms, three seeds each, 50,000 moments)

`out/tmp/ablation_queue.sh`, finished 2026-11-22 10:28. Each arm: the clean lives' launch line with one part taken
away, from birth, kept in the pages stage (W2, until now called "two words") to 50,000 moments. Control: the clean
lives, then `cleanw`, at the same age. Score above the blind floor, seeds 0 / 1 / 2 (mean).

| | first words | pages, all | naming a thing on a page | "is there a …?" | the adults' short lessons |
|---|---|---|---|---|---|
| **control** | 0.38 / 0.44 / 0.43 (0.42) | 0.37 / 0.36 / 0.36 (0.36) | 0.31 / 0.39 / 0.37 (0.36) | 0.52 / 0.36 / 0.45 (0.44) | 0.30 / 0.32 / 0.26 (0.29) |
| never-developed cortex | 0.41 / 0.41 / 0.41 (0.41) | 0.20 / 0.19 / 0.16 (0.18) | 0.07 / 0.02 / 0.03 (0.04) | 0.30 / 0.27 / 0.24 (0.27) | 0.24 / 0.27 / 0.20 (0.24) |
| no relay | 0.35 / 0.36 / 0.34 (0.35) | 0.23 / 0.26 / 0.32 (0.27) | 0.21 / 0.19 / 0.19 (0.19) | 0.27 / 0.36 / 0.58 (0.40) | 0.21 / 0.24 / 0.21 (0.22) |
| no episodic store | 0.41 / 0.44 / 0.47 (0.44) | 0.37 / 0.34 / 0.41 (0.37) | 0.50 / 0.40 / 0.45 (0.45) | 0.39 / 0.39 / 0.55 (0.44) | 0.21 / 0.24 / 0.24 (0.23) |
| no looking | 0.45 / 0.42 / 0.41 (0.42) | 0.19 / 0.23 / 0.20 (0.21) | 0.03 / 0.01 / 0.02 (0.02) | 0.27 / 0.42 / 0.39 (0.36) | 0.26 / 0.25 / 0.19 (0.23) |
| no match neurons | 0.41 / 0.42 / 0.43 (0.42) | 0.26 / 0.37 / 0.25 (0.30) | 0.36 / 0.31 / 0.32 (0.33) | 0.15 / 0.49 / 0.24 (0.29) | 0.28 / 0.31 / 0.20 (0.26) |

First words was passed (old mark, 0.25) at 15,000 moments by every arm, except two seeds without the relay (20,000),
like one control seed.

- **The relay** is the only part whose absence costs first words (0.35 against 0.42, all three seeds), and it halves
  naming on a page (0.19 against 0.36). As on the benchmark.
- **Looking** is what naming a thing on a page is made of (0.02 without it) and costs single pictures nothing.
- **The cortex's development**: nothing for single pictures, everything for a thing on a page (0.04, the level of a
  brain that does not look), and "is there a …?" falls to 0.27. See below.
- **Match neurons**: "is there a …?" falls on two of three seeds (0.15, 0.24 against 0.49 on the third); naming is
  untouched. A weak result: the spread between seeds is as large as the effect.
- **The episodic store buys nothing that these tests ask for**, and naming a thing on a page is *higher* without it
  on all three seeds (0.45 against 0.36). Its job, a name heard once, is not in any stage's test (it is measured
  apart: 0.9 against 0.1). A store that holds 55% leisure talk and answers by likeness of whole views may be
  getting in the way on pages. To be looked at before anything is claimed.

### Why is a thing on a page not named with never-developed kinds? Not answered yet

- The gaze is not the reason: it works from colour (`organic/gaze.py`) and does not use the kinds.
- In the probe (`look2_probe.py`, wide pages, 6,000 moments lived half single pictures and half pages, seeds 0 / 1)
  never-developed kinds do nearly as well: single pictures 0.530 / 0.563 (developed 0.520 / 0.587), a fixated thing
  on a page **0.370 / 0.450** (developed 0.467 / 0.530), "is there" 0.733 / 0.737 (0.747 / 0.727).
- In the life the same cortex gives 0.04 where developed kinds give 0.36, and its answers look like those of a brain
  that does not look: wrong names and run-together ones ("elephantbirdbanana"). So the probe does not reproduce the
  life. What differs: in the life the pages come after first words, the pages are the world's, and the store, the
  chooser and the mouth are there. **Next: on the saved brains (out/ablrand_s0.pkl, out/cleanw_s0.pkl), view by
  view on the test pages: what does the word cells' drive say for the whole page, and for each fixation?**

## 2026-11-23: the lateral network's settled activities as the code (seed 0, 10,000 pictures)

`out/bench/stream_settled_s0.log`. The Hebbian/anti-Hebbian network (Pehlevan & Chklovskii's rules) developed on
unlabelled pictures; the summary is made of its settled activities.

| after 10,000 pictures (5,000) | settled activities | winner-take-all kinds, usual response |
|---|---|---|
| word cells, no relay | **0.595** (0.578) | 0.500 (0.486) |
| word cells + batch relay | **0.670** (0.629) | 0.623 (0.599) |
| nearest class mean, no relay / + relay | 0.364 / 0.593 | 0.312 / 0.559 |
| streaming LDA, no relay / + relay | 0.668 / 0.662 | 0.664 / 0.670 |

- **Learned lateral inhibition does part of the relay's work, by a local rule, online**: without any relay the word
  cells gain 9.5 points on its code (0.500 → 0.595); the batch relay is worth 12 on the old code. A relay on top
  still adds 7.5 (0.670, above every word-cell number at 10,000 pictures so far).
- Streaming LDA is unmoved (0.66–0.67 either way): the code holds no more than before; it is better conditioned for
  a learner that sees each picture once. That is the same thing the relay does.
- One seed, a short stream, read once. Not a claim yet; and by rule 7 the number is not the point: the point is that
  the one part computed in batch has a local, learned form that goes most of the way. Next: three seeds on the whole
  stream; lateral inhibition among the 4,096 summary neurons (what the relay itself is) by the same rule.

## 2026-11-23 (afternoon): it is the store's key that fails on pages (`experiments/page_views_probe.py`)

Two things in the ablation table had no explanation: a never-developed cortex names single pictures and not a thing
on a page; and naming on a page is higher without the episodic store. Saved brains, 100 held-out pages each, nothing
learning. Share of pages named right:

| brain (seed) | the brain as it lives | the same, its store silent | some fixation alone | the coarse whole page alone |
|---|---|---|---|---|
| control (0 / 1 / 2; age 115,000–120,000) | 0.36 / 0.45 / 0.41 | **0.48 / 0.49 / 0.50** | 0.54 / 0.52 / 0.48 | 0.08 / 0.11 / 0.10 |
| never-developed cortex (0 / 1 / 2; 50,000) | 0.15 / 0.07 / 0.12 | **0.43 / 0.33 / 0.44** | 0.46 / 0.39 / 0.44 | 0.02 / 0.05 / 0.03 |
| no relay (0) | 0.27 | 0.38 | 0.39 | 0.10 |
| no match neurons (0) | 0.38 | 0.45 | 0.53 | 0.08 |
| no store (0) | 0.52 | – | 0.51 | 0.08 |

- **Every brain names the fixated thing alone about as well as it can name at all** (developed 0.48–0.54,
  never-developed 0.39–0.46). The never-developed cortex is not blind to a thing on a page.
- **With the store silent every brain with a store names more pages**: the control +0.04 to +0.12, the
  never-developed cortex +0.26 to +0.32. Its "failure" on pages was the store's.
- **Why:** the store is written and read under the whole view only (`organic/words.py`: `x[0]`). A page is seen
  whole only coarsely, and coarse pages look alike: 6–8% of pairs of test pages are alike enough for the store to
  answer (likeness in the top 5% of pairs: 0.84; with never-developed kinds up to 0.999), where no two single
  pictures are (top 5%: 0.05–0.07). So on a page the store gives back the name heard on another page, at full
  strength, against the fixation's right one. When the brain is wrong although a fixation alone is right, what it
  says is most often the whole page's answer.
- So, corrected: **development of the kinds is worth 0.05–0.10 on a fixated thing, not everything**; and **the
  store, as keyed, costs naming on a page in every brain that has it.** The ablation table stands as measured; its
  reading changes.

**Proposal: the store keeps a moment under what the eye was on** (queue item 10, now with a measured reason).
Organic check, written before building: (1) Organic? In primates the hippocampus is tied to where the eyes are:
cells that answer to the place looked at rather than the place stood in, and more fixations at study go with more
hippocampal activity and better memory [check: Rolls's spatial view cells; Liu, Shen, Olsen & Ryan 2017]. A moment
kept under the fixated window is closer to that than a moment kept under a blurred whole scene. It wires no
behaviour: the store still only says "a moment like this one came with these words". (2) Data, training or size?
None; a key. (3) The motive is not a score: the store's own job (a name heard once) cannot work on a page as it is,
and it harms what the slow cells know. Rule: a moment is kept under the windows the eye fixated, or under the whole
view when it fixated nothing; it is compared with what the eye is on now, likewise. To be tried first on the saved
brains (their stores hold each moment's fixations since 2026-11-18), nothing learning: pages as above, single
pictures, and the name-heard-once test.

### The store keyed by what the eye is on, on the saved brains (`out/tmp/page_views3.log`)

Same brains, same 100 pages and 180 single pictures, nothing learning; only how the store compares moments.

| share named right | pages: store as it is | store silent | store keyed by what the eye is on | single pictures: as it is / silent / keyed |
|---|---|---|---|---|
| control, seed 0 | 0.36 | 0.49 | 0.49 | 0.56 / 0.55 / 0.56 |
| control, seed 1 | 0.45 | 0.49 | 0.49 | 0.55 / 0.55 / 0.55 |
| control, seed 2 | 0.41 | 0.50 | 0.50 | 0.59 / 0.58 / 0.59 |
| never-developed cortex, seed 0 | 0.15 | 0.43 | 0.43 | 0.53 / 0.52 / 0.53 |

Keyed by what the eye is on, the store no longer harms pages (it reads exactly as a silent store there: the test
pictures were never seen, so no stored moment is like a fixated one, which is right) and single pictures are as
before. Whether it then does its own job on a page (a name heard once) is `experiments/fastmap_page_probe.py`,
running on three seeds.

### A name heard once for a thing on a page (`experiments/fastmap_page_probe.py`, three seeds)

The word cells live 6,000 naming moments (single pictures and pages, looking as the brain does). Then 75 kinds of
things never named: a picture of the kind is on a page and its name is heard once. Said again when asked (at once;
after 20 ordinary moments in brackets), seeds 0 / 1 / 2:

| | the very page | the picture on another page, another place | the picture alone, filling the view | another picture of the kind |
|---|---|---|---|---|
| word cells alone | 0.00 / 0.01 / 0.00 | 0.00 / 0.00 / 0.01 | 0.00 | 0.00 |
| + store keyed by the whole view | 0.81 / 0.88 / 0.88 (0.79 / 0.85 / 0.83) | 0.13 / 0.11 / 0.15 (0.12 / 0.11 / 0.15) | 0.00 | 0.00 |
| + store keyed by what the eye is on | 0.77 / 0.84 / 0.92 (0.76 / 0.81 / 0.85) | **0.85 / 0.92 / 0.91** (0.85 / 0.89 / 0.87) | 0.00 / 0.01 / 0.00 | 0.00 |

Held-out first words before and after the 75 new names: single pictures unchanged for all (0.62 → 0.61, 0.54 →
0.54, 0.61 → 0.62); on a page the keyed store reads as the cells alone (0.60 → 0.59, 0.54 → 0.59, 0.59 → 0.55), the
whole-view store a little lower before (0.53, 0.55, 0.57).

- **Keyed by what the eye is on, the name goes with the thing**: the same picture on another page, in another place,
  is named 0.85–0.92 of the time; keyed by the whole view, 0.11–0.15 (the store remembered the page, not the thing).
- **It still does not reach the picture filling the view (0.00), nor the kind (0.00).** The fixated window is the
  thing with the page's margin around it, at another size than the picture alone; the store compares steeply and
  finds no likeness. It is a memory of what was fixated, as it was fixated. Stated as a limit, not hidden.
- Nothing is lost on the names already known.

So the proposal holds on both counts measured: it takes the store's harm away on pages (saved brains, above) and it
lets the store do its own job there. **Not adopted in lives yet: it changes the brain, and that is Kris's word.**

## 2026-11-23 (evening): the lateral network's code on a second seed

`out/bench/lateral_settled_s1.log`, `stream_settled_s1.log`, `stream_developed10k_s1.log` (seed 2 is being coded).
The network itself comes out the same on seeds 1 and 2 as on seed 0: stable (inhibition mean 0.08–0.09, max 1.1–1.2),
no kind silent, filters more alike than the winner-take-all ones (962–1,108 pairs above 0.8; 649 for seed 2's
winner-take-all kinds) and settled activities apart (top 1% alike 0.40).

| after 10,000 pictures | seed 0: settled / usual | seed 1: settled / usual |
|---|---|---|
| word cells, no relay | 0.595 / 0.500 | 0.619 / 0.532 |
| word cells + batch relay | 0.670 / 0.623 | 0.651 / 0.632 |
| nearest class mean, no relay / + relay | 0.364, 0.593 / 0.312, 0.559 | 0.372, 0.595 / not run |
| streaming LDA, no relay / + relay | 0.668, 0.662 / 0.664, 0.670 | 0.666, 0.655 / not run |

- **Reproduced on seed 1: without any relay the word cells gain 9 points on the settled code** (0.532 → 0.619;
  seed 0: 0.500 → 0.595). The local, learned inhibition closes most of the gap the batch relay closes (to 0.632).
- With the batch relay on top the gain is smaller and less even (+4.7 on seed 0, +1.9 on seed 1): once the summary
  is decorrelated in batch, there is less left for the inhibition among kinds to do.
- Streaming LDA again unmoved (0.66–0.67): the code holds no more; it is better conditioned for a learner that sees
  each picture once.
- Still 10,000 pictures. Seed 2, then the reading on three seeds. The paper's figure of the ablation arms
  (`docs/paper/fig_ablation.pdf`) was drawn from the lives' logs today; it shows the seeds, and with them how weak
  the match-neuron row is (0.15 / 0.24 / 0.49) beside the rows for looking and the relay.

## 2026-11-23 (evening): what preschool is made of (`experiments/preschool_sources.py`; queue item 8)

No brain and no judge: the stage's held-out items, and what blind children get by the letter-level grade
(out/tmp/preschool_sources.log; kindergarten in kinder_sources.log). Preschool has **15 sources that take turns**, so
each gets one moment in fifteen whatever its size, and the 300-item test asks **20 of each**.

| source | to live | different answers (test) | commonest answer | best blind child |
|---|---|---|---|---|
| coco:count ("how many people are there?") | 4,204 | 4 | "zero" 0.57 | 0.57 |
| coco:presence ("is there a giraffe?") | 11,468 | 2 | "no" 0.60 | 0.60 |
| coco:where ("where is the person?") | 1,049 | 3 | "in the middle" 0.56 | 0.56 |
| vqa:yesno | 8,944 | 2 | even | 0.50 |
| page:count | 20,000 | 3 | "one" 0.35 | 0.35 |
| vqa:count | 2,483 | 35 | "2" 0.27 | 0.27 |
| vqa:color | 1,897 | 14 | "white" 0.21 | 0.21 |
| coco:name, cifar:name (new names) | 39,870, 8,069 | 15, 18 | 0.11, 0.07 | 0.11, 0.07 |
| text:mind, text:story, the adults' lessons | 863, 878, 1,519 | 90, 96, 145 | 0.02–0.06 | 0.06, 0.02, 0.14 (last word) |
| text:dialogue | 812 | 86 | 0.03 | 0.50 (the last word said back) |
| text:chat | 2,816 | 228 | "yes please" 0.03 | 0.25 |
| dailydialog | 11,464 | 1,028 of 1,255 | "thank you" 0.01 | 0.04 |

Faults, all of them in the world and none in the brain:

1. **One question, two ways of answering.** "how many giraffes are there?" is answered "2" in one source and "how
   many bears are there?" "one" in two others. An adult says a number; digits belong to reading.
2. **Three sources we build are lopsided by how we build them**, not by how the world is: "how many people are
   there?" is asked mostly of photographs with nobody in them ("zero" 0.57; and who asks that of an empty room?),
   "is there a …?" is "no" six times in ten, "where is the …?" is "in the middle" more than half the time. A blind
   child gets 0.56–0.60. The score above the floor already takes the floor away, but its chance error is divided by
   what is left above the floor, so these sources say little with 20 items.
3. **Three sources have no answer to judge against.** A reply in a conversation ("right, congratulations." → "thank
   you"; 1,028 different replies among 1,255) is not right or wrong by an answer key; the judge, strict on meaning,
   can only refuse nearly everything. This is talk: to be lived and listened to, like leisure talk, not tested.
4. **Twenty items a source** is too few to read a source by.
5. Kindergarten has one of the same kind: "what is the biggest thing?" is "person" a third of the time.

**Built the same evening as world 5** (`organic/curriculum.py`, a block on `world >= 5`; world 4 stays exactly as it
is, three lives are in it: `experiments/world_hash.py 4` gives the same fingerprint of lived items, held-out items
and preschool's test before and after):

- a number is said as a word in every source ("how many giraffes are there?" → "two"; the few answers above ten
  are left out);
- the three sources made from annotations are drawn evenly over their answers, on the lived and the held-out side
  alike (the commoner answers thinned, always the same items): "how many people" 0.25 each of zero to three (804
  to live), "is there a …?" 0.50 (8,858), "where is the person?" 0.33 (798);
- conversation (`TALK`: dailydialog, text:chat, text:dialogue) is lived as before, and its held-out items are kept
  apart (`Curriculum.talk`), not in the test. What a brain says back in a conversation is to be read, not marked;
- so the test asks 12 sources, 25 items each at 300. Still few for one source: a life can ask more (`TEST_N`).

Left as the photographs are: "what colour" (white 0.21), the VQA counts (two 0.30), kindergarten's "biggest thing"
("person" 0.34). These are how the pictures are, not how we asked.

Check against the rules: nothing here touches the brain or is chosen to move a number; it is the ruler and the
lessons (training program before architecture). The plateau lives will enter world 4's preschool as it is, which is
an observation worth having beside the rebuilt one. **New lives: `WORLD=5`.**

## 2026-11-23 (18:00): the lateral network's code, three seeds

`out/bench/stream_settled_s{0,1,2}.log`, `stream_developed10k_s{1,2}.log`, 10,000 pictures.

| word cells after 10,000 pictures | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| usual code, no relay | 0.500 | 0.532 | 0.508 | 0.513 |
| **settled activities of the lateral network, no relay** | 0.595 | 0.619 | 0.602 | **0.605** |
| usual code + batch relay | 0.623 | 0.632 | 0.627 | 0.627 |
| settled activities + batch relay | 0.670 | 0.651 | 0.648 | 0.656 |

- **Three seeds: the published Hebbian/anti-Hebbian network, by local rules and online, gives the word cells 9.2
  points without any relay (+9.5, +8.7, +9.4), four fifths of the 11.4 the batch relay gives.** The batch relay on
  top adds another 5.
- Streaming LDA: 0.664–0.668 on the settled code without a relay (0.664 on the usual code, seed 0), and *lower*
  with the relay on the settled code (0.635–0.662). The code holds no more than before; it is better conditioned.
- Still a fifth of the stream; the windows' own whitening is still computed in batch. Next: the whole stream
  (about six hours a seed to code), and the same rule among the 4,096 numbers of the summary, where the relay sits.

## 2026-11-23 (18:00): the paper rewritten; what a referee found

Kris on the first PDF: the title is dull, the writing cold and clipped ("slop, AI-ish"), it must be fun to read.
Three independent readers were brought in (a survey of how well-known papers in the field are written; a hostile
referee with access to the notebook and the logs; a cold reader marking sentences). The draft was rewritten whole
(docs/paper/DRAFT.md; working title "It Was the Code, Not the Rule").

The referee's points, each checked against the logs before it was accepted. **All held:**

1. **Two parts are computed in batch, not one**: the windows' whitening as well as the relay (`organic/cortex2.py`,
   both `eigh`). "Local rules only" was false as written. A centre-surround retina in place of the windows'
   whitening had cost 0.065 of the linear reference (2026-11-17 table).
2. **Like for like, backpropagation wins**: without the relay the word cells reach 0.591 and one-pass
   backpropagation 0.616. The lead is a whitening step the reference was not given. **Owed: an end-to-end reference
   with a whitening layer before its readout.**
3. **Streaming LDA does not need the relay** (0.722 without, 0.741 with, seed 0), and **development is worth two
   points to it** (0.739 against 0.719 with never-developed filters, three seeds): "development buys nothing" was
   true of our reader only.
4. **The store's 1.00 on a picture shown once is true by construction**, and "nobody reaches the kind" may be an
   artefact of 20 pictures against 550 in 4,096 dimensions. Claimed more modestly now.
5. **The life is not "each moment once"**: the world draws from finite pools with replacement. Owed: how often a
   picture comes again.
6. **"First words improves while pages is lived"** is at least partly continued training (30% of moments go back).
   Owed: a life with no going back.
7. **One number was tuned** (the relay's shrinkage, four values on held-out naming), beside "nothing was tuned".
8. The test uses 2,000 of the 10,000 test pictures (chance error 0.010): differences under 0.02 are not read.
9. Missing for a reader: a drawing of the learner, the rules as equations, a table of settings, compute, curves of
   a life, the judge's instruction, examples.

Kris, on the second PDF's way: a part that *shows* what was built (babbling, a timeline of a life, its size,
pictures with its answers) and how long it takes to answer quick and slow questions. Probe:
`experiments/showcase_probe.py` (a saved learner asked item by item; ticks of looking before it speaks, ticks of
speaking, eye moves; its size organ by organ).

## 2026-11-23 (19:20): how often a thing comes again in a life; the like-for-like reference, first reading

**Recurrence** (the lives' own counts of lessons lived by source, beside the number of different items the world
holds; `plat_s0` to 45,000 moments, 35,000 of them in first words):

| first words source | lessons lived | different items | each met on average |
|---|---|---|---|
| CIFAR names | 6,043 | 7,206 | 0.84 times |
| COCO cut-outs | 5,915 | 50,655 | 0.12 |
| Commons photographs | 5,887 | 3,102 | 1.9 |
| the adults' spoken lessons | 6,026 | 1,076 | 5.6 |
| social routines | 5,896 | 90 | 66 |

In the pages stage held for 95,000 moments (`cleanw_s0`): a page 1.1 times, a spoken lesson 10 times. So for
photographs a life is close to one pass, and for the spoken lessons it is not: the sources take turns whatever
their size, and the small ones come round. The paper says so now. It also bears on text being the weak part
(0.11–0.33 above the floor): that is after five to ten meetings, not one.

**Backprop with the relay step, tuning so far** (seed 0, the first 10,000 pictures, 1,000 test pictures;
`out/bench/backprop_relay_tune_s0.log`): relay fixed, rate 0.001: **0.597**; rate 0.003: 0.572. The reference
without a relay had 0.40–0.53 in its tuning and 0.523 at 10,000 in its full run; the word cells with the relay have
0.623 there. If this holds on three seeds and the whole stream, most of the lead over end-to-end backprop was the
relay it had not been given, as the referee said. Four tuning runs and the six full runs are still to come.

### 2026-11-23 20:20: no pass mark, 45,000 moments of first words and still rising; backprop's tuning grid widened

**Lives `plat_s0..2`** (`GATE=plateau`): first words, above the blind floor, test by test from 15,000 lived:

| seed | 15k | 20k | 25k | 30k | 35k | 40k | 45k | 50k | 55k |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.33 | 0.35 | 0.36 | 0.41 | 0.41 | 0.43 | 0.44 | 0.47 | 0.49 |
| 1 | 0.32 | 0.33 | 0.38 | 0.39 | 0.44 | 0.46 | 0.40 | 0.46 | |
| 2 | 0.41 | 0.41 | 0.43 | 0.43 | 0.44 | 0.49 | 0.47 | 0.50 | |

The rule asks whether the mean of the last three tests is above the mean of the three before by more than the
chance error of that difference (0.030 here). It still is, by 0.049 / 0.038 / 0.052, so none has left the stage.
The earlier lives left it at 0.27–0.29 after 5,000–10,000 moments, on our mark. What this says so far: the stage
had nearly twice as much to give as the mark took; and the rise is slow (about one point per 5,000 moments), close
to the rule's own resolution, so a life may leave on a chance dip (seed 1's 0.40 at 45,000 nearly did it one test
later: margin 0.038 against 0.030). Not changed: the rule is what Kris asked for and it is running; the margin is
recorded with each test for the reading.

**Backprop with the relay, tuning** (seed 0, first 10,000 pictures, 1,000 test pictures, so ±0.015):

| relay | rate 0.001 | 0.003 | 0.01 |
|---|---|---|---|
| fixed | 0.597 | 0.572 | 0.527 |
| settled again every 5,000 | 0.602 | 0.564 | (running) |

Fixed and tracked do not differ. The best rate is the lowest on the grid, and the first reference had been tuned
over batch size as well. A reference whose best setting sits on the edge of its grid is not yet tuned, so
`out/bench/backprop_relay_queue2.sh` adds rates 0.0003 and 0.0001 and batch 8 after the first queue, and runs a
winner on three seeds. The paper keeps its pending mark and says the search is being widened.

**The drawing of the learner** (referee: the paper had no picture of what was built): `fig_architecture` in
`experiments/paper_figures.py`. One box a part, filled by how the part comes to be what it is: learned online by a
local rule (visual cortex, ear, word cells, store, mouth), computed in batch before learning (the two whitening
steps), learned from reward (the gating policy), a fixed reflex (where the eye goes). The two batch boxes are the
ones the paper's result rests on, and the drawing shows that at a glance.

### 2026-11-23 21:08: the first stage left with no pass mark; the word cells' step, never stated before

`plat_s1` left first words at 55,000 lived: the mean of its last three tests (0.40, 0.46, 0.49 → 0.45) was above
the three before (0.39, 0.44, 0.46 → 0.43) by 0.025, and the chance error of that difference is 0.030. Its newest test was its highest, and the low test at 45,000 is what closed the gap: a leave
on one noisy test, as feared. The rule was not changed. What it shows about the rule: with 300 questions and
three tests a side it cannot tell a rise slower than about 0.03 per 15,000 moments from none, and late first
words rises at about that pace. More tests a side (PLATEAU_SPAN) or a slope over the whole stage would see
finer; that is a choice for the next lives, with Kris.

**The word cells' step.** Writing the paper's table of settings: the step of each synapse is not one number. It
is the cell's error times the input, divided by (1 + u/5), u the running sum of that input's squared activity
(`maturation = 5` in organic/words.py), and the correction is shared among the kinds of input active in the
moment. A synapse that has carried much signal moves little: consolidation with use, local, and the same idea as
AdaGrad's per-coordinate step. The paper described a plain delta rule; it now says this. It matters for the
paper's account: a per-input step evens out inputs of different *size* on its own, so what the relay adds is the
removal of their *correlations*; and the "gradient readout does as well" control (Adam) had a per-coordinate step
too, so that comparison was like for like in this respect.

### 2026-11-23 21:45: a learned gaze and a cortex from biology: the organic checks (Kris's directions, rule 8)

Three surveys: docs/RESEARCH_GAZE.md, docs/RESEARCH_CORTEX.md, docs/RESEARCH_RECENT.md. What follows is the plan
they lead to. Every item names the biology it rests on; none is there because it scores. Each is built as a
probe, measured (also by what it does to the code: sparseness, near copies, selectivity, how many looks an
answer takes), taken out again, and adopted on its biological ground whatever the number (rule 8).

**What "looking" is today.** One glimpse hands the word cells the whole view and up to four fixated windows at
once; each cell takes the view that drives it most; match neurons count in how many views the heard word's cells
are driven. No order, no cost, no choice. Nothing to learn.

**The learned gaze, in the order of how well each part is established:**
1. *Looking becomes serial* (wiring; a priority map and winner-take-all: Bisley & Goldberg 2010). One fixation a
   tick. Before a place is fixated it is seen only as the coarse whole view shows it (the periphery); the fovea's
   detail comes with the fixation. A look costs a tick, as waiting does now, so how many looks an answer takes
   is something the learner pays for and can learn about. The reflex (organic/gaze.py) stays as the salience
   term: it is the newborn's gaze.
2. *The heard word biases the map* (Tanenhaus et al. 1995; Bergelson & Swingley 2012; Bichot et al. 2015;
   McMurray et al. 2012). No new rule: the cells of the words just heard are already driven by what is seen
   (their own delta rule). Their drive from the coarse view of each candidate place is added to its salience.
   The eye then goes first where the named thing probably is. Organic check: this is the template account, and
   the learning is the word cells' own; what is wired is only that the drive reaches the map.
3. *Value opens the gate* (Kawagoe et al. 1998; Kim & Hikosaka 2013; Yagishita et al. 2014). A three-factor
   rule: a trace on the synapses from what was seen at a place (coarse) and what was heard to the choice of that
   place; the moment's reward prediction error (r − V, the signal the gating policy already has) turns traces
   into changes. A fast and a slow memory of value, as in the caudate's head and tail.
4. *Staying and letting go* (Hanes & Schall 1996; Johnson, Posner & Rothbart 1991). The learner already speaks
   when its confidence passes a learned patience; with serial looking the same rise to a threshold decides
   between another look and an answer. Inhibition of return is in the reflex already.
5. Later, and marked as less established as a learning signal: the information a look gave (Horan et al. 2019)
   as part of the reward.
What emerges and is only observed (never a target): the order of looks, how many looks by kind of question, the
time to the named thing by age (children: 984 → 769 ms from 15 to 25 months).
First probe (`experiments/look3_probe.py`, to write): on pages, with a saved learner, nothing learning except
the gaze: how many looks until the named thing is fixated under the reflex alone, with the word's bias, with
value; naming when only the first k looks are had.

**The cortex against biology** (docs/RESEARCH_CORTEX.md, table C). In the order of how well established and how
far off we are:
1. *Divisive normalisation* in place of "response above the mean, rectified" (Carandini & Heeger 2012; Burg et
   al. 2021): R = Dⁿ / (σⁿ + pool), n about 2, the pool over similar filters and the near surround. Ours is
   subtractive and far too dense (about half the units on; V1: 0.5% strongly, lifetime sparseness 0.4–0.9).
2. *Complex cells and stages* in place of one 4×4 average (Rust et al. 2005; Serre et al. 2007): squared
   responses summed over about twice the field within a feature, then stages that each about double the field.
3. *Several patch sizes at each place* (De Valois et al. 1982).
4. *A retina* in place of exact batch whitening of patches (Pitkow & Meister 2012; Mante et al. 2005):
   centre-surround ON and OFF and colour-opponent channels, local gain control, a threshold. This also takes
   away one of the paper's two batch stand-ins. (Tried on 2026-11-17 as a bare centre-surround without gain
   control or threshold: 0.469 against 0.537 for whitening. Under rule 8 the number does not decide; the
   faithful version is owed.)
5. *Soft competition in development* (queue E 27) and *the relay by interneuron gains* (queue E 28).
6. *Filters before experience*: the same rule first run on wave-like spontaneous patterns (Albert, Schnabel &
   Field 2008; Espinosa & Stryker 2012), then on pictures.
7. *No shared weights* (Pogodin et al. 2021): the largest change in cost; last.
Width: V1 is 25 to 356 times its input, so 256 filters are not too many; more is a question of the machine, not
of biology, and is not sought for a score (rule 7).

**Order of work.** The gaze first (Kris named it first, and it needs no new cortex). Then the cortex items one
at a time, each as a probe on 10,000 pictures beside the present cortex. Memory decides when: 2.3 GB are free
beside three lives and the backprop queue, so small probes only until a queue ends.

### 2026-11-23 22:29: the heard word leads the eye (look3_probe.py, seeds 0 and 1; seed 2 running)

Word cells raised as the learner is now (the whole view and every fixated place at once; 6,000 moments of single
pictures, pages, "is there a …?", "how many …?"). Then nothing learns, the looks are taken one at a time, and the
same candidate places are ordered three ways. Held-out pages where one candidate place holds the named thing and
another does not (111 of 300 "is there" pages), "is there a …?":

| order of looks | named thing in the fovea on the first look (s0 / s1) | mean look | right with one look only |
|---|---|---|---|
| reflex (strongest contrast first; today) | 0.46 / 0.46 | 1.73 / 1.73 | 0.600 / 0.610 |
| the heard word's pull on the periphery | 0.82 / 0.85 | 1.23 / 1.19 | 0.671 / 0.700 |
| random | 0.38 / 0.41 | 1.78 / 1.78 | 0.571 / 0.595 |

(with all looks: 0.710 / 0.743; the commonest answer 0.52.) "How many …?", seed 0: first look 0.92 against 0.57.
So the word cells' own visual synapses, read on a blurred view of each place before the eye moves, already know
where the named thing probably is. No new learning rule was needed for that: it is the template account, and the
learner had the template all along without using it to move its eye. The reflex is no better than chance at
finding what was asked for, as it should be: it knows nothing of the question.
What this does not show yet: that the answer gets better. With one look the word's order answers 7–9 points
better than the reflex's; with all looks the orders are the same by construction (the same views). The gain of a
serial, word-led gaze is in looks saved, which matters only once a look costs time.

**What the learner's code has, read for the next step.** There are two eyes in `organic/speech.py`. The old one
(`self.eye`, a grid of targets at three levels of detail) is moved tick by tick by the gating policy, which
chooses among eye commands with the contrast reflex as a bias and is credited by reward (`_choose`, `_credit`):
a learned gaze of the kind the survey describes, feeding the gating policy only. The new one (`_glimpse`, 2026-11-17)
hands the word cells the whole view and up to four fixated windows before the first tick: no order, no cost. The
word cells are the part that names things, and they are served by the eye that does not learn. The step is to
join them: the places the word cells get become the places the eye commands choose among, one a tick, with the
reflex and the heard word's pull as the biases and reward as the teacher. When to stop looking is the second
question (a rise to a threshold): the extended probe reads how sure the word cells are look by look
(`organic-look3b`).

### 2026-11-23 23:45: the gaze read-outs on three seeds, and the serial gaze in the learner (behind a switch)

**The read-outs** (look3_probe.py, seeds 0 / 1 / 2; "is there a …?", 111 of 300 held-out pages where one candidate
place holds the named thing and another does not):

| order of looks | named thing in the fovea on the first look | right with one look only |
|---|---|---|
| reflex (today) | 0.46 / 0.46 / 0.46 | 0.600 / 0.610 / 0.629 |
| the heard word's pull | 0.82 / 0.85 / 0.81 | 0.671 / 0.700 / 0.714 |
| random | 0.38 / 0.41 / 0.38 | 0.571 / 0.595 / 0.571 |

With all looks 0.710 / 0.743 / 0.793 in every order (the same views). "How many …?": first look 0.92 / 0.90 / 0.91
against 0.57, and no gain in the answer from any order, as it must be: to count, every place has to be seen.

**When to stop looking is the weak part.** How sure the word cells are, look by look (the reflex's order):
- it rises with the look that brings the named thing into the fovea: 0.388 → 0.434, 0.374 → 0.434, 0.421 → 0.464
  ("how many": 0.308 → 0.322, 0.325 → 0.331, 0.307 → 0.332);
- it separates right answers from wrong ones only a little: 0.443 against 0.390, 0.447 / 0.390, 0.475 / 0.439
  ("how many": 0.338 / 0.328, 0.352 / 0.345, 0.354 / 0.347);
- a rule "another look until sure enough" stops after nearly the same number of looks in every order (word against
  reflex: 1.19 against 1.23, 1.29 / 1.32, 1.22 / 1.22 at the middle level) with a little more right at the low
  level (0.667 / 0.643, 0.740 / 0.727, 0.743 / 0.697).
So the word cells know *where* far better than they know *whether they have seen enough*. The step 4 of the plan
(staying as a rise to a threshold on the word cells' sureness) has little to stand on in what the word cells now
carry. Not built. What decides between another look and an answer stays what it is: the gating policy's own
patience, taught by reward and the cost of time.

**The serial gaze, as built** (`organic/speech.py`, `gaze = "serial"`; life env `GAZE=serial`; off by default):
- `_glimpse` codes the whole view and finds the candidate places (organic/gaze.py, as before), and codes each as
  the periphery has it (the whole view's own coarseness, that place enlarged). The word cells get the whole view.
- Each tick's eye command is the gating policy's, as it always was for the old eye. The map it chooses on gets,
  at the target of each candidate place: half its standing-out (the reflex), plus the heard words' pull (the drive
  of the heard words' cells from the place's coarse view, the probe's measure), or −1 where the eye has been
  (inhibition of return). When the eye lands on a candidate place the word cells get the fovea's view of it and
  recall again.
- The eye commands are credited as before: an eligibility on the active clusters, fading back from the reward,
  times the reward's error against that command's own running value. That is a three-factor rule already; the
  plan's step 3 was half there. Not there: a fast and a slow memory of value.
- When an utterance is heard while looking (the adult's reply, an unasked comment, what is said back in leisure
  talk), the eye goes once more, to the place its words pull most, before the word cells learn. This one is a
  reflex, not a choice (no vote of the gating policy, nothing credited).
Organic check. What is wired: that the word cells' drive reaches the eye's map (the template account: Bichot et
al. 2015; a heard word draws the eye: Tanenhaus et al. 1995, Bergelson & Swingley 2012), inhibition of return
(in the reflex already), that the fovea's view arrives when the eye lands. What is learned: where to look and
whether to look again, by the gating policy from reward; what a word's thing looks like, by the word cells.
Nothing says "look at the named thing".
Two choices that are stand-ins, written down as such:
1. *The periphery is the coarse page with the place enlarged.* A real periphery does not enlarge; a real template
   acts on features wherever they are. Our word cells' synapses are tied to position in the view, so the place
   has to fill the view to be compared. A cortex with position-tolerant stages (queue E 24, complex cells) would
   make this unnecessary.
2. *The fovea hands the word cells a view only at places that stand apart.* A real fovea always delivers. On
   2026-11-17 handing the word cells parts of a single thing cost its naming, and the `apart` rule came from
   that; the serial gaze keeps it so that this step changes one thing. Owed: the fovea always delivering.
What the learner may lose: today it gets every candidate place for nothing. With the serial gaze a look costs
ticks, the old eye takes three ticks at a place before the next command, and an answer given before the first
command has seen the page only coarsely. Whether the patience learns to wait for a look is the question.

Checked: with the switch off, a learner of 300 moments says and does exactly the same before and after the change
(one digest over all answers, ticks, rewards and eye paths).
**Running** (`experiments/gaze_life_probe.py`, unit `organic-gaze1`): a small learner raised twice a seed on the
same moments, both ways. Read: looks before answering, the first look on the named thing, grades on pages, ticks.
A small learner from a late start is not a life; the same from a grown learner's checkpoint is owed when memory
allows, then lives.

### 2026-11-24 00:58: the serial gaze as first built is never used; the question must draw the eye too

`experiments/gaze_life_probe.py`, seed 0 (out/tmp/gaze1.log): a small learner (1,000 clusters, an ear that has
heard speech; 2,000 moments of first words, then pages) raised twice on the same moments. Held-out, nothing learning:

| moments lived | pages, all at once | pages, serial | "is there" right, all at once | serial | looks before answering (serial) | eye moves (both) |
|---|---|---|---|---|---|---|
| 3,000 | 0.30 | 0.31 | 0.45 | 0.47 | 0.03 | 0.04–0.08 |
| 4,000 | 0.45 | 0.35 | 0.59 | 0.56 | 0.01 | 0.00–0.01 |
| 5,000 | 0.58 | 0.29 | 0.67 | 0.59 | 0.02 | 0.02–0.06 |
| 6,000 | 0.56 | 0.44 | 0.67 | 0.55 | 0.01 | 0.07–0.08 |

One seed of a small learner, so only the plain facts are read:
1. **The learner stops moving its eye once it answers.** In both arms the eye commands fall from two a moment
   at 1,000 moments to none by 3,000. This is the old eye too: where looks are free (all at once) a command only
   costs time, and the gating policy learns that. A gaze learned from reward and a cost of time alone learns
   not to look.
2. **With the serial gaze it answers before its first eye command.** The old eye spends three ticks at a place
   before a command can be given, and the answer begins at about the third tick. So the word cells answer from
   the coarse page and the pages fall behind (0.44 against 0.56). What kept the word cells learning the pages at
   all is the look the adult's reply draws (0.95 a page).
3. Nothing ties the decision to speak to what a look would bring: the gating policy's patience is on its own
   motor confidence, and how sure the word cells are is a weak signal (the read-out above).
What was wrong in the build, on its own ground and not by the number: the reply drew the eye by reflex and the
question did not. That is not what is known. A heard word draws the eye to what it names within about 200 ms,
before any answer and without being asked to (Tanenhaus et al. 1995; in infants Bergelson & Swingley 2012), and
a speaker fixates a thing close to a second before naming it (Griffin & Bock 2000). The first look after a
question is the reflex's, led by the question's words; what can be learned is what follows it.
**Changed** (`look_on_question`, on with the serial gaze): on hearing the question the eye goes once to the
candidate place that stands out and that the question's words pull most, before anything is said; it takes one
tick. The eye commands after it stay the gating policy's. Also: the serial gaze starts again at each new moment
(a scene asked about twice kept its looks).
Organic check: is this the designer saying "look at the named thing"? The reflex is "the eye goes where the
heard words' cells are driven"; where that is, the word cells have learned. Before they know a word the look
goes to what stands out, as a newborn's. Rule 7: the change is not made to raise the pages' grade; the grade is
read, and the arm without it is kept as `serial_noq`.
Queued: `organic-gaze2` (first a check that the default path still gives the same digest as before the serial
gaze existed; then the serial arm again, seed 0 → out/tmp/gaze2.log).

### 2026-11-24 10:34: the eye's map without a hand-set weight; the question's look, second reading

**The serial gaze with the question drawing the eye** (gaze2, seed 0; small learner; held-out, nothing learning):
looks before answering 0.87–0.93 a page; pages 0.33 / 0.39 / 0.55 / 0.50 at 3,000–6,000 moments (all at once 0.30
/ 0.45 / 0.58 / 0.56; without the question's look 0.31 / 0.35 / 0.29 / 0.44); "is there" 0.62 at 6,000 (0.67;
0.55). But the first look was on the named thing on only 0.46–0.51 of the pages where it matters: the reflex's level.
**Why** (out/tmp/pull_size.py on the read-out's word cells, three seeds): the pull is 0.06 at the named place and
0.02 at the others, and half the standing-out varies by 0.19 across a page. "0.5 × standing-out + pull" was my
number, set without measuring either term; it drowned the word (first look on the named thing 0.50–0.53; by pull
alone 0.81–0.85; standing-out × pull 0.79–0.82; each scaled to its largest on the page and added 0.79–0.83).
Kris: "why fixed, research a better solution". The survey is in docs/RESEARCH_GAZE.md (second part).
**As built now** (`organic/speech.py`, `_priority`, `_eligible`, `_credit_map`):
- Within each source the places compete: a place's value is its share of the source's whole activity over the
  candidate places (divisive normalisation across alternatives: Louie et al. 2011 in LIP; Mysore & Knudsen 2011 and
  Banerjee et al. 2026 in the colliculus and tectum). This takes the raw sizes out.
- The map is the sources' weighted average (Ipata et al. 2009; Wolfe 2021), the weights non-negative on a fixed total.
- The weights are learned from reward (Chelazzi et al. 2014; Hikosaka et al. 2014), by the rule the eye commands
  already learn by: for each look taken before the answer, an eligibility (how much more than the other places
  still unseen the source favoured the place looked at) times the reward's error against the usual reward of such
  moments; the same two rates as `_credit` (0.1 and 0.02), no new number.
- The learner is born with the reflex alone (weights 1 and 0): a three-month-old's looking follows salience
  (Frank, Vul & Johnson 2009). Whether and when the words' pull takes over is observed, never a target; infants
  show word-driven looks from six months.
Organic check: nothing says how much the word counts. If going where the words pull earns more than usual, the
pull's weight grows; if the word cells know nothing yet, it has nothing to grow on. What is ours and not a
finding: two scalar weights for the balance. What is left out: the time course (fast looks follow salience, later
ones the goal), which a tick does not resolve.
Caution written before the reading: the teaching signal is small. In the read-out an answer after one look was
7–9 points better when the look was the word's, on a third of the pages. The weight may move slowly or not at all
in 4,000 moments of a small learner; a life is the test.
Default path checked again (digest c52819a8c3a93a7b as before). **Running: `organic-gaze3`** (serial, seed 0 →
out/tmp/gaze3.log; it prints the map's two weights at each test).

**Architecture ideas from 2023–2026 work**: docs/RESEARCH_ARCH.md (second survey); queue E 29.

### 2026-11-24 11:16: the word's weight in the eye's map grows from reward (gaze3, seed 0); divisive normalisation on the stream

**The map's weights, learned** (`experiments/gaze_life_probe.py`, serial arm, seed 0; small learner; out/tmp/gaze3.log).
Held-out, nothing learning; "first look" on the 37 "is there" pages where one candidate place holds the named
thing and another does not:

| moments lived | weight of what stands out | weight of the words' pull | first look on the named thing | pages | "is there" right |
|---|---|---|---|---|---|
| 1,000 | 1.00 | 0.00 | 0.49 | 0.04 | 0.00 |
| 2,000 (pages begin) | 0.99 | 0.01 | 0.51 | 0.11 | 0.01 |
| 3,000 | 0.85 | 0.15 | 0.41 | 0.36 | 0.51 |
| 4,000 | 0.54 | 0.46 | 0.62 | 0.45 | 0.61 |
| 5,000 | 0.42 | 0.58 | 0.65 | 0.57 | 0.68 |
| 6,000 | 0.25 | 0.75 | 0.65 | 0.53 | 0.66 |

The same seed's other arms at 3,000–6,000, pages: all at once 0.30 / 0.45 / 0.58 / 0.56; serial with my fixed half
0.33 / 0.39 / 0.55 / 0.50; serial without the question's look 0.31 / 0.35 / 0.29 / 0.44. "Is there": all at once
0.45 / 0.59 / 0.67 / 0.67.
What this says, on one seed of a small learner:
1. **The shift from looking at what stands out to looking at what was named came by itself**, from reward, in
   about 3,000 moments of pages. Nothing set how much the word counts. An infant's looking makes the same shift
   between three and nine months; ours is only observed, and the likeness is not evidence of the same mechanism.
2. The first look goes to the named thing on 0.65 of the pages, up from the reflex's 0.49, and still under the
   0.82 of the read-out with grown word cells: this learner's word cells are young (first words 0.30).
3. **With one look before it answers (0.87 a page) the serial learner is level with the learner that is handed
   every place for nothing** (pages 0.53 against 0.56, "is there" 0.66 against 0.67; one seed, 100 items: not
   told apart). It pays about one tick more a moment.
4. The caution I wrote beforehand was wrong in its size: the weight moved fast. Why it can: when the reflex's
   choice happens to be the place the words favoured, the answer is better than usual, and when it is not,
   worse; both push the words' weight up. A drift from the rule itself would favour the reflex (its eligibility
   is always positive while reward is still rising), so the rise is not such a drift. **The control is running**
   (`serial_shuf`: the pull shuffled over the places; its weight should stay near nothing).
Running: `organic-gaze4` (the control on seed 0; seeds 1 and 2, all at once and serial).

**Divisive normalisation, the stream** (10,000 pictures, seed 0; out/bench/stream_{norm0,divnorm,divnorm_s}10k_s0.log):

| after 10,000 pictures | response as now | divisive, pool at the place | pool also over the surround |
|---|---|---|---|
| word cells with the relay | 0.613 | 0.624 | 0.610 |
| word cells without | 0.502 | 0.519 | 0.506 |
| ridge with the relay (reference) | 0.659 | 0.654 | 0.637 |
| ridge without | 0.646 | 0.638 | 0.634 |

Level, within what one seed and 2,000 test pictures can tell (about ±0.011). The code itself changed as cortex's
does: sparser (0.83 against 0.71 over the kinds at a place; 3.9% of kinds above half the strongest against 8.5%)
and a response that grows with contrast and saturates, where ours did not depend on contrast at all. The surround
pool is the weaker of the two here and no sparser over kinds. Under rule 8 the divisive response is the one to
carry forward, on its ground; the surround waits for three seeds. Seeds 1 and 2 are queued behind the gaze runs.

### 2026-11-24 11:28: a second stage that learns from the eye's drift (queue E 29a): the organic check and the probe

**What is established.** Complex cells answer to their feature over a range of positions; simple cells do not.
The account with the most support for how that tolerance is acquired is temporal: what the eye sees a moment
later is nearly always the same thing slightly moved, and a cell that has just fired is still marked when the
next view comes, so it learns that view as well (Földiák 1991; Wallis & Rolls 1997). The monkey experiments that
swap one object for another during a saccade change what IT cells treat as the same thing, in the direction
this predicts (Li & DiCarlo 2008). Halvagal & Zenke 2023 give a modern local rule built around the same term
and show that without it a stack of local layers learns nothing useful. The source of the small movements is the
eye itself: within a fixation it drifts by about a receptive field (Rucci & Poletti 2015).
**What we have.** Stills, and a fixed 4 × 4 average as the only tolerance to position. No cell has learned what
goes together across a small movement.
**The probe** (`experiments/cortex_trace_probe.py`). The first stage stays (developed filters, pooled 2 × 2). A
second stage, the one of 2026-11-17 (256 kinds in 16 groups of 16, each listening to 32 kinds of the first over
3 × 3 places), develops by the first stage's own rule on fixations of four views, the picture sliding a pixel at
a time. Two arms on exactly the same views: *plain* (only the winner of a view learns it) and *trace* (a cell
that won an earlier view at that place learns the present one too, in proportion to a trace that keeps 0.7 a
view). Read: how alike a cell's answers are to a picture and to it two pixels aside; sparseness; then the stream
on the second stage's summary beside the first stage's (0.613 on these pictures).
**What is wired and what is learned.** Wired: that a cell's having fired lingers (a trace), and that the eye
drifts. Learned: which first-stage cells a second-stage cell pools. Nothing names a feature or a tolerance.
**Stand-ins, written down.** The trace's persistence (0.7 a view) is borrowed from the persistence the eye's own
memory of what it saw has in the learner (`seen = 0.7 * seen`); it is a time constant, and I have not derived it
from a tick's length. The drift is a random walk of one pixel a view, kept within three pixels. The second
stage's summary alone goes on to the relay (both stages together make a code twice as wide; later).
Noted for the whitening item (queue E 26): the same drift is, with the retina's transient response, a known
source of spatial whitening (Rucci & Victor 2015; not yet read by us), a local alternative to our batch step.

### 2026-11-24 12:28: the control fails the first rule for the map's weights; the rule corrected

**The control** (`serial_shuf`, seed 0, out/tmp/gaze4_shuf.log): the words' pull shuffled over the candidate
places, so that it says nothing about where the named thing is.

| moments lived | weight of the pull, real | weight of the pull, shuffled | first look on the named thing, real | shuffled |
|---|---|---|---|---|
| 3,000 | 0.15 | 0.00 | 0.41 | 0.49 |
| 4,000 | 0.46 | 0.22 | 0.62 | 0.35 |
| 5,000 | 0.58 | 0.35 | 0.65 | 0.49 |
| 6,000 | 0.75 | 0.53 | 0.65 | 0.35 |

**A source that says nothing gained more than half the map's weight.** So the rise of the real pull's weight was
not, as I wrote at 11:16, the reward teaching the eye to trust the word: at most the part by which the real arm
is ahead (0.75 against 0.53) is that, and one seed cannot say. What I claimed there ("the shift came by itself,
from reward"; "a drift from the rule would favour the reflex") is withdrawn. The second seed of the real arm
(0.25 / 0.54 / 0.67 at 3,000–5,000) repeats the rise and repeats nothing more.
**What was wrong with the rule** (a mistake of mine in the rule, found by its control):
1. The place of highest priority was always taken, and a source's eligibility was "how much more than the other
   places it favoured the place taken". For the place that wins the map that is positive more often than not
   for any source with weight: the choice selects for it. With reward still rising (its error positive on
   average while the learner improves), every source with weight is pushed up.
2. A weight at nothing cannot go below it, so noise alone lifts it; and the pull's shares of a page are more
   extreme than the reflex's (0.77 against 0.23; the reflex about 0.6 against 0.4), so its steps are larger.
3. The rate (0.1, the eye commands') is far too fast for a signal this weak: by my estimate the noise alone
   moves a weight by several tenths in 4,000 moments.
**Corrected** (`organic/speech.py`, `_led_look`, `_credit_map`, `gaze_rate`):
- While living, the question's look is drawn among the places by their priority (the learner's own exploration
  and its temperature, as its eye commands are drawn); tested, it takes the highest.
- A source's eligibility is how much more it favoured the place taken than it favoured the places on average
  under that draw. For a source that says nothing this is nothing on average, whatever the reward does. (The
  textbook form of a three-factor rule; AuGMEnT's tags are of this kind: Rombouts et al. 2015.)
- Only the question's look teaches the weights; the eye commands have their own credit.
- The rate is that of a stable value, a hundredth of the eye commands' (0.001): values that move the eye by
  reflex are learned slowly and kept (Kim & Hikosaka 2013). At this rate my estimate is a rise of about 0.02 in
  10,000 moments of pages for a real pull and a wander of about 0.01 for a shuffled one: the small learner may
  show little, and a life would show it late. That is what a weak teaching signal means.
Running: `organic-gaze5` (real and shuffled at the slow rate, 12,000 moments, weights printed to three places);
after the drift probe, the same pair at the eye commands' rate (`organic-rest1`), to see what the correction of
the eligibility alone does. The seeds 1 and 2 of the first rule were stopped; seed 1's all-at-once arm stands
(pages 0.51, "is there" 0.74 at 6,000).
What stays true from 11:16: with the question's look the serial learner looks before it answers and is near the
learner that gets every place for nothing. What moved the first look to the named thing was a weight that the
rule inflated; whether reward alone can do it is open.

### 2026-11-24 14:30: reward does not teach the balance; the balance by its time course

**The corrected rule at the slow rate, with its control** (gaze5_slow.log, seed 0, 12,000 moments):

| moments lived | the pull's weight, real | shuffled | first look on the named thing, real | shuffled |
|---|---|---|---|---|
| 4,000 | 0.000 | 0.001 | 0.49 | 0.49 |
| 8,000 | 0.005 | 0.002 | 0.49 | 0.49 |
| 12,000 | 0.010 | 0.006 | 0.51 | 0.51 |

The real pull's weight is not told apart from that of a pull that says nothing. What both show is a weight that
cannot go below nothing being lifted by noise. So: **at a rate a stable value could have, reward does not teach
the eye to trust the word within 10,000 moments of pages**; at the eye commands' rate the first rule inflated the
weight and a clean one would let it wander (that pair still runs, `organic-rest1`, for the record). The teaching
signal is too weak for this: where the first look lands changes the answer on few pages and by little. A
negative result, and it stays in the record with its control.
**What I had passed over.** The survey gave three grounded ways to set the balance. I built the one it marked as
our own reading (two learned weights) and left out the one with the plainest evidence, because "a tick does
not resolve it". That was the wrong call: what stands out moves the eye only in the first quarter of a second
after a scene appears, and later looks follow what is sought (van Zoest, Donk & Theeuwes 2004; Donk & van Zoest
2008; Siebold et al. 2011; Schütz et al. 2012). A look drawn by a heard word comes after the word, long after.
**As built now** (`gaze_balance = "time"`, the default of the serial gaze; "learned" kept for its controls): what
stands out counts as a transient that fades from the moment the scene appeared, exp(−t / 100 ms); the heard
words' pull counts as it is; each source's places still compete among themselves. A tick is taken as 200 ms
(the learner hears and says one unit a tick; people say four or five syllables a second). Nothing about the
balance is learned and nothing is set by a weight of mine: after a question of three units what stands out
counts 0.002 of itself. Before the word cells know a word their pull is nothing and the reflex decides, as in
a newborn; as they come to know it, the look follows it. What moves with age is the word cells' knowledge.
**Stand-ins:** the two times (200 ms a tick, 100 ms for the transient) are taken from the reports above as the
survey gave them and are not derived in the learner; no one has opened those papers yet. With a noisy young
pull the first look is led by noise rather than by what stands out; the read-out's 0.82 needs grown word cells.
Default path unchanged (digest c52819a8c3a93a7b). Running: `organic-gaze6` (serial, seed 0 → out/tmp/gaze6.log).

### 2026-11-24 15:18: the balance by its time course, first reading (gaze6, seed 0)

| moments lived | first words | first look on the named thing | pages | "is there" right | looks before answering |
|---|---|---|---|---|---|
| 1,000 | 0.10 | 0.51 | 0.04 | 0.00 | 1.22 |
| 2,000 (pages begin) | 0.17 | 0.57 | 0.03 | 0.00 | 1.24 |
| 3,000 | 0.22 | 0.62 | 0.32 | 0.45 | 0.89 |
| 4,000 | 0.28 | 0.59 | 0.45 | 0.54 | 0.89 |
| 5,000 | 0.24 | 0.65 | 0.51 | 0.50 | 0.87 |
| 6,000 | 0.27 | 0.70 | 0.50 | 0.68 | 0.86 |

All at once, same seed, at 3,000–6,000: pages 0.30 / 0.45 / 0.58 / 0.56, "is there" 0.45 / 0.59 / 0.67 / 0.67.
The first look goes to the named thing more often as the word cells come to know the words (0.51 → 0.70; the
reflex 0.49; the read-out with grown word cells 0.82), with nothing learned about the balance and no weight set.
With that one look the serial learner is near the learner that is handed every place (pages 0.50 against 0.56,
"is there" 0.68 against 0.67), on one seed. Seeds 1 and 2 are queued (`organic-rest1`, after the drift probe).
The learned-weights pair at the eye commands' rate is not run: the question it would answer is settled by the
slow pair and the first rule's control, and the machine's hours go to the cortex.
**The drift probe, first arm** (trace_probe.log, seed 0): answers to a picture and to it two pixels aside are
alike 0.847 in the first stage (after its 2 × 2 pooling) and 0.799 in a second stage developed without a trace
(sparser: 0.818 against 0.687). The trace arm is developing.

### 2026-11-24 16:18: the second stage and the trace, read (drift probe, seed 0)

| | first stage alone (as now) | second stage, plain | second stage, trace |
|---|---|---|---|
| answers to a picture and to it two pixels aside, alike | 0.847 | 0.799 | 0.826 |
| sparseness over the kinds at a place | 0.687 | 0.818 | 0.814 |
| stream, 10,000 pictures: word cells with the relay | 0.613 | 0.646 | 0.631 |
| word cells without the relay | 0.502 | 0.513 | 0.509 |
| ridge with the relay (reference) | 0.659 | 0.655 | 0.640 |
| ridge without | 0.646 | 0.638 | 0.630 |

One seed; the stream's chance error is about ±0.011.
1. **The trace does what it is for**: cells that keep a trace of having fired answer more alike when the picture
   slides (0.826 against 0.799 for the same stage without it), with the same sparseness.
2. **It does not reach the tolerance the fixed 2 × 2 average already gives the first stage** (0.847). A second
   stage looks at 3 × 3 places of the first, so it is more particular about position than what it is built on;
   the trace wins back a little over half of that.
3. **On the stream the second stage helps the word cells (0.646 and 0.631 against 0.613) and not the ridge.** The
   trace arm is the lower of the two second stages by about 0.015 for both learners: single centred pictures do
   not ask for tolerance to position, and tolerance is paid for in detail. Where tolerance should pay is where
   the eye lands a little off a thing, on pages; that is not measured here.
4. The second stage's gain for the word cells is larger than the +0.01 measured on 2026-11-17 for the same layer;
   then it was added beside the first stage's summary, here it replaces it.
Under rule 8 the trace is carried forward on its ground (the established account of how tolerance is acquired),
with its cost on this benchmark stated. Seeds 1 and 2 are queued (`organic-trace2`, after `organic-rest1`).
Owed before anything is adopted in a learner: three seeds; the tolerance read on pages with the serial gaze.

### 2026-11-24 16:48: a blurred start (queue E 29b): the organic check and the probe

**What is established.** Before the eyes open the retina drives the cortex with its own waves; a newborn resolves
about a fortieth of an adult's detail and little colour; acuity rises over months, about evenly in the logarithm.
Models: V1-like filters from spontaneous patterns alone (Albert, Schnabel & Field 2008); a start without fine
detail gives more reliance on shape and more robustness when pictures are degraded (Vogelsang et al. 2018; Lu et
al. 2025; Ligeralde et al. 2023). **What we have:** filters that develop on sharp, coloured pictures from the
first step.
**The probe** (`experiments/cortex_diet_probe.py`): the same rule, steps, first weights and window decorrelation
as the present cortex of the seed; only what is seen while developing differs: an eighth of the steps on waves
(blobs of slow noise with soft edges, no colour), then pictures whose blur falls evenly in the logarithm from a
fortieth of full detail to none, colour coming in with the detail. Read: the filters' spatial frequencies and
how much of their pattern is colour; the stream with sharp test pictures; and for both cortices the stream with
the test pictures blurred a little (one pixel), which is where the literature says a blurred start should show.
**Wired / learned.** Wired: that the eye matures (a schedule on what reaches the cortex). Learned: the filters,
by the rule they had. **Stand-ins:** the share of steps on waves and the fit of the whole schedule to 16,000
steps are mine; the order (waves, coarse, fine) is what is established. The window decorrelation is the present
one (settled on sharp pictures), so that this changes one thing; a retina that matures with the eye is item 4.
Running: `organic-diet1` (seed 0), beside the gaze seeds (memory checked: 3.7 GB free, the probe takes about 1.2).

### 2026-11-24 17:31: the serial gaze on three seeds; the blurred start on one

**The serial gaze, balance by its time course, against all at once** (small learner, 6,000 moments, seeds 0 / 1 / 2;
100 held-out items each):

| at 6,000 moments | all at once (the lives' looking) | serial |
|---|---|---|
| pages | 0.56 / 0.51 / 0.59 (mean 0.55) | 0.50 / 0.61 / 0.59 (mean 0.57) |
| "is there" right | 0.67 / 0.74 / 0.69 (mean 0.70) | 0.68 / 0.73 / 0.72 (mean 0.71) |
| views of places the word cells get | every candidate, for nothing | 0.86 / 0.88 / 0.88 looks before answering |
| first look on the named thing (where it matters) | – | 0.70 / 0.70 / 0.68 (the reflex: 0.49) |

**Level on three seeds.** A learner that looks at one place, the one the heard words pull, answers as well as one
that is handed every place at once. The first look goes to the named thing more often as the word cells learn
(0.51–0.62 at 3,000 moments, 0.68–0.70 at 6,000). Nothing was learned about where to look except what words'
things look like, and no weight was set. What it costs: about one tick more a moment.
Not shown: a grown learner (the read-out promises 0.82 first looks); counting, where every place must be seen and
the eye commands would have to learn to go on looking (they die out here, as before).
**A blurred start, seed 0** (diet_probe.log; stream at 10,000 pictures):

| | present cortex (sharp from the first step) | blurred start |
|---|---|---|
| filters' mean spatial frequency (cycles a pixel) | 0.296 | 0.278 |
| share of the filters' pattern that is colour | 0.560 | 0.612 |
| word cells with the relay, sharp test | 0.613 | 0.616 |
| ridge with the relay, sharp test | 0.659 | 0.663 |
| word cells with the relay, test blurred by a pixel | 0.588 | 0.585 |
| ridge with the relay, test blurred | 0.619 | 0.625 |

Nothing moved: the filters end a little coarser and more coloured, and the stream is level with sharp and with
blurred test pictures. The robustness the literature reports for a blurred start is not there in one layer of
5 × 5 filters that go on learning through the sharp end of the schedule: by the end they have relearned the sharp
world. The reports are for deep networks, where early coarse learning shapes later stages. Kept as measured; the
schedule costs nothing and is the eye's own, so it stays for the next cortex (rule 8), where a second stage
gives it something to shape. Seeds 1 and 2 running (`organic-diet2`).


### 2026-11-24 18:45: a sparse expansion before the cells that associate (queue E 29f): the organic check and the probe
**What biology has.** In front of the cells that learn associations three structures put a wide layer that is
sparsely active: the fly's mushroom body (about 50 glomeruli onto about 2,000 Kenyon cells; each cell has about
7 claws on glomeruli picked at random, with no structure anyone has found; 5–10% of the cells answer an odour,
held there by one inhibitory neuron that all of them drive and that inhibits all of them), the cerebellum's
granule layer (4 dendrites a cell, the most numerous neurons of the brain) and the dentate gyrus in front of CA3
(several times its input's cells, 2–5% active). Litwin-Kumar et al. 2017 show the small number of inputs a cell
is close to the best for telling patterns apart. The wiring is not learned in the fly; the surveys of 2025
(RESEARCH_ARCH.md, 6) say the layers are plastic in mammals.
**Organic check.** No rule of behaviour and nothing designed for a task: random wiring and one inhibitory loop,
each with numbers taken from the structures themselves (inputs a cell 7, share active 5%, 4 to 10 times more
cells than inputs; the fly's ratio of 40 is beyond this machine for a first reading). The honest limit of the
ground: these layers stand in front of *associative memories* (odour to value, the hippocampus's store), so
the ground is strongest for the store's key and weaker for the word cells, which stand in for cortex. Both are
read. A second limit: the relay's numbers are signed, a firing rate is not; the cells here sum signed inputs
(a stand-in for on and off channels).
**What is read, in this order** (`experiments/expand_probe.py`; codes of the present cortex, 10,000 pictures):
1. the code itself: how alike two different pictures look, two of one class against two of different classes,
   how many directions the code uses;
2. the stream: word cells, and a ridge on everything seen (in its dual form; the code is too wide for the
   other); a picture seen once and asked about twenty later;
3. controls: as many cells as inputs (sparseness with no expansion), and the same wiring with no inhibition
   (expansion with no sparseness: a linear map);
4. one more wiring with 100 inputs a cell (the dentate gyrus's cells have thousands), seed 0.
Smoke test on 1,000 pictures (says little): word cells 0.474 on the relay's code, 0.363 with as many sparse
cells, 0.386 with four times as many, 0.470 with no inhibition; the ridge 0.470, 0.361, 0.414, 0.434. So at
1,000 pictures the sparse layer loses more than its width gives back. The full reading is queued
(`organic-expand1`, three seeds).

### 2026-11-24 19:33: divisive normalisation and the blurred start on three seeds; the sparse layer's first two
All on the stream at 10,000 pictures, 2,000 test pictures, seeds 0 / 1 / 2 (mean); the logs are
out/bench/stream_{norm0,divnorm,divnorm_s,diet,dietb,norm0b}10k_s{0,1,2}.log, norm_probe*.log, diet_probe*.log.

**Divisive normalisation** (queue E 24, item 1):

| | the response as now | divided by the pool at the place | pool over the surround too |
|---|---|---|---|
| sparse over kinds at a place | 0.712 / 0.711 / 0.710 | 0.827 / 0.825 / 0.826 | the same |
| kinds above half the strongest | 8.5% / 8.6% / 8.5% | 3.9% / 4.0% / 3.9% | the same |
| sparse over places for a kind | 0.719 / 0.721 / 0.717 | 0.873 / 0.868 / 0.868 | 0.896 / 0.892 / 0.891 |
| response at a quarter, at half the contrast (of full) | 1.00, 1.00 | 0.13–0.28, 0.41–0.64 | 0.14–0.24, 0.45–0.60 |
| word cells with the relay | 0.613 / 0.616 / 0.631 (0.620) | 0.624 / 0.636 / 0.631 (0.630) | 0.610 / 0.618 / 0.614 (0.614) |
| word cells, no relay | 0.502 / 0.522 / 0.524 (0.516) | 0.519 / 0.529 / 0.532 (0.527) | 0.506 / 0.518 / 0.518 (0.514) |
| ridge with the relay | 0.659 / 0.662 / 0.672 (0.664) | 0.654 / 0.647 / 0.648 (0.650) | 0.637 / 0.642 / 0.637 (0.639) |
| ridge, no relay | 0.646 / 0.644 / 0.655 (0.648) | 0.638 / 0.637 / 0.636 (0.637) | 0.634 / 0.628 / 0.629 (0.630) |

What it does to the code is the same on every seed and is what the physiology says: far fewer kinds answer at
a place (sparseness 0.71 → 0.83), and the response grows with contrast and saturates where the present one does
not know contrast at all. On the stream it is small both ways: the word cells gain 0.010 (0.011 without the
relay; below the 0.02 we read as a difference, though in the same direction on five of six comparisons) and the
ridge loses 0.011–0.014 on all three seeds. So the division helps the reader that sees each picture once a
little and costs the reader that inverts a covariance a little: it throws away some of what a second-order
reader could use and conditions the rest. The pool over the surround does nothing for either and costs the
ridge more. **Adopted for the next cortex on its ground (the pool at the place); the surround pool is not.**

**A blurred start** (queue E 29b):

| | sharp from the first step | blurred start |
|---|---|---|
| filters' mean spatial frequency (cycles a pixel) | 0.296 / 0.297 / 0.295 | 0.278 / 0.281 / 0.275 |
| share of the filters' pattern that is colour | 0.560 / 0.558 / 0.556 | 0.612 / 0.593 / 0.614 |
| word cells with the relay, sharp test | 0.613 / 0.616 / 0.631 (0.620) | 0.616 / 0.618 / 0.626 (0.620) |
| ridge with the relay, sharp test | 0.659 / 0.662 / 0.672 (0.664) | 0.663 / 0.661 / 0.654 (0.659) |
| word cells with the relay, test blurred by a pixel | 0.588 / 0.573 / 0.589 (0.583) | 0.585 / 0.574 / 0.582 (0.580) |
| ridge with the relay, test blurred | 0.619 / 0.627 / 0.623 (0.623) | 0.625 / 0.615 / 0.624 (0.621) |

**Nothing moved, on three seeds.** The filters end a little coarser and more coloured, every time, and no
reading of the stream changes, with sharp or with blurred test pictures. The reading of 17:31 stands: one layer
of small filters that goes on learning through the sharp end of the schedule has relearned the sharp world by
the end. The schedule stays for a cortex with a second stage (rule 8), with no gain claimed.

**The sparse layer, seeds 0 and 1** (expand_probe_s{0,1}.log; seed 2 and the wiring with 100 inputs running):

| at 10,000 pictures | the relay's code | as many sparse cells | 4 × | 10 × | 4 ×, no inhibition |
|---|---|---|---|---|---|
| two different pictures alike (mean, unsigned) | 0.035 | 0.018 | 0.014 | 0.013 | 0.039 |
| two of one class alike / of different classes | +0.012 / −0.002 | +0.006 / −0.001 | +0.006 / −0.001 | +0.006 / −0.001 | +0.014 / −0.002 |
| directions the code uses | 375 / 380 | 857 / 838 | 1,021 / 1,038 | 1,072 / 1,089 | 321 |
| word cells name a new picture | 0.613 / 0.616 | 0.522 / 0.501 | 0.569 / 0.561 | 0.576 / 0.570 | 0.605 |
| ridge on everything seen | 0.659 / 0.662 | 0.516 / 0.498 | 0.573 / 0.558 | 0.600 | 0.658 |
| a picture seen once, asked 20 later | 0.890 / 0.860 | 0.887 / 0.857 | 0.943 / 0.923 | 0.953 / 0.957 | 0.913 |

The probe's own relay rows repeat the stream's (0.613, 0.616; ridge 0.659, 0.662), so the two are one test.
What the sparse layer does is what it is said to do in the hippocampus and the mushroom body, and it is the
opposite of what a namer of new pictures needs: two pictures look less alike (0.035 → 0.013) and the code uses
three times the directions, and the likeness of two pictures of one class is halved with it. Naming new
pictures falls (0.61 → 0.57 at ten times the cells; 0.51 with no expansion) and the memory of one picture seen
once rises (0.88 → 0.95). The control with no inhibition changes nothing either way, as a linear map should.
So the ground was read right in the check above: **this layer belongs in front of the store, as its key, and
not in front of the word cells.** Next for it: the store keyed by the sparse code against the store as it is
(the held-out-class test and "seen once, 20 later" with the store), after seed 2.

### 2026-11-24 19:36: the sparse layer as the store's key: a first test that is too easy
`experiments/expand_store_probe.py` (out/bench/expand_store_probe.log; three seeds, no learner): 1,000 test
pictures stored; asked with the same pictures blurred by a pixel and with 1,000 never stored. The question
behind it: the store's steepness (exp(12 × (likeness − 1)), set by hand) is what makes it a memory of moments;
the dentate gyrus is said to do that by its wiring. Would a sparse key need no exponential?

| key and answer | blurred picture finds its own moment | its own moment answers | strongest answer to a picture never stored |
|---|---|---|---|
| relay's code, exponential (the store as it is) | 1.000 | 0.71 | 0.000 |
| relay's code, the likeness itself | 1.000 | 0.97 | 0.24 |
| 4 × sparse cells, the likeness itself | 1.000 | 0.92 | 0.17 |
| 4 × sparse cells, exponential | 1.000 | 0.45 | 0.000 |

(The same on the three seeds to the second decimal.) **The test tells nothing apart**: a picture blurred by a
pixel is 0.97 like itself on the relay's code and no other picture comes near, so every key is perfect. What
it does show: the sparse layer by itself is not steep enough to stand in for the exponential (a picture never
stored still draws 0.17 of a wrong reply, against 0.24 on the relay's code), and it costs some of the match to
a changed view (0.92 against 0.97). So the hand-set steepness stays for now, named as such. A test that could
tell the keys apart needs stored moments that are alike and must not be confused: pages of one scene with
different things asked about, in a learner. Owed, with the other gaze work on pages.

### 2026-11-24 20:08: the sparse layer on three seeds; a hundred inputs a cell changes the reading (one seed)
expand_probe_s{0,1,2}.log (7 inputs a cell), seeds 0 / 1 / 2 (mean), 10,000 pictures:

| | the relay's code | as many sparse cells | 4 × | 10 × | 4 ×, no inhibition |
|---|---|---|---|---|---|
| word cells name a new picture | 0.613 / 0.616 / 0.631 (0.620) | 0.522 / 0.501 / 0.504 (0.509) | 0.569 / 0.561 / 0.568 (0.566) | 0.576 / 0.570 / 0.587 (0.578) | 0.605 / 0.612 / 0.622 (0.613) |
| ridge on everything seen | 0.659 / 0.662 / 0.672 (0.664) | 0.516 / 0.498 / 0.487 (0.500) | 0.573 / 0.558 / 0.568 (0.566) | 0.600 / 0.606 / 0.611 (0.606) | 0.658 / 0.664 / 0.664 (0.662) |
| a picture seen once, asked 20 later | 0.890 / 0.860 / 0.797 (0.849) | 0.887 / 0.857 / 0.843 (0.862) | 0.943 / 0.923 / 0.927 (0.931) | 0.953 / 0.957 / 0.947 (0.952) | 0.913 / 0.883 / 0.853 (0.883) |

Three seeds agree with the two: with seven inputs a cell, naming falls by 0.04 at ten times the cells and the
memory of one picture rises by 0.10.
**A hundred inputs a cell, seed 0** (expand_probe_fan100_s0.log; planned as the second wiring in the check):

| seed 0 | the relay's code | 4 ×, 7 inputs | 4 ×, 100 inputs | 10 ×, 7 inputs | 10 ×, 100 inputs |
|---|---|---|---|---|---|
| two different pictures alike | 0.035 | 0.014 | 0.010 | 0.013 | 0.008 |
| directions used | 375 | 1,021 | 1,480 | 1,072 | 1,574 |
| word cells name a new picture | 0.613 | 0.569 | 0.590 | 0.576 | 0.612 |
| after 500 pictures | 0.344 | 0.232 | 0.190 | 0.233 | 0.185 |
| a picture seen once, asked 20 later | 0.890 | 0.943 | 0.967 | 0.953 | 0.983 |

With a hundred inputs a cell the layer separates pictures more (0.008 alike) and still names new ones as well
as the relay's code at ten times the cells (0.612 against 0.613), with the memory of one picture at 0.98. So
the cost on naming belonged to the thin wiring, not to the sparse layer. In proportion a hundred of 4,096 (2.4%)
is the nearer to biology: a Kenyon cell takes 7 of about 50 channels (14%), a dentate granule cell thousands
of its input's cells. The reading of 19:33 ("not in front of the word cells") is therefore taken back to
"undecided": one seed. The price that remains is a slow start (0.19 against 0.34 after 500 pictures).
Queued: seeds 1 and 2 of this wiring (`organic-expand2`). Not done, on purpose: a search over the number of
inputs; if the two seeds agree, one more wiring at the fly's proportion (14%) would say whether the trend goes
on, and that is all.

### 2026-11-24 20:13: the reward's error against its own usual size (queue E 29d): the organic check and the probe
**What 29d said, and what of it was already settled.** "Surprise sets how much is learned": three things in the
survey. (1) A step scaled by the cell's own error: the word cells had it (`plastic_by_error`) and it was taken
out on 2026-11-17 because it counts the error twice and cost 0.03; a broadcast surprise multiplying every step
would be the same thing a third time. Not built. (2) A step that shrinks as evidence gathers (precision): the
word cells' synapses have it (each synapse's step falls with its use). (3) An error taken against the spread
of its context. **That one the gating policy lacks**: its error is the reward minus the command's value, in
the reward's own units, times a rate of 0.1 that I set; so the size of a reward, which the world's designer
chose, sets how fast the policy learns.
**What biology has.** Dopamine neurons do not code a reward's error in absolute units: their response to a
given error scales with the spread of the rewards the animal has come to expect, so that the neuron's range
covers the range of outcomes (Tobler, Fiorillo & Schultz 2005, adaptive coding; among the best replicated
findings about these cells). The form the cortex uses for the same job is divisive normalisation, which is
also how value is coded in parietal cortex (Louie & Glimcher). [Both references to be opened by hand.]
**The mechanism** (`organic/loop2.py _credit`, behind `reward_spread`; None by default, the default path's
digest unchanged, c52819a8c3a93a7b; life env `RPE=normalised`): each command keeps the usual size of its own
error (a running average of the unsigned error, at the rate its value already learns), and the policy takes
the error divided by that usual size plus the error's own size. It lies between −1 and 1 whatever the reward's
scale; an error of the usual size counts a half. Nothing of mine sets the scale, and a world that pays twice
as much teaches exactly as before (true by construction, so not tested).
**Organic check.** No rule of behaviour; one local quantity per command, learned; a canonical form. What is
still a number of mine: the rate 0.1 (now in units that mean something: a share of the usual error) and the
averaging rate 0.02 (the value's own).
**The probe** (`organic-rpe1`: the small learner of the gaze probes, 6,000 moments, three seeds, arm
"all+rpe"), **with its control from the start**: in the smoke test the normalised errors are about twice the
size of the raw ones (0.42 against 0.22), so the policy's step is about doubled; an arm with the raw error and
a step larger by exactly the factor measured in the first arm ("all+gain") tells the normalisation from a
larger step. Baseline: the "all" arm of the gaze probes (same seeds and moments). Read: grades, ticks before
answering, eye moves, patience. Expectation, stated before: little moves at this age; the ground is the
mechanism, and what is watched is whether anything breaks.

### 2026-11-24 21:05: the sparse layer with a hundred inputs a cell, three seeds
expand_probe_fan100_s{0,1,2}.log; 10,000 pictures; seeds 0 / 1 / 2 (mean):

| | the relay's code | 4 × the cells | 10 × the cells | 4 ×, no inhibition |
|---|---|---|---|---|
| two different pictures alike | 0.035 | 0.010 | 0.008 | 0.139 |
| directions used | 375 / 380 / 383 | 1,480 / 1,489 / 1,489 | 1,574 / 1,580 / 1,585 | 28 |
| word cells name a new picture | 0.613 / 0.616 / 0.631 (0.620) | 0.590 / 0.580 / 0.602 (0.591) | 0.612 / 0.603 / 0.612 (0.609) | 0.602 / 0.599 / – |
| after 500 pictures | 0.344 / 0.398 / 0.434 (0.392) | 0.190 / 0.266 / 0.326 (0.261) | 0.185 / 0.273 / 0.331 (0.263) | 0.258 / 0.360 / – |
| ridge on everything seen | 0.659 / 0.662 / 0.672 (0.664) | 0.569 / 0.571 / 0.567 (0.569) | 0.617 / 0.618 / 0.612 (0.616) | 0.656 / 0.657 / – |
| a picture seen once, asked 20 later | 0.890 / 0.860 / 0.797 (0.849) | 0.967 / 0.970 / 0.957 (0.965) | 0.983 / 0.980 / 0.963 (0.975) | 0.897 / 0.873 / – |

Seed 0's reading holds on three: at ten times the cells the word cells name new pictures at 0.609 against
0.620 (not a difference we read; down on two seeds, level on one) and the memory of one picture is 0.975
against 0.849. What it costs: a slow start (0.26 against 0.39 after 500 pictures) and what a second-order
reader could have used (the ridge 0.616 against 0.664). With no inhibition and a hundred inputs of one sign
the cells all carry the inputs' common sum (28 directions): the inhibition is what makes the layer a code.
**Decision.** Not put in front of the word cells: in a life a word is heard a few times, which is the stream's
first 500 pictures and not its 10,000th, and the layer is slowest exactly there; and the learner already has a
memory of single moments (the store). Not adopted as the store's key either until a test can tell keys apart
(pages of one scene, in a learner). Kept as measured: the mechanism does what biology says it does (a memory
of moments out of slow cells), and the cost of a thin wiring is known. No further wirings are tried (that
would be a search).

### 2026-11-24 22:13: the drift stage on a second seed; the reward's error, first seed; the reward lives at 65,000
**The drift stage, seed 1** (trace_probe_s1.log, stream_l2{plain,trace}10k_s1.log; 10,000 pictures). A picture and
itself two pixels aside are answered 0.849 alike by the first stage, 0.811 by a second stage developed plainly,
0.837 by one developed with the trace (seed 0: 0.847 / 0.799 / 0.826): the trace again gives back most of what a
plain second stage loses, and no more than the first stage already had. On the stream the two second stages do
not differ: word cells with the relay 0.624 plain, 0.626 trace (seed 0: 0.646, 0.631); ridge with the relay 0.635
and 0.639 (seed 0: 0.655, 0.640). Seed 0's lower trace arm was one seed's noise. Seed 2 is developing.
**The reward's error against its usual size, seed 0** (rpe_s0.log, rpe_gain_s0.log; 6,000 moments). The raw
errors grow as the learner starts to earn (0.22 at 1,000 moments, 0.37 at 6,000); the normalised ones stay near
0.41–0.46 throughout, as the form says they must. So the step it gives is 1.9 times the raw one early and 1.3
times late: a rate that is largest while little is known and falls by itself. The fixed-gain control used the
mean factor, 1.59. At 6,000 moments: first words 0.31 (baseline 0.35), pages 0.61 (0.56), "is there" on a page
0.70 (0.67), ticks 5.0 (5.1), patience 0.524; the control at 5,000: 0.23, 0.42, 0.68. One seed: nothing to read
but that nothing broke. Seeds 1 and 2 run.
**The reward lives (lenient all life) at 65,000 moments**, beside the plateau lives at the same age, above blind
answering (the records' `above`): first words 0.46 / 0.49 / 0.46 against 0.53 / 0.46 / 0.48. Seeds 0 and 2 left
first words at 55,000 lived (45,000 there; the plateau lives: 70,000 / 45,000 / 50,000 there) and are in pages
at 0.27 / 0.24 after 10,000 moments there (the plateau lives after 10,000 there: 0.32 / 0.26 / 0.27). Seed 1 is still in
first words (0.49). **No difference so far from a kinder reward.**

### 2026-11-24 22:13: what comes back in sleep is marked at the moment (queue E 29e): the organic check and the test
**Where this stands.** Replay from the fast store was built on 2026-11-18 and did not protect the earlier
stages in a life (docs/RESEARCH_MEMORY.md): the store was more than half leisure talk, the store's surprise
read 0.83–0.87 for every moment so that "the more strongly held, the more often" chose nothing, and the earlier
stages were about 3% of what came back. The lives since then run without it.
**What biology has.** The hippocampus does not replay a day evenly. While the animal pauses at a reward, sharp
wave ripples replay the path just taken, and their number follows the size of the reward (Singer & Frank 2009;
Ambrose, Pfeiffer & Foster 2016). Those awake ripples select what sleep replays: trials followed by them come
back in the night's sleep, trials without them do not (Yang et al. 2024). Against a simple reading: replay also
favours what was experienced less (Gupta et al. 2010; Carey, Tanaka & van der Meer 2019), and in people rest
replays the weakly learned (Schapiro et al. 2018). So "reward at the moment marks it for sleep" is established;
"only the rewarded returns" is the strong form of it. [All to be opened by hand.]
**The mechanism** (`organic/episodes.py`: `tag`, `rewarded`; one line in `organic/speech.py`; life env
`REPLAY_TAG=reward`; None by default, the default path's digest unchanged, c52819a8c3a93a7b). A stored moment
keeps the reward that came with it (the reward the gating policy already gets for that answer; nothing below
zero). At rest a moment comes back in proportion to how strongly it is held times that reward. The moment's
strength, its fading and what leaves a full store are as they were.
**Organic check.** No rule about kinds of moment: leisure talk, comments and overheard speech are never graded,
so no reward comes at them and they do not return, which is what removes the talk from sleep. No new number.
The reward is one the learner already receives. What stays set by hand is the old debt (the store's size, how
long a moment lasts, 0.3 returns for every lived moment). Not taken from the survey: recent and old memories
in separate blocks of sleep (Chang et al. 2025); one thing at a time.
**What it would choose, read from the records before anything is run** (the plateau lives in preschool; a
stage's mean reward taken as its judged grade): of the store's 2,000 moments 1,091 are talk. Among the lessons
kept, first words and pages are 0.08 and 0.15 by count; weighted by reward they would be 0.11–0.18 and
0.21–0.29, preschool 0.54–0.60 (a fifth of preschool's answers are right without looking, so its moments are
marked too). Earlier stages would be about four parts in ten of what returns, against about one in ten when it
was tried. With 0.3 returns a lived moment that is about 0.12 more practice of the earlier stages on top of the
0.30 the world's mix already gives.
**Stated before the test:** a small effect at most. The delta rule learns nothing from a moment it still
answers; a marked moment teaches only once the cells have drifted from it. And the rule rehearses what was got
right and never what was got wrong, the opposite of Schapiro's finding; if earlier stages are held but the new
stage slows, that is the cost to look for.
**The test** (`organic-tag1`, out/tmp/tag_queue.sh; the lesson of 2026-11-18 was a test with nothing to lose in
its control). Where the loss is real: preschool. Each arm is a life continued from a plateau life's level as it
left pages (out/levels/plat_s*_W2.pkl), 20,000 moments of preschool, the plateau lives' launch line with one
thing changed: `tagn` nothing (the same life lived again from its level: how far two lives from one level
differ; the plateau life itself is a second sample), `tagr` replay with the mark, `tagb` replay as built. Read:
first words and pages above blind answering over the four tests, preschool itself, and the record of whose
moments came back. In the plateau lives first words fell by 0.05 / 0.00 / 0.09 and pages by 0.06 / 0.08 / 0.08
within 10,000 moments. One life at a time and only with 5.7 GB available, so it waits behind the reward lives;
about four hours an arm, nine arms.
**Smoke test** (a small brain, 400 moments; out/tmp/smoke_tag.py): 97 of 480 stored moments marked, all of them
lessons, none of the comments; 50 asked back, 50 came, every one marked.

### 2026-11-24 23:11: the relay by a local rule, learned as the pictures come (queue E 26 and 28): the organic check and the probe
**What is owed.** The relay between the cortex and the word cells is the paper's main finding and it is a batch
stand-in: one matrix computed at once from 6,000 summaries, with a constant of mine (`shrink`, 0.0005 of the
largest variance in the benchmark's cortex) that keeps the weak directions from being blown up. The lateral
network among a place's kinds (2026-11-23) did four fifths of the relay's work by local rules, but one stage
earlier and on a fifth of the stream. The relay's own place, among the summary's 4,096 numbers, has not been
tried.
**What biology and the literature have** (the equations read in the papers tonight, not recalled). Whitening
by inhibitory interneurons: principal cells excite interneurons, which inhibit them back; the synapse between
a principal cell and an interneuron grows with the product of their activities and decays (Lipshutz, Pehlevan
& Chklovskii 2022, Algorithm 2: z = Wᵀy; y settles at x − Wz; W += η(yzᵀ − W); with as many interneurons as
principal cells it converges to the symmetric whitening the relay computes). With the principal cells' own
leak (Duong, Simoncelli, Chklovskii & Lipshutz 2023, equation 6) the settled state is y = (I + WWᵀ)⁻¹x. The
gain-only circuit of Duong et al. 2023a (queue 28 as first written) needs N(N+1)/2 interneurons for N cells:
8.4 million here. Not built; the synaptic form is the one that fits the relay's size.
**Where the hand-set constant goes.** Inhibition can only take away (WWᵀ is never negative) and the leak is 1,
so a direction whose variance is under 1 cannot be raised: it is passed as it is; every direction above is
brought down to 1. With each cell's input scaled so that the cells' variance is 1 on average (a running
average, no constant), that floor sits at the mean variance. **Measured on the benchmark's summaries (seed 0):
the mean variance is 0.00094 of the largest; my `shrink` was 0.0005.** The circuit's floor falls within a
factor of two of the constant I had tuned by hand; 183 directions stand above it (241 above the batch floor).
The constant is not needed: the leak and the direction of inhibition do its work.
**Organic check.** Published rules, unchanged; every change depends on the two cells at a synapse's ends; no
constant of mine but the papers' rate (0.001 a picture). The authors' own caveats hold here too: the two
directions of an interneuron's connection are taken as equal (weight transport), and the weights are not
sign-constrained. Departures of the probe, said in its header: a block of 50 pictures settles with the
synapses as they were when the block began, the settled state is computed in closed form, and each cell's
average and scale are running averages over all it has seen.
**The probe** (`experiments/relay_local_probe.py`; `organic-relay1` behind the drift probe: a smoke test on
1,000 pictures, then seed 0 on 10,000). Arms: the batch relay; *settled*, where the circuit's rule comes to
rest, computed at once (does the new floor cost anything?); *developed*, the circuit after one sight of the
stream's pictures without their names, then held still (the stream's own pictures, since the pictures the
batch relay was settled on were not kept as codes: said here, to be replaced); *plastic*, the circuit learning
from nothing along the stream while the word cells learn from what it sends. Read first on the code (two
pictures alike, directions used, pairs of cells that still go together), then the word cells.
**Stated before:** settled should be near the batch relay (0.620); developed a little under it (one pass, the
weak directions slow); plastic lower still at the start and unknown at the end: the word cells' early
associations were made in a code that has since moved.
**The reward's error, seed 1** (rpe_s1.log; gain for its control 1.70): first words 0.28 (baseline 0.25), pages
0.50 (0.51), "is there" 0.71 (0.74) at 6,000 moments. Two seeds: as the baseline. Noted to watch: with either
larger step (normalised, or the fixed gain) the eye moves on "is there" pages fall to 0.00 at the end, where
the baselines have 0.07–0.08.


### 2026-11-25 00:10: the drift stage on three seeds; the reward's error on three seeds; the reward lives at 75,000
**The drift stage, three seeds** (trace_probe.log, trace_probe_s{1,2}.log, stream_l2{plain,trace}10k_s{0,1,2}.log;
10,000 pictures; seeds 0 / 1 / 2, mean ± sd):

| | one stage (norm0) | a second stage, plain | a second stage, with the trace |
|---|---|---|---|
| a picture and itself two pixels aside, alike | 0.847 / 0.849 / 0.849 (0.848) | 0.799 / 0.811 / 0.804 (0.805) | 0.826 / 0.837 / 0.825 (0.829) |
| word cells + relay | 0.613 / 0.616 / 0.631 (0.620 ± 0.010) | 0.646 / 0.624 / 0.644 (0.638 ± 0.012) | 0.631 / 0.626 / 0.646 (0.634 ± 0.010) |
| word cells, no relay | 0.502 / 0.522 / 0.524 | 0.513 / 0.519 / 0.551 | 0.509 / 0.523 / 0.554 |
| ridge + relay | 0.659 / 0.662 / 0.672 (0.664 ± 0.007) | 0.655 / 0.635 / 0.645 (0.645 ± 0.010) | 0.640 / 0.639 / 0.652 (0.644 ± 0.007) |
| ridge, no relay | 0.646 / 0.644 / 0.655 | 0.638 / 0.620 / 0.643 | 0.630 / 0.626 / 0.638 |
| a picture seen once, asked 20 later (cells + relay) | 0.890 / 0.860 / 0.797 | 0.887 / 0.860 / 0.850 | 0.890 / 0.873 / 0.830 |

What holds on every seed: (1) a second stage developed plainly is less tolerant of a two-pixel shift than the
first stage was (0.805 against 0.848), and the trace gives back about half of that (0.829), never more than the
first stage had; (2) the trace changes nothing on the stream (word cells 0.634 against 0.638, ridge 0.644
against 0.645); seed 0's point and a half for the plain stage was one seed's noise; (3) a second stage of
either kind gives the word cells one to three points on every seed (+0.033 / +0.008 / +0.013 plain,
+0.018 / +0.010 / +0.015 trace: under the 0.02 we read as a difference, but of one sign six times in six) and
costs the ridge about two (−0.004 / −0.027 / −0.027 and −0.019 / −0.023 / −0.020).
**Decision.** The trace rule is kept as the rule of any second stage: its ground is the eye's own drift and a
well-established learning rule, it does to the code what it should, and it costs nothing on the stream. No
gain is claimed for it. The second stage itself is not put into the lives' cortex on this evidence: what it
gives the word cells is small and what it costs a second-order reader is as large; it waits for 29c (critical
periods in sequence) and for a test where views differ by more than two pixels (the serial gaze's successive
looks), which is where tolerance should matter. The paper's row and paragraph are filled with the three seeds.

**The reward's error against its usual size, three seeds, with the control** (rpe_s{0,1,2}.log,
rpe_gain_s{0,1,2}.log; baselines: the "all" arms of gaze1.log, gaze4.log, gaze4_s2all.log; 6,000 moments; seeds
0 / 1 / 2):

| at 6,000 moments | baseline | error against its usual size | raw error, step larger by the same factor |
|---|---|---|---|
| first words | 0.35 / 0.25 / 0.34 | 0.31 / 0.28 / 0.33 | 0.31 / 0.32 / 0.27 |
| pages | 0.56 / 0.51 / 0.59 | 0.61 / 0.50 / 0.60 | 0.57 / 0.57 / 0.60 |
| "is there" on a page | 0.67 / 0.74 / 0.69 | 0.70 / 0.71 / 0.69 | 0.71 / 0.71 / 0.68 |
| the eye moves on such a page | 0.08 / 0.07 / 0.07 | 0.00 / 0.00 / 0.02 | 0.00 / 0.01 / 0.02 |
| ticks before answering, living | 5.1 / 5.0 / 5.0 | 5.0 / 4.9 / 5.3 | 4.9 / 5.0 / 5.0 |

(the factor: 1.59 / 1.70 / 1.62.) Grades as the baseline on three seeds, as stated before the probe. The raw
error grows on every seed as the learner starts to earn (to 0.37 / 0.33 / 0.35 at 6,000) while the normalised
one stays at 0.45–0.46: the step is largest when little is known and falls by itself. One thing does move, on
every seed and in the control too: the eye stops moving on "is there" pages (0.07–0.08 → 0.00–0.02) with the
share right unchanged. It is the larger step that does it, not the normalisation: the policy learns sooner that
a move there costs time and pays nothing.
**Decision (rule 8).** Adaptive coding of the reward's error is among the best established facts about dopamine
cells, it takes a number of the world's designer (the size of a reward) out of the learner's rate, and nothing
breaks: **adopted for lives born from now on** (`-E RPE=normalised`). The lives running and the ramp lives
queued behind them stay as they are (they are one comparison). To be watched in the first life that has both
this and the serial gaze: whether the eye, which the same policy moves, is taught to stay still too soon.

**The reward lives (lenient all life) at 75,000 moments.** All three are now in pages (seed 1 left first words at
70,000 lived, 60,000 there). Above blind answering, by time in pages: after 10,000 there 0.27 / – / 0.24, after
20,000 there 0.28 / – / 0.29; seed 1 has 0.18 after its first 5,000. The plateau lives at the same times in
pages: 0.32 / 0.26 / 0.27 and 0.30 / 0.29 / 0.33. First words at 75,000: 0.45 / 0.48 / 0.49. **Still no
difference from a kinder reward**, in the time a stage takes or in where it levels off.

**The relay by a local rule, first numbers** (relay_local_smoke.log, relay_local_s0.log; seed 0). The circuit's
resting point, computed at once, against the batch relay: two pictures alike 0.038 against 0.035, directions
used 316 against 350, pairs of cells that go together 0.027 against 0.024; word cells 0.609 against 0.613 at
10,000 (0.305 against 0.344 at 500), a picture seen once 0.883 against 0.890. **The floor at the mean variance,
which nobody set, costs nothing we can read against the hand-set one.** The circuit after only 1,000 pictures
(smoke test): directions used 201–207, pairs that go together 0.060–0.064; word cells at 1,000 pictures 0.440
held still and 0.444 learning along (the batch relay: 0.473). The arms on 10,000 pictures are running.
**The flaw named last night is being mended:** `experiments/bench_dev_codes.py` codes, with the same cortex, the
very 6,000 unlabelled pictures the batch relay was settled on and 14,000 more of the same kind
(out/bench/cifar100_norm0_s{seed}_dev.npy), so that the resting point and the developed circuit can be given
the relay's own pictures and never the stream's.

### 2026-11-25 00:25: the published soft winner-take-all rule, unchanged (queue E 27): the organic check and the probe
**Why.** The paper says that filters developed without labels do no better than filters never developed. Our
development is a *hard* winner-take-all with a conscience. Journé, Garcia Rodriguez, Guo & Moraitis (ICLR 2023)
report for one convolutional layer under a trained linear classifier: hard winner-take-all 62.7, random weights
63.9, their soft rule 71.1. If that holds in our measure, our sentence is about one rule and must say so; and
my soft attempt of 2026-11-17 ("my version", worse than the hard rule) proves nothing about theirs.
**The rule, read tonight from the paper and from the authors' code** (the two differ in places that matter, so
the code decides): the drive of a kind at a place is u = w · x on the standardised window, weights neither
normalised nor biased; the kinds at a place compete by a softmax of the drives (inverse temperature 1); the kind
with the largest drive changes by +y (x − u w), every other kind by −y (x − u w); the changes of a batch of 10
pictures are summed, **divided by the largest absolute change in the whole layer** (in the code, not in the
paper), and applied at 0.08 × |length of the neuron's weights − 1|^0.5; the weights are born with length about
25 and learning dies away by itself as each neuron's length comes to 1; every picture is seen once. The
response sent on is (u − the kinds' mean at the place)+ raised to 0.7, averaged over 4 × 4 at stride 2. Each
colour is standardised first (batch norm with no learned weights). No whitening of windows anywhere.
**Organic check.** What is local: each change needs the window, the neuron's own activity and drive and its
own weights; the rate needs the neuron's own weight length; the competition is the canonical one (divisive,
among neighbours at a place); a cell that is driven but not the strongest unlearns, which is the sign rule of
BCM in other clothes. What is not: the division of every change by the largest in the layer is a number no
synapse has; the weights are shared over places (as ours are); the batch statistics of the standardiser. No
number in it is mine. It is built as published **to be measured**, not adopted: if it earns a place, the
non-local division is the first thing to replace.
**The probe** (`experiments/softhebb_probe.py`, `organic-sh1`, seed 0 first). The layer as published at
32 × 32, developed on the 50,000 unlabelled pictures (CIFAR-100's: the stream's own are never seen before the
stream; the authors develop on CIFAR-10's), with 256 kinds as our cortex has and with 96 as published. Arms:
developed and as born (their "random weights"). Read first on the code, then as every code of ours is read:
the map summarised in 4 × 4 regions, the relay settled on 6,000 unlabelled pictures, the stream of 10,000
pictures with the word cells and the ridge. **The control that it is their rule and not again mine:** their
finding in their own measure, in small: a ridge on the layer's whole map (24,576 numbers a picture, as their
classifier reads it) from 5,000 labelled pictures, developed against as born.
Smoke test (2,000 unlabelled pictures only, seed 9; not a result): the weights' lengths fall as published
(25 → 6 on average, the first neurons at 1), no two kinds alike above 0.9, the whole-map ridge 0.529 developed
against 0.517 as born.
**Stated before:** in their measure in small the developed layer will be ahead by a few points (less than their
seven: 5,000 labelled pictures, a ridge). In ours I expect less than 0.02 between developed and as born once
the relay has decorrelated the summary, because on 2026-11-17 most of what a first layer gave was in the
decorrelation, the graded response and the number of kinds; and both under our cortex (0.620 word cells, 0.664
ridge), whose windows are decorrelated before the filters see them. If the developed layer is clearly ahead in
our measure too, the paper's sentence is rewritten and the rule is tried on our decorrelated windows next.

### 2026-11-25 00:26: the relay circuit at the papers' rate does not come to rest at this size; why, and the check
**What came back** (relay_local_s0.log; seed 0; the first run, developed on the stream's own pictures without
names). The circuit after 10,000 pictures, held still: two pictures alike 0.055, directions used 170, pairs of
cells that go together 0.078; word cells 0.584 at 10,000 pictures and 0.860 for a picture seen once. Its own
resting point computed at once: 0.038, 316, 0.027; 0.609 and 0.883. The batch relay: 0.035, 350, 0.024; 0.613
and 0.890. And after only 1,000 pictures (the smoke test) the circuit had 207 directions and 0.060: **it is
further from its resting point after ten times the pictures.** So the rule's resting point is as good as the
batch relay, and the rule as I ran it does not get there.
**Why (worked out, then checked small).** One picture changes the synapses by rate × (y zᵀ − W): along the
picture's own activity the synapses are multiplied by 1 + rate × |y|². The papers' rate, 0.001, is for
circuits of tens of cells, where |y|² is tens and the step is a few hundredths. With 4,096 cells |y|² is in
the hundreds, so one picture moves the synapses along itself by tens of per cent: not a slow rule any more.
`experiments/relay_small_check.py` (256 made-up cells, variances falling as in the summary): with
rate × |y|² = 0.08 the circuit comes to its resting point (variance 1.0 along every direction above the floor,
the input's own under it, by 5,000 pictures); with rate × |y|² = 1 the strongest directions stay at 2.5 to 4
times the floor and never settle; with 4 they wander.
**Next** (`organic-relay2`, `experiments/relay_local_spectrum.py`): the same reading at full size, direction by
direction, at the papers' rate and at a tenth of it, with |y|² printed. If it confirms, the rate is stated as
what it is, a share by which one picture may move a synapse along itself (rate × |y|², the papers' regime:
a few hundredths), so that it means the same at any size; then the arms again on three seeds, developed on the
relay's own unlabelled pictures (seed 0's are coded and match the relay's average to 2e-7).
**A fault of mine, said:** I took "the papers' rate" as if a rate had no units. The first run's word-cell
numbers for the circuit (0.584) are a run at the wrong step and are not the circuit's result.

### 2026-11-25 01:14: the relay circuit read direction by direction; the soft rule in its authors' measure
**The relay circuit at full size** (relay_spectrum_r1e-3_s0.log, relay_spectrum_r1e-4_s0.log; seed 0; the batch
relay's own unlabelled pictures, read on 1,000 test pictures along the eigen-directions of the input). The
input's variance, in the circuit's units, along directions 1 / 3 / 10 / 30 / 100 / 183 / 300 / 1,000 / 3,000:
1,020 / 367 / 69 / 21 / 3.4 / 1.05 / 0.31 / 0.016 / 0.0003. The resting point brings the first six to 1 and
leaves the last three.

| the output's variance there | 1 | 3 | 10 | 30 | 100 | 183 | 300 | 1,000 | 3,000 | \|y\|² |
|---|---|---|---|---|---|---|---|---|---|---|
| resting point | 1 | 1 | 1 | 1 | 1 | 1 | 0.31 | 0.016 | 0.0003 | |
| rate 0.001, after 2,000 pictures | 1.15 | 1.05 | 1.32 | 1.12 | 1.18 | 0.90 | 0.35 | 0.023 | 0.0015 | 356 |
| rate 0.001, after 20,000 | 1.36 | 1.22 | 1.40 | 1.13 | 1.18 | 1.05 | 0.42 | 0.026 | 0.0015 | 383 |
| rate 0.0001, after 2,000 | 1.81 | 1.72 | 2.33 | 2.26 | 1.29 | 0.48 | 0.15 | 0.018 | 0.0084 | 358 |
| rate 0.0001, after 6,000 | 1.12 | 0.95 | 1.24 | 1.20 | 1.08 | 0.55 | 0.18 | 0.015 | 0.0035 | 279 |
| rate 0.0001, after 20,000 | 1.06 | 0.90 | 1.13 | 0.99 | 0.98 | 0.74 | 0.26 | 0.019 | 0.0014 | 306 |

As worked out: at the papers' rate one picture's step is 0.38 (rate × |y|²) and the circuit stops moving after
2,000 pictures with the strong directions 10–40% above the floor, wandering, and the weakest raised five times
by what leaks into them. At a tenth of the rate the step is 0.03 and the circuit goes on approaching its
resting point for all 20,000 pictures: the strong directions within 13% of the floor, the ones near the floor
still held down by synapses the circuit was born with (they fade at the rule's own slow pace), the weakest
still falling. **The rule comes to rest when its step is in the papers' regime; it is slow near the floor.**
**Decided.** The rule stays the papers', unchanged; the rate is 0.0001, said for what it is (a step of 0.03 a
picture at this size). I did not add a mechanism that scales the rate by the population's activity: that
would be mine. The arms run again on three seeds at this rate, developed on the batch relay's own unlabelled
pictures (`organic-relay3` → relay_local_r1e-4_s{0,1,2}.log): the batch relay, the resting point, developed on
the relay's 6,000 pictures, on 20,000, learning from nothing along the stream, and developed on 6,000 then
learning on. **Stated before:** developed on 20,000 within 0.01 of the resting point; on 6,000 a little under;
from nothing along the stream clearly under at the start (the first thousand pictures are coded by a circuit
far from rest) and under at the end; developed-then-learning-on between the two developed arms.
The first run's last arm, for the record (rate 0.001, learning from nothing along the stream): 0.587 at 10,000
pictures, 181 directions; a run at the wrong step, as its developed arm.

**The published soft rule, seed 0, on the code and in its authors' measure** (softhebb_probe_s0.log; 50,000
unlabelled pictures seen once).

| | 256 kinds, developed | 256, as born | 96, developed | 96, as born |
|---|---|---|---|---|
| two pictures' summaries alike | 0.150 | 0.293 | 0.216 | 0.298 |
| directions the summary uses | 27 | 8 | 14 | 8 |
| the same through the relay | 0.025, 577 | 0.045, 236 | 0.035, 361 | 0.048, 213 |
| a ridge on the whole map, 5,000 labelled pictures | | | **0.616** | **0.515** |

**The control passes: it is their rule.** In their measure in small the developed layer is ten points above the
layer as born (they report seven with ten times the labels and a trained classifier). My statement before the
run ("a few points, less than their seven") was too low. And the code itself is different in kind from
anything our hard rule gave: the summaries of two pictures are half as alike and use three times the
directions before any relay, and after the relay more than twice the directions. Noted against the paper's
words: after 50,000 pictures the kinds' weight lengths are at 1.0 for the first and 6.8 on average (4.3 with
96 kinds), not all at 1; and in the last batch one kind is the strongest at a third of all places.
The reading in our measure (the stream; word cells and ridge) is running (`organic-sh2`; the first queue
stopped there on a fault of mine in how the file was written).

### 2026-11-25 02:19: the published soft rule in our measure, seed 0: my expectation was wrong; the relay circuit at the right step, seed 0
**The soft rule on the stream** (stream_{sh,shr,sh96,shr96}10k_s0.log; 10,000 pictures; seed 0). Beside our
cortex on the same seed and stream (norm0: hard winner-take-all on decorrelated windows, 256 kinds).

| after 10,000 pictures | soft rule, 256 kinds | the same layer as born | soft rule, 96 kinds | as born, 96 | our cortex |
|---|---|---|---|---|---|
| word cells + relay | **0.689** | 0.549 | 0.662 | 0.544 | 0.613 |
| word cells, no relay | 0.579 | 0.411 | 0.590 | 0.412 | 0.502 |
| ridge + relay | 0.700 | 0.603 | 0.666 | 0.579 | 0.659 |
| ridge, no relay | 0.689 | 0.597 | 0.669 | 0.565 | 0.646 |
| word cells + relay after 1,000 pictures | 0.531 | 0.421 | 0.532 | 0.413 | 0.473 |
| a picture seen once, asked 20 later | 0.903 | 0.843 | 0.830 | 0.790 | 0.890 |

**I said before the run: under 0.02 between developed and as born, and both under our cortex. It is 0.14 between
them, and the developed layer is 0.076 above our cortex for the word cells and 0.041 for the ridge.** One seed;
seeds 1 and 2 are running (`organic-sh4`). What it already says, if they agree:
- "Filters developed without labels do no better than filters never developed" is a finding about **our hard
  rule**, as Journé et al. found for theirs. A soft competition in which the losers unlearn develops filters
  that are worth fourteen points to the word cells. The paper's sentence and section must change.
- At 10,000 pictures the word cells on this layer are where they were at 50,000 on ours (0.687), above the
  streaming discriminant with the relay (0.672) and one-pass backprop of the same shape (0.510; 0.604 with the
  relay), all on our cortex's code: to be compared again like for like, each on this code.
- The relay still matters on it (0.579 → 0.689 for the word cells) and hardly for the ridge (0.689 → 0.700),
  as before.
- The layer as born is worse than our never-developed filters (0.549 against about 0.64): with no
  decorrelation of the windows, random filters see mostly the windows' common brightness. The rule finds its
  way out of that by itself.
**What is not known yet, and the order:** (1) seeds; (2) which part of the rule does it, and above all whether
it survives without its one non-local step (the division by the largest change in the layer); (3) the rule in
our cortex's place (the eye's 64 × 64 pictures, decorrelated windows). Nothing is adopted before (2).

**The relay circuit at a step of 0.03, seed 0** (relay_local_r1e-4_s0.log; all arms on the batch relay's own
unlabelled pictures; 10,000 pictures of the stream).

| | two pictures alike | directions | pairs of cells together | word cells at 1,000 | at 10,000 | seen once |
|---|---|---|---|---|---|---|
| the batch relay | 0.035 | 350 | 0.024 | 0.473 | 0.613 | 0.890 |
| the circuit's resting point | 0.040 | 289 | 0.031 | 0.469 | 0.608 | 0.883 |
| developed on the relay's 6,000 pictures, held still | 0.045 | 236 | 0.041 | 0.470 | 0.600 | 0.883 |
| developed on 20,000, held still | 0.040 | 282 | 0.031 | 0.468 | 0.601 | 0.880 |
| from nothing, learning along the stream | 0.040 | 285 | 0.030 | 0.441 | 0.603 | 0.890 |
| developed on 6,000, then learning on | 0.039 | 302 | 0.028 | 0.476 | **0.613** | 0.887 |

At the right step the circuit does the relay's work: every arm is within 0.013 of the batch relay, and the arm
a brain would have (developed before the lessons, never held still) equals it on this seed. Against what I
stated: developed on 20,000 is within 0.01 of the resting point (yes); from nothing it is under at the start
(0.441 against 0.473 at 1,000) and 0.010 under at the end (yes, barely); developed-then-learning-on is not
between the developed arms but above both. **The word cells are not hurt by a code that keeps moving slowly
under them.** Seeds 1 and 2 are running.

**The reward lives at 85,000 moments:** pages above blind answering 0.31 / 0.23 / 0.31 (after 30,000 / 15,000 /
30,000 moments there; the plateau lives after as long: 0.33 / 0.28 / 0.35). First words 0.45 / 0.46 / 0.46.
Nothing new.

### 2026-11-25 02:23: which part of the soft rule does the work: the check and the probe
Three arms on seed 0, 256 kinds, each the published layer with one part taken out and nothing tuned
(`DISSECT=1` in experiments/softhebb_probe.py; `organic-sh5`, behind the seeds):
- **hard**: only the strongest kind learns, by 1 (the authors' own hard baseline in their pipeline). If it is
  near the layer as born, as they report, the soft layer's gain is not the pipeline's.
- **no unlearning**: the soft competition, the losers learning with the winner's sign. Is it the softness or
  the unlearning?
- **no layer-wide division**: the one step no synapse could take. Every change of a batch is divided by the
  largest change in the layer, which is the same as one rate shared by all synapses that moves with time. In
  this arm it is held at its median over the published run. If the layer develops as well, the division was
  only a rate and the rule is local as it stands; if the weights run away or the code is worse, the moving
  rate is doing work and needs a counterpart in biology (a broadcast gain on plasticity) before anything is
  adopted. Not tuned: one value, from the published run itself.
**Stated before:** hard near as born; no unlearning clearly worse than published (kinds pile onto the same
windows); the constant factor: I do not know. The weights are born long (25) and the first changes are a
hundred times the later ones, so a constant may be too large at the start; a smoke test showed the factor
between 9e-7 and 1e-4 within the first 1,500 pictures.

### 2026-11-25 03:27: the soft rule on seed 1; the relay circuit on seed 1; the references for the soft layer
**The published soft rule, seed 1** (softhebb_probe_s1.log, stream_{sh,shr,sh96,shr96}10k_s1.log; 10,000
pictures of the stream; seed 0 beside it).

| | word cells + relay | word cells | ridge + relay | ridge | seen once |
|---|---|---|---|---|---|
| 256 kinds developed, seed 0 / seed 1 | 0.689 / 0.684 | 0.579 / 0.591 | 0.700 / 0.700 | 0.689 / 0.697 | 0.903 / 0.890 |
| the same layer as born | 0.549 / 0.546 | 0.411 / 0.412 | 0.603 / 0.607 | 0.597 / 0.597 | 0.843 / 0.817 |
| 96 kinds developed | 0.662 / 0.655 | 0.590 / 0.597 | 0.666 / 0.664 | 0.669 / 0.662 | 0.830 / 0.843 |
| 96 kinds as born | 0.544 / 0.537 | 0.412 / 0.414 | 0.579 / 0.567 | 0.565 / 0.557 | 0.790 / 0.727 |

Their measure in small on seed 1 (a ridge on the whole map, 96 kinds, 5,000 pictures): 0.613 developed against
0.515 as born (seed 0: 0.616 / 0.515). The code: two summaries alike 0.151 developed against 0.289 as born,
directions 27 against 8, through the relay 576 against 240. The weights after 50,000 pictures are 1.00 / 6.92 /
14.03 long (least, mean, most): as on seed 0, most kinds have not come to length 1. Seed 1 says what seed 0
said, to the second decimal. Seed 2 is in its streams.

**The relay circuit at a step of 0.03, seed 1** (relay_local_r1e-4_s1.log; word cells at 10,000 pictures; seed 0
beside it): the batch relay 0.613 / 0.616; the circuit's resting point 0.608 / 0.601; developed on the relay's
6,000 pictures and held 0.600 / 0.590; developed on 20,000 and held 0.601 / 0.598; from nothing along the
stream 0.603 / 0.606; developed on 6,000 and then learning on 0.613 / 0.608. On seed 1 the arm a brain would
have is 0.008 under the batch relay, not equal to it; every arm is within 0.026. Seed 2 so far: batch 0.631,
resting point 0.618, developed on 6,000 and held 0.615. The reading waits for seed 2.

**The reward lives at 90,000 moments** (seed 0 at 85,000): pages above blind answering 0.31 / 0.27 / 0.33, first
words 0.45 / 0.47 / 0.46. Nothing new.

**The references for the soft layer** (written now, so that they are ready when seed 2 is in; `organic-softbase1`,
out/tmp/soft_baselines_queue.sh, behind the probes that are running):
- `experiments/bench_soft_codes.py`: the layers the probe kept code the whole CIFAR-10 stream (50,000 pictures,
  the 10,000 test pictures), because the paper's headline is read there and the probe kept 10,000 only. Nothing
  is developed again.
- `experiments/bench_backprop_soft.py`: backprop of the soft layer's shape, as `bench_backprop.py` is of our
  cortex's: the same standardised colours, 5 × 5 windows, 256 filters, the same response, the same two averages,
  unit length, a linear readout; Adam, one pass. Filters born random (unit length) or the soft rule's, frozen or
  not; with and without the fixed relay. Backprop is given the response's power as a choice (0.7, the layer's
  own, or 1, the usual rectifier), and rate × batch × power are tuned on the first 10,000 pictures of seed 0 as
  the first reference was. **Checked before any run:** its way in with the developed filters gives the probe's
  own codes (largest difference 6e-5 on 200 pictures, which is the codes' half precision); its way back against
  small changes in double precision agrees to six decimals at power 1 and within 0.5% at power 0.7 (where the
  slope at zero is infinite and is taken at no less than 0.001 above zero). The way back through the relay is
  checked as the queue's first step (backprop_soft_check.log).
- Then, for each seed on the whole stream: word cells, ridge and streaming LDA on the developed layer's code,
  with and without the relay; backprop at its best settings; and the same readouts on the layer as born.
**What these can change in the paper.** If the three seeds agree, the sentence "random filters do as well as
the filters we developed" becomes a statement about our hard rule, and the comparison with backprop has to be
made on the better layer. I do not know which way that comparison goes: the soft layer gives the word cells
0.07 more than our cortex does, but backprop of the soft layer's shape may gain as much or more from the same
way in (raw colours and a softer response in place of our decorrelated windows).

### 2026-11-25 04:11: three seeds: the soft rule develops filters that help; the relay circuit does the relay's work
**The published soft rule, three seeds** (softhebb_probe_s{0,1,2}.log, stream_{sh,shr,sh96,shr96}10k_s{0,1,2}.log;
10,000 pictures of the stream; mean ± spread over seeds; our cortex is the norm0 code of the same seeds).

| | word cells + relay | word cells | ridge + relay | ridge |
|---|---|---|---|---|
| soft rule, 256 kinds developed | **0.689 ± 0.005** (0.689 / 0.684 / 0.693) | 0.591 ± 0.013 | 0.700 ± 0.001 | 0.693 ± 0.004 |
| the same layer as born | 0.548 ± 0.002 (0.549 / 0.546 / 0.549) | 0.419 ± 0.013 | 0.605 ± 0.002 | 0.598 ± 0.001 |
| soft rule, 96 kinds developed | 0.659 ± 0.004 | 0.598 ± 0.009 | 0.667 ± 0.003 | 0.668 ± 0.006 |
| 96 kinds as born | 0.539 ± 0.004 | 0.421 ± 0.013 | 0.570 ± 0.008 | 0.561 ± 0.004 |
| our cortex (hard rule, decorrelated windows) | 0.620 ± 0.010 | 0.516 ± 0.012 | 0.664 ± 0.007 | 0.648 ± 0.006 |

Their measure in small (a ridge on the whole map, 96 kinds, 5,000 pictures): 0.616 / 0.613 / 0.613 developed
against 0.515 / 0.515 / 0.521 as born.
- **Development by the soft rule is worth 0.14 to the word cells and 0.10 to ridge, on every seed.** "Developed
  filters do no better than random" is a finding about our hard rule, not about Hebbian development. My stated
  expectation (under 0.02 in our measure) was wrong on all three.
- The soft layer is 0.07 above our cortex for the word cells and 0.04 for ridge, with raw standardised colours
  in and no decorrelation of the windows. With 96 kinds it is still 0.04 above our 256.
- The relay still earns its place on the better layer for the word cells (0.591 → 0.689) and hardly matters to
  ridge (0.693 → 0.700).
- Not known yet: the whole stream, and backprop of this layer's shape (`organic-softbase1`).
**Decision:** nothing is adopted before the dissection says which part does the work and whether the layer-wide
division is needed. The paper says what is measured: the abstract and the section (now "Our Hebbian development
did nothing for the learner") carry the three seeds; the headline comparison stays on our cortex until the
like-for-like numbers on the soft layer are in. Draft 8 sent to Kris.

**The dissection, first arm (seed 0, softhebb_dissect_s0.log):** with the competition made hard (only the
strongest kind learns), two kinds' weights are alike 0.402 on average and 916 pairs are above 0.9 (published
soft rule: 0.09 and 2–4 pairs); the summary uses 7 directions (27), 144 through the relay (577). All weights
come to length 1 within 10,000 pictures. The hard rule in their pipeline fails the way ours failed in our
pipeline (658 near-copy pairs among our developed filters): kinds pile onto the same windows. The streams and
the other two arms are running.

**The relay circuit at a step of 0.03, three seeds** (relay_local_r1e-4_s{0,1,2}.log; word cells).

| | at 1,000 pictures | at 10,000 | seen once, 20 later |
|---|---|---|---|
| the batch relay | 0.473 / 0.463 / 0.508 | 0.613 / 0.616 / 0.631 (0.620 ± 0.010) | 0.890 / 0.860 / 0.797 |
| the circuit's resting point | 0.469 / 0.464 / 0.505 | 0.608 / 0.601 / 0.618 (0.609 ± 0.009) | 0.883 / 0.860 / 0.780 |
| developed on the relay's 6,000 pictures, held | 0.470 / 0.457 / 0.498 | 0.600 / 0.590 / 0.615 (0.602 ± 0.013) | 0.883 / 0.853 / 0.777 |
| developed on 20,000, held | 0.468 / 0.465 / 0.503 | 0.601 / 0.598 / 0.622 (0.607 ± 0.013) | 0.880 / 0.853 / 0.783 |
| from nothing, learning along the stream | 0.441 / 0.443 / 0.476 | 0.603 / 0.606 / 0.616 (0.608 ± 0.007) | 0.890 / 0.837 / 0.783 |
| developed on 6,000, then learning on | 0.476 / 0.459 / 0.506 | 0.613 / 0.608 / 0.621 (0.614 ± 0.007) | 0.887 / 0.853 / 0.770 |

- Every arm of the circuit is under the batch relay on every seed but one (seed 0, developed then learning on:
  equal). The arm a brain would have is 0.000 / 0.008 / 0.010 under: 0.006 on average, inside the spread
  between seeds (0.010) but always on the same side. The resting point itself is 0.011 under, so most of what
  is lost is the resting point's (it leaves 3,900 directions alone that the batch relay also flattens), not the
  learning's.
- From nothing, the circuit costs 0.02–0.03 at 1,000 pictures and 0.012 at 10,000: a relay can be learned along
  the lessons, but one that developed before them is better early. That is the order a brain has.
- A picture seen once is named 20 later as often (0.84 against 0.85 on average; 0.027 under on seed 2).
**Decision (rule 8: biology decides, and the score is reported).** The circuit is the relay's biological form:
interneurons, a local rule, no eigendecomposition, no constant set by hand. It costs 0.006 ± 0.005 on the
benchmark. It is taken as the relay for the next lives **if a life-scale test shows it does no harm there**,
which is not known: in a life the store's keys and the word cells' inputs are in the relay's code and that code
would keep moving for 100,000 moments, not 10,000. The work, in order: the circuit in `organic/cortex2.py`
behind a switch that leaves today's lives untouched (the digest must stay c52819a8c3a93a7b); a probe on a grown
learner's level (as the gaze probes); then lives. The paper has the three seeds in "The relay, learned locally:
two attempts" with three cautions (a fifth of the stream; the settled activity by a matrix solve, not by running
the dynamics; the rate lowered tenfold and why).

### 2026-11-25 04:20: the relay circuit in the cortex's code, behind nothing that changes today's lives
- **`organic.cortex2.RelayCircuit`**: the circuit of experiments/relay_local_probe.py as a class that stands
  where `Relay` stands (called with a summary, returns the code at unit length). It learns from the views it
  carries, once every 50 views, as in the probe; its rate is the probe's (1e-4, for 4,096 cells: the rate has
  units, and the docstring says so). No new mechanism.
- **A relay that learns does not learn from a test.** One hook in `organic/speech.py` (where the look fetches
  the relay): if the relay has a `plastic` flag it is set to the brain's own `learning` flag, the one that
  already stops the eye's adaptation during a test. The batch relay has no such flag, so nothing changes for
  it. This follows the code's existing convention (an instrument does not change what it measures); a brain's
  interneurons would not know a test from a lesson, and the views of a test carry no names, so either choice
  is defensible. I chose the one that keeps measures clean.
- **`experiments/make_cortex2_circuit.py`**: a life's cortex with the circuit in the batch relay's place,
  developed before any lesson on the very 6,000 pictures of the lived side the batch relay was settled on (the
  same draw), then saved as data/visual_cortex2_circuit.pkl (git-ignored). A life takes it with
  `CORTEX=data/visual_cortex2_circuit.pkl`; nothing else changes. Running now (`organic-mkcircuit1`).
- **The default path is unchanged, checked side by side:** the committed code and the edited code give the same
  digest of 150 moments (2baf9589ac5900bc) when run at the same time with the same data.
- **What the check showed about itself.** The digest written down on 2026-11-24 (c52819a8c3a93a7b) is no longer
  what either code gives. The likely reason, not verified: the world's files (data/world/text_qa.jsonl and
  curriculum.jsonl, last written tonight) grow while the daemons run, and a world is read as it is when a life
  starts. So (1) a digest is only good against another run at the same time: the check from now on is the
  committed code (`git archive HEAD organic` into out/tmp/, with data/ linked beside it) against the working
  code; (2) **each generation of lives starts in a slightly larger world than the last** (the plateau lives,
  the reward lives, the ramp lives at 09:00). The comparisons between generations in JUDGING.md carry that
  caution; how much the world grew between them is not measured yet. A frozen copy exists for controlled lives
  (`WORLD_DIR`, data/world_ctl) and was not used for these. Put to Kris in LOOP.md.
- **Not done:** the life-scale test. Three lives fill the memory until the ramp lives end; the circuit's first
  life-scale reading waits for a slot (a probe on a grown learner's level, as the gaze probes, needs 5 GB).

### 2026-11-25 05:07: the dissection of the soft rule (seed 0); its organic check part by part; the circuit for a life
**The dissection** (softhebb_dissect_s0.log, stream_{shh,shp}10k_s0.log; 256 kinds, 50,000 unlabelled pictures,
one part taken out at a time, nothing tuned; 10,000 pictures of the stream).

| | near-copy pairs of kinds (of 32,640) | directions, summary / through relay | word cells + relay | ridge + relay |
|---|---|---|---|---|
| published (soft, losers unlearn) | 2 | 27 / 577 | **0.689** | 0.700 |
| the layer as born | – | 8 / 236 | 0.549 | 0.603 |
| hard: only the strongest kind learns | 916 | 7 / 144 | 0.512 | 0.577 |
| soft, no unlearning (losers learn with the winner's sign) | 19,031 | 6 / 81 | 0.444 | 0.508 |
| no layer-wide division (factor held at its median, 0.00086) | the weights ran away | | | |

Against what I stated: hard near as born (it is under: 0.512 against 0.549); no unlearning clearly worse than
published (yes, and worse than hard); the constant factor: I did not know, and it diverged.
- **The unlearning does the work.** Take it away and development is worse than no development: the kinds pile
  onto the same windows (916 near copies when hard, 19,031 of 32,640 pairs when soft without unlearning). This
  is the failure our own hard rule had (658 near-copy pairs), and our conscience term did not prevent it.
  Softness alone does not help; softness is what gives the losers something to unlearn by.
- **The division by the largest change in the layer is not only a rate.** Held at its median the weights
  diverge. But that test is blunt: the factor is 1e-5 in the first 100 batches and 1.3e-3 in the last, so the
  median is eighty times too large at the start (as I wrote beforehand it might be). What the test shows is
  that the published rule needs a step that follows the size of the changes, not that the step has to be
  shared by the layer.
**The rule's parts against the organic philosophy** (what a cell could do; one seed, so this guides the next
probe and adopts nothing).
| part | in a brain | verdict |
|---|---|---|
| each colour standardised by running averages | adaptation in the retina | local |
| a soft competition among the kinds at a place | lateral inhibition, divisive normalisation | local, canonical |
| the winner strengthens, active losers weaken | plasticity whose sign depends on how strongly the cell fired (weak firing depresses, strong firing potentiates: Bienenstock, Cooper & Munro 1982; Artola, Bröcher & Singer 1990) | local; **to be read in the papers, cited from memory here** |
| the decay in proportion to the cell's own response | Oja's term, synaptic scaling | local |
| a rate that falls as the cell's weights near length 1 | a cell knows its own total synaptic weight | local to the cell |
| every change divided by the largest change in the layer | no cell has this number | **not local** |
| one filter for all places | none (ours shares them too) | a shortcut, as in our cortex |
**The next arms** (`organic-sh6`, out/tmp/softhebb_dissect2_queue.sh → softhebb_dissect2_s0.log,
stream_{sho,she}10k_s0.log), both without tuning:
- **own**: each kind divides by the largest change among its own synapses. A quantity one cell has, in the
  layer-wide one's place; the largest change of any synapse of a cell is then its rate, as in the published
  rule it is for the one cell with the largest change.
- **early**: the factor held at its early value (the median of the first 100 batches, about 1e-5), where a
  constant is safe.
**Stated before:** own: I do not know. Every kind then moves at its full rate where in the published rule most
move far slower than the fastest, so it may be too fast; my guess is within 0.03 of the published 0.689.
early: stable, and too slow to finish in 50,000 pictures; between as born and developed (0.60–0.65).
If own holds, the rule is local as a cell could run it and goes to its test in our cortex's place.

**The relay circuit for a life** (out/tmp/make_circuit.log): developed on the 6,000 pictures of the lived side
the batch relay was settled on (401 s); the average it keeps equals the batch relay's (6.5e-8). On 500 further
pictures: two views alike 0.221 with no relay, 0.031 through the batch relay, 0.040 through the circuit;
directions used 13 / 271 / 208. The same picture as on the benchmark's developed arm (350 against 236):
the circuit after 6,000 pictures has not flattened as many directions as the batch relay; on the benchmark it
closed most of that while learning on. Saved as data/visual_cortex2_circuit.pkl; waits for a life.

**The reward lives at 95,000 moments.** Seeds 0 and 2 have left pages for preschool (seed 1 is still in
pages: 0.29 above blind answering). In preschool: 0.04 / 0.01; pages now 0.23 / 0.18 and first words 0.41 /
0.37, both under where they were at 90,000 (pages 0.31 / 0.33, first words 0.45 / 0.46). Earlier stages give way
when a new one starts, as in the plateau lives.

### 2026-11-25 05:09: what the sources say about the soft rule's unlearning (read tonight, not recalled)
- **Its authors do not ground it in biology.** Journé et al. (arXiv 2209.11883, section 3, read in the ar5iv
  rendering as far as the start of section 4): they "introduce a new, simple form of anti-Hebbian plasticity
  for soft WTA networks, that simply negates SoftHebb's weight update ... in all neurons except the maximally
  activated one", and give earlier machine-learning work as the reason (Krotov & Hopfield 2019; Grinberg et
  al. 2019). The biology they cite is for the competition (Douglas et al. 1989; Douglas & Martin 2004), not
  for the unlearning. The rate that falls as a kind's weights near length 1 is theirs too, given for
  robustness and speed. **The division by the largest change in the layer is not in the text that was read**
  (it is in their code); the appendix was not read word for word.
- **Bienenstock, Cooper & Munro (1982)**, *Journal of Neuroscience* 2(1), 32–48, doi
  10.1523/JNEUROSCI.02-01-00032.1982 (reference and statement read on the encyclopedia page, not in the paper):
  a cell in a high-activity state potentiates its active synapses and in a lower-activity state depresses
  them, and the threshold between the two slides with the cell's own average activity.
- **Artola, Bröcher & Singer (1990)** could not be opened tonight (the record's page did not load). It stays
  cited from memory and unverified, and goes into no document for readers until it is read.
- **What follows for the organic check.** "Active losers weaken" has a biological form, but not the published
  one. In the published rule a kind's sign depends on whether it is the strongest at that place: a fact about
  the others. In the biological form the sign depends on the cell's own activity after the competition
  against its own sliding threshold: a fact the cell has. With a sharp competition the two agree (the winner
  is high, the rest low). **A third arm is therefore owed after own/early: the sign by the cell's own activity
  against its own running threshold** (the form in Bienenstock et al., to be taken from the paper itself, not
  designed here). If it develops the layer as the published rule does, every part of the rule is something a
  cell could do.

### 2026-11-25 06:03: the layer-wide division cannot simply be made a cell's own; what the BCM papers give
**The dissection's second step, seed 0** (softhebb_dissect2_s0.log, stream_{sho,she}10k_s0.log; all rows seed 0).

| | weight lengths (least / mean / most) | near-copy pairs | directions, summary / relay | word cells + relay | ridge + relay |
|---|---|---|---|---|---|
| published (division by the largest change in the layer) | 1.00 / 6.76 / 12.19 | 2 | 27 / 577 | **0.689** | 0.700 |
| the layer as born | 25 | – | 8 / 236 | 0.549 | 0.603 |
| own: each kind divides by its own largest change | 1.00 / 21,520 / 90,157 | 31,376 | 6 / 29 | 0.222 | 0.352 |
| early: the factor held at 1.06e-5 | 2.09 / 8.65 / 15.37 | 0 | 9 / 296 | 0.561 | 0.605 |
| the factor held at its median (0.00086) | ran away | | | | |

Against what I stated: own "within 0.03 of 0.689, but I do not know": wrong, it destroys the layer. early
"stable and too slow, 0.60–0.65": stable and slower than that (0.561, hardly above the layer as born).
- **Why own fails** (my reading, not tested further): at a place most kinds are losers with a share near zero,
  so their changes are tiny. Dividing each kind's changes by its own largest makes every kind move at full
  rate whatever it had to learn; the difference in size between the winner's change and the losers' is the
  rule's information, and a division made cell by cell erases it.
- **So the published layer is not local as it stands, and the two obvious local stand-ins do not work.** The
  step has to follow the size of the changes (which falls a hundredfold as the weights shorten from 25 towards
  1) and has to be the same for all kinds. This is not in the paper's text as far as it was read; it is in the
  code. If the dissection holds on seeds 1 and 2 (`organic-sh7`, all five arms, about 3.5 hours) the paper
  should say so in one sentence.
- **No third stand-in of mine.** A step shared by the layer that follows the changes' size could be imagined
  (a broadcast signal), and I could design one; that would be hand-design. The question goes to biology instead.
**What the BCM papers give** (read in the paper: Blais, Intrator, Shouval & Cooper 1998, *Neural Computation* 10,
1797–1813; the 1982 and 1994 papers could not be opened tonight, the server asks for a browser check).
- Their words for the family: modification functions "characterized by a negative region for small
  postsynaptic depolarization, a positive region for large postsynaptic depolarization, and a threshold that
  moves and switches between the Hebbian and anti-Hebbian regions."
- The quadratic form (from Intrator & Cooper 1992, as given in their equations 3.11–3.12): with c the cell's
  rectified activity, the change of a synapse is c (c − θ) σ′ times its input, θ = E[c²], the averages kept as
  running averages with a time constant. "Unlike the measures [kurtosis, skewness], the QBCM rule does not
  require any additional stabilization."
- They study one cell on purpose ("we eliminate possible network-lateral interaction effects"); many cells need
  lateral inhibition, which that paper does not give. The network form is in Intrator & Cooper 1992 (not opened).
- **What this means for us.** The rule that biology has for "active losers weaken" carries its own stability
  (the sliding threshold) and needs neither weights born long, nor a rate tied to their length, nor a
  division by anything shared. That is the candidate: cells with the quadratic BCM rule and lateral
  inhibition, in the soft layer's pipeline (same way in, same response and averages out), so that only the
  rule differs. **Not built tonight:** the network form with inhibition has to be read first (Intrator &
  Cooper 1992; Castellani et al. 1999), and its constants (the threshold's time constant, the inhibition's
  strength, the rate) taken from there. Building from the one-cell paper plus my own inhibition would be a
  design of mine.
**Backprop of the soft layer's shape, first numbers** (backprop_soft_tune_s0.log; seed 0, first 10,000 pictures of
the whole stream, 1,000 test pictures; not yet like for like with the table above, which has batch 1 and 2,000):
without the relay the best of 12 settings is 0.477 (the usual rectifier, rate 0.001, 8 pictures a step); the
layer's own power 0.7 gives at most 0.446. The way back through the relay is exact at power 1 (six decimals;
run tonight as `organic-bpcheck2`) and within 3% at power 0.7 but for one tiny weight (the guard at zero). The
relay arms and the whole stream are running.

### 2026-11-25 06:07: the BCM network form, read in the paper
Intrator & Cooper (1992), *Neural Networks* 5, 3–17 (the authors' copy, read tonight; equation numbers theirs).
- **One cell** (their 4.1–4.2): activity c = σ(x·m), σ a smooth sigmoid; threshold θ = E[c²] (in the 1982
  form it was a power of the average activity; they give the difference: "The original BCM form requires that
  a history of activity ... be stored and then via a non-linear process produce the modification threshold.
  The present form ... requires that the non-linear process occur first"; "Either form seems consistent with
  presently available experimental results"); change of the synapses: μ φ(c, θ) σ′ x with φ(c, θ) = c(c − θ)
  (equation 5). The threshold "will always move itself to a position such that the distribution will never be
  concentrated at only one of its sides": that is the rule's own stability. The rate "has to decay in time".
- **Many cells, inhibition fed forward** (4.3): c̃_k = σ(c_k − η Σ_{l≠k} c_l), θ̃_k = E[c̃_k²], and
  ṁ_k = μ { E[φ(c̃_k, θ̃_k) σ′(c̃_k) x] − η Σ_{j≠k} E[φ(c̃_j, θ̃_j) σ′(c̃_j) x] } (equation 11).
- **Many cells, lateral inhibition** (2.3, 4.5; Scofield & Cooper 1985, Cooper & Scofield 1988): c = M d + L c,
  that is c = (I − L)⁻¹ M d; the fed-forward form is its first-order approximation and "generates the same
  synaptic modification equations ... within a scaling factor" in the mean-field approximation.
- **What the paper does not give:** numbers. No strength of inhibition, no rate, no time constant for the
  threshold's average, no particular sigmoid.
**The organic check for a BCM layer, as far as tonight's reading goes.**
- The rule on a cell's own inhibited activity (the first term of equation 11) is local: the cell's activity,
  its own running average, its input.
- **The second term of equation 11 is not obviously local**: a cell's synapses change by the *modification
  signals* of the other cells (their φ σ′), carried with the inhibition's weight. It comes from descending one
  objective for the whole network. Whether the lateral network of Cooper & Scofield has it has to be read
  there (1988, *PNAS*; not opened). Two arms follow from the reading, not from me: with the term (the
  objective's rule) and without (each cell by itself).
- **The constants have to come from a source** (Law & Cooper 1994 or the 2004 book's simulations, or the
  authors' published simulator and its defaults), read, not chosen by me. If none can be opened, the arm is
  run at the values of whatever published simulation can be read and that is said; no search over them.
- The way in and out stays the soft layer's (standardised colours, 5 × 5 windows, the same response and
  averages), so that only the rule differs from the published soft rule and from the layer as born.
**Stated before anything is built:** I do not know whether BCM cells with inhibition develop 256 different
filters on raw standardised windows; the natural-image work reads oriented fields from whitened or
centre-surround input, one cell at a time. If the first reading shows near copies (the hard rule's failure),
the inhibition is not doing the unlearning's work and the arm is reported as it is.

### 2026-11-25 07:30: BCM's numbers found in its authors' simulator; the first arm, and what I expect of it
**Where the numbers are.** The papers read so far give none (Intrator & Cooper 1992; Blais et al. 1998; and,
opened this morning, Blais, Intrator, Shouval & Cooper 1997, *NIPS* 10, 423–429, "Receptive Field Formation in
Natural Scene Environments: Comparison of Single Cell Learning Rules": the same rules, 13 × 13 round patches
from 12 natural images after a centre-surround filter or whitening, and no rate or time constant). They are
in the authors' simulator (B. Blais, *Plasticity*, a public repository; `plasticity/train.pyx`,
`plasticity/utils.py` and `run_sim.py` read in the source this morning, not run):
- **The rule in code:** `phi = eta*y*(y-th)`; `w += phi*x`; then `th += (y*y-th)/tau`. The derivative of the
  sigmoid is off by default. Weight decay is zero by default: the rule has no other stabiliser.
- **Its defaults:** `eta = 8e-6`, `tau = 1000`, weights born uniform in [−0.05, 0.05], thresholds born uniform
  in [0.01, 0.02], the sigmoid `tanh(u)` below zero and `50 tanh(u/50)` above, 500 × 500 presentations, one
  cell.
- **Between cells:** nothing by default. Two lateral forms exist. One takes a matrix that the user supplies
  (no default). One is a uniform strength that learns (`y_i = σ(σ(w_i·x) + L_i ȳ)`, `L_i += eta*(−phi_i ȳ − L_i)`
  with ȳ the cells' mean activity); `phi` already carries `eta`, so at the defaults that strength sits near
  `eta` times a product of activities, about 1e-5: no interaction to speak of.
- A search result names a table of a later paper of the group with a rate of 1e-6, a first threshold of 0.7
  and a memory constant of 300–1000; the page asks for a browser check and was **not opened**, so those
  numbers are not used.
- **Not opened:** Udeigwe, Munro & Ermentrout 2017 (*J. Math. Neurosci.*; when the threshold's time scale keeps
  the rule stable): the publisher redirects to a login. Law & Cooper 1994 and Cooper & Scofield 1988 as before.

**So the inhibition between BCM cells still has no published number that I can read**, and equation 11's
strength η stays unknown. What can be built with nothing of mine in the rule is the simulator's default: **a
layer of BCM cells that do not interact**, which differ only by how they were born.

**Organic check.** Local in every part: a cell's activity, its own running average of its squared activity,
its input. No normalisation, no step shared by the layer, no winner chosen by anyone. The biology is among the
best supported we have (the sliding threshold: Kirkwood, Rioult & Bear 1996 is the usual citation; **not
opened by me**, so it is not cited for readers yet). What is ours is the input (the soft layer's standardised
5 × 5 colour windows where theirs is a 13 × 13 centre-surround patch); the 1997 paper says of this rule that
it "produce[s] oriented RFs even when the data is not whitened". The rate's units: both inputs have unit
variance at each point and the cell's own threshold sets the scale of its activity, so the rate is taken as
it stands; whether 250,000 presentations are enough on our input is read from how far the weights still turn.

**Stated before it is run** (`experiments/bcm_probe.py`, seed 0): the cells will be stable (lengths bounded,
no run-away) because that is what the rule's threshold is for. Shown the same windows with nothing between
them, most cells will end as copies of a few filters: **I expect fewer than 40 different filters among 256**
(grouping at 0.9) and, in the stream, word cells + relay **at or under the layer as born (0.549), between 0.45
and 0.55**, the hard rule's kind of failure. If so, the reading says what inhibition would have to do and
that its strength is the missing number. If instead the cells stay different on their own, that is news and
the arm goes to three seeds.

### 2026-11-25 07:40: independent BCM cells, seed 0: stable, slow, and most of them the same two filters
`experiments/bcm_probe.py` (the single window checked against the soft layer's: equal). Logs:
`out/bench/bcm_develop_s0.log`, `bcm_develop_2500000_s0.log`.
- **At the simulator's 250,000 presentations the layer has hardly begun.** Weight lengths 0.202 / 0.289 /
  0.725 (least, mean, most; born near 0.25), one cell of 256 longer than twice its birth, 256 different
  filters, and the turning between readings still growing (0.043 since 100,000). The simulator's number is for
  one cell on its own input; on ours it is not enough, so the same run was read on to 2,500,000 (nothing else
  changed; said in the file).
- **At 2,500,000 presentations:** lengths 0.218 / 1.867 / 8.935, thresholds up to 13.5, largest activity 42.7
  (the sigmoid's top is 50): **stable, as I said**. 90 cells have developed; the others still sit near birth.
  Near copies: 43 pairs above 0.9 at 500,000, 900 at 1,000,000, **2,034 at 2,500,000; 134 different filters of
  256 and falling**; the weights still turn (0.186 since 1,000,000).
- **What the developed cells are** (`docs/img/bcm_filters_2500000_s0.png`, sorted by length): of the 90 developed cells 55 are one
  colour-opponent blob (red and blue against green in the middle of the window), 24 its opposite (green
  against red and blue), **two are an oriented grey bar**, and the other nine fall in six small groups
  (grouping at 0.9, signed). So the rule does find what the
  retina's cells find in colour, and nearly every cell finds the same thing.
- **Against what I said:** the direction is as stated (copies of a few filters; fewer than 40 different ones
  is where it is heading, not yet where it is: 134 at 2,500,000, with the developed 90 in nine groups, 79 of them in two).
  The stream reading is owed (`organic-bcm3`, behind the dissection: the layer at 250,000, 2,500,000 and
  10,000,000 presentations and as born); `organic-bcm4` reads the development on to 10,000,000.
- **What it says about the missing part:** with nothing between the cells, what decides a cell's filter is the
  input's strongest non-Gaussian direction, the same for all. Inhibition between the cells (or the soft rule's
  unlearning by the losers, which is inhibition's effect written into the rule) is what has to make them
  different, and its strength is the number no source I could open gives. Two honest ways on, neither a
  constant of mine: (a) an inhibition whose strength is learned by a published local rule (the anti-Hebbian
  lateral weights of Földiák 1990, and of the similarity-matching circuits already read for the relay:
  Pehlevan & Chklovskii), with BCM on the feed-forward synapses; (b) the centre-surround or decorrelated input
  the BCM work used, which removes the direction all cells now share. Each needs its reading first (Földiák
  1990 has **not been opened**). Nothing built until then.

### 2026-11-25 07:20: an inhibition that learns its own strength, read in the paper (Földiák 1990)
Földiák (1990), "Forming sparse representations by local anti-Hebbian learning", *Biological Cybernetics* 64,
165–170 (a scanned copy with a text layer, read this morning; symbols as printed, where the scan allows).
- **The network:** n units over m inputs; feed-forward weights q, feedback weights w between the units,
  a threshold t for each unit. The output settles by dy*/dt = f(Σ_j q_ij x_j + Σ_j w_ij y*_j − t_i) − y*_i with
  f(u) = 1 / (1 + exp(−λu)), then is rounded to 0 or 1. Feed-forward weights born random (uniform on [0, 1],
  unit length), feedback weights born at 0.
- **Three rules, each with terms of the two cells a connection joins only** ("all the information necessary
  for the modification is available locally at the site of the connection"):
  anti-Hebbian between units Δw_ij = −α (y_i y_j − p²), kept at or under zero and zero for i = j;
  Hebbian into a unit Δq_ij = β y_i (x_j − q_ij); threshold Δt_i = γ (y_i − p).
- **Its numbers:** lines on an 8 × 8 grid, 16 units: α = 0.1, β = γ = 0.02, λ = 10, p = 1/8, thresholds settled
  first (100 cycles at γ = 0.1 with α = β = 0); letters: α = 0.01, β = 0.001, γ = 0.01, λ = 10, p = 0.1, 8,000
  presentations.
- **For us:** the strength of inhibition is not a number here, it is learned (two units that fire together
  more than chance come to inhibit each other), which is what the BCM layer lacks. What is set by hand is p,
  the share of presentations a unit should fire at; the units are binary and the inputs are made-up patterns.
- **What follows, and what does not.** Putting BCM's feed-forward rule under Földiák's feedback rule would be
  a combination of mine: not built. The published line that carries this network to natural images is, as I
  remember it and **have not opened**, Falconbridge, Stamps & Badcock (2006, *Neural Computation*) and
  Zylberberg, Murphy & DeWeese (2011, *PLoS Computational Biology*; open access): complete layers with these
  three local rules and their own constants, on whitened image patches. **Next: open the 2011 paper, read its
  rules and numbers, write the organic check; if it stands, that layer, unchanged, is the next arm in the soft
  layer's place** (its own input as one arm, the soft layer's windows as the other).

### 2026-11-25 08:31: BCM at 10,000,000 presentations; dissection seed 1; a published layer with learned inhibition (SAILnet)
**The BCM layer read on** (`out/bench/bcm_develop_10000000_s0.log`): at 5,000,000 presentations 19 different
filters of 256 (8,970 near-copy pairs), **at 10,000,000 ten different filters (14,345 pairs)**: what I
stated (fewer than 40). **A correction to "stable, as I said":** the weights' lengths go on growing (mean 1.87
at 2,500,000, 3.45 at 5,000,000, 8.82 at 10,000,000; longest 30.6) and the largest activity has reached the
sigmoid's top (49.07 of 50). Nothing ran away to infinity, but I have not seen the lengths level off, so
"stable" is not shown on our input; what bounds the cells here may be the sigmoid and not the threshold. The
stream reading comes with `organic-bcm3`.

**The dissection, seed 1** (logs `softhebb_dissect_s1.log`, `stream_sh{h,p,e,o}10k_s1.log`; word cells + relay /
ridge + relay at 10,000 pictures; published 0.684 / 0.700, as born 0.546 / 0.607): hard 0.496 / 0.567 (930
near-copy pairs); no unlearning 0.423 / 0.495 (19,033 pairs); the factor at its median (0.00085): ran away; each
kind's own division 0.256 / 0.359 (weights 72,000 long); **the factor at its early value (1.49e-5): 0.600 /
0.620**, where seed 0 (1.06e-5) gave 0.561 / 0.605. Same picture as seed 0 in every arm; the early-value arm
gets further on this seed, with a step 40% larger. That a fixed step somewhere between "early" and "median"
might do is a thought I am not following: it would be a search for a constant.

**A published layer whose inhibition learns its strength.** Zylberberg, Murphy & DeWeese (2011), "A Sparse
Coding Model with Synaptically Local Plasticity and Spiking Neurons Can Account for the Diverse Shapes of V1
Simple Cell Receptive Fields", *PLoS Computational Biology* 7(10): e1002250 (the preprint's text read this
morning, arXiv 1109.2239v1; **the journal's own pages still to be compared**), and the authors' published code
(a public repository: `init.m`, `SAILnet.m`, `activities.m`, read, not run).
- **Cells:** leaky integrate-and-fire. For 50 steps a cell's potential follows u ← 0.9 u + 0.1 (Q x − W a),
  a the cells that fired on the step before; a cell fires when u reaches its threshold θ and starts again
  from 0; its answer to an image is its count of spikes n.
- **Three rules** (the paper's equation 1; Földiák's, with counts): between cells ΔW_im = α (n_i n_m − p²), no
  self-connection, never below zero; into a cell ΔQ_ik = β n_i (x_k − n_i Q_ik) (Oja's form of Hebb's rule);
  threshold Δθ_i = γ (n_i − p). "each of our learning rules is 'synaptically' local: the information required
  ... is merely the activity of the pre- and post-synaptic units."
- **Numbers.** Code: Q born Gaussian with each cell's weights at unit length, W born 0, θ born 2, α = 1,
  β = 0.01, γ = 0.1, p = 0.05 spikes an image, batches of 100 patches (the batch's average change), 25,000
  batches. Paper: θ born 5; "we start the training with large values for the learning rates, and these are
  eventually reduced. For the last 10⁴ batches ... (α, β, γ) = (0.1, 0.001, 0.01)"; about 10⁷ presentations to
  settle. Input: 16 × 16 patches of whitened natural images, each patch at zero mean and unit deviation; 256
  cells in the code, 1,536 in the paper.
- **What the authors call unrealistic themselves:** the cells inhibit each other directly (no interneurons),
  learning in batches, continuous inputs, inference and learning in turns.

**Organic check.** Every change uses the two cells a connection joins, or the cell's own count: local, with
no step shared by the layer, no normalisation by anyone, and the strength of inhibition learned (two cells
that fire together more than chance come to inhibit each other): the part the BCM layer lacked. Spiking cells
are closer to biology than anything else in our cortex. **What is set by hand is p**, a cell's target rate;
it is a set point each cell keeps for itself (the well-known homeostasis of firing rate), not a weight that
balances two mechanisms, and it is the authors' number. The batch is an average of local changes. Direct
inhibition between excitatory cells is not Dale's law; the authors' later work is said to give it
interneurons (King, Zylberberg & DeWeese 2013: **not opened**). It passes for an arm.

**The arm** (`experiments/sailnet_probe.py`): the code's layer unchanged, 256 cells; 25,000 batches at the
code's rates, then 10,000 at the paper's last rates (both published; nothing else). **Ours, and said:** the
input is the soft layer's (5 × 5 colour windows, each colour standardised; not whitened, not brought to unit
deviation patch by patch), and the first reading takes only the learned feed-forward weights Q through the
soft layer's own response (as every arm of the dissection was read); the layer's own answer, spike counts
under inhibition, would be a second and much dearer reading.

**Stated before it is run:** the cells will stay different (fewer than 10 near-copy pairs above 0.9 among
256), because that is what the learned inhibition is for. In the stream, word cells + relay at 10,000
pictures: **over the layer as born (0.549) and under the published soft rule (0.689); my guess 0.58 to 0.66.**
What could make it fail: the weights were learned for spiking cells under inhibition and are read here
without either; and raw colour windows may give mostly colour blobs.

### 2026-11-25 09:18: SAILnet on the soft layer's raw windows: one filter. My expectation was wrong; the reason is the input
`out/bench/sail_develop_s0.log` (seed 0; 25,000 batches at the code's rates, 10,000 at the paper's last):
- **All 256 cells end as the same filter** (two cells' weights alike 0.997; every one of the 32,640 pairs
  above 0.9). I had said fewer than 10 pairs. The collapse is there after 1,000 batches (7 different filters)
  and never undone.
- The layer never reaches its own targets: a cell fires 0.068 times an image (target 0.05), two cells fire to
  the same image 0.034 (target 0.0025, fourteen times over); inhibition has grown to 1,300 on average and
  thresholds to 140 (born at 2) and both were still rising when the rates were lowered.
- **Why (read from the numbers, not yet tested):** the rule's constants are in the units of its input, a
  whitened patch with its mean removed and at unit deviation, where a cell's drive is about ±1 and a
  threshold of 2 is rarely crossed. The soft layer's window keeps its mean and its correlations: one direction
  (brightness) carries most of the variance, the drive along it is in the tens, every cell fires many times to
  the same windows, and Oja's term with squared counts pulls all of them onto that direction faster than
  inhibition, capped at stopping a cell, can part them. I named this risk beforehand as "ours, and said" and
  still guessed 0.58–0.66; the stream reading (`organic-sail2`) will give the number, and it will be one
  filter's worth.
- **A lesson already on the list, met again:** a published rule's numbers hold for its input. The soft rule
  brings its own input normalisation; this one expects the retina's (centre-surround, contrast) to be done.

**The arm that follows: the same layer in our cortex's place.** Our cortex's first layer already gives its
filters the published kind of input: each 5 × 5 colour window with its mean removed, decorrelated (the settled
Z), at unit length (`Layer2._drive`). So: SAILnet, unchanged, develops 256 cells on those windows (times
√75, so that a component has unit deviation as in the code: a change of units, said), taken as our own
development takes them (windows of 64 × 64 unlabelled CIFAR-100 pictures with contrast above 0.3 of the
picture's largest); its Q then stands where our hard rule's filters stand, and the benchmark's own pipeline
codes the pictures (`experiments/sailnet_cortex_probe.py`; variants `sailc` as developed, `sailcu` the same
at unit length as every filter of our cortex is). Beside it, already measured: our cortex with filters as born
or developed by our hard rule, word cells + relay 0.620 ± 0.010 at 10,000 pictures (seed 0: 0.613).
**Stated before it is run:** on this input the layer will hold its targets (rate within a fifth of 0.05, the
pair rate within a factor of two of 0.0025) and its cells will stay different (fewer than 10 pairs above
0.9). In the stream: **over our cortex's 0.613 on seed 0, my guess 0.63 to 0.67**; under the soft rule in its
own pipeline (0.689). If the cells stay different and the stream does not move, then different filters are
not what our cortex lacks, and that is worth as much.

### 2026-11-25 10:03: the dissection on three seeds; SAILnet in our cortex's place after 25,000 batches; the soft layer's references, seed 0
**The dissection, three seeds** (word cells + relay / ridge + relay at 10,000 pictures; logs
`softhebb_dissect_s{0,1,2}.log`, `softhebb_dissect2_s0.log`, `stream_sh{h,p,e,o}10k_s*.log`):
| the rule | word cells + relay | ridge + relay | near-copy pairs (of 32,640) |
|---|---|---|---|
| published | 0.689 ± 0.005 | 0.700 ± 0.001 | 2 (seed 0) |
| layer as born | 0.548 ± 0.002 | 0.605 ± 0.002 | 0 |
| hard: only the winner learns | 0.502 ± 0.009 | 0.571 ± 0.005 | 916 / 930 / 903 |
| soft, losers learn with the winner's sign | 0.433 ± 0.011 | 0.499 ± 0.008 | 19,031 / 19,033 / 19,610 |
| the layer-wide factor held at its early value (1.06e-5 / 1.49e-5 / 1.16e-5) | 0.578 ± 0.020 | 0.616 ± 0.009 | 0 |
| the factor held at its median | weights not finite, all three seeds | | |
| each kind divides by its own largest change | 0.238 ± 0.017 | 0.352 / 0.359 / (running) | 31,376 / 31,376 / 20,262 |
Every arm says on three seeds what it said on one. In the paper as a paragraph (the own-division weights are
hundreds of times their birth length on average, 11,305 to 21,520 against 25: I first wrote "ten-thousand-fold"
and took it out before it left this machine).

**SAILnet in our cortex's place, seed 0, after the 25,000 batches at the code's rates**
(`out/bench/sailc_develop_s0.log`; the last 10,000 batches run):
- **It holds its targets, as stated:** a cell fires 0.049 times an image (target 0.05), two cells to the same
  image 0.0047 (target 0.0025; within the factor of two I allowed, at its edge).
- **It does not keep its cells different, against what I stated:** 5,797 near-copy pairs, **92 different
  filters among 256** (1,367 pairs after 1,000 batches, 3,997 after 5,000, 5,010 after 10,000: slowing).
  Inhibition goes on growing (87 on average, the strongest 1,739).
- **A reading I had not thought of beforehand:** a 5 × 5 colour window with its mean removed has 74
  dimensions. 256 cells are three and a half cells to a dimension, and the layer has made about one filter per
  dimension and a quarter more, the rest copies that take turns (the inhibition between two copies makes them
  fire to different images, which is all the rule asks: uncorrelated counts, not different weights). The
  paper's layers are 1 to 6 cells per dimension on 256-dimensional patches, where there is far more to tell
  apart. Whether this is the reason is a guess until tested; the test that needs no constant of mine is the
  same layer with as many cells as it can use, or on our second layer's larger input. First the stream
  reading (`organic-sailc2`).

**The soft layer's references, seed 0, whole stream** (`stream_shfull_s0.log`, `backprop_soft_full.log`; 50,000
pictures once, 2,000 tests): word cells + relay 0.729 (no relay 0.675); ridge + relay 0.740 (0.745 without);
streaming LDA + relay 0.743 (0.741). Backprop of the layer's shape, tuned on the first 10,000 as the first
reference was (24 settings): filters learned from random, no relay 0.547; with the fixed relay 0.597; the
developed filters kept and only a softmax readout learned by gradient, no relay 0.623; with relay: running.
The same order as on our cortex (local readers on local filters over one-pass backprop), with a wider gap.
Seeds 1 and 2 follow; nothing in the paper's table until they are in.

### 2026-11-25 10:50: the BCM layer in the stream (first readings); a gradient readout on the soft layer's frozen filters
- **BCM, seed 0, word cells + relay / ridge + relay at 10,000 pictures** (`stream_bcm*10k_s0.log`): after the
  simulator's 250,000 presentations **0.443 / 0.506**; after 2,500,000 **0.371** (ridge running); 10,000,000 and
  the layer as born still to come. Under the range I gave (0.45 to 0.55). Even the first reading, where one
  cell of 256 had developed and all filters still counted as different, is ten points under the soft layer as
  born (0.549): through the relay its code uses 77 directions where the same cells as born use 254
  (`bcm_code_s0.log`). A few long filters and a common drift are enough to spend the code on a few directions.
- **SAILnet in our cortex's place, final** (35,000 batches): rates on target (0.049), 98 different filters of
  256, 5,592 near-copy pairs. Stream reading queued.
- **The soft layer's last reference on seed 0:** the developed filters frozen, a softmax readout trained by
  gradient through the fixed relay: **0.737 at 50,000**, over the word cells + relay (0.729) and level with
  streaming LDA (0.743). A softmax readout's gradient is a delta rule on one layer of synapses, so this is not
  backprop through anything; it says our reader leaves about a point on the table on this code, where on our
  cortex the two were level. Seed 1 of the whole stream: word cells + relay 0.734 (no relay 0.671).
- **A caution on one row:** the frozen-filters run without the relay (0.623) used the response tuned for
  backprop without relay (the plain rectifier, power 1), not the layer's own (power 0.7); the run with the
  relay used 0.7. So 0.623 is not the soft code read by a gradient readout, and only the relay row (0.737) is
  like for like with the word cells.


### 2026-11-25 11:42: the BCM layer and SAILnet on raw windows in the stream, seed 0
Word cells + relay / ridge + relay at 10,000 pictures (`stream_{bcm,bcm_2500000,bcm_10000000,bcmr,sail,sailr}10k_s0.log`):
| layer (the soft layer's pipeline) | word cells + relay | ridge + relay |
|---|---|---|
| BCM cells as born | 0.566 | 0.611 |
| BCM after 250,000 presentations (the simulator's number) | 0.443 | 0.506 |
| BCM after 2,500,000 | 0.371 | 0.468 |
| BCM after 10,000,000 (ten different filters) | 0.471 | 0.537 |
| SAILnet's cells as born | 0.551 | 0.620 |
| SAILnet developed on raw windows (one filter) | 0.213 | 0.351 |
| for comparison: the soft layer as born / published rule | 0.549 / 0.689 | 0.603 / 0.700 |
- **BCM:** every state is under its own birth (0.566). I had said "at or under the layer as born, 0.45 to
  0.55": right in direction; the two earlier states are lower than my range. The layer is worst in the
  middle of its development (2,500,000: 90 cells developed, 79 of them two blobs) and recovers a little when
  all cells have settled into ten filters, since ten different filters is more than two.
- **SAILnet on raw windows:** one filter's worth, as the weights said.
- **Both controls agree with the soft layer as born** (0.566 and 0.551 against 0.549): random filters of any
  birth give about 0.55 in this pipeline, so each arm's loss is its rule's.
**Where queue E 27 stands.** Three published local rules have now been put where the soft rule stands: BCM
without interaction (copies), SAILnet on raw windows (one filter), and SAILnet on our cortex's windows (98
filters; stream pending). The only rule that has made filters worth having is still the one with a non-local
step.

### 2026-11-25 12:15: SAILnet in our cortex's place, in the stream; what its filters show; and a fault in our benchmark's input
**The stream** (`stream_sailc10k_s0.log`; seed 0, 10,000 pictures): Q as developed, word cells + relay **0.484**,
ridge + relay 0.571, against our cortex's 0.613 and 0.659 on the same seed. I had said 0.63 to 0.67: wrong, by
fifteen points. (The same Q at unit length, `sailcu`, is in the stream now.)
**What the filters are** (`docs/img/sailc_filters_s0.png`, sorted by length; lengths 0.99 to 6.69):
- 107 of the 256 cells are one filter, and they are the long ones. Further groups of 14, 13, 6 and 5; 75 cells
  are alone. All the filters together span 3 directions by the participation measure (as born: 58).
- **The short filters are oriented edges, bars and gratings, grey and colour-opponent**: what the paper shows
  and what V1's simple cells look like. So the rule does on our input what it is published to do, for about a
  third of the cells.
- The long filter answers above 0.5 to 16% of the benchmark's windows and to 3.7% of the windows of COCO
  photographs at the same size. That sent me to look at the windows themselves.

**The fault: the benchmark's pictures are blown up, and a 5 × 5 window of a blown-up picture is nearly empty.**
The benchmark shows CIFAR's 32 × 32 pictures to the cortex at 64 × 64 (bilinear), as the lives' world shows
CIFAR, because the learner's eye is 64 × 64. Measured this noon (`experiments/upsample_probe.py`; 600 pictures
each, windows as the cortex's development takes them, mean removed, unit length):
| pictures | directions holding 99% of the windows' variance | directions above the decorrelation's floor (a thousandth of the largest) | participation after the cortex's decorrelation |
|---|---|---|---|
| CIFAR blown up to 64 (the benchmark; CIFAR in the lives) | 14 | 22 of 75 | 19 |
| CIFAR at its own 32 | 33 | 47 of 75 | 33 |
| COCO photographs brought down to 64 (most of the lives' pictures) | 34 | 48 of 75 | 35 |
A 5 × 5 window at 64 sees two and a half of CIFAR's own pixels across; blown up, it holds less than half of
what a window of a real picture holds. **Every filter rule we tried on the benchmark in our cortex's place,
our hard rule included, had 22 directions to tell 256 filters apart with.** That does not excuse the hard rule
(its near copies are its own) and it does not touch the lives' COCO pictures, but it bears on three things
the paper says: (1) "developed filters do no better than random" on our cortex was measured on windows with
little in them; (2) our cortex's 0.620 against the soft layer's 0.689 compares a cortex looking at blown-up
pictures with a layer looking at the pictures themselves; (3) SAILnet's collapse onto one filter here may be
the input's and not the rule's.
**The control, with nothing new in it** (`experiments/cortex_native_probe.py`): the benchmark's own cortex,
code and stream, the pictures at their own 32 × 32 (decorrelation settled on them, the relay too); filters as
born (`nat_r`) and developed by our hard rule (`nat_d`); seed 0, then three seeds if it moves.
**Stated before it is run:** as born, word cells + relay at 10,000 pictures **over 0.613, my guess 0.64 to
0.68** (twice the directions in a window; but a coarser map, 14 × 14 after pooling where the blown-up picture
gives 30 × 30, which may cost some of it back). Developed against as born: **within 0.01 either way**, since
the hard rule's failure is its near copies. If the developed filters now do better than random, the paper's
sentence about our development was about the blown-up input and has to say so.

### 2026-11-25 12:58: the benchmark's cortex on CIFAR at its own size, seed 0: the hard rule's development helps there
`out/bench/native_code_s0.log`, `stream_{nat_r,nat_d}10k_s0.log` (word cells + relay / ridge + relay at 10,000):
| our cortex, seed 0 | word cells + relay | ridge + relay | near-copy pairs among the filters |
|---|---|---|---|
| blown up to 64 (the benchmark), developed by our hard rule (`norm0`) | 0.613 | 0.659 | 658 |
| at 32, filters as born | 0.627 | 0.633 | 0 |
| **at 32, developed by our hard rule** | **0.656** | **0.667** | 100 (194 different filters) |
| SAILnet's filters at unit length, blown up (`sailcu`) | 0.588 | 0.635 | 5,592 |
- **As born:** 0.627 against 0.613. I had said 0.64 to 0.68: less than I thought (the coarser map takes back
  most of what the fuller windows give; ridge is even lower at 32, 0.633 against 0.659).
- **Developed against as born: +0.029 for the word cells, +0.034 for ridge.** I had said within 0.01. On
  pictures at their own size our hard rule makes a sixth of the near copies it made on blown-up ones (100
  pairs against 658) and its filters are worth three points. **One seed.** If seeds 1 and 2 agree
  (`organic-native2`), the paper's "our Hebbian development did nothing for the learner" was a finding about
  blown-up pictures, and has to be said so; the hard rule would still be seven points under the published
  soft rule's gain (0.548 → 0.689 in its pipeline), but not nothing.
- `sailcu`: SAILnet's filters at unit length give 0.588 on the blown-up benchmark, under our cortex's 0.613;
  with a hundred cells alike it could hardly do otherwise. Its test on real-sized pictures is the one that
  counts now, after the three seeds of the control.
- **A hole in this table, seen after writing it:** there is no reading of the blown-up cortex *as born* on
  these same 10,000 pictures (0.613 is the developed one; the whole-stream runs, on other pictures, gave
  0.636 as born and 0.623 developed at 10,000). So "as born 0.627 against 0.613" compared unlike things, and
  the clean rows are owed: the same script with the pictures blown up (`SIZE=64`, variants `big_r`, `big_d`),
  three seeds, queued with seeds 1 and 2 of the native size (`organic-native4`, each step only when 3.3 GB
  are free). What stands from seed 0 without that hole: at native size, developed beats as born by 0.029
  (same script, same pictures, same seed).


### 2026-11-25 13:52: the soft layer on the whole stream, three seeds of the local readers; the replay test's first reading
**The soft layer, 50,000 pictures once** (`stream_shfull_s{0,1,2}.log`; mean ± deviation over three seeds):
| reader on the soft layer's code | 500 | 5,000 | 10,000 | 50,000 |
|---|---|---|---|---|
| word cells + relay | 0.416 ± 0.055 | 0.654 ± 0.010 | 0.687 ± 0.011 | 0.730 ± 0.004 |
| word cells, no relay | 0.384 ± 0.041 | 0.574 ± 0.003 | 0.602 ± 0.014 | 0.671 ± 0.005 |
| ridge + relay | 0.456 ± 0.005 | 0.663 ± 0.006 | 0.699 ± 0.007 | 0.744 ± 0.006 |
| ridge, no relay | 0.471 ± 0.001 | 0.662 ± 0.005 | 0.691 ± 0.002 | 0.744 ± 0.003 |
| streaming LDA + relay | 0.439 ± 0.014 | 0.617 ± 0.008 | 0.681 ± 0.006 | 0.748 ± 0.005 |
| streaming LDA, no relay | 0.455 ± 0.013 | 0.666 ± 0.006 | 0.700 ± 0.004 | 0.747 ± 0.006 |
| backprop of the layer's shape, filters from random, no relay (three seeds) | 0.271 ± 0.007 | 0.419 ± 0.026 | 0.479 ± 0.008 | 0.562 ± 0.013 |
| the same with the fixed relay (seeds 0, 1) | 0.309 | 0.479 | 0.524 | 0.604 |
| soft filters frozen, gradient softmax readout, relay (seeds 0, 1) | 0.431 | 0.635 | 0.673 | 0.737 |
- The word cells on the soft layer end at 0.730 where they end at 0.687 on our cortex; the relay is worth six
  points to them here (0.671 without) and nothing to ridge or streaming LDA, as on our cortex.
- **What this comparison is and is not.** Backprop learns its filters from the stream's 50,000 labelled
  pictures, once; the soft layer developed its filters beforehand on 50,000 other pictures without labels.
  The local learner has seen twice the pictures. That is the setting the paper states (development on
  unlabelled pictures, then the stream), but it has to be said beside the number. The row that tells the two
  apart is backprop *started from* the developed soft filters, which the first queue did not have:
  **`organic-softbase2`** (out/tmp/soft_baselines2_queue.sh → backprop_soft_full2.log; behind softbase1 and
  native4, memory-gated; about six hours): filters developed then learned on, with and without the relay,
  and the frozen filters without relay at the layer's own response (the first queue's row used power 1).
**The replay test's first arm, first checkpoint** (`tagn_s1`: the plateau life of seed 1 lived again from its
level at 90,000 with nothing changed): at 95,000, first words 0.45, pages 0.25, preschool 0.01; the plateau
life itself at that age: 0.40, 0.18, 0.02. Continuing from a level works with today's code, and **two lives
from one level differ by 0.05 to 0.07 after 5,000 moments**: that is the size an arm of this test has to beat.

### 2026-11-25 14:55, the soft layer's references, all three seeds

`out/bench/backprop_soft_full.log` is complete (softbase1's backprop part). Right on the 2,000 test pictures,
mean ± deviation over the three seeds:

| reference, on the soft layer's shape | 500 | 5,000 | 10,000 | 50,000 |
|---|---|---|---|---|
| backprop, filters from random, no relay (power 1, rate 0.001, batch 8) | 0.271 ± 0.007 | 0.419 ± 0.026 | 0.479 ± 0.008 | 0.562 ± 0.013 |
| backprop, filters from random, fixed relay (power 0.7, rate 0.001, batch 32) | 0.304 ± 0.009 | 0.475 ± 0.007 | 0.518 ± 0.011 | 0.606 ± 0.008 |
| soft filters frozen, gradient readout, no relay (power 1: not the layer's own) | 0.292 ± 0.036 | 0.478 ± 0.045 | 0.553 ± 0.012 | 0.619 ± 0.016 |
| soft filters frozen, gradient readout, fixed relay | 0.432 ± 0.001 | 0.635 ± 0.001 | 0.673 ± 0.005 | 0.739 ± 0.003 |

(The two-seed numbers of the last entry, 0.604 and 0.737, become 0.606 and 0.739.) The tuning on the first
10,000 pictures of seed 0 covered twelve settings for each kind of relay: 0.380–0.477 without, 0.446–0.528 with.
The best rate (0.001) is the middle of the three; batch 8 is the smaller of the two tried, without the relay.

- **The reader.** Gradient readout on the frozen soft filters behind the relay: 0.739 ± 0.003; word cells there:
  0.730 ± 0.004. Under the 0.02 we interpret. The same finding as on our cortex (0.694 against 0.687): the
  local reader equals the gradient reader on the same code.
- **Backprop of this shape is below backprop of ours** (0.562 against 0.616 on the whole stream). The soft
  layer's shape has no whitening of the window before the filters; ours has. It fits what the paper says about
  one-pass learning and correlated input; it was not run as a test of it (the two shapes differ in more).
- **In the paper** (docs/paper/DRAFT.md, new section "The same contest on the better layer"): the table of ten
  rows on three seeds, the caveat about the 50,000 unlabelled pictures in the text beside the number, and one
  pending mark naming the three rows still running (the layer as born on the whole stream: softbase1's last
  part, first seed at 0.609 after 20,000; backprop started from the soft filters and the frozen readout at
  power 0.7: softbase2). The abstract is not changed until those rows are in: its headline stays the one on
  our cortex.

**tagn_s1 at 100,000:** first words 0.42, pages 0.23, preschool 0.00 (plat_s1 at that age: 0.46, 0.23, 0.01).
The two lives from one level cross each other from one test to the next; the 0.05–0.07 of the first reading
is the noise of a single test and not a drift apart.

**Not opened:** King, Zylberberg & DeWeese (2013, J. Neurosci. 33:5475), the interneuron form of SAILnet: the
journal refuses automated readers and there is no preprint. It stays on the list for Kris's hand or a library.

### 2026-11-25 16:55, the soft layer as born, on the whole stream

`out/bench/stream_shrfull_s{0,1,2}.log` (softbase1's last part; the word cells' rows on three seeds, the
second-order readers' on two so far, seed 2 running):

| on the layer as born | 500 | 5,000 | 10,000 | 50,000 |
|---|---|---|---|---|
| word cells + relay | 0.354 ± 0.020 | 0.530 ± 0.006 | 0.565 ± 0.011 | 0.630 ± 0.004 (0.633, 0.626, 0.631) |
| word cells, no relay | 0.270 ± 0.009 | 0.386 ± 0.015 | 0.424 ± 0.003 | 0.503 ± 0.010 (0.493, 0.503, 0.513) |
| ridge + relay / ridge (seeds 0, 1) | | | | 0.661 / 0.641 |
| streaming LDA + relay / without (seeds 0, 1) | | | | 0.662 / 0.655 |

- **The like-for-like contest on this layer is the one as born** (no pictures seen by anybody but the relay's
  6,000, which backprop's relay has too). With the relay: word cells 0.630 against backprop 0.606 ± 0.008, two
  points, the edge of what we interpret. Without: 0.503 against 0.562, backprop ahead by six. The order flips
  with the relay, as on our cortex (0.687 / 0.591 against 0.688 / 0.616).
- **Development by the soft rule** is worth 0.100 to the word cells with the relay (0.630 → 0.730) and 0.168
  without (0.503 → 0.671); to ridge about 0.08–0.10. Paid for in 50,000 unlabelled pictures.
- In the paper: two rows added to the second table and a paragraph; the pending mark now names only softbase2's
  rows. **Draft 9 sent to Kris 16:55** (20 pages).

### 2026-11-25 17:45, softbase1 ended; the replay test's first arm ended

**The layer as born, second-order readers, three seeds** (stream_shrfull_s*.log, complete): ridge + relay
0.666 ± 0.008, ridge 0.646 ± 0.007; streaming LDA + relay 0.663 ± 0.003, without 0.657 ± 0.003 (developed:
0.744 / 0.744 / 0.748 / 0.747). The soft rule's development is worth about 0.08 to them (0.10 to the word cells).
On the as-born layer the relay is worth 0.02 to ridge and nothing we interpret to streaming LDA. One sentence
added to the paper's section; no new rows.

**The replay test's first arm, whole** (`tagn_s1`: seed 1's plateau life lived again from its level at 90,000,
nothing changed, to 110,000), beside the plateau life at the same ages (first words / pages / preschool):

| age | tagn_s1 | plat_s1 |
|---|---|---|
| 95,000 | 0.45 / 0.25 / 0.01 | 0.40 / 0.18 / 0.02 |
| 100,000 | 0.42 / 0.23 / 0.00 | 0.46 / 0.23 / 0.01 |
| 105,000 | 0.40 / 0.19 / 0.02 | 0.42 / 0.25 / 0.00 |
| 110,000 | 0.35 / 0.20 / 0.03 | 0.37 / 0.19 / 0.02 |
| mean | 0.405 / 0.218 / 0.015 | 0.413 / 0.213 / 0.013 |

Two lives from one level, same code: at one test they differ by up to 0.07, over the four tests their means
agree to 0.01. **So an arm of the replay test is read on its mean over the four tests,** and with one pair as
the only measure of the noise, a difference under about 0.03 on that mean says nothing. (The babbling stage's
number jumps between 0.03 and 0.41 from test to test in both lives: not a measure to read here.)

### 2026-11-25 18:50, the native-size control on three seeds; SAILnet at native size, said before it runs

**Our cortex on CIFAR at its own size (32), three seeds** (`native_code_s{0,1,2}.log`,
`stream_{nat_r,nat_d}10k_s*.log`; 10,000 pictures, 2,000 tests):

| our cortex at 32 | word cells + relay | word cells | ridge + relay | ridge | near-copy pairs / different filters |
|---|---|---|---|---|---|
| as born (`nat_r`) | 0.626 ± 0.002 | 0.556 ± 0.005 | 0.632 ± 0.003 | 0.633 ± 0.006 | 0 / 256 |
| developed by our hard rule (`nat_d`) | 0.649 ± 0.008 | 0.580 ± 0.021 | 0.666 ± 0.001 | 0.657 ± 0.002 | 100, 111, 96 / 194, 195, 194 |
| developed minus born, by seed | +0.029, +0.013, +0.028 | +0.007, +0.026, +0.039 | +0.034, +0.038, +0.031 | +0.016, +0.031, +0.026 | |

- **Three seeds agree: at the pictures' own size our development helps.** Every one of the twelve differences is
  positive. For ridge behind the relay it is 0.034 and the same on every seed; for the word cells behind the
  relay 0.023, with one seed at 0.013 (the sampling error of one number is 0.010). I had said "within 0.01".
- The paper's sentence "our Hebbian development did nothing for the learner" was measured on pictures blown up to
  twice their size. Whether it still holds there on these same 10,000 pictures is the three `big_r` / `big_d`
  steps now running (about 21:15); the section is reworded when they are in, with both sizes side by side.
- Beside the soft layer at the same size and the same 10,000 pictures (word cells + relay): ours as born 0.626,
  ours developed 0.649, soft as born 0.548, soft developed 0.689. Our cortex is born better (it decorrelates
  each window before the filters; the soft layer does not) and develops less (0.023 against 0.141).
- A question this opens for the lives, not answered: their cortex (`data/visual_cortex2.pkl`) developed on the
  world's pictures at 64, some of them CIFAR-100 blown up and some photographs with real detail at that size.

**SAILnet in our cortex's place, at native size** (the same published layer and numbers as 09:45; only the
pictures at 32 and the base `cortex_nat_r_s0.pkl`; `SIZE=32`, names `sailn` / `sailnu`). The organic check is
the one of 08:40: Földiák's three local rules on spiking cells, every constant the authors', inhibition that
learns its own strength; nothing of ours but the input's units (× √75, said). On blown-up pictures it reached
its rates but ended with 98 different filters of 256 and read 0.484 / 0.588 (as developed / at unit length)
against the hard rule's 0.613. A window at native size holds 47 directions where the blown-up one holds 22.
**Said before:** more different filters than 98 (I expect 150–220); `sailnu` between 0.62 and 0.65, that is
at the level of the layer as born (0.626) and not above our hard rule (0.649). I have been wrong on five of
five such statements in this thread; it is said so that it can be wrong again.

### 2026-11-25 19:50, SAILnet at native size, seed 0: a fully local rule that develops filters worth having

`sailn_develop_s0.log`, `sailn_code_s0.log`, `stream_{sailn,sailnu}10k_s0.log` (10,000 pictures, 2,000 tests;
seed 0; the same base cortex as `nat_r`, only the 256 filters differ):

| our cortex at 32, seed 0 | word cells + relay | word cells | ridge + relay | ridge | near-copy pairs / different filters |
|---|---|---|---|---|---|
| as born (`nat_r`) | 0.627 | 0.551 | 0.633 | 0.639 | 0 / 256 |
| our hard rule (`nat_d`) | 0.656 | 0.558 | 0.667 | 0.655 | 100 / 194 |
| SAILnet, as developed (`sailn`) | 0.633 | 0.522 | 0.658 | 0.642 | 258 / 197 |
| SAILnet, filters at unit length (`sailnu`) | 0.641 | 0.550 | 0.661 | 0.647 | the same |

- **The layer does what its paper says, on this input.** Every cell fires at its target (0.027 / 0.050 / 0.085
  spikes an image: least, mean, most), two cells fire together less often than chance would have it (0.0016
  against the target 0.0025), thresholds and inhibition settle (unchanged between 25,000 and 35,000 batches),
  weights stay near unit length (0.63–1.97). **197 different filters of 256**, where the same layer on pictures
  blown up to 64 ended with 98. On the input the rule was written for, with nothing else changed, the collapse
  is gone.
- **In the stream:** behind the relay, ridge reads 0.661 from these filters, 0.633 from the filters as born and
  0.667 from our hard rule's; the word cells 0.641, 0.627 and 0.656. So on one seed SAILnet's filters are worth
  about what our hard rule's are to ridge (a difference of 0.006), and they sit between born and hard rule for
  the word cells, with both gaps (0.014, 0.015) under what we interpret. Seeds 1 and 2 are running
  (`organic-sailn2`).
- **What I said before:** 150–220 different filters (197), `sailnu` between 0.62 and 0.65 (0.641), not above
  the hard rule (it is not). The first of six such statements in this thread to hold.
- **What this is, if the seeds agree:** every step of this layer is local (a cell's own rate, its own input,
  the firing of the two cells an inhibitory weight joins), nothing is divided by a number from elsewhere in
  the layer, and its constants are its authors'. It would be the fully local filter rule that queue E 27 asked
  for, at our hard rule's level, not at the soft rule's (0.689 on its own pipeline). The hard rule's conscience
  (a count of wins kept per kind) is local too; what SAILnet adds is that the competition itself is learned
  inhibition between cells and not a maximum taken over the layer by nobody.
- **Not read yet:** the layer's own answer (spike counts under its inhibition) in place of our response; here
  only its feed-forward weights stand in our cortex.

**The filters** (`experiments/filters_picture.py`, new: each filter as the raw 5 × 5 colour window it prefers,
W @ Z): docs/img/sailn_filters_s0.png, nat_d_filters_s0.png, nat_r_filters_s0.png. Both developed layers are
oriented black-and-white edges and bars at many angles and colour-opponent edges (blue/orange, green/magenta);
SAILnet's forty or so longest filters are the least clean (mixed colour blobs).

**The enlarged like-for-like rows, seed 0** (`big_r`, `big_d`; the same script at `SIZE=64`): as born 0.631 /
0.647 (word cells + relay / ridge + relay), developed 0.617 / 0.659; 277 near-copy pairs, 167 different
filters. On enlarged pictures development moves the word cells by −0.014 and ridge by +0.012: the paper's
"did nothing", on these 10,000 pictures. Seeds 1–2 running.

### 2026-11-25 20:30, real photographs as the stream (Kris: "should use all type of pictures")

**Kris, 20:20:** "should use all type of pictures honestly why only cifar those pictures are shit we should have
also wiki pics etc". He is right, and today's measurements say why: CIFAR's pictures are 32 pixels, the eye is
64, and an enlarged picture has half the detail a real one has in every window. Our development looked useless
and a published local rule collapsed on that input; both work on pictures at their real size.

**What is on disk, real at 64 pixels:** 95,249 objects cut from COCO photographs, each named with one of 35
words; 3,482 Wikimedia Commons photographs named with one of 16 words; 118,287 whole COCO photographs with no
labels. No download was needed.

**The photo stream** (`experiments/bench_photos.py`; decisions, each said):
- *Classes:* the 33 words with at least 1,000 COCO crops (mouse and sports ball fall out; so does rabbit, which
  only Commons has, 73 pictures). A word's Commons pictures are added to it.
- *Held out:* a twentieth of the COCO photographs with every crop cut from them, and a twentieth of the Commons
  pictures: 4,884 tests (170 from Commons). No photograph gives crops to both sides.
- *Stream:* 50,000 of the remaining 93,000, drawn once (1,747 from Commons), the size of CIFAR-10's, each seen
  once in an order set by the seed. Words are as common as they are in the pool (the commonest is 5.1% of the
  tests, so always saying it would be right 5.1% of the time).
- *Development:* the cortex develops, and the relay settles, on the unlabelled whole photographs, minus the
  2,812 held-out ones (115,475). Whole scenes and cut-out objects are not the same statistics; it is what a
  learner that looks around before it is taught would have.
- *Readers and references:* the same as on CIFAR (`SET=photos` in bench_stream.py, default unchanged). 4,000 of
  the tests are read at each point (sampling error about 0.008). Backprop of the same shape on this stream is
  the next piece (bench_backprop.py needs the same switch); until it has run this is not yet a comparison
  with the current paradigm, only our learner on better pictures.
- *Not in it yet:* CIFAR-100's pictures as the lives see them, a per-source reading (Commons alone is 170
  tests), other sources. Commons is 3.5% of what we have; more Wikimedia pictures would need a slow, polite
  download and are worth it if the first numbers are.

**Said before it runs (seed 0):** a window of a COCO picture at 64 holds 48 directions, as CIFAR at 32 does, so
I expect our development to help here as it does there: developed above born by 0.02–0.04 for ridge behind the
relay and 0.01–0.03 for the word cells. Level: cut-out objects in 33 classes are harder than CIFAR's ten; I
expect the word cells behind the relay between 0.30 and 0.40 after 50,000, ridge a few points above.

### 2026-11-25 21:35, both sizes on the same 10,000 pictures, three seeds (native4 ended)

`big_code_s{0,1,2}.log`, `stream_{big_r,big_d}10k_s*.log` beside the native rows of 18:50 (word cells + relay /
ridge + relay after 10,000 pictures; ± over three seeds):

| our cortex | as born | developed by our hard rule | developed minus born, by seed | near-copy pairs / different filters |
|---|---|---|---|---|
| CIFAR enlarged to 64 | 0.643 ± 0.010 / 0.656 ± 0.008 | 0.621 ± 0.005 / 0.665 ± 0.007 | −0.014, −0.032, −0.019 / +0.012, +0.001, +0.014 | 277, 282, 257 / 167, 172, 173 |
| CIFAR at its own 32 | 0.626 ± 0.002 / 0.632 ± 0.003 | 0.649 ± 0.008 / 0.666 ± 0.001 | +0.029, +0.013, +0.028 / +0.034, +0.038, +0.031 | 100, 111, 96 / 194, 195, 194 |

- **On enlarged pictures development costs the word cells two points** (negative on every seed) and gives ridge
  one; at the pictures' own size it gives two and three. The paper's "did nothing" was kind to the enlarged
  case at this point of the stream (at 50,000 the paper's table has 0.687 against 0.693).
- As born, the enlarged picture is the better input for the word cells (0.643 against 0.626): there are four
  times the windows to average over. Developed, the picture at its own size is better (0.649 against 0.621).
- **Into the paper:** a paragraph and this table in "Our Hebbian development did nothing for the learner"; the
  comparison with the soft layer corrected (0.689 against our 0.649 at the same size, not 0.620); the abstract's
  sentence qualified. 21 pages. Not sent: draft 10 goes when the photo stream has numbers.
- **Not explained:** the paper says 658 near-copy pairs among the benchmark cortex's developed filters; the same
  rule through `cortex_native_probe.py` at 64 ends with 257–282. Same accuracy (0.620 / 0.621), different count:
  the two developments are not the same run (pictures drawn, how long). To look at before the 658 stays in the
  paper.

**SAILnet at native size, seeds 1 and 2 so far:** different filters 201 and 192 (seed 0: 197); word cells +
relay as developed 0.633, 0.625, 0.645 (0.634 ± 0.010); at unit length 0.641, 0.636, seed 2 running. Against
as born on the same seeds, ridge + relay at unit length: +0.028, +0.024; against our hard rule: −0.006, −0.014.

**The photo stream, first log** (`photos_code_ph_r_s0.log`): the cortex as born coded 50,000 + 4,884 photographs
in 35 minutes; its code reads same class +0.0854, different −0.0006 (the as-born cortex on CIFAR at 32: +0.0747
/ −0.0075). `ph_d` is being coded; the streams follow; `organic-softbase2` started at 21:04 when native4 ended.

### 2026-11-25 22:35, SAILnet at native size on three seeds; softbase2's first seed; the photo stream's first rows

**SAILnet in our cortex's place, CIFAR at 32, three seeds** (`stream_{sailn,sailnu}10k_s{0,1,2}.log`; 10,000
pictures; word cells + relay / ridge + relay):

| filters | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| as born (`nat_r`) | 0.627 / 0.633 | 0.627 / 0.628 | 0.624 / 0.634 | 0.626 ± 0.002 / 0.632 ± 0.003 |
| our hard rule (`nat_d`) | 0.656 / 0.667 | 0.640 / 0.666 | 0.652 / 0.665 | 0.649 ± 0.008 / 0.666 ± 0.001 |
| SAILnet at unit length (`sailnu`) | 0.641 / 0.661 | 0.636 / 0.652 | 0.651 / 0.658 | 0.643 ± 0.008 / 0.657 ± 0.005 |
| SAILnet as developed (`sailn`) | 0.633 / 0.658 | 0.625 / 0.647 | 0.645 / — | 0.634 ± 0.010 / — |

Different filters of 256: 197, 201, 192. Against as born, by seed: word cells +0.014, +0.009, +0.027; ridge
+0.028, +0.024, +0.024. Against our hard rule: word cells −0.015, −0.004, −0.001; ridge −0.006, −0.014, −0.007.
**Three seeds agree: a fully local published rule develops filters worth about what our hard rule's are**
(ridge clearly above as born on every seed; under the hard rule by less than we interpret). Queue E 27's
question, "a fully local filter rule", has an answer at the hard rule's level and none at the soft rule's.
Into the paper (a paragraph after the dissection). Still not read: the layer's own spikes as the response.

**softbase2, seed 0** (`backprop_soft_full2.log`; the soft layer's shape, whole stream):

| backprop on the soft shape, seed 0 | 10,000 | 50,000 | beside it (seed 0) |
|---|---|---|---|
| started from the developed soft filters, fixed relay | 0.616 | 0.701 | from random 0.597; filters frozen 0.737; word cells 0.729 |
| started from the developed soft filters, no relay (power 1) | 0.541 | 0.606 | from random 0.547; frozen (power 1) 0.623; word cells 0.675 |
| soft filters frozen, no relay, power 0.7 | 0.511 | 0.616 | at power 1: 0.623 |

One seed, so only as a first look: (1) most of backprop's gap was the filters, that is the unlabelled
pictures: started from the soft filters it goes from 0.597 to 0.701; (2) letting one-pass backprop change
those filters is worse than freezing them (0.701 against 0.737); (3) the word cells (0.729) stay above backprop
started from the same filters; (4) the frozen readout without the relay is the same at the layer's own
exponent (0.616) as at 1 (0.623). Seeds 1 and 2 follow (about 90 minutes a seed).

**The photo stream, seed 0, filters as born** (`stream_photos_ph_r_s0.log`; 33 words, 4,000 tests; the
commonest word is 5.1%): word cells + relay 0.179 / 0.359 / 0.391 / 0.439 after 500 / 5,000 / 10,000 / 50,000;
without the relay 0.134 / 0.272 / 0.308 / 0.379. A picture seen once is named right twenty pictures later
0.893 of the time behind the relay. Backprop of the same shape, first tuning run (rate 0.001, batch 8, no
relay): 0.301 at 10,000. What I said: word cells between 0.30 and 0.40 after 50,000; it is 0.439 as born.

### 2026-11-25 23:35, the photo stream, seed 0: as born whole, developed for the word cells

`stream_photos_ph_r_s0.log` (complete), `stream_photos_ph_d_s0.log` (word cells done, the rest running); 33
words, 4,000 held-out photographs, after 10,000 / 50,000 pictures seen once:

| photo stream, seed 0 | as born (`ph_r`) | developed by our hard rule (`ph_d`) |
|---|---|---|
| word cells + relay | 0.391 / 0.439 | 0.381 / 0.443 |
| word cells, no relay | 0.308 / 0.379 | 0.306 / 0.369 |
| ridge + relay | 0.412 / 0.459 | running |
| ridge | 0.407 / 0.452 | |
| streaming LDA + relay | 0.395 / 0.476 | |
| streaming LDA | 0.419 / 0.475 | |

- **What I said before it ran was wrong for the word cells:** developed above born by 0.01–0.03; it is +0.004
  after 50,000 and −0.010 after 10,000, on one seed. On real photographs at the eye's size our development
  gives the word cells nothing, as on enlarged CIFAR and unlike CIFAR at 32 (+0.023 at 10,000). So "the
  enlargement hid the development's worth" does not carry over as a general statement: the gain at CIFAR's own
  size may belong to CIFAR at 32 and not to "pictures with real detail". Ridge's rows and two more seeds
  before anything is concluded; the paper's new paragraph says only what was measured on CIFAR, and stays.
- Checked, because it would have explained it: the COCO crops are not enlarged (an outline of at least 80
  pixels on its shorter side, the square around it cut at the photograph's resolution and reduced to 64;
  `experiments/world/coco_crops.py`). The developed filters here: 177 different of 256, 180 near-copy pairs
  (CIFAR at 32: 194 and about 100; CIFAR enlarged: about 170 and 270). They develop on whole photographs and
  read cut-out things; that difference is in the design and is a candidate, not a finding.
- The relay is worth six points to the word cells here (0.439 against 0.379), one to ridge, nothing to streaming
  LDA: the same order as on CIFAR. A picture seen once is named right twenty pictures later 0.893 of the time
  (0.843 on the developed filters).
- **Backprop of the same shape, tuning so far** (first 10,000, no relay, batch 8): 0.301 / 0.231 / 0.302 at rates
  0.001 / 0.0003 / 0.003; the word cells without the relay are at 0.308 there, with it at 0.391.

**softbase2, seed 1** (soft shape, whole stream): backprop started from the soft filters with the relay 0.708
(seed 0: 0.701), without relay 0.615 (0.606); frozen without relay at power 0.7: 0.583 (0.616). Two seeds agree
on the first look of 22:35: started from the soft filters backprop ends 0.70, under the same filters frozen
(0.737) and under the word cells (0.729, 0.734).

**Ramp lives at 50,000:** still in first words (0.43 / 0.48 / 0.49; pages 0.15 / 0.13 / 0.13).
`organic-commons`: 1,432 photographs in its first hour.

### 2026-11-26 00:40, the photo stream, seed 0, whole; SAILnet on real photographs, said before it runs

`stream_photos_ph_d_s0.log` is complete. After 50,000 photographs seen once (33 words; 4,000 tests):

| photo stream, seed 0 | as born | developed by our hard rule | difference |
|---|---|---|---|
| word cells + relay | 0.439 | 0.443 | +0.004 |
| word cells, no relay | 0.379 | 0.369 | −0.010 |
| ridge + relay | 0.459 | 0.482 | +0.023 |
| ridge | 0.452 | 0.462 | +0.010 |
| streaming LDA + relay | 0.476 | 0.499 | +0.023 |
| streaming LDA | 0.475 | 0.495 | +0.020 |

- **The entry of 23:35 was half the picture.** On real photographs development is worth two points to the readers
  that keep second moments and nothing to the word cells. That is what the paper's first table already shows on
  enlarged CIFAR (streaming LDA 0.739 against 0.719 with random filters; the word cells 0.687 against 0.693),
  and it is what I had said for ridge here (+0.02–0.04; +0.023) and not for the word cells (+0.01–0.03; +0.004).
  One seed. CIFAR at 32 remains the one place where the word cells gained too.
- So across three kinds of picture the developed filters hold something a first-order reader does not get
  out. What that is (the near copies make the code's directions unequal again after the relay? the gain sits
  in directions with little variance?) is a question about our reader, and a better one than "which size".
- Seeds 1 and 2 queued (`organic-photos2`).

**SAILnet on real photographs** (`organic-sailp1`: `SET=photos` in sailnet_cortex_probe.py, checked against the
kept codes: the as-born cortex's code of a stream picture is reproduced exactly; the layer develops on the
unlabelled photographs, 64 pixels, nothing enlarged; base `cortex_ph_r_s0.pkl`; `sailp` / `sailpu`). **Said
before:** it keeps its filters apart as on CIFAR at 32 (at least 180 different of 256; on enlarged CIFAR 98);
at unit length ridge and streaming LDA behind the relay read about 0.02 above as born (0.459, 0.476), the
word cells within 0.01 of as born (0.439).

**The photo daemon is back** (00:02 new day; `inclusionai/ling-3.2-flash:free` in the withdrawn adult's place):
20 questions written, 15 judged right, 13 requests; two batches got no reply after retries. The health alert
for it is gone.

### 2026-11-26 01:35, softbase2 complete: backprop handed the soft filters; SAILnet's filters on photographs

**`backprop_soft_full2.log`, three seeds** (the soft layer's shape, whole stream; 500 / 5,000 / 10,000 / 50,000):

| reference | 500 | 5,000 | 10,000 | 50,000 |
|---|---|---|---|---|
| backprop started from the soft filters, fixed relay | 0.276 ± 0.013 | 0.544 ± 0.003 | 0.609 ± 0.007 | 0.703 ± 0.004 (0.701, 0.708, 0.701) |
| backprop started from the soft filters, no relay (power 1) | 0.297 ± 0.045 | 0.471 ± 0.049 | 0.551 ± 0.009 | 0.620 ± 0.017 (0.606, 0.615, 0.638) |
| soft filters frozen, gradient readout, no relay, power 0.7 | 0.270 ± 0.026 | 0.429 ± 0.057 | 0.525 ± 0.016 | 0.607 ± 0.021 (0.616, 0.583, 0.621) |

- **Better filters or more pictures: mostly the filters.** From a random start backprop behind the relay ends at
  0.606; handed the soft filters, at 0.703. The word cells on those filters: 0.730 ± 0.004, ahead on every seed
  (0.729 / 0.701, 0.734 / 0.708, 0.727 / 0.701), by 0.027.
- **One pass of backprop through good filters makes them worse than leaving them:** 0.703 against 0.739 ± 0.003
  with the filters frozen and only the readout trained. Without the relay: 0.620 learned on, 0.607 frozen
  (power 0.7), the word cells 0.671.
- Not done for this row: its settings are those tuned from a random start; its relay is settled on the soft
  filters and not again as they move.
- **Into the paper:** the second table is complete (14 rows, no pending mark), the reading rewritten around the
  two rows that separate filters from pictures, the abstract has the sentence (73% / 56% / 70%). 21 pages.

**SAILnet on real photographs, its development** (`sailp_develop_s0.log`): rates on target (0.030 / 0.050 /
0.091), two cells together 0.0021 (target 0.0025), **145 different filters of 256** (1,712 near-copy pairs),
weights 0.65–5.68 long. I had said at least 180. Between CIFAR at 32 (197) and enlarged CIFAR (98); the
"directions in a window" count (48 for these photographs, as for CIFAR at 32) does not predict it. The codes
and streams follow.

**Backprop on the photo stream, tuning** (first 10,000 of seed 0; no relay): batch 8: 0.301 / 0.231 / 0.302 at
rates 0.001 / 0.0003 / 0.003; batch 32: 0.267 / 0.165 / 0.316. Fixed relay, first run (0.001, batch 8): 0.403.
The word cells at that point: 0.308 without the relay, 0.391 with it. Five runs to go.

### 2026-11-26 08:00, the photo stream on three seeds; SAILnet on photographs; a hypothesis about the word cells

(The loop was idle from 02:28 to 07:37: the session's model allowance ran out; Kris switched the model and the loop
went on. Every queue ran meanwhile.)

**The photo stream, three seeds** (`stream_photos_{ph_r,ph_d}_s{0,1,2}.log`; streaming LDA on the developed code has
two seeds, the third running; 33 words; 4,000 held-out photographs; after 50,000):

| photo stream | as born | developed by our hard rule | developed minus born, by seed |
|---|---|---|---|
| word cells + relay | 0.446 ± 0.006 | 0.445 ± 0.002 | +0.004, −0.004, −0.004 |
| word cells, no relay | 0.389 ± 0.009 | 0.373 ± 0.004 | −0.010, −0.020, −0.017 |
| ridge + relay | 0.461 ± 0.004 | 0.486 ± 0.003 | +0.023, +0.029, +0.022 |
| ridge, no relay | 0.457 ± 0.006 | 0.467 ± 0.004 | +0.010, +0.013, +0.006 |
| streaming LDA + relay | 0.476 ± 0.007 | 0.501 (0.499, 0.503) | +0.023, +0.034 |
| streaming LDA, no relay | 0.476 ± 0.005 | 0.497 (0.495, 0.499) | +0.020, +0.027 |

Three seeds agree with seed 0: on real photographs development is worth 2.5 points to ridge and streaming LDA behind
the relay and nothing to the word cells (without the relay it costs them 1.6). A photograph seen once is named
right twenty pictures later 0.89–0.93 of the time on the filters as born, 0.84–0.89 on the developed ones.

**Backprop of the same shape on photographs, seed 0** (`backprop_photos_full.log`): without the relay, at the best
of twelve settings on the first 10,000 (rate 0.003, batch 32: 0.316 there), 0.387 after 50,000; the word cells
without the relay 0.389 ± 0.009, with it 0.446 ± 0.006. Level, like for like without the relay; with the relay
the run is in work. **The best setting was the largest rate and the largest batch of the search, an edge**, so
the search goes on past it (`organic-photos-bp2`: rate 0.01 / batch 128 without the relay, batch 128 with it),
as the CIFAR reference's search went on past its smallest rate; the whole stream is run again if a wider setting wins.

**SAILnet on photographs, seed 0** (`sailp_code_s0.log`, `stream_photos_{sailp,sailpu}_s0.log`): 145 different
filters of 256. At unit length, after 50,000: word cells + relay 0.427, without 0.350; ridge + relay 0.474;
streaming LDA + relay 0.498 (as born on that seed 0.439 / 0.379 / 0.459 / 0.476; our hard rule 0.443 / 0.369 /
0.482 / 0.499). What I said: at least 180 different filters (wrong: 145); ridge and streaming LDA about 0.02
above as born (+0.015, +0.022: held); word cells within 0.01 of as born (−0.012: just outside). As developed (not
at unit length) it is worse everywhere (0.386 word cells + relay). The same pattern as our rule's: development
helps the second-order readers and not ours.

**A hypothesis, from five codes** (developed minus born, behind the relay; and how many directions the developed
code spreads over after the relay, the participation ratio read on 1,500 test pictures):

| code | directions after the relay, as born → developed | word cells + relay | ridge + relay |
|---|---|---|---|
| CIFAR at 32, our hard rule | ~790 → ~535 | +0.023 | +0.034 |
| CIFAR at 32, SAILnet at unit length | ~790 → ~497 | +0.017 | +0.025 |
| photographs, our hard rule | ~603 → ~393 | −0.001 | +0.025 |
| CIFAR enlarged, our hard rule | ~575 → ~354 | −0.022 | +0.009 |
| photographs, SAILnet at unit length (one seed) | 585 → 307 | −0.012 | +0.015 |

Ridge gains every time. The word cells gain only where the developed code still spreads over about 500 directions
after the relay, and the order of the five follows the count but for one pair. This is what the paper's account
of the relay predicts for a one-pass first-order reader: it learns along the strong directions and hardly along
the weak, and a developed code (its near copies) puts its variance into fewer directions than the relay, settled
on 6,000 pictures for 4,096 numbers, takes back out. It is a reading of five points, not a finding. **The test,
queued** (`organic-relaymore1`, `experiments/relay_more_probe.py`): the same codes of seed 0 with the relay settled
on 20,000 unlabelled photographs in place of 6,000, for the developed and the as-born cortex; nothing else changes.
**Said before:** if the hypothesis holds, the developed code spreads over clearly more directions after the new
relay and the word cells gain from development (developed above born by 0.01 or more); if neither moves, the
relay's estimate is not the reason and the next candidate is the reader itself (a first-order rule seeing each
picture once).

**The photo daemon** wasted most of its day: since midnight 6 batches written and 27 failed ("no reply after
retries", with no reason given), and since every attempt counts against the allowance, the day's share was gone at
04:02. Which adult fails, and why, was not in the log: `organic/adults.py` now says the model and the reason (the
HTTP code, or a timeout and after how long) in the failure, and annotate.py names the writer and the checker. A
logging change only; the photo daemon restarted (it sleeps until midnight); the text and curriculum daemons pick it
up when they next restart. Tonight's log will say which adult to look at.

**Ramp lives at 80,000** (first words / pages): 0.42 / 0.30, 0.35 / 0.25, 0.41 / 0.28. Reward lives at that age
0.45 / 0.33, 0.49 / 0.20, 0.48 / 0.28; plateau lives 0.53 / 0.14, 0.47 / 0.32, 0.49 / 0.33. No visible effect so
far; read whole at 120,000. `organic-commons`: 11,807 photographs in nine hours.

### 2026-11-26 09:10, backprop on photographs; the relay's sample size ruled out; the relay's floor as the next candidate

**Backprop of the same shape on photographs, seed 0** (`backprop_photos_full.log`): with the fixed relay (rate 0.001,
batch 32, the best of six with the relay on the first 10,000) **0.462** after 50,000 (0.146 / 0.344 / 0.395 at
500 / 5,000 / 10,000). The word cells behind the relay, filters as born: 0.446 ± 0.006 (seed 0: 0.439); ridge +
relay 0.461. So on photographs backprop with the relay is ahead of the word cells by two points at the end (one
seed) and behind early (500: 0.146 against 0.170; 5,000: 0.344 against 0.363). Without the relay the two are level
(0.387 against 0.389). The widened search (`organic-photos-bp2`): rate 0.01 / batch 32 → 0.262, rate 0.003 / batch
128 → 0.221 (both under the chosen 0.316); three runs to go.

**The photo stream is complete on three seeds** (seed 2's streaming LDA in): developed, streaming LDA + relay 0.499
± 0.004, without 0.495 ± 0.005.

**The relay on 20,000 photographs in place of 6,000** (`relay_more_ph_d_s0.log`, `stream_photos_ph_dm20k_s0.log`):
the developed code spreads over 396 directions after it (386 with 6,000), and the word cells read 0.441 (0.443
with the relay of 6,000). **What I said: if neither moves, the relay's estimate is not the reason. Neither moved.**
The number of pictures the relay settles on is ruled out.

**The spectrum of the codes themselves** (`experiments/spectrum_probe.py`, `spectrum_s0.log`; 4,000 codes of each,
seed 0). The relay levels a direction only if its variance is not far below 0.0005 of the largest; the
participation ratio of the levelled spectrum says over how many directions the code is spread once the relay has
done what it can:

| code | spread after the relay, as born → developed | directions above the relay's floor | word cells + relay, developed minus born |
|---|---|---|---|
| CIFAR at 32 | 1,597 → 917 | 594 → 423 | +0.023 |
| photographs | 1,103 → 593 | 360 → 280 | −0.001 |
| CIFAR enlarged to 64 | 933 → 518 | 323 → 238 | −0.022 |

Development roughly halves the spread in all three (×0.57, ×0.54, ×0.56): the developed filters put more of the
code's variance into its largest directions (the top ten hold 0.50 against 0.46, 0.58 against 0.57, 0.62 against
0.59). The word cells gain from development where the developed code still spreads over about 900 directions, and
not where it spreads over 500–600. Three points, one seed: a reading. Its mechanism is the relay's floor, which is
set relative to the largest variance: a developed code, whose largest direction is larger, has more of its
directions left under the floor. **The test, queued** (`organic-relayshrink1`): the same relay with a floor five
times lower (0.0001), for the developed and the as-born photo cortex. **Said before:** if the floor is what hides
the development from the word cells, the developed photo code spreads over clearly more directions with it and the
word cells gain from development (developed above born by 0.01 or more); ridge should change little. If the word
cells still gain nothing, the floor is not the reason and the reader itself is (the delta rule's step on
thousands of nearly equal directions). Either way this is a diagnosis, not a setting to adopt: the floor is a
hand-set constant, and if it matters, the answer is a relay that sets its own (the local circuit of queue E 28),
not a better number.

### 2026-11-26 09:55, a cochlea; the relay's sample size ruled out on both codes; the floor's prediction

**A cochlea** (queue E 32; `organic/cochlea.py`, a new organ not used by any life yet). The organic check: before any
learning, the inner ear splits sound by frequency along the basilar membrane, with bands that widen with frequency
(equivalent rectangular bandwidths, Glasberg & Moore 1990) and a compressive loudness (outer hair cells; Stevens'
exponent). So: 64 bands evenly spaced on the ERB-rate scale from 80 Hz to 7,600 Hz, a frame every 10 ms from 25 ms
of sound, each band's energy to the power 0.3. A fixed front end, as the retina's whitening is. Checked: tones of
200 / 1,000 / 4,000 Hz land in the bands centred at 187 / 1,014 / 4,081 Hz; silence gives 0.
**The videos' sound** (`experiments/world/video_audio.py`, service `organic-video-audio`): the same 120 seconds of
each kept Commons video, through the cochlea, ten sound frames under every picture frame →
`~/datasets/commons_video/audio_64.f16` and `audio.jsonl` (videos without sound marked). Of the first five raw
files, three had a sound track. This is the data for the first test of senses teaching each other.

**The relay on 20,000 pictures, the as-born code too:** 597 directions after it (585 with 6,000), word cells 0.439
(0.439). Sample size ruled out on both codes.

**The relay's floor, predicted by the spectrum before the run** (`spectrum_shrink1e-4_s0.log`): with the floor at
0.0001 of the largest variance, the developed photo code spreads over 1,000 directions after the relay (593 at the
benchmark's 0.0005), more than the developed CIFAR-at-32 code does at 0.0005 (917), where the word cells gained.
So if the reading of 09:10 is right, `organic-relayshrink1` should show the word cells gaining from development on
photographs.

**Backprop's widened search on photographs** (5 of 6 runs): every wider setting is worse than the chosen ones
(no relay: 0.262, 0.221, 0.204 against 0.316; relay, batch 128: 0.379 against 0.406). The chosen settings stand.

### 2026-11-26 11:10, the relay's floor (first half); two senses teaching each other, said before it runs

**The relay's floor, the developed photo code** (`organic-relayshrink1`, seed 0; relay settled on 20,000 photographs
at 0.0001 of the largest variance instead of 0.0005): the code spreads over 572 directions after it (386 at the
benchmark's relay; the spectrum had predicted about 1,000, so the levelled spectrum overstates the spread of the
codes themselves). Word cells + relay **0.452** against 0.443 (0.441 with 20,000 at the old floor); ridge + relay
0.481 against 0.482. So the word cells move by about a point and ridge does not move. Whether that makes a gain
from development waits for the as-born half (ph_rz1, running): as born at the old floor, word cells 0.439. One seed.
**The relay on 20,000 pictures, as born:** ridge + relay 0.460 against 0.459. Nothing moves, on both codes.
**Backprop's widened search on photographs, complete:** the last run (relay, rate 0.003, batch 128: 0.401) is also
below the chosen one (0.406). The chosen settings stand; seeds 1 and 2 at them are queued (`organic-photos-bp3`).

**Two senses teaching each other: the first probe** (queue E 32; `experiments/av_probe.py`). Kris's hypothesis is
that streams interconnected make a more reliable brain. Before any brain hears real sound, the question this can
answer on its own: does what the eye sees in a video carry what the ear hears at that moment, on videos never seen,
and can the senses learn that from each other by the local rules the brain already has?

The organic check. (1) Multisensory association is learned from what occurs together: neurons of the superior
colliculus in cats reared in the dark, or with sight and sound paired at random, do not come to integrate the two,
and come to integrate whatever pairing they were reared with (Wallace & Stein 2007; Xu et al. 2012; to verify).
(2) The two senses can be each other's teacher without labels: de Sa (1994), "learning classification with
unlabeled data", let each modality's units label the other's. (3) The hearing uses the eye's own rule:
winner-take-all Hebbian learning with a conscience on windows of the cochlea's output read as a picture (80 ms ×
8 bands), with its decorrelation; efficient codes of natural sounds give filters like the auditory system's (Smith
& Lewicki 2006; Carlson, Ming & DeWeese 2012). (4) The association is Hebb's rule (the running mean of the product
of the two codes) and the word cells' delta rule; after the relays decorrelate both codes, Hebb's rule comes close
to regression, which is one more reason the relay is there. Least squares on everything at once is the
reference. What the paradigm does here is learn both encoders by backprop on "do these belong together"
(Arandjelović & Zisserman 2017, "Look, Listen and Learn"); that reference is for later.

The material: 597 of the first 768 Commons videos have sound; titles naming an animation, a render, a time-lapse,
a game, an interview or a lecture are left out (47 of 1,356 kept videos have such titles; a rough count, the
deep categories hold more talking heads). A fifth of the videos is held out whole. One frame every two seconds and
the second of sound around it.

**What I expect, before it runs.** Which sound (the true second against another held-out video's, chance 0.5),
from sight: Hebb about 0.70 (0.62–0.80), the delta rule within 0.03 of Hebb, least squares at most 0.05 above.
When (the same video's sound 5 s or more away, chance 0.5): about 0.55; Commons sound is mostly steady within a
video (music, wind, an engine). Developed hearing above hearing as born by 0.02 or more; the cochlea's band means
alone below both. What (the word a video was fetched under, nearest class mean, balanced): sight above hearing;
sight and hearing together above sight by a point or more; the sound that sight predicts below sight. If "which
sound" sits near 0.5, Commons videos are the wrong material (music laid over pictures) and CORe50 has no sound:
then the next material is filmed sound with its picture, not a different rule.

### 2026-11-26 11:40, the relay's floor is not why the word cells miss development's worth

`organic-relayshrink1` complete (seed 0; relay settled on 20,000 photographs; photo stream, 50,000 pictures seen once,
4,000 tests):

| photo code, seed 0 | directions after the relay | word cells + relay | ridge + relay | seen once, 20 later |
|---|---|---|---|---|
| as born, floor 0.0005 (the benchmark) | 585 | 0.439 | 0.459 | 0.893 |
| developed, floor 0.0005 | 386 | 0.443 | 0.482 | 0.843 |
| as born, floor 0.0001 | 816 | 0.448 | 0.460 | 0.917 |
| developed, floor 0.0001 | 572 | 0.452 | 0.481 | 0.863 |

A lower floor lifts the word cells by about a point on both codes alike, and ridge not at all; the gain from
development stays where it was for both readers (word cells +0.004, ridge +0.021). **What I said before was wrong:**
I expected the word cells to gain a point or more from development once the developed code spread like CIFAR's
did. It now spreads over more directions (572) than the as-born code did at the old floor (585), and the word cells
still take nothing from development. So the spread after the relay is not the reason; the reading of 08:00 and 09:10
(the word cells gain only where the developed code stays spread) fits five codes and fails the sixth. One seed, but
the difference it was looking for is not there.

What is left is the reader. Ridge and streaming LDA gain two points from development on photographs and the word
cells gain nothing; on CIFAR at its own size all three gained. Ridge and LDA use the second moments of the code;
the word cells learn by the delta rule with a step that shrinks with use. The clean next step costs nothing new:
the nearest class mean through the relay (Hebbian running means of each word's code, no error, no second moments),
on both codes and three seeds (`organic-ncm1`, queued after the census). If it gains from development like ridge,
the delta rule's step is why the word cells do not; if it gains nothing too, what development adds on photographs
lives in the code's covariance, which only readers with second moments use.

A side reading worth keeping: the lower floor is better for the word cells on both codes (+0.009) and for how well
they keep a picture seen once (0.917 against 0.893 as born). The floor is a hand-set constant; a relay that sets its
own floor (the local circuit of queue E 28) is where that point belongs, not a new number.

### 2026-11-26 12:50, an ear that hears sound: the plan (Kris: no transcription, the ear develops its own cortex)

Kris, 12:50: "for audio what i want is not to transcribe to text then work on text similar to how words are just
bytes audio can be fed in the ear develop it's own cortex with interconnectivity and just work". The organic check
is easy here: it is how hearing works and how infants come to words. The cochlea splits sound by frequency before
any learning (organic/cochlea.py); auditory cortex develops its receptive fields from the sounds an animal grows up
with (rat pups reared with one tone over-represent it in A1: Zhang, Bao & Merzenich 2001, to verify); infants find
word forms in continuous speech by its statistics (Saffran, Aslin & Newport 1996) and come to their language's sound
categories by the distribution of what they hear (Maye, Werker & Gerken 2002, to verify). Nothing in that chain is
text. The same three steps, with rules the brain already has:

1. **Hearing cortex**: the eye's layer on the cochlea's output read as a picture (80 ms × 8 bands windows,
   decorrelated, winner-take-all Hebbian with a conscience), developed on unlabelled speech; then its own relay.
   Running already on Commons video sound (`experiments/av_probe.py`).
2. **Units from sound**: the byte ear's principle on the hearing cortex's stream: a predictor of the next 10 ms from
   the last few (a delta rule); a unit starts where its error jumps; each stretch's code goes to the unit that
   matches it, or makes a new unit when none matches well enough (as the gate's clusters are born). The ear's
   lexicon grows by use.
3. **Grounding**: the units heard in a spoken caption are the word cells' cues and the picture is their content, as
   today; the store and the gate unchanged.

Material: SpokenCOCO (people reading the COCO captions aloud; CC BY-SA; fetching, docs/WORLD.md), heard with the COCO
photographs we hold. The captions' text is used only to measure: which sound unit comes with which word, and how
well a caption is found from its picture and back. The paradigm's answer to the same question is visually grounded
speech learned by backpropagation (Harwath, Torralba & Glass 2016; Harwath et al. 2018: retrieval of the right
spoken caption among 1,000 for a picture), our reference row.

**What I expect, before building:** units found in sound will mostly be syllable-sized pieces, not words; a few
frequent words ("a", "of", "the", and object words said often, "dog", "pizza") will each have a unit that comes with
them in most captions that contain them; picture-to-caption retrieval by local rules will sit well under backprop's
numbers. The point is that the ear gets there without ever being given a letter.

### 2026-11-26 13:20, the reader's verdict, first seed; real video for the eye, said before it runs

**Nearest class mean through the relay on the photo stream, seed 0** (`organic-ncm1`): as born 0.419, developed
**0.401**. Development costs the purest Hebbian reader two points, while ridge and streaming LDA gain two (0.459 →
0.482, 0.476 → 0.499) and the word cells sit between (0.439 → 0.443). Of the two readings I wrote at 11:40 this is
the second: what development adds on photographs is not in the words' mean codes (those come closer together)
but in the code's second moments, which only readers that use them can take. The word cells' delta rule uses them
partly (a delta rule tends to least squares, slowly, one picture at a time), which fits their place between. Seeds 1
and 2 running. What follows if they agree: the reader, not the cortex and not the relay, is where the photo stream's
missing two points are; the candidates are a delta rule that keeps its step longer on the faint directions, or a
local reader of second moments (the streaming discriminant's form, as dendritic or interneuron circuits), each to
pass its organic check first.

**Real video for the eye: CORe50 in order against shuffled** (queue E 31 (b); `experiments/core50_trace_probe.py`).
CORe50 is downloaded (5.9 GB, CC BY 4.0) and being turned into 64 × 64 frames in order
(`experiments/world/core50_frames.py`). The probe is the one written into the queue on 2026-11-26 08:05: the same
frames developed in their order and shuffled, time the only difference. The first stage is the photo cortex as it
is; a second stage (the one of 2026-11-24's drift probe) develops on 4,000 fixations of four frames 100 ms apart from
the eight training sessions, in three arms: in order with the trace, shuffled with the trace, in order without it.
Read on the three held-out sessions (other backgrounds, other light): the nearest frame from another session, is it
the same object (chance 0.02) and the same kind (0.10); and the stream of 10 kinds (word cells, ridge, through
relays settled on COCO, never on CORe50).
**What I expect.** Same object elsewhere: the trace in order above the same stage without it by 0.02 or more, and
above the shuffled trace by more (the shuffled trace binds unrelated frames, so it should be the worst of the
three); the first stage alone between. The stream of kinds: within 0.01 for all; the drift probe's cost of the trace
(0.015 on stills) should not appear on video, where tolerance is what the test asks for. If the order makes no
difference, the trace at this level (3 × 3 places of the first stage) is too local for what video teaches, and the
next place for it is a stage that sees the whole view (as IT does).

### 2026-11-26 14:45, an ear that hears sound: the first design failed its smoke test; three levels instead

Smoke test of `experiments/sound_units_probe.py` as first written (600 SpokenCOCO recordings): units cut at peaks
of a next-frame predictor's error came 1.9 to a syllable (the "largest within 50 ms" rule fires at the window's own
pace, not the speech's), and a lexicon that founds a new unit below a vigilance filled its 2,000 places within 480
utterances at every vigilance tried (0.65, 0.75, 0.85): every stretch was new, nothing recurred. Two of my own
choices, two failures; rewritten closer to biology and to the byte ear, each level the rule one level down:
- **syllables by rhythm**: auditory cortex follows the slow loudness rhythm of speech, 4 to 8 a second (Giraud &
  Poeppel 2012, to verify); a stretch ends at a trough of the loudness and at silence;
- **syllable kinds by the eye's own rule**: 512 kinds, winner-take-all with a conscience, the step falling with a
  kind's wins (no vigilance);
- **words by the byte ear's rule**: counts of which kind follows the last one or two; a word starts where the next
  kind's uncertainty jumps up, and at silence (Saffran, Aslin & Newport 1996).
Smoke test of the new design (600 recordings, 64 kinds): **0.89 stretches a syllable**, words found 0.63 a written
word, two stretches long on average: the scales are right; too few held-out utterances to read word-likeness.
Running: the whole validation part (25,035 recordings: 20,028 to learn from, 5,007 held out; 512 kinds) →
out/bench/sound_units_val_s0.log. What I expect is as written at 12:50: syllable-sized kinds, some of them coming
with one frequent word ("a", "of", and object words said often); word-like words found well above the shuffled
count but most found "words" not words.

### 2026-11-26 15:45, the stream brain runs; two senses on 1,399 videos (seed 0, which sound)

**Kris, 15:15: "every neuron ticks and send info ... stream in stream out"** (docs/ASYNC.md). The first stream brain
(`organic/stream.py`): a 10 ms clock; populations that follow their input with their own time constants and pass on
only what changed; outputs as races to a threshold (the strongest drive crosses in about 1 / rate seconds, the rate
taken from what the act is known to do; noise lets a weaker one win). The eye: nine places of the current frame,
their centre-surround salience as drive, inhibition of return fading over a second, 30 ms blind in a saccade, the
fixated half of the frame through the photo cortex. Hearing: the cochlea's rows every 10 ms through the eye's own
layer developed on other videos' sound. A babbling mouth (habits, nothing linking it to the ear yet). One plastic
link: sight and sound by Hebb's rule while both stream.
**First run** (`experiments/stream_probe.py`, 3 Commons videos, 85 simulated seconds): 3.31 saccades a second,
fixations median 290 ms (quartiles 250–340; people's are about 200–400 and more skewed); 0.54 of saccades away from
the 50 ms after a new frame (chance 0.50: the eye moves by its own race, not on the world's clock); 2.21 sounds a
second, 0.53 of them while the world was loud (chance 0.50: the mouth ignores the world, as built); CPU 6.5 s a
simulated second on a crowded machine, most of it the sight–sound link updated every 10 ms. The rhythm's mean is
set (the fixation rate); its spread and its independence from the frames are the mechanics. Nothing learned yet but
the link. Next for it: the link updated at its own pace (cheaper), sound and sight driving the eye's race (a sound
pulls the eye: the audio-visual orienting of the colliculus), the mouth linked to the ear (babbling heard), then the
word cells as a population.

**Two senses, seed 0, which sound** (`experiments/av_probe.py`; 1,399 videos with sound, a fifth held out whole;
42,938 moments): the true second against another video's (chance 0.5):

| hearing | sight → sound: Hebb / delta / least squares | sound → sight: Hebb / delta / least squares |
|---|---|---|
| as born (random kinds) | 0.611 / 0.530 / 0.586 | 0.626 / 0.534 / 0.600 |
| developed (the eye's rule on sound) | 0.622 / 0.527 / 0.584 | 0.631 / 0.532 / 0.619 |
| the cochlea's band means alone | 0.600 / 0.527 / 0.579 | 0.607 / 0.534 / 0.591 |

Against the same video 5 s away: 0.50–0.51 everywhere (chance). **What I said before: Hebb about 0.70; delta within
0.03 of Hebb; least squares at most 0.05 above; developed above born by 0.02.** Measured: Hebb 0.61–0.63, below; the
delta rule far under Hebb (0.53: a step that falls with use, on 34,000 moments of two 4,096- and 1,024-number codes,
hardly moves); least squares *under* Hebb (the relays decorrelate both codes, so Hebb is already close to regression,
and regression overfits); developed hearing one point above born, born one point above the cochlea alone. So: sight and
sound on Commons video share something real but small, Hebb's rule finds it best, and the hearing's development adds
a little. One seed; seeds 1–2 queued; the word test of this seed still running.

### 2026-11-26 16:30, the ear that hears sound, first reading: the right scale, nothing recurs

`experiments/sound_units_probe.py` on SpokenCOCO's validation part (20,028 recordings heard, 5,007 held out, 1,793
speakers; 512 syllable kinds): **0.90 stretches a syllable**, words found 0.65 a written word, two stretches long on
average; but **18,877 different "words" in 5,007 utterances**: nearly every one found once. No syllable kind and no
word found comes with one written word in half its utterances (0 word-like, shuffled 0). What I said at 12:50:
syllable-sized pieces, yes; a few frequent words with their own unit, no.
**Why, as far as I can tell:** the same syllable said by different people lands in different kinds, so sequences of
kinds hardly ever repeat, and the byte ear's rule needs repetition. Two changes, each on biological ground:
1. **The ear adapts to the speaker.** Listeners hear a vowel relative to the voice it comes in (Ladefoged & Broadbent
   1957, to verify); auditory neurons adapt their range to the recent statistics of the sound (Dean, Harper & McAlpine
   2005, to verify). Each band of the cochlea is taken relative to its own running level over the utterance (its mean
   and spread), before the hearing cortex.
2. **Words by the eye's rule, not by exact sequences.** Kinds of two- and three-stretch windows (the codes of
   neighbouring syllables side by side), winner-take-all with a conscience, as the syllable kinds are; a word is a
   kind of such windows. Repetition is then likeness, not identity.
And a softer reading beside the strict one: for each kind, the written word it lifts most (how much more often that
word is in its utterances than in all), against the same with the kinds shuffled.

**Two senses, seed 0, the word test** (`av_probe_s0_what.log`, from the cached codes after the OOM): videos fetched
under 15 words, 114 held-out videos (cat 47, tree 12, dog 10, train 10, the others 1–7), 2,281 moments; the word by
the nearest class mean, balanced over words (chance 0.067): **sight 0.239; hearing developed 0.201, as born 0.191, the
cochlea alone 0.164; sight and hearing together 0.264; the sound that sight predicts 0.164.** What I said before:
sight above hearing (yes), both together above sight by a point or more (yes, 2.5), the sound sight predicts below
sight (yes). The order developed > as born > cochlea holds here too. The first hint in this project for Kris's view
that interconnected streams make a more reliable brain: two senses name a video better than either. One seed and few
videos a word; seeds 1 and 2 running (`organic-av2`).

**The reader's verdict, three seeds** (`organic-ncm1` complete): nearest class mean through the relay on the photo
stream, as born 0.419 / 0.422 / 0.421 (0.421 ± 0.002), developed 0.401 / 0.401 / 0.407 (0.403 ± 0.003). Development
costs the purest first-order reader two points on every seed, where ridge and LDA gain two and the word cells
nothing. Written into the paper's photo section with the two explanations ruled out (the relay's sample; its floor).
What follows for the learner (not done yet; each needs its organic check): a reader of second moments that is local,
or a delta rule that keeps learning along the faint directions longer.

### 2026-11-26 17:05, the stream brain babbles with a vocal tract and begins to imitate

`organic/vocal.py`: a source-filter vocal tract (glottal pulses at the voice's pitch; F1, F2 and a fixed F3 as
two-pole resonators in cascade, as in Klatt's 1980 synthesizer, much simplified; the lips' radiation). Checked: /i/
above /u/ at 2,265 Hz, /u/ above /i/ at 790–841 Hz (their second formants), 0.4 ms a vowel. In the stream brain
(`organic/stream.py`, switches `tract`, `habit_step`): each of 32 motor units is a gesture, a random place in the vowel
space; when the mouth's race fires, the gesture's sound joins what the ear hears; Hebb's rule keeps, for each gesture,
the mean hearing code while it sounded (the auditory–motor map; DIVA: Guenther, Ghosh & Tourville 2006); what is heard
drives the gestures whose learned sound matches it above the average gesture (covert imitation). Hearing: the eye's
rule on speech (SpokenCOCO), 8 band stretches. `experiments/babble_probe.py`, 300 s of babbling in silence (about two a
second, all 32 gestures heard), then 64 vowels at random places of the vowel space:

| | same voice (280 Hz, the child's) | another voice (140 Hz, an adult's) |
|---|---|---|
| the map, given a vowel at each gesture's place: picks that gesture | 29 of 32 (rank 1.1) | 13 of 32 (rank 2.4: its neighbours) |
| the next sound made, among the three gestures nearest the heard vowel | **0.22** (imitation off 0.06; chance 0.09) | **0.22** (off 0.06) |
| its rank among 32 by nearness | 14.1 (off 16.7; chance 16.5) | 11.7 (off 16.7) |

Two failures on the way, both mine: habit that grows with every gesture made let 14 of 32 gestures take all the
babbling (so the map covered half the space); and imitation taken relative to the best match alone was flat, since
every gesture's learned sound is much alike: taken above the average gesture (the filters' own response) it works.
**Reading:** babbling teaches the mouth what its gestures sound like, well within its own voice and partly across
another; a heard vowel triples the chance that the next sound is a near imitation. The bottleneck is now the race:
the imitation drive lasts as long as the hearing population's 50 ms, and a gesture already half-way to threshold
often wins first. Next: the drive from a slower trace of what was heard (echoic memory lasts seconds; the copy route
of organic/speech.py does this for text), and habits that come from what is heard (covert imitation), not from the
mouth's own repetition. One seed.

**Imitation, three seeds, and the echo** (`out/bench/babble_seeds.log` and the seed-0 runs above; 64 heard vowels a
seed and voice, both voices pooled):

| | imitation on | off |
|---|---|---|
| plain (the 50 ms hearing population): mean rank of the next sound among 32 (chance 16.5) | **13.1** | 16.3 |
| plain: the next sound among the three nearest gestures (chance 0.09) | **0.18** | 0.09 |
| with an echo (the last whole stretch heard, fading over 1 s): mean rank | 14.4 | 16.9 |
| with the echo: among the three nearest | 0.14 | 0.05 |

The map, every seed: 27–30 of 32 gestures recognised by their sound in the child's own voice, 10–13 in an adult's
(their picks at rank 2.4–3.3: neighbours). **Imitation is real on every seed and in both voices** (the next sound
three ranks nearer the heard vowel, twice as often among the three nearest), and smaller than seed 0 alone said (0.22
was the lucky end). **The echo adds nothing.** Two echoes tried: a fading average (the old sound mixes with the new:
imitation at chance) and the last whole stretch (as good as the plain version or a little worse). The echo stays
behind its switch, off. What limits imitation is then not how long the heard sound lasts; the next guesses are the
race (a gesture already half-way to threshold wins first) and the map across voices (the adult's vowels land on
neighbours). Habits from what is heard (`heard_habit`) are built and not yet tried.

**Two senses, seed 1** (`av_probe_s1.log`; another fifth of the videos held out, another hearing development): the
true sound against another video's, Hebb, sight → sound / sound → sight: as born 0.602 / 0.606, developed 0.607 /
0.608, the cochlea alone 0.582 / 0.582 (seed 0: 0.611 / 0.626, 0.622 / 0.631, 0.600 / 0.607); the same video 5 s away
at chance again. The word test (19 words, 128 held-out videos, chance 0.053): sight 0.186, hearing developed 0.136, as
born 0.117, cochlea 0.101, **sight and hearing together 0.206**, the sound sight predicts 0.155. **Two seeds now:**
both senses together name a video better than sight alone (+0.025, +0.020), and on both the hearing's own development
is worth something (word test +0.010, +0.019 over hearing as born; which sound +0.011, +0.004). Seed 2 running.

### 2026-11-26 20:45, babbling drifts toward the language heard (stream brain; exploratory, three seeds)

Infants' babbling comes to resemble the sounds of the language around them before they speak (de Boysson-Bardies
and colleagues, 1980s; to verify). `experiments/babble_drift_probe.py`: the stream brain with its vocal tract babbles
for 600 s while an adult (140 Hz) says vowels of one small language about every 1.5 s; habits grow only from what is
heard (`heard_habit` 0.01: a heard sound strengthens the habits of the gestures whose learned sound matches it), not
from the mouth's own repetition. Two languages of three vowels ("i a u", "e o y"), a brain on each, and one in
silence; measured in the last 200 s: the babbled gestures' mean distance (log formants) to each language's nearest
vowel. **No prediction was written before the first run: this is exploratory.**

| raised on (seeds 0 / 1 / 2) | to "i a u" | to "e o y" |
|---|---|---|
| silence (habits stay flat: top gesture 0.031 = 1/32) | 0.301 / 0.378 / 0.371 | 0.291 / 0.282 / 0.257 |
| "i a u" | **0.301 / 0.315 / 0.341** | 0.281 / 0.286 / 0.263 |
| "e o y" | 0.317 / 0.426 / 0.386 | **0.250 / 0.188 / 0.213** |

Each brain babbles nearer its own language than the other brain does on every seed (6 of 6); against silence the
drift is −0.060 toward "e o y" and −0.031 toward "i a u" (whose vowels sit at the corners of the vowel space, where
few of the 32 random gestures lie). Two leaks of mine on the way, both of the same kind: the habits fed on the
mouth's own voice (while the hearing population still held its tail), so the silent brain drifted at random too;
fixed by taking habits only while sound comes in from the world and not within 100 ms of one's own gesture (the
suppression of one's own voice). The 32 gestures are random places in the vowel space; nothing about vowels was
given but the tract's physics.

### 2026-11-26 21:55, two senses on three seeds: a correction

Seed 2 (`av_probe_s2.log`; 17 words, 125 held-out videos, chance 0.059): sight 0.254, hearing developed 0.114, as
born 0.115, cochlea 0.109, **sight and hearing together 0.218**, the sound sight predicts 0.200. **Together is under
sight alone on this seed.** Over three seeds the difference is +0.025, +0.020, −0.036: no gain established. I wrote at
19:52 and told Kris that both senses together name a video better than either, "two seeds now"; that was early, and
it does not hold. What holds on all three seeds:

| (Hebb, seeds 0 / 1 / 2) | sight → sound | sound → sight |
|---|---|---|
| hearing as born | 0.611 / 0.602 / 0.619 | 0.626 / 0.606 / 0.642 |
| hearing developed | 0.622 / 0.607 / 0.629 | 0.631 / 0.608 / 0.639 |
| the cochlea alone | 0.600 / 0.582 / 0.610 | 0.607 / 0.582 / 0.618 |

Sight and sound on Commons video share something real (0.60–0.64 against 0.5, the same video 5 s away at chance);
any hearing cortex beats the cochlea's band means (+0.009 to +0.024), and its development adds about a point on
sight → sound (+0.011, +0.005, +0.010) and nothing clear the other way. The word test is too thin to carry claims:
one word (cat) holds a third to a half of the held-out videos and most words have one to six. Joining the two codes
side by side for a nearest class mean is also the crudest of combinations; a learner that weighs each sense by how
much it carries is the fair test, with more videos a word (the Commons fetcher keeps adding them).

### 2026-11-26 22:25, real video for the eye, seed 0: time teaches this second stage nothing

`experiments/core50_trace_probe.py`, seed 0 (`out/bench/core50_trace_s0.log`); CORe50's three held-out sessions (other
backgrounds and light), the nearest frame from another session:

| code | same object (chance 0.02) | same kind (chance 0.10) |
|---|---|---|
| the first stage alone (the photo cortex) | 0.117 | 0.226 |
| second stage, frames in their order, with the trace | 0.116 | 0.203 |
| second stage, the same frames shuffled, with the trace | 0.119 | 0.210 |
| second stage, in order, no trace | 0.113 | 0.205 |

**What I said before (13:20): the trace in order above the stage without it by 0.02 or more, and above the shuffled
trace by more. Wrong:** the three arms are level (0.113–0.119) and none is above the first stage alone. Real time,
with objects turning in the hand, teaches this second stage nothing it does not learn from the same frames in any
order. The reading written before the run applies: a stage that looks at 3 × 3 places of the first is too local
for what video teaches (a turning object changes the whole view, not a 3 × 3 patch of it); the place for the trace
is a stage that sees the whole view, as inferotemporal cortex does (Li & DiCarlo 2008 changed IT cells by swapping
objects across saccades; Wallis & Rolls 1997 put the trace at the top of the hierarchy). Next for E 31: the trace in
a whole-view stage (on the relay's code: a population that sees the whole picture, learning by the same rule with
the trace), in order against shuffled. Seeds 1–2 of this probe and its streams are queued as they were. One seed.

**The trace in a whole-view stage, first try** (`experiments/core50_it_probe.py`, three seeds; 256 cells on the first
stage's relayed whole-view code of every tenth frame, half a second apart, one pass over 12,000 frames): same object
from another session 0.033–0.041 in every arm (order 0.035 / 0.033 / 0.041, shuffled 0.039 / 0.038 / 0.038, plain
0.034 / 0.034 / 0.033), same kind 0.134–0.146, against **0.117 and 0.226 for the code it reads**. The stage as built
throws away most of what it is given (256 winner-take-all cells, about 47 frames each, one pass), so the comparison
of orders says nothing either way: this is an inconclusive build, not a finding about time. What a fair test needs:
a stage that keeps what it reads (graded answers of many more cells, or the trace on top of the full code, e.g. as a
learned slow projection), and measures first what the trace is for (the same cells answering across a video's
frames), then whether that carries to another session. Left in the queue (E 31); the 3 × 3 second stage's seeds 1–2
and the streams keep running.

### 2026-11-26 23:10, the ear that hears sound finds word-like units (second reading)

`experiments/sound_units_probe.py`, SpokenCOCO's validation part (20,028 recordings heard, 5,007 held out, 1,793
speakers), with the three changes of 16:30: the ear adapted to each voice (each band relative to its level and
spread over the utterance), 8 band stretches instead of 4, and words as kinds of two- and three-syllable windows by
the eye's rule (1,024 kinds each). The transcripts are read only here, to measure. Held out, nothing learning:

| level | kinds heard in 20+ utterances | word-like (shuffled) | lift of the best word, median (shuffled) |
|---|---|---|---|
| syllable kinds (512) | 512 | 8 (0) | 4.48 (3.73) |
| words by the byte ear's rule | 353 | 5 (0) | 2.38 (2.09) |
| two-syllable kinds | 1,006 | **50 (0)** | 3.73 (2.83) |
| three-syllable kinds | 993 | **61 (0)** | 3.13 (2.53) |

Word-like: the kind's commonest written word is in half its utterances or more, where that word is in a fifth or
fewer. The strongest: "group" 0.94 of the kind's utterances (in 4% of all), "woman" 0.94, "people" 0.92, "person"
0.88, "next" 0.89, "sitting" 0.88, "table" 0.86, "street" 0.83, "top" 0.78. By lift, rarer words: a three-syllable
kind with "motorcycles" in 31% of its utterances (81 times its share), "computer" in 75% (48 times), "skateboard" 56%,
"walking" 64%, "laptop" 48%; two-syllable kinds with "hydrant", "giraffe", "umbrella", "airplane", "keyboard".
**From 0 word-like units in the first reading to 111**, with no text anywhere in the ear: a cochlea, the eye's own
rule on its output, adaptation to the voice, and the eye's rule again on neighbouring syllables. What I said at 12:50
("syllable-sized pieces; a few frequent words with a unit that comes with them in most captions that contain them"):
the units' utterances mostly contain their word (measured), how many of a word's utterances contain its unit is not
measured yet. Caveats: one seed; speakers are shared between the heard and the held-out utterances (a voice-kind
would show here only if it followed one word, which the 20-utterance floor makes unlikely: about three held-out
utterances a speaker); the measure is at the utterance, not where in the utterance the word is.
Next: SpokenCOCO's training part through the cochlea (running, about an hour; every training image is among our COCO
thumbnails) → do these sound units go with what the pictures show (the word cells with sound units as cues and the
photo cortex as content), against the paradigm's visually grounded speech.

**CORe50 seed 0, the stream of 10 kinds** (12,000 training frames seen once; 4,500 frames of the held-out sessions;
chance 0.10): word cells + relay / ridge + relay: first stage 0.363 / 0.373, second stage in order with the trace
0.374 / 0.383, shuffled 0.371 / 0.376, in order without it 0.365 / 0.374. The order is a point above shuffled for both
readers, inside a seed's noise; the same verdict as the nearest-frame reading. Seeds 1–2 queued.

**CORe50 seed 1** (`core50_trace_s1.log`): same object from another session: first stage 0.120, in order with the trace
0.109, shuffled 0.102, in order without it 0.106 (same kind 0.231 / 0.202 / 0.189 / 0.201). Two seeds agree: no arm of
the 3 × 3 second stage is above the first stage, and the order adds a point at most (0.109 against 0.102 here,
0.116 against 0.119 on seed 0). Seed 2 and the streams follow in the queue.

### 2026-11-27 02:25, the trace on top of the whole code (slow directions), said before it runs

The whole-view stage of 22:25 threw away what it read, so it could not test time. `experiments/core50_slow_probe.py`
keeps it: 256 linear cells read the first stage's relayed code (whitened by the relay) and learn by Hebb's rule on a
trace of their own answer (Földiák 1991), against what the cells together already give back (Oja's 1989 subspace
rule: dW = rate · trace · (x − Wᵀy)ᵀ, local to each cell but for the shared feedback). On whitened input, Hebb on a
trace finds the directions that change slowly (slow feature analysis, Wiskott & Sejnowski 2002; its derivation from
spike-timing plasticity, Sprekeler, Michaelis & Wiskott 2007; to verify); shuffled frames have nothing slow in them.
Three arms on the same frames, one pass (order, shuffled, no trace). Measured first what the trace is for (how alike a
frame's answer is to the next frame of its video, 5 s later, the same object elsewhere, another object of its kind,
another kind), then the nearest frame from another session.
**What I say before it runs:** in order the cells' answers are more alike across a video's frames than shuffled (by
construction if anything is learned at all); the test is the same object in another session: in order above shuffled
by 0.01 or more on two of three seeds. Whether 256 slow directions beat the 4,096 they read (0.117): I guess not.
Risk: 12,000 frames for 4,096 inputs is thin, and with one pass the cells may barely move (measured: how like their
start they still are).

**First reading (seeds 0 and 1, `out/bench/core50_slow_s012.log`; seed 2 running).** The slow stage learns (its cells
end 0.33–0.70 like their random start, kept apart: overlap 0.005) and in order its answers are a little more alike
across a video's frames than shuffled (0.5 s apart 0.623 / 0.618 against 0.615 / 0.609), but nothing carries to
another session: same object elsewhere, in order 0.103 / 0.100, shuffled 0.098 / 0.100, no trace 0.104 / 0.105,
against 0.117 / 0.120 for the code read. The prediction (order above shuffled by 0.01 on two seeds) fails on both.
A relay adapted to CORe50's own training frames (seed 0) makes everything worse (code 0.072; arms 0.064–0.072).
**What the measure of alikeness shows, and it is the finding:** in the first stage's code **another kind of object
filmed in the same session is more alike (0.151 / 0.147) than the same object filmed in another session (0.033 /
0.030).** The scene (background, light, the operator's hand) rules the whole-view code; within a video it never
changes, so the slowest thing in every video is the scene, and a slowness rule (any trace) raises exactly that
(same session, other kind: 0.177 in order, 0.172 shuffled, 0.157 without the trace). This is a blind spot of
learning from time with a still camera and a whole-frame eye, not of the trace: a child's eyes fixate the thing in the
hand, and the fovea gives it most of the cortex.

### 2026-11-27 02:55, the eye on the thing (CORe50 through a fixation), said before it runs

`experiments/core50_fovea_probe.py`: the same cortex (the photo benchmark's, unchanged) reads the fixated half of each
frame enlarged to fill the view (as the stream brain's eye does), with a relay settled on COCO thumbnails seen the
same way: the middle place (the object is held before the camera: a stand-in for pursuit), the place that stands out
most (the reflex's own pick), and the whole frame and the salient fixation side by side. **What I say before:** the
middle fixation lowers the scene's share (another kind in the same session less alike than now, relative to the same
object elsewhere) and lifts the same object from another session above the whole frame's 0.117 by 0.02 or more; the
salient pick less surely (it may land on clutter or the hand); the two side by side at least as good as the better.
If the middle fixation helps, the slow stage is tried again on it (CODE=centre).

**Seed 0** (`out/bench/core50_fovea_s0.log`; the slow stage on three seeds: `core50_slow_s012.log`):

| code (seed 0) | same object elsewhere (chance 0.02) | same kind (0.10) | alike: next frame | same object, other session | other kind, same session (the scene) |
|---|---|---|---|---|---|
| the whole frame | 0.117 | 0.226 | 0.563 | 0.033 | 0.158 |
| **the middle fixation** | **0.164** | **0.283** | 0.319 | 0.031 | **0.054** |
| the reflex's pick (stands out most) | 0.075 | 0.171 | 0.232 | 0.019 | 0.065 |
| whole and reflex's pick side by side | 0.102 | 0.204 | 0.397 | 0.026 | 0.112 |

**The eye on the thing: +0.047** on the same object from another session (0.164 against 0.117), the scene's share down
by two thirds (another kind in the same session 0.054 against 0.158). Prediction (+0.02 or more): held, one seed. The
reflex's pick does worse than the whole frame: in a still, centre-surround contrast lands on clutter, edges and the
hand, and jumps from frame to frame (next frame 0.232); the side-by-side code with it is worse than the whole frame
alone (prediction "at least as good as the better": failed). The middle is a stand-in for an eye that follows the
held object; the organic version is an eye drawn by what moves (the held object turns, the scene stays): `VIEWS=moving`
(the place that changed most in the last 100 ms), running on seed 0.
**The slow stage, three seeds, the whole frame** (the prediction of 02:25 fails): same object elsewhere in order
0.103 / 0.100 / 0.107, shuffled 0.098 / 0.100 / 0.099, no trace 0.104 / 0.105 / 0.107, the code read 0.117 / 0.120 /
0.118; order above shuffled by 0.005 / 0.000 / 0.008, never the 0.01 said, and level with no trace at all. **On the
middle fixation's code** (seed 0): in order 0.137, shuffled 0.130, no trace 0.137, the code read 0.164. With the scene
gone the trace still adds nothing the same cells learn without it, and 256 slow directions keep less than the 4,096
they read. The finding of the night is the fixation, not the trace. Queued (`organic-fovea-q`): the fixations of seeds
1 and 2, the stream of 10 kinds on them (word cells and ridge), the slow stage on the middle fixation, seeds 1–2.

**The eye drawn by what moves, seed 0** (`core50_fovea_moving_s0.log`): same object elsewhere **0.093** (same kind
0.194), its next frame alike 0.158 (it jumps: the operator's hand moves as much as the object), side by side with the
whole frame 0.107. Better than the salient pick (0.075), worse than the whole frame (0.117), far under the middle
(0.164). **So the night's result splits in two.** An eye on the held thing makes recognition across scenes much
better (one seed so far; seeds 1–2 queued); neither reflex we have finds the thing (contrast lands on clutter, motion on
the hand, both jump every frame). What the middle stands for, honestly: CORe50's frames are cropped to 128 × 128
around the object, so the middle is where the dataset's crop puts it, not where an eye of ours went. A child's eyes get
there by holding the thing before them (in toddlers' head-camera views held objects fill the middle: Yu & Smith 2012;
to verify) and by pursuit, which keeps a fixation on a moving thing and does not jump to every change; a learned value
of places (queue E 23) is the other road. Next for the eye (E 34): a fixation that persists and follows (pursuit),
measured by how often it lands on the dataset's middle, before any reading of recognition.

**Seed 1 agrees** (`core50_fovea_s1.log`): the middle fixation 0.164 against the whole frame's 0.120 (+0.044; seed 0
+0.047), the scene's share again down by two thirds (0.051 against 0.153); the salient pick 0.077; **the whole frame
beside the middle fixation adds nothing** (0.162): the periphery's code carries the scene, not the thing.

**Where an eye that follows lands** (`experiments/core50_pursuit_probe.py`, held-out sessions, all 44,972 frames at 20 a
second; measured before any code is read: how far the fixated half is from the middle, where the crop put the object):

| eye | within 8 px of the middle | mean distance | moves a frame |
|---|---|---|---|
| jumps to the place that changed most | 0.26 | 13.8 px | 9.1 px |
| pursuit (follows the motion in its own field, τ 100 ms) | 0.34 | 10.9 px | 1.5 px |
| jumps, motion against the view's own flow | 0.29 | 13.1 px | 10.5 px |
| **pursuit, motion against the view's own flow** | **0.45** | 8.8 px | 1.6 px |

Why plain motion misleads: the camera is hand-held, so the whole background moves (38% of all change falls in the
middle quarter of the frame, against 25% by area; the whole-frame centre of change is within 8 px of the frame's centre
59% of the time). Taking out the view's own flow (the best whole-pixel shift between frames, the job of the optokinetic
system and MST; to verify) and following instead of jumping brings the eye near the thing in 45% of frames. The
catch-up saccade (when the eye's own half moves less than the average place for 300 ms) never fired. **Reading:** the
eye on the thing is worth +0.045 on recognition across scenes (two seeds); reflexes alone (contrast, motion, motion
against flow, pursuit) get the eye there at best half the time. The other half is for a gaze that learns where looking
pays (queue E 23: a learned value of places), which is also the stream brain's road. Not pursued further tonight.

**Seed 2 agrees: three seeds now** (`core50_fovea_s2.log`): the middle fixation 0.168 against the whole frame's 0.118
(+0.050; seeds 0 and 1 +0.047, +0.044), the scene's share down by two thirds again (0.051 against 0.150), the salient
pick 0.077 (0.075, 0.077, 0.077), the whole frame beside the middle 0.162 (adds nothing, three seeds).

### 2026-11-27 04:48, what the middle fixation is worth to the word cells (stream of 10 kinds, first two seeds)

`bench_stream.py SET=core50` on the fixated codes, beside the whole frame's of 2026-11-26 (12,000 training frames seen
once, 4,500 tests from the held-out sessions, 10 kinds, chance 0.10; `stream_core50_c50{first,centre,salient}_s*.log`):

| code | word cells + relay, after 10,000 (s0 / s1) | ridge + relay (s0 / s1) | seen once, named 20 later (s0 / s1) |
|---|---|---|---|
| the whole frame | 0.363 / 0.347 | 0.373 / 0.368 | 0.897 / 0.900 |
| the middle fixation | 0.348 / 0.332 | **0.398 / 0.396** | 0.827 / 0.837 |
| the reflex's pick | 0.212 / … | 0.234 / … | 0.773 / … |

**Not what the nearest frame said.** The middle fixation lifts the linear readout of kinds by +0.025 / +0.028 (the
ridge, the reference), but the word cells name kinds *worse* on it (−0.015 on both seeds), and a frame seen once is
remembered less well 20 frames later (0.83 against 0.90). Why is not known yet. (A first reading, that the cells follow the last frames and neighbouring frames of a video are more alike on the whole frame, is wrong: the stream is shuffled, bench_stream.py permutes the training frames.) Two readings left, told apart by a cheap check: if nearest class mean (running means, no gradient, no covariance) also gains on the fixation, the word cells' delta rule (one full-size correction a frame) is what fails to take the gain; if it loses too, the gain needs a readout that weighs directions by how much they vary within a kind (the ridge does, means alone do not). Queued: `ARMS=ncm+relay` on the whole frame and the middle fixation, seeds 0–1 (`stream_core50_ncm_*`). Seed 2 and the reflex's pick on seeds 1–2 are running (`organic-fovea-q`).

**References of tonight's CORe50 entries, found on Crossref (2026-11-27 04:56; that they exist and where, not their
content: what I say they show is from memory until Kris or I read them):** Yu & Smith 2012, Embodied attention and word
learning by toddlers, Cognition, 10.1016/j.cognition.2012.06.016 (held objects fill the toddler's view); Lisberger 2010,
Visual guidance of smooth-pursuit eye movements: sensation, action, and what happens in between, Neuron,
10.1016/j.neuron.2010.03.027 (pursuit driven by image motion near the fovea); Wiskott & Sejnowski 2002, Slow feature
analysis: unsupervised learning of invariances, Neural Computation, 10.1162/089976602317318938; Sprekeler, Michaelis &
Wiskott 2007, Slowness: an objective for spike-timing-dependent plasticity?, PLoS Computational Biology,
10.1371/journal.pcbi.0030112 (Hebb on a trace of whitened input as slow feature analysis).

### 2026-11-27 04:57, the stream brain's eye drawn by motion and following it (two reflexes), said before it runs

Kris asked (2026-11-26) how fixations move and whether they move at all. In the stream brain (`organic/stream.py`)
the eye jumps three or four times a second among nine places, drawn only by how much a place stands out from its
surround, and stays still between jumps. Tonight's probes on CORe50 said what that costs: contrast alone lands on
clutter (0.075 on the same object elsewhere, against the whole frame's 0.117 and an eye on the held thing's 0.164),
and an eye that follows motion against the view's own flow is near the thing in 45% of frames against 26% for
jumps. **Two reflexes move into the stream brain, behind switches (default unchanged, checked by digest):**
- `motion_drive`: what moves draws the saccade too. The superior colliculus's map of where to look takes in motion
  and onsets as well as contrast (a priority map: Fecteau & Munoz 2006; to verify). The two sources are each taken
  relative to their own strongest and then added, the convention of the mouth's drives (no weight between them).
  Motion is the change against the view's own flow (the best whole-pixel shift of the frame before, within 3 px:
  the self-motion the optokinetic system and MST discount; to verify).
- `pursuit_tau`: between saccades the eye drifts toward the centre of that motion within its own field (the fixated
  half and a quarter around it), with time constant 100 ms, every 10 ms step; the view is coded again when the eye
  has moved 2 px. Pursuit is driven by image motion on and near the fovea (Lisberger 2010) and develops in the
  first months of life (to verify). A saccade puts the eye on a place; pursuit holds it there if the thing moves.
Nothing here is learned: both are reflexes, given as the retina is. **Measured** on CORe50's held-out sessions (all
frames at 20 a second, the brain's clock at 10 ms, three seeds of cortex and noise): where the eye is at each
coded frame (within 8 px of the middle, where the crop puts the object) and the code of what it is on, read as
before (the same object in another session, chance 0.02; the same relay for every eye). **What I say before:**
today's eye near the salient pick (0.07–0.08); with both reflexes above today's eye by 0.02 or more and within 8 px
of the middle in at least 35% of frames; reaching the whole frame's 0.117 I give even odds; the middle's 0.164 not.
