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
- **And it made copies.** About 330 pairs of kinds are near copies of each other (cosine 0.89), where no two random
  kinds are alike beyond 0.29. Winner-take-all with a conscience sends the kinds to where the windows are dense, and
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
