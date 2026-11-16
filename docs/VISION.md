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
