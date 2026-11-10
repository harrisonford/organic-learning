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
