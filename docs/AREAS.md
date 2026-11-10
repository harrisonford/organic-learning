# Words made of the ear's own units (2026-11-10)

Steps 2 and 4 of [REVIEW_2026-11-10.md](REVIEW_2026-11-10.md): dissolve the hub (count tables,
adults' answer strings as "words", kinds clustered in a batch) into something that grows, with words
made of the units the ear itself discovered, and let the brain learn from unasked comments as well
as from lessons. Code: `organic/words.py`. Probes: `experiments/areas_probe.py`,
`experiments/observation_probe.py`. Everything here is offline; the lives are in docs/LOG.md.

## What was tried first, and why it failed

**One tissue, three areas** (`organic/areas.py`, kept for the record). Three areas of the
orchestrator's own cluster tissue (visual, heard-unit cues, echo of the exchange), all learning to
predict the unit heard next, the echo area's expectation acting as the gain of the others.

| first words only, 9,000 moments, seed 0 | right | same answer ×3 | adult items |
|---|---|---|---|
| hub (unlimited visual cells) | 0.39 | 0.99 | — |
| three areas, as first built | 0.19–0.22 | 0.28–0.49 | 0.42 |
| + lateral "company" links, no cues | 0.33 | 0.997 | 0 |
| + lateral links, with cues | 0.31 | 0.66 | 0.46 |

Two things were learned from it:

1. **Lateral company links do what "answer kinds" did, online.** When a unit is heard, a link grows
   from each unit the context was expecting to the unit that came (Hebbian: fired × expected). Units
   that replace each other (the names; the colours; yes and no) come to share expectation. Same answer
   across three wordings went from 0.28 to 0.997 with no clustering step and no threshold.
2. **The cluster tissue is an exemplar memory, and words need more than that.** A cluster's
   expectations followed its last ~100 moments (a fixed plasticity floor), so a cue heard in every
   question kept re-learning the last reply. And remembering pictures one by one reads the eyes badly:

| reading the same visual code (first-words pictures, 780 held out) | named right |
|---|---|
| nearest remembered picture, 2,000 remembered | 0.36–0.42 |
| nearest remembered picture, 9,000 remembered | 0.43 |
| hub's visual cells with a life's budget (1,500–3,000 cells) | 0.28–0.29 |
| batch linear readout (ridge), 2,000 pictures | 0.54–0.57 |
| **local delta rule, online, one pass of 9,000 moments** | **0.47–0.50** |
| **the same, 27,000–90,000 moments (a life's worth)** | **0.54–0.56** |

The review said "first words are capped by the eyes at about 0.4". That was half wrong: about
0.15 of the cap was the way the eyes were read. A slow statistical pathway (each word cell's
dendrite learning, by its own error, which visual neurons predict its firing) reads 0.55 from the
same code with a few thousand synapses per word instead of a memory of every picture. The brain
has both kinds of learning (fast episodic, slow statistical: McClelland, McNaughton & O'Reilly
1995); the project had only the fast one, and the hub's count tables were quietly playing the slow one.

## The design: unit cells with two dendrites

A word form is one of the ear's own units. Every unit heard in a reply or a comment has a cell,
which fires when its unit is heard. Its dendrites learn to predict that firing with one local rule
(dendritic prediction of somatic firing, Urbanczik & Senn 2014; for cues it is the Rescorla–Wagner
rule of animal learning):

    change of a synapse = cell's plasticity × input's plasticity × (fired − expected by this dendrite) × input activity

- **Content dendrite**: the visual cortex's neurons (each passing on its deviation from its own
  average), the ear's unit cells heard in the utterance (cues), and the cell's own baseline. Predicts
  whether the unit will be in the reply. Inputs compete to explain it: "what" and "is", heard before
  every reply, end up explaining nothing; the picture or the informative cue explains the word.
- **Context dendrite**: the ear's order-aware echo of the exchange so far (what was heard, the
  silence that ended it, what has been said since). Predicts the unit that comes next: which kind of
  reply this kind of utterance gets, and in what order. Lateral company links (above) spread its
  expectation over units that replace each other.
- **Recall**: a cell is active as far as its content dendrite is driven above its baseline AND its
  context dendrite expects it now (coincidence of two compartments, as in pyramidal cells). The most
  active unit is heard inwardly, joins the echo, and the next is predicted, until the silence cell
  wins. A cell that has just fired is refractory while the reply lasts. The reply is a sequence of the
  brain's own units.
- **Plasticity, no schedule**: the cell's own recent error (a new cell is fully plastic; one whose
  predictions come true stops changing) × the input's consolidation with its own use, 1/(1 + use/5),
  the orchestrator's law. A new word on a new cue is learned at once; an input present in thousands
  of moments carries long-run averages.

What it replaces: the hub's count tables (cue competition), its answer strings (unit cells), answer
kinds and their sleep clustering (context dendrite + company links), the order of words in a reply
(the echo of what was just said), and the special status of a lesson (see observation below).

### What failed on the way (all measured, seed 0)

- Plasticity = own error alone: the context dendrite never settles, because which name follows a
  naming question is never predictable; it chases the last reply (right 0.04).
- Consolidation by the cell's experience alone: the visual synapses freeze too early (0.38–0.47).
  Both factors together work; the consolidation belongs to the input, not to the cell.
- One fixed wording per picture (the world, not the brain): a wording's own words became cues for
  "its" pictures' names. Naming questions are now worded afresh at every moment (`Curriculum.wording`).
- Cue cells passing on their deviation from their own average (as visual neurons do): no difference,
  removed.
- Without refractoriness, a reply could repeat its last unit until cut ("yellowwwwwww").
- Without the company links, on mixed stages: right 0.17–0.21, same ×3 0.28–0.38.

## Results (3 seeds; the hub has a life's budget of 1,500 visual cells)

First words only (9,000 moments; mean ± spread over 3 orders of moments; 300 pictures × 3 wordings, 150 adult items):

| | pictures named right | same answer ×3 | right ×3 | adult items (the word is in the question) |
|---|---|---|---|---|
| hub | 0.287 ± 0.012 | 0.970 | 0.283 | 0.474 |
| hub + kinds | 0.287 ± 0.012 | 0.970 | 0.283 | 0.184 |
| **word cells**, one pass | **0.389 ± 0.010** | 0.733 | 0.333 | 0.402 |
| **word cells**, three passes | **0.444 ± 0.015** | 0.836 | **0.410** | 0.445 |

First words and two words mixed, as a life at two words lives them (18,000 moments; 2 orders so far, the third is running):

| | first-words pictures right | same answer ×3 | right ×3 | adult items | two words (colour, yes/no, counting, phrases) |
|---|---|---|---|---|---|
| hub | 0.105 | 0.956 | 0.102 | 0.266 | 0.356 |
| hub + kinds | 0.228 | 0.835 | 0.215 | 0.124 | 0.426 |
| **word cells**, one pass | **0.360** | 0.593 | 0.280 | 0.314 | **0.507** |
| **word cells**, three passes | **0.416** | 0.734 | **0.362** | 0.357 | **0.514** |

The hub's naming falls when two-words moments are mixed in (0.29 → 0.11; with kinds 0.23); the word
cells' barely moves (0.39 → 0.36), and two words is at its pass mark offline. By kind of question
(seed 0): is-there 0.62–0.65, how-many 0.70, yes/no 0.58–0.65, phrases 0.41, colour 0.24. What the
context dendrite expects after "how many people are there?" is one, two, three, zero; after "what
color is the bus?" brown, black, yellow, white; nobody listed those.

The price: the same answer across three wordings is 0.59–0.84 against the hub's 0.84–0.97.
Speed and size: 3 ms per moment; a few thousand synapses per word instead of a remembered picture
per cell (the hub's 1,500 visual cells are 440 MB; 400 word cells are 12 MB).

## Observation before interrogation

`experiments/observation_probe.py`. For the word cells an observation is nothing special: an
utterance heard after silence while a picture is seen ("look, a dog!"; 66 ways written by the adults,
`data/world/observation_frames.json`). The first-words pictures were split: 10% asked about, 90% only
commented on.

| (mean ± spread over 3 seeds; 300 new pictures × 3 wordings) | new pictures named right | same answer ×3 |
|---|---|---|
| asked only (10% of the pictures) | 0.215 ± 0.011 | 0.36 |
| asked + observed (the other 90% only commented on) | 0.379 ± 0.013 | 0.50 |
| all asked | 0.411 ± 0.008 | 0.78 |

Names form from co-occurrence alone, and questions then show what was learned. In lives:
`OBSERVE=1` (one unasked comment per moment; from first grade also whole descriptions of photos).

## Limits known now

- Unit cells are exact units: "dog", "dog!" and "a dog" are three cells. The ear's sound code cannot
  tell punctuation from a plural ("dog!" 0.81, "dogs" 0.77 alike to "dog"), so merging by sound is not
  the answer; the ear's cuts are (docs/EAR_SEGMENTATION.md steps 2–5 are still open).
- Yes/no questions about a picture ("is there a dog?") need the asked thing AND the picture together;
  additive content cannot do that. They are answered at base rates (0.6–0.65).
- Same answer across wordings is 0.6–0.85, below the hub's 0.95+: a wording's words carry a little of
  the names' base rates, differently per wording. It rises with experience (more passes).
- The orchestrator still chooses between copying, its own motor habits and recall by hand-wired
  routes. Recall as an inner echo that the copy route follows would remove one route.
