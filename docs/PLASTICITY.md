# Every organ learns, and settling has to emerge (2026-11-06, design)

The user: "maybe it shouldnt be fixed and everything be trained at the same time eye
ear and all clusters and every organ because organs fix would be similar to images
and bytes downstreams… organs should settle but we should see that emerge not force
if it doesnt emerge we iterate on the principles of the neurons behaviour". Also: wins
are hints of generalization, not scores, and score drops are acceptable.

## Where things stand

| part | during a life | biology |
|---|---|---|
| raw sensors: pixels, bytes → fixed random neuron patterns | fixed | the physics of transduction: photoreceptors and hair cells do not learn. **Stays fixed.** |
| eye, V1 parts bank | frozen after development | V1 stays plastic for life (less after critical periods, which close *through experience*) |
| visual cortex, layers 1–3 | frozen after development (decaying rate, then only read) | plastic, more so higher up |
| ear: where words split (byte statistics) | learns | statistical learning, for life |
| ear: sounds above the "cochlea" | no learned layer: the unit's code is a sum of fixed byte/n-gram patterns | auditory cortex: learned, plastic |
| mouth | learns | learns |
| orchestrator (all clusters) | learns, grows, myelinates, prunes | learns |

The frozen eye and visual cortex are what the user describes: fixed feature extractors
feeding fixed codes downstream, no different in kind from feeding the orchestrator
pre-made features.

## The principle to test

**A neuron's plasticity depends on its own history, not on a clock.** In the brain,
critical periods close because experience matures the circuit (inhibition, perineuronal
nets: Hensch 2005); dark-reared animals stay plastic longer; enrichment and novelty
reopen plasticity in adults. Metaplasticity (Abraham & Bear 1996, BCM) does the same
at single neurons: how much a neuron changes depends on its past activity.

The candidate rule for every learned organ (the same rule everywhere, no schedule):

- each neuron (a V1 part, a visual-cortex kind, an auditory kind) keeps a running
  measure of how well it fits the inputs it wins (its own mismatch);
- it learns in proportion to that mismatch: a neuron whose inputs keep looking the same
  stops changing on its own; one whose inputs change (a new kind of photo, new speech)
  becomes plastic again.

Settling is then something we **measure**, not impose: if the organ's code drifts less and
less as the life goes on, settling emerged. If it does not (the code keeps drifting, or
it settles too early and the brain cannot take in a new kind of input), the neurons'
principle is what we iterate on.

## Steps (each measured before the next; small controlled lives before any lineage)

1. **Measure drift.** At every checkpoint, the codes each organ gives for a fixed probe
   set (pictures, utterances) are compared with the previous checkpoint's. A drift curve
   per organ. For frozen organs it is 0, the baseline.
2. **A plastic visual cortex and V1**, with the rule above, learning on what the eye sees
   during life. Offline first: does the code stay usable for naming (nearest-neighbour
   probe) while it keeps learning, and does its drift fall on its own? Then two small
   controlled lives, frozen vs plastic vision, judged on generalization signals as well as
   scores.
3. **A learned auditory cortex** above the fixed "cochlea" (the byte patterns): kinds
   that learn the sounds of units by the same competitive Hebbian rule and the same
   plasticity principle. This is where a frame and a word could come apart in the code.
4. **Everything together** in a new small life, then in the lineage, only once the
   steps show the principle works.

## What counts as a win (generalization signals, recorded at every checkpoint since 2026-11-06)

- **Wording invariance:** the same test picture asked three different ways. Does the
  brain give the same answer each time, and the right one? (It understood the request,
  not the sound of one frame.)
- **Response time per task and per source:** quick and slow responses for different
  kinds of questions, as seen before.
- **Organ drift curves** (step 1): whether settling emerges.
- Scores are kept and reported, but a drop is not a reason to roll back a richer world.
