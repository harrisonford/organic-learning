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

## Result, step 2 offline (2026-11-06): settling emerges, to a floor; vision learning is no better than random for naming

`experiments/plastic_vision_probe.py`: the same stream of life pictures to three
cortices. Drift is measured on 100 fixed pictures over each block of ~2,100 pictures:

| pictures seen | frozen (developed) | plastic (developed) | scratch (random, learning only in life) |
|---|---|---|---|
| 2k | drift 0, naming 0.417 | drift 0.017, naming 0.407 | **drift 0.400**, mismatch L3 0.48, naming 0.417 |
| 4k | 0 | 0.017 | **0.052**, L3 0.35 |
| 6k | 0 | 0.018 | **0.025**, L3 0.30 |
| 8k | 0 | 0.016, naming 0.393 | **0.014**, L3 0.30, naming 0.403 |
| 12.5k | 0 | 0.016, naming 0.397 | **0.015**, L3 0.285, naming 0.410 |

1. **Settling emerges without a clock.** A random cortex learning only from what the eye
   sees drops from drift 0.40 to 0.015 within ~8k pictures, as each kind's mismatch falls.
   It then sits at the same floor as the developed cortex (~0.016 per 2k pictures) and
   stays there: plasticity is proportional to each kind's mismatch, which never goes
   below the natural spread of the windows it wins (~0.17 / 0.25 / 0.28 per layer). So
   the organ never fully settles; it keeps drifting slowly in a world that does not change.
   Next iteration on the principle: a kind learns in proportion to its *unexpected*
   mismatch (above its own usual spread), so it is still in a familiar world and plastic
   again when something new comes.
2. **The vision rule learns nothing useful for naming beyond random filters.** A random
   cortex gives the same 1-NN naming (0.42) as the developed one, before and after
   learning. The competitive Hebbian rule finds local textures and edges, and the
   naming code is carried by colour and layout, which random filters keep as well. Real
   inferotemporal cortex learns invariance (the same object across views); the trace
   rule (experiments/trace_it_probe.py) is the candidate principle for that.
   This is where vision needs iteration, more than in freezing or not.

## Trouble 3 offline: cross-situational links between organ cells (2026-11-06)

`experiments/cross_situational_probe.py`, first-words world: 11,246 training moments; test
pictures (CIFAR and Commons) each asked 3 different ways; 115 adult-written items.

| | picture right | same answer, 3 wordings | right all 3 times | adult items (word in the question) |
|---|---|---|---|---|
| whole-situation memory (question sound + picture → word; roughly our clusters) | 0.18 | **0.06** | 0.03 | 0.36 |
| vision only (nearest remembered picture) | 0.36 | – | 0.36 | – |
| cross-situational links from every input (sound units, broad visual features) | 0.08 | 0.70 | 0.07 | 0.51 |
| **cross-situational links from sound units and visual cells** | **0.39** | **0.90** | **0.38** | **0.51** |
| **the hub as built (organic/hub.py): the visual area is a Cortex, the orchestrator's own neurons, seeing only the picture (5,662 cells grown)** | **0.39** | **0.99** | **0.38** | **0.51** |

- The whole-situation memory reproduces kid_s3's frame-bound naming offline (same answer
  across wordings 6%).
- Cross-situational links (Smith & Yu 2008), where each active input keeps Hebbian links
  to the words it co-occurs with, normalised by how often it is active, make frame words
  favour nothing without anyone marking them: consistency 0.06 → 0.70, adult items
  0.36 → 0.51.
- But broad visual features (each co-occurs with every word) cast flat votes, so pictures
  collapse. When vision first forms its own cells (vision-only memories, like clusters
  grown from what is seen) and those cells link to words across situations, pictures are
  named as well as vision allows (0.39), the same way 90% of the time, whatever the
  wording; adult items stay at 0.51.

**The principle this points to:** each organ grows its own cells (visual cells from what is
seen, sound cells from what is heard), and words link to organ cells across situations,
instead of one pool of clusters each storing a whole situation. This matches the brain:
separate sensory cortices and an association hub (temporal lobe, hippocampus), where
the word-thing link is cross-modal. It is also what lets the varied world (56 wordings,
many breeds) help instead of hurt.

Noise to iterate on: rarely heard units ("hmm?", "!") get sharp links by chance from a
few co-occurrences; a link should become trusted only with experience, which has to
come from the neuron rule, not a cutoff.

## The hub in the speaking brain (2026-11-07)

`organic/hub.py`, `SpeakingBrain(hub=True)` (`HUB=1` in curriculum_life): a visual area
(a Cortex fed only the whole view), sound cells (heard units), and lexical neurons (word forms
the adults said), linked across situations. The hub proposes a word; speaking it is a third
learned route next to copy and own ("speak:recall", credited by outcome like the other
two; the hub's confidence enters that choice as its own drive, as the eye's salience
enters its movement choice). The recalled word's sound primes the mouth through the
same route copying uses. Old brains (kid_s3) have no hub and are unchanged.

## The hub in a life: controlled pair (2026-11-08)

`experiments/hub_pipeline.sh`: two small lives (8 neurons, 2,000 clusters, seed 0, the wider
world, 1 overheard sentence per moment), with and without the hub.

| at first words (W1) | W1 score | same answer, 3 wordings | right all 3 times | W1 passed |
|---|---|---|---|---|
| no hub (15k–80k) | 0.12–0.22 (mean ~0.15) | **0.00** at every checkpoint | 0.00 | no |
| **hub** (15k–50k) | **0.27–0.41 (mean ~0.36)** | **0.77–0.97** | **0.13–0.33** | **yes, at 50k (0.41), after eight checkpoints at 0.33–0.41** |

The hub more than doubles first words and turns wording consistency from never into
almost always. It passes W1 on a sustained level, not a lucky checkpoint.

**But after moving on to two words, its picture naming collapses** (CIFAR and Commons
naming → ~0.0) while its answers stay consistent across wordings (0.6–0.97): it now says
the same *wrong* thing for a picture whatever the wording. W2 asks about the same kinds of
pictures with other answers ("yes", "two", "brown"), and cross-situational links from a
picture's cells now favour those words. Additive votes cannot express a conjunction ("a
naming question AND this picture → its name; a colour question AND this picture → its
colour"): the picture's vote outweighs the question's. The whole-situation clusters did
conjunctions but were over-specific; the hub generalises but cannot do conjunctions.
Next iteration: organ evidence that combines multiplicatively (each organ's vote scales
the others', like coincidence detection in dendrites), so a question kind that never
goes with "yes" can veto it. Tested offline first.

kid_s3 ended on 2026-11-08 at age 745k: "stops at W2 (two words): 0.35 < 0.5 after 400,000
moments". Its lineage released L1 babbler, L2 echo and L3 first words.
