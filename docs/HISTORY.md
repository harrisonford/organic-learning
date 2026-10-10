# History: the designed brain and the hierarchy experiments

This keeps the record of earlier designs and experiments. The current
structure is described in [STRUCTURE.md](STRUCTURE.md).

## The designed brain (`organic/brain.py`, `organic/eye.py`)

Before the raw organs, the brain was built with a lot of designer help.
This inventory explains why it was set aside as the main line.

### 1. Innate organs (fixed, hand-built)

| organ | what it hard-codes |
|---|---|
| Eye (`organic/eye.py`) | finds salient blobs (fixed threshold), groups parts enclosed by another blob, saccades largest-first, centers and scales each object; splits each object into pre-made streams: form (luminance + 4 edge orientations), color (12 hue-tuned neurons + saturation + brightness), where (x, y, size), number (count of fixations); a coarse 8x8 glance (color, where, form). Arrival ticks: glance 1, color/where 2, form 3, +1 per extra object. |
| Ear (`organic/ear.py`) | raw UTF-8 bytes; Hebbian next-byte counts over the last 1-4 bytes; a new unit starts where surprise rises > 1.2 bits; bytes and 2-3 byte sequences excite fixed random neurons (by position and anywhere); frequent units habituate; fixed echo shapes for "what was asked" (8 units deep) and "what I said" (3 units). |

Debt: **the eye hands the brain exactly the attributes the questions ask
about** (color, shape, place, size, count). This is the largest piece of
guidance in the system. Next step: rebuild the eye as a raw, retina-like organ.

### 2. Generic learning rules (closest to the original idea)

- Sensory areas: a new assembly when nothing responds above *vigilance*;
  otherwise the winners drift toward the stimulus at rate 1/n.
- Speech cortex: a cluster is 8 neurons. Problem affinity = stimulus affinity
  × whether its outputs lean to the heard unit. No affinity: neurogenesis.
  Only clusters with affinity learn (instar/outstar), with plasticity scaled
  by surprise and falling with use.
- Per-cluster myelin (up when helping a correct answer, down when voting wrong)
  and stability.
- Sleep: near-identical clusters merge, useless ones die; hippocampal replay.
- Hebbian traces between what is seen and what is heard; homeostatic word
  baselines.

### 3. Wiring designed by hand (task-shaped guidance)

| decision | added for | how much it steers |
|---|---|---|
| one sensory area per attribute (shape / color / where / number / glance) | easy naming | high: mirrors the questions |
| vision reaches speech only as priming (gain on word-forms), never as input | held-out color+shape pairs | high |
| match / mismatch neurons (counts of heard-and-seen, heard-but-absent) | yes/no and logic | high: logic mostly rides on this |
| "objects not yet named" signal into speech | stopping description loops | high |
| attention: inhibition of return once the shape and color words are said; joint attention; heard words bias object competition | multi-object and relation tasks | high: designer chose which areas "name" an object |
| emergent word categories + activation spread within them + category-colored self-echo | unseen combinations | medium |
| word fatigue | repetition | low-medium |
| only answer words bind to vision, not question words | cleaner lifts | medium |
| pathway myelin vs an integration deadline | biological flavour | low |
| patience gate with designer-chosen update signs; confidence = least certain word of the plan | system 1 / system 2 | medium |
| area sprouting (off by default) | structural growth experiment | n/a |

### 4. Parameters tuned by looking at accuracy (designer optimization)

| parameter | value | where |
|---|---|---|
| category spread | 0.3 | speech |
| "not yet named" weight | 0.6 | speech |
| priming strength λ | 1.5 | speech |
| category blend | 0.5 | speech |
| competition temperature | 0.02 | speech |
| familiarity threshold | 0.97 | speech |
| maturation | 5 | speech |
| vigilance | 0.86-0.9 | sensory areas |
| sound-overlap cutoff | 0.15 | priming, binding |
| grounded / supported thresholds | 3.0 / 2.7 | match / mismatch neurons |
| attention bias margin | 1.2 | attention |
| segmentation rise | 1.2 bits (the default actually used; 0.5 was explored but never set) | ear |

### 5. Environment

- Learning signal: imitation (hearing the human answer = supervised next-unit
  learning) plus a scalar reward for timed tasks.
- Data: templated scenes and questions, hidden deadlines, held-out color+shape
  combinations and phrasings.

## Hierarchy experiments: how should hierarchy appear?

The raw brain's V1 area memorizes views, not things (thousands of
assemblies, few reused). Biology gets invariance from **temporal contiguity**:
successive glimpses in a moment are usually of the same thing, so an area that
learns from the *recent history* of a lower area's activity groups the views
of one object (Földiák 1991; Li & DiCarlo 2008). Two ways to let that happen:

**Easier: a fixed rule.** Every area's leaky state feeds one higher area that
grows by novelty like any other. The designer decides that hierarchy exists
and how deep it goes. Generic (no task knowledge), but structural guidance.

**Harder: the brain decides.** An area watches its own novelty. If, long after
infancy, most of what it sees still gives birth to new assemblies ("everything
looks new"), it sprouts a higher area fed by the recent history (leaky state)
of its own activity, read through its assemblies' fixed axon patterns. The new
area is subject to the same rule, so depth can grow where needed and nowhere
else. Earlier sprouting in the designed brain (random expansion triggered by
speech surprise) failed or was neutral; this version differs in what triggers
it (local, persistent novelty), what it reads (a source's history, not random
mixtures) and whom it serves (any downstream reader, not only speech).

Plan: explore the harder option first, time-boxed. It counts as fruitful if,
on a probe that needs invariance (naming shapes and colors at positions never
seen during life), brains that sprout beat brains that do not on held-out
positions across two seeds, or the sprouted area's assemblies are measurably
more position-invariant than their source's. If not, switch to the easier
rule and record why.

### Harder option, round 1: not fruitful (runaway chain)

Novelty-driven sprouting, children fed by the parent's leaky state, same
growth rule everywhere. Probe: shape/color naming, objects seen only in the
left and middle columns, tested also in the right column. 6000 moments, 2 seeds.

| | shape, trained positions | shape, new positions | areas |
|---|---|---|---|
| no sprouting | 0.80 / 0.65 | 0.66 / 0.38 | 3 |
| sprouting | 0.20 / 0.20 | 0.04 / 0.16 | 8 (chain) |

Each child found its input *more* novel than its parent did (novelty 0.16, 0.25,
0.29, 0.33, 0.35), so it sprouted again: five generations of ~9000 assemblies.
Their noisy priming ruined speech. No invariance appeared: ~25 assemblies per shape
in every area. Reason: reading a history of views does not make things invariant if the growth rule
still makes a new assembly for every new combination. Temporal contiguity needs
the *trace rule* (Földiák 1991): the assembly that was just active stays
favoured for a moment, so the next view joins it.

### Harder option, round 2: trace rule in every area

Generic change to competition in all areas: an assembly's lingering activity
(the leaky state the brain already keeps) adds a bias of 0.2 (times its
normalized state) to its affinity, both for who wins and for whether a new
assembly is born.

Result (6000 moments, 2 seeds):

| | shape, trained positions | shape, new positions | assemblies per shape |
|---|---|---|---|
| trace rule, no sprouting | 0.77 / 0.65 | 0.66 / 0.34 | v1 ~24 |
| trace rule + sprouting | 0.16 / 0.28 | 0.04 / 0.20 | every area ~24 |

Not fruitful either. The chain still runs five generations deep (children are
smaller: ~3000 assemblies instead of ~9000). No area becomes invariant: a 0.2
bias cannot bridge successive views of one object (an edge, then a corner), which
differ more than that. Common cause of both rounds: a novelty trigger cannot
tell "I need a new level" from "my input is simply rich", so it builds chains.

Per the plan, the harder option is paused. Next: the easier rule (one fixed
higher level per area), where the open question is no longer *when* to grow
a level but *how strong* the trace must be for views to group at all.
