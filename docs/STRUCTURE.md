# Structural decisions and configuration

An honest inventory of what is designed by hand versus what emerges. The aim
of the project is a brain that is barebones and raw: simple growth rules,
with structure that evolves. Anything listed under sections 3 and 4 below is
guidance from the designer and should shrink over time.

Policy going forward:

- **Organs** (eye, ear) may be fixed in behaviour, like what evolution hands
  an animal. They are improved separately, as organs, never to make one task
  work.
- **The rest of the brain** should only have simple, generic rules. Task-shaped
  circuits and tuned parameters are debts to be removed.

## 1. Innate organs (fixed, hand-built)

| organ | what it hard-codes |
|---|---|
| Eye (`organic/eye.py`) | finds salient blobs (fixed threshold), groups parts enclosed by another blob, saccades largest-first, centers and scales each object; splits each object into pre-made streams: form (luminance + 4 edge orientations), color (12 hue-tuned neurons + saturation + brightness), where (x, y, size), number (count of fixations); a coarse 8x8 glance (color, where, form). Arrival ticks: glance 1, color/where 2, form 3, +1 per extra object. |
| Ear (`organic/ear.py`) | raw UTF-8 bytes; Hebbian next-byte counts over the last 1-4 bytes; a new unit starts where surprise rises > 0.5 bits; bytes and 2-3 byte sequences excite fixed random neurons (by position and anywhere); frequent units habituate; fixed echo shapes for "what was asked" (8 units deep) and "what I said" (3 units). |

Debt: **the eye hands the brain exactly the attributes the questions ask
about** (color, shape, place, size, count). This is the largest piece of
guidance in the system. Next step: rebuild the eye as a raw, retina-like organ.

## 2. Generic learning rules (closest to the original idea)

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

## 3. Wiring designed by hand (task-shaped guidance)

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

## 4. Parameters tuned by looking at accuracy (designer optimization)

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
| segmentation rise | 0.5 bits (picked because units looked word-like) | ear |

## 5. Environment

- Learning signal: imitation (hearing the human answer = supervised next-unit
  learning) plus a scalar reward for timed tasks.
- Data: templated scenes and questions, hidden deadlines, held-out color+shape
  combinations and phrasings.

## Possible barebones reset

1. One generic sheet per organ (raw pixels at two speeds, raw bytes), so any
   specialization into color / shape / where must emerge.
2. Remove the match/mismatch neurons, the "not yet named" signal, the
   naming-based attention rules and the category spread/blend. Keep growth,
   Hebbian association, surprise and reward.
3. Freeze all remaining parameters at round, untuned values.
4. Keep the tasks and experiments as probes. Performance will drop. What we
   watch is whether structure differentiates as tasks are added.

## Status

- The raw eye exists (`organic/retina.py`, V1 frozen in `data/v1.npy`).
- `organic/raw_brain.py` implements steps 1-3 of the reset (generic areas per
  organ stream, no designed circuits, round parameters) and adds leaky
  activity over ticks. The designed brain (`organic/brain.py`) is kept for
  comparison.
- Still designed in the raw brain: the patience update rule, fatigue,
  resonant binding through shared auditory neurons, homeostatic word
  baselines, and the ear's segmentation rule.

## Open decision: how hierarchy appears

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
