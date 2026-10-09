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
