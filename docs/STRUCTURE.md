# The current structure: what it is, what it does, and why

This describes the main line: the barebones brain on raw organs
(`organic/raw_brain.py`). The earlier designed brain and the hierarchy
experiments are recorded in [HISTORY.md](HISTORY.md).

## What we are studying

1. **Time constraints.** One structure should handle both fast, rough answers
   (system 1) and slow, careful ones (system 2), without being told which a
   question needs.
2. **Evolving structure.** New kinds of problems should make the structure
   grow and reorganize through simple rules, not through circuits we design
   per task.

Performance is not a goal at this stage. Real competence probably needs a
structure and a life orders of magnitude larger.

## Policy

- **Organs** (eye, ear) may be fixed, like what evolution hands an animal.
  They are improved separately, as organs, never to make a task work.
- **The rest of the brain** uses only generic rules with round, untuned
  numbers. Anything designed for a task is a debt (listed at the end).

## A moment of life, step by step

1. A scene appears and a question is heard (raw bytes).
2. The eye looks for 9 ticks: 3 fixations of 3 ticks each. Each tick it sends
   what it sees.
3. Each brain area updates its activity: half of the old activity remains and
   the new response adds to it. A percept builds up over the moment.
4. At every tick the speech area silently plans an answer from the current
   activity. If it is confident enough (its *patience* is met), it speaks; at
   the last tick it speaks anyway.
5. Afterwards it hears what a person would have said (imitation) and gets a
   scalar reward (right, wrong, late). It learns from both.

## The organs (fixed)

### Eye (`organic/retina.py`)

| part | what it does | design decisions |
|---|---|---|
| receptors | a log-polar grid around the point of fixation: 14 rings × 16 directions, dense and sharp in the center, sparse and blurred toward the edge, covering the whole view | grid size and growth rate chosen by hand |
| cones | three types (L, M, S) computed from RGB | fixed mixing weights |
| ganglion cells | each receptor compares its center with a 3× wider surround: brightness ON/OFF, red-green ON/OFF, blue-yellow ON/OFF | standard retina model |
| magno pathway | brightness change only; fires at the start of each fixation (transient) and arrives first | |
| parvo pathway | all six channels, sustained, arrives one tick later | |
| saccades | starts at the center; after 3 ticks jumps to the strongest peripheral contrast, avoiding the last 4 places it looked (within 5 px) | fixation length and inhibition-of-return radius chosen by hand; no notion of objects |
| eye position | where the eye points, as a population code (7 + 7 neurons) | |
| V1 | 16 kinds of simple cells tile the map, each looking at a 3×3 window; 2 strongest kinds respond per location; responses scale with local contrast | learned once, then frozen (see below) |

**How V1 developed:** competitive Hebbian learning on contrast-normalized
windows (only well-driven windows teach, winners that win too often are
handicapped). First 1500 spontaneous "retinal waves" (drifting gray
blobs), then 600 crops of natural photographs, never the task scenes. Result:
oriented edges in several directions and color-opponent cells. Stored in
`data/v1.npy`.

### Ear (`organic/ear.py`)

| part | what it does | design decisions |
|---|---|---|
| input | raw UTF-8 bytes, then silence | |
| statistics | counts which byte follows the last 1-4 bytes (Hebbian) | context length 4 |
| segmentation | a new unit starts where the surprise of the next byte jumps by more than 1.2 bits, or where it has no idea at all; at birth every byte is its own unit | 1.2 bits is the value actually used (0.5 was explored in a test but never set) |
| sound | each byte and each 2-3 byte sequence excites a fixed random set of neurons, by position and anywhere in the unit | |
| habituation | units heard often excite less | |
| echoes | "what was asked": the last 8 units, fading, plus the gist; "what I said": the last 3 units | echo depths chosen by hand |

## The brain (`organic/raw_brain.py`)

### Areas: one per organ output stream

`v1` (the V1 map, 3584 signals), `magno` (224) and `gaze` (14).
No areas for color, shape or place.

- **Growth by novelty:** if no assembly matches an input closely enough
  (vigilance 0.8), a new assembly is born for it; otherwise the best matches
  respond and drift toward it (more slowly the more experienced they are).
- **Activity in time:** each area keeps a leaky state: ×0.5 per tick plus the
  new response.
- **Axon patterns:** each assembly projects through a fixed random pattern
  (256 signals), so an area's whole state can be read by another area.

### Binding vision to words

- After a moment, every assembly that was active during it strengthens its
  link to the word-forms of the answer (a running average of how often each
  word-form is heard while it is active).
- Each word-form keeps its own average activity (homeostasis). An area then
  raises a word-form by how much *more* it expects it than usual. It never
  lowers one; competition does that.
- A word-form is also raised when another word-form that shares its sounds is
  raised ("red?" when "red" is).

### Speech: the affinity / neurogenesis cortex (`organic/cortex.py`)

This is the original idea of the project.

- Input: only the echo of the question and the echo of what it has said so far.
  Vision acts only by raising word-forms.
- A cluster is 8 neurons with an identity (what situation it stands for).
- **Affinity to a problem** = how close the situation is to the cluster's
  identity × how much its outputs already lean to the word that came.
- **No affinity → neurogenesis:** a new cluster is born for the situation.
  Also when clusters that know the answer feel less familiar than the ones
  that got it wrong.
- A very familiar situation with a new outcome widens its expectations
  instead of growing.
- Only clusters with affinity learn: inputs move toward the situation, outputs
  toward what happened. Learning is scaled by surprise (a global signal) and
  slows down as a cluster gets used.
- Myelin per cluster: up when it helps a correct answer, down when it votes
  wrong.
- Word-forms that just fired are tired for a moment (fatigue).
- Sleep every 2000 moments: nearly identical clusters merge.

### Time: patience and reward (system 1 / system 2)

- Confidence of a plan = the probability of its least certain word.
- Every cluster has a *patience* (starts at 0.5). The plan is spoken when its
  confidence reaches the patience of the clusters that shaped it.
- After answering, a scalar reward: right on time = 1, late = halved per tick
  over the hidden deadline, wrong = 0, minus 0.02 per tick waited.
- Patience update: right but late → lower; wrong after answering early →
  higher; right and on time → slightly lower.

## Environment (the world the brain lives in)

- 40×40 scenes with 1-3 objects (6 shapes × 6 colors × 2 sizes × 9 places).
- Questions about a scene (color, shape, where, size, funny, yes/no, counting,
  logic, relations) and plain chat; answers are what a person would say.
- Hidden deadlines: color, size, where, "is it red?": 1 tick. Shape, what,
  funny, "am i ...": 3 ticks. Logic, counting, relations: none.
- The brain never sees the question type, the deadlines or any label.

## What has emerged so far

- **V1 edge and color-opponent cells** from waves and photographs.
- **Word-like units** from raw bytes.
- **Hasty and patient clusters (designed brain):** with hidden deadlines, the
  clusters that plan answers to fast questions became hastier (patience
  0.39-0.42 vs 0.45-0.47 without deadlines), and the ones that plan precise
  questions stayed patient (~0.6).
- **Growth bursts per new task (designed brain):** each new kind of question
  set off a burst of new clusters, while earlier tasks were kept.
- **A "where" area in the raw brain:** the gaze area came to predict place
  words about twice as strongly as anything else, with nothing telling it to.
- **Not emerging yet:** invariance (V1 memorizes views, ~24 assemblies per
  shape) and useful new areas (sprouting built runaway chains).

## Remaining debts (designed, not emerged)

| debt | where | note |
|---|---|---|
| patience update signs and "confidence = least certain word" | speech | chosen by the designer |
| the brain knows the 0.02 tick cost when judging "late" | speech | it should infer it from reward alone |
| fatigue (0.9, halves per tick) | speech | generic but hand-set |
| raising word-forms through shared sounds (cutoff 0.15) | binding | generic but hand-set |
| one area per organ stream | brain layout | follows the organ's outputs, not the tasks |
| imitation of the person's answer | learning signal | the main teacher; reward only tunes timing |
| cortex numbers: vigilance 0.8, familiarity 0.95, problem threshold 0.55, competition 0.04, learning 0.25, maturation 5 | everywhere | round values, but chosen by the designer |
| fixation length 3, 9 ticks per moment, decay 0.5 | time | hand-set |
| trace rule (off by default) and sprouting (off by default) | areas | see HISTORY.md |
