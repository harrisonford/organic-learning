# Running log (scale-up phase)

Plan agreed 2026-10-17: free models act as adults in the brain's world
(write and check questions, grade, compare), never inside the brain; half of
the account's limits (<=10 req/min, <=500 req/day); Jev capped at $1.50;
everything detached and resumable; long lives.

## 2026-10-17

- **Adults** (`organic/adults.py`): rate-limited, cached OpenRouter client;
  per-purpose daily shares: photo annotation 340, text world 100, reserve 60.
  Nemotron 3.5 Super VL writes, Dots 3 Note checks (both free, see images).
  Inkling Small free is gated to agent apps (403), so it is not used. Jev is
  `typesafe/jev-router`: a router running on TypeSafe's system-one model that
  can pick paid models, so it is capped.
- **Pilot:** 4 COCO photos at the eye's resolution, 5 questions each: the two
  adults agreed on 19/20 answers.
- **Daemons running:** COCO photo annotation (~700 photos/day), text world
  (spoken Q&A for the ear, 50 pairs per request).
- **World** (`organic/world.py`): COCO val2017 photos at 64x64 with questions
  from human annotations (presence, counting, where, biggest thing), adult
  questions (agreed items only), reading (words drawn on photos), eyes-closed
  talk, and CIFAR-100 once downloaded. Stable photo-level test split;
  kind-balanced sampling; hidden deadlines per kind.
- **Sleep consolidation** (`Cortex.consolidate`): merges a new cluster into an
  existing one only if they stand for the same situation *and* expect the same
  outcome; retires long-idle, barely used clusters. Finding: new clusters'
  nearest neighbours have similar identities (median 0.90) but different
  expected outcomes (output similarity 0 in >90% of pairs), so growth is not
  duplication; merging is rare by design and retirement does most of the work.
- **Runs:** sleep-cleanup lives on the Yarbus task (design 2 + contrast reflex
  + adaptation, 80k moments, 2 seeds); photo-world lives (300k moments,
  2 seeds); counterpart benchmark scheduled for tomorrow's reserve.

### Photo world, first checkpoints (age 10,000-20,000 of 300,000)

Baseline = always giving the most common training answer for that kind.

| kind | brain | baseline | answer tick |
|---|---|---|---|
| reading (48 words) | 0.18 -> 0.39 | 0.02 | ~10-11 |
| presence | 0.61-0.66 | 0.61 | ~1.1 |
| where is the person | 0.38-0.52 | 0.49 | 3-4 |
| counting | 0.49-0.51 | 0.56 | 3-4 |
| identity ("biggest thing", 83 answers) | 0.20-0.23 | 0.32 | 7-8 |

Reading is clearly learned early; presence is answered almost instantly and
reading / identity late, matching their hidden deadlines. Other kinds are at or
below the majority baseline so far. Color, gist, activity and talk have too
few test items yet (3-5) to judge; they grow as the adults write more.

## 2026-10-18

### Sleep cleanup (design 2 + contrast reflex + adaptation + sleep, age 80,000)

| | color | shape | gaze -> asked side (shape) | clusters |
|---|---|---|---|---|
| without cleanup (seeds 0/1) | 0.78 / 0.83 | 0.68 / 0.62 | 0.82 / 0.90 | 17,930 / 16,266 |
| with cleanup, seed 1 (age 80k) | 0.78 | 0.61-0.64 | 0.88-0.89 | 15,202 |
| with cleanup, seed 0 (age 74k) | 0.78 | 0.54-0.68 | 0.62-0.86 | 17,634 |

Cleanup trims growth modestly (seed 1: -7%) without hurting accuracy; it
cannot stop growth, because new clusters are distinct memories, not copies.

### Counterpart benchmark (80 held-out items, 4 per request)

| model | presence | count | identity | where | reading | s / request |
|---|---|---|---|---|---|---|
| Nemotron 3.5 Super VL (free) | 0.94 | 0.69 | 0.31 | 0.00* | 1.00 | 17.1 |
| Dots 3 Note (free) | 0.94 | 0.06 | 0.31 | 0.00* | 1.00 | 16.2 |
| Jev router (paid flash models) | 0.81 | 0.38 | 0.25 | 0.00* | 1.00 | 5.1 |

*"where" scores 0 only because the models answer "left" and the expected
answer is "on the left" (exact matching); not a real failure. Jev routed to
DeepSeek / Gemini / GLM flash models and cost $0.035 for 20 requests.

### Photo world at age 100,000: not learned

| kind | seed 0 / 1 | baseline |
|---|---|---|
| presence | 0.51 / 0.52 | 0.61 |
| counting | 0.46 / 0.49 | 0.56 |
| identity | 0.21 / 0.19 | 0.32 |
| where | 0.47 / 0.42 | 0.49 |
| reading | 0.24 / 0.39 | 0.02 |

Diagnosis (nearest-neighbour probe on the eye's output): for real photos the
parts-code eye carries no usable information for presence, identity or where
(probe at or below baseline); only reading (0.32). The eye organ, built for
coloured shapes on black, is the bottleneck: the brain cannot learn what the
eye does not deliver.

Same probe on the retina eye with its developed V1 (stored as `retina-eye-v1`):

| question | baseline | 1 fixation | 3 fixations |
|---|---|---|---|
| is there a person? | 0.57 | 0.57 | 0.57 |
| reading | 0.02 | 0.36 | 0.36 |
| biggest thing | 0.26 | 0.14 | 0.14 |
| where is the person | 0.44 | 0.35 | 0.34 |

No better. For real photos, no raw retina-like code (with or without V1)
makes these questions separable by similarity. In animals that job belongs to
a deep, wide visual cortex grown during development (V1 -> V2 -> V4 -> IT; the
fly's 77,000-neuron optic lobes); this brain has nothing between the organ and
the orchestrator. Proposed: let such a hierarchy self-organize on natural
photos with the same growth and Hebbian rules.
