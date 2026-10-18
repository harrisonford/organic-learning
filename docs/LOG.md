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
