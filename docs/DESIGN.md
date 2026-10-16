# Design after the fly connectome

Current design of the loop brain (`organic/loop2.py`), and the finding that
motivated each piece ([FLY.md](FLY.md)). The previous version
(`organic/loop.py`) is kept for comparison.

```
                question (raw bytes) ──► ear ─────────────┐
                                                          ▼
 eye ──parallel channels──────────────────────► ORCHESTRATOR (clumps that grow by affinity)
  ▲   (coarse, medium, fine; noisy)                │          │            │
  │                                                │ commands │ answer     │ patience gate
  │  control ports:                                ▼          ▼            ▼
  ├── where   (gaze target map) ◄────── command library: 25 targets,   speak / wait
  ├── how long (stay)          ◄─────── stay, 8 gains, no gain
  └── which   (8 gain neurons) ◄──────  (efference copy of each command
  ▲                                       goes back into the orchestrator)
  └── reflex (organ-local): reacts to change in the coarse glance only
```

| fly finding | design |
|---|---|
| feedback = few neurons, broad reach, mostly inhibitory/modulatory | three control ports driven by the orchestrator: *where* (target map), *how long* (stay), *which* (8 gain neurons, each boosting a fixed, broad third of the eye's output channels) |
| direct eye -> motor shortcuts are few and for looming/collision | the reflex reacts only to *change* in the coarse glance (onset, motion, looming); in a still scene it is nearly silent |
| light reaches motor commands in 3-4 hops; breadth, not depth | the orchestrator reads the eye's parallel channels directly; no hierarchy of clumps between |
| descending neurons are a library of small command types that talk to each other | every eye command is an output neuron of the orchestrator, all chosen by competition and credited by one rule |
| drive to the eye's feedback neurons is shared with motor commands | efference copy: the last command issued is part of the orchestrator's situation |

Supporting change: the eye has photoreceptor noise, so *staying* on a spot
genuinely averages evidence. That gives "how long" a real value for fine
distinctions (shape) and none for coarse ones (color).

Learning, unchanged in spirit:
- **Commands** are learned only from the outcome (right/wrong, how long),
  through eligibility traces and a reward prediction error per command.
- **Answers** are learned by imitation.
- **Patience** is tuned by the outcome.
- **The orchestrator** grows clusters by affinity.

## Results: Yarbus test, design 1 vs design 2 (20,000 moments, 2 seeds each)

Design 1 (`organic/loop.py`), age 20,000:

| brain | color left / right | shape left / right | gaze -> asked side, color q's | gaze -> asked side, shape q's |
|---|---|---|---|---|
| both loops | 0.82-0.87 / 0.77-0.79 | 0.27-0.37 / 0.36 | 0.90-1.00 / 0.80-1.00 | 0.50-0.78 / 0.47-0.53 |
| goal loop only | 0.52-0.77 / 0.65-0.74 | 0.27-0.44 / 0.20-0.40 | 0.46-0.93 / 0.69-0.72 | 0.63-0.66 / 0.33-0.43 |
| reflex only | 0.45 / 0.47-0.56 | 0.27-0.32 / 0.18-0.30 | 0.30 / 0.60-0.65 | 0.48 / 0.41-0.45 |

- Question-dependent gaze emerged from outcome-only learning (eligibility
  traces, per-command prediction error). With both loops the eye goes to the
  asked side for color questions 90-100% of the time; the reflex-only eye keeps
  one habit whatever the question.
- Both loops beat either alone: the reflex explores early, the goal loop takes over.
- Shape stays weak (~0.3-0.4) even when the eye looks at the right side: the fine
  view / parts code does not separate the shapes well enough.
- Answers come late (tick 8-11 of 12), even for color.

Design 2 (`organic/loop2.py`) regressed: color 0.52-0.79, shape 0.18-0.35, less
consistent gaze, and the "stay" command was almost never used (0-0.1 per
moment). Candidate causes (untested): photoreceptor noise blurs weak shape
information; 35 commands instead of 25 adds early randomness and credit noise;
staying costs time at once while its payoff is rare and delayed.

## Does more training help? (long lives to age 80,000) and design-2 diagnosis

Long lives (2 seeds, average of the last three checkpoints):

| | color | shape | gaze -> asked side | clusters |
|---|---|---|---|---|
| design 1, age 20,000 | 0.80-0.83 | 0.32-0.37 | 0.74-0.75 | ~3,400 |
| design 1, age 80,000 | 0.74-0.82 | 0.30-0.37 | 0.66-0.77 | ~12,400 |
| design 2, age 20,000 | 0.58-0.78 | 0.23-0.30 | 0.50-0.60 | ~4,100 |
| design 2, age 80,000 | 0.57-0.69 | 0.18-0.25 | 0.62-0.71 | ~15,000 |

Four times more life did not help: design 1 plateaued by age 8,000-20,000;
design 2's gaze improved slightly, its answers did not. Both kept growing
clusters (3-4x) without improving: memorizing situations, not generalizing.

Design 2, one change at a time (20,000 moments, 2 seeds):

| design 2 with... | color | shape | gaze -> asked side |
|---|---|---|---|
| none of the new features | 0.65-0.68 | 0.24-0.34 | 0.51-0.70 |
| noise only | 0.56-0.71 | 0.26-0.28 | 0.38-0.59 |
| stay only | 0.46-0.71 | 0.34-0.35 | 0.32-0.55 |
| gains only | 0.55-0.74 | 0.20-0.32 | 0.51-0.67 |

No single new feature explains the regression; even design 2 without them is
below design 1, so the cause is shared: the change-only reflex (no early pull
toward the objects), the efference copy, or the fixation timing. "Stay" was
discovered by one seed (~2 per moment) but did not help.

Diagnosis: the ceiling is structural. Profiling showed 3,200 of 3,640
orchestrator clusters feel familiar with every situation; the question echo and
the overall look of the scene are shared by all moments, so affinity is not
selective, competition does all the work and growth turns into memorization.
Next: adaptation of the orchestrator's input (respond to what differs from the
usual, as the ear's habituation does), the contrast reflex back in design 2, then
the eye's shape perception.

## Input adaptation and the contrast reflex (20,000 moments, 2 seeds)

Adaptation: each input to the orchestrator slowly learns its own average
(over ~1000 moments) and passes on only the deviation from it (stimulus-specific
adaptation, like the ear's habituation). Contrast reflex: design 2 with design
1's static-contrast reflex instead of the change-only one.

Average of the last three checkpoints (chance 0.17; 12 ticks per moment, the
first fixation is ticks 1-3):

| | color | shape | answer tick, color | answer tick, shape | familiar fraction |
|---|---|---|---|---|---|
| design 1 | 0.80-0.82 | 0.32-0.35 | 8.6-9.8 | 10.5-10.6 | ~0.85 |
| design 1 + adaptation | 0.78-0.83 | 0.46-0.59 | 2.5-3.4 | 7.2-9.0 | 0.02 |
| design 2 + contrast reflex | 0.75-0.78 | 0.28-0.35 | ~11 | ~11 | ~0.85 |
| design 2 + contrast reflex + adaptation | 0.71-0.83 | 0.45-0.59 | 3.4-5.3 | 8.1-9.3 | 0.02 |

- System 1 / system 2 emerged in one structure: with adaptation, color questions
  are answered during the first fixation (ticks 2.5-5, from the coarse glance)
  and shape questions only after the eye has moved to the object (ticks 7-9).
  Nothing labels the question type or its deadline; it is the same patience
  mechanism, working now that affinity is selective.
- Shape accuracy rose from ~0.33 to ~0.5: non-selective affinity was the main
  bottleneck (familiar fraction 0.85 -> 0.02).
- The design-2 regression came from the change-only reflex: with the contrast
  reflex back it matches design 1.
- Cost: about twice as many clusters (~6,800 vs ~3,400).
- "Stay" is still never used.
