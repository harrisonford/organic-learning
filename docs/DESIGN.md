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
