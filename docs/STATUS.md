# Status (2026-10-21): where the project stands

Start here when picking the project back up. Details: [DESIGN.md](DESIGN.md)
(current brain and its experiments), [LOG.md](LOG.md) (scale-up phase),
[FLY.md](FLY.md) (connectome lessons), [STRUCTURE.md](STRUCTURE.md) and
[HISTORY.md](HISTORY.md) (earlier brains and the inventory of designed vs
emergent parts).

## The idea

A brain without backpropagation. Clusters of neurons grow when nothing has
*affinity to the problem*; only clusters with affinity learn, with local rules.
Organs (eye, ear) are fixed like evolution's gift and improved separately; the
rest should stay generic and let structure emerge. The two things under study:
**one structure for fast and slow thinking**, and **structure that evolves as
tasks are added**. Accuracy is not the goal yet.

## Current architecture (main line: `organic/loop2.py`)

```
 question (raw bytes) ─► EAR ─► echo of the question ──────────────┐
                                                                    ▼
 photo ─► EYE: per fixation, coarse / medium / fine views ─► ORCHESTRATOR (Cortex)
  ▲        (raw pixels, noisy, coded by a frozen bank of 64 parts)  │  clusters grow by affinity;
  │                                                                 │  input adapts (only what differs
  │   control ports ◄── commands: 25 gaze targets, stay, 8 gains ◄──┤  from usual gets through);
  │   reflex (inside the eye): pull toward static contrast          │  sleep merges / retires clusters
  │                                                                 ├─► answer (word-form)
  └── efference copy of the last command ───────────────────────────┤
                                                                    └─► patience gate: speak when sure
```

| part | what it does | file |
|---|---|---|
| ear | raw UTF-8 bytes; units where next-byte surprise jumps (BLT-style); fading echo of the question | `organic/ear.py` |
| eye | per fixation 3 ticks: whole image coarse, region medium, region fine (8x8 each), photoreceptor noise, frozen parts code; gaze map, stay, 8 broad gain neurons as control ports; contrast reflex | `organic/loop2.py`, `organic/peye.py` |
| orchestrator | `Cortex`: clusters of 8 neurons; affinity x outcome-compatibility decides who learns; neurogenesis when nothing fits; instar/outstar; myelin; patience per cluster | `organic/cortex.py` |
| input adaptation | each input neuron subtracts its own slow average (stimulus-specific adaptation) | `_adapt` in loop2 |
| learning signals | answers by imitation (what a person would say); eye commands and patience by outcome reward (right/wrong/late) via eligibility traces and per-command prediction error | `organic/loop2.py` |
| sleep | merges a new cluster into one with the same situation and the same expected outcome; retires long-idle, barely used clusters | `Cortex.consolidate` |
| adults | free models as people in the world: write and check questions, compare; capped and cached | `organic/adults.py` |

Alternatives kept for comparison: `organic/loop.py` (design 1), `organic/brain.py`
(the earlier designed brain), `organic/raw_brain.py` + `organic/retina.py`
(retina eye with developmental V1, tag `retina-eye-v1`).

## What emerged (no labels, no rules for it)

- **Fast vs slow in one structure:** on the left/right scene task, color is
  answered during the first fixation (~tick 3), shape only after the eye moved
  to the object (~tick 7-8); on real photos, presence ~tick 1, reading and
  identity late.
- **Question-dependent gaze (Yarbus):** the eye goes to the asked side (up to
  0.88-0.90 for shape questions) learned only from outcomes.
- **Selective affinity** once inputs adapt: familiar fraction 0.85 -> 0.02,
  shape 0.33 -> 0.5, rising to 0.62-0.73 with longer life alone.
- **Growth bursts** when a new kind of task arrives (earlier brain).
- **Reading** words on photos: 0.31-0.40 vs a 0.02 baseline.

## What is not working

- **Real photos:** at 64x64 the eye's output carries no usable information
  for "is there a person?", "biggest thing", "where" (nearest-neighbour probe
  at or below baseline, for this eye and for the retina eye). After 300,000
  moments the brain is still at baseline on those. Needs a visual cortex
  between eye and orchestrator (proposed, not built).
- **Growth never levels off:** ~30,000 clusters at 300,000 moments; new
  clusters are distinct memories, so sleep trims only ~7%.
- **"Stay" is never used.**

## Running now

- COCO annotation daemon (restarted 2026-10-19 after a crash on 10-18):
  ~1,500 checked items so far; `data/world/coco_qa.jsonl`.
- Text world daemon (spoken Q&A): 970 pairs; `data/world/text_qa.jsonl`.
- Photo-world lives finished at 300,000 moments (`out/world_s*.pkl`).

## Budgets

OpenRouter: <=10 requests/min, <=500/day (photo 340, text 100, reserve 60);
Jev capped at $1.50 (spent $0.035). All long jobs run with `setsid nohup`.

## Latest (2026-10-21)

Visual cortex built (`organic/visual_cortex.py`): three self-organizing
layers between eye and orchestrator. Long lives finished (1M and 500k
moments, ~1.19B and ~0.76B numbers). On one fixed test set the cortex brains
are better at color, identity and gist, slightly worse at reading and where;
presence, gist and counting remain below baseline. Most learning happens in
the first 50-100k moments. Details in [LOG.md](LOG.md).
