# Speaking: a speech motor area built by babbling (design, 2026-10-25)

This is the first architecture step the curriculum calls for (stages B, W1, W2; see [CURRICULUM.md](CURRICULUM.md) and [LANGUAGE_BIOLOGY.md](LANGUAGE_BIOLOGY.md), gaps 1 and 2).

Today the brain answers by choosing one output neuron per whole answer ("word:on the left"). It cannot repeat a word it has never said, and it cannot put words together.

## Biology it follows

- **Infants babble before they speak** (canonical babbling, 6-10 months). Producing sounds and hearing them builds a map between articulations and their sounds: an inverse model, so you know how to make a sound you hear. Deaf infants' babbling fades.
- **Repetition runs on that map** (the dorsal stream: auditory cortex -> Spt -> arcuate fasciculus -> premotor/Broca). New words can be repeated because they are made of known sound pieces. Nonword repetition is a standard test of it in children.
- **Speech is produced in time, one gesture after another.** Speakers hear themselves and use an efference copy to know what they have just said. Frequent sequences become motor chunks ("thank you" is one gesture for an adult).
- **Babbling drifts toward the sounds of the language heard**, before words.

## Design

```
                 ┌──────────── self-hearing (the ear hears what was said) ───────────┐
                 ▼                                                                   │
heard units ─► EAR ── auditory echo ──► ORCHESTRATOR ── chooses ──► SPEECH MOTOR AREA ─► bytes out
                 │                         ▲   (one motor unit per tick)  │   motor units = byte gestures,
                 │                         │                              │   chunks grow from use
                 └─► AUDITORY-MOTOR MAP ───┘ (primes the motor units      │
                     (learned by babbling)    that would say what is      │
                                              echoing: the copy route)    │
                                   efference copy of what I just said ◄───┘
```

1. **Motor units.** Each produces a short byte sequence: at birth, one per byte that occurs in speech. Motor chunks grow like the ear's units: a sequence of motor units said together often becomes one unit. So "thank you" can become one gesture. A motor-unit budget replaces the answer budget.
2. **Babbling (stage B, before any question):** spontaneous, noisy activation of motor units. The ear hears the result. A Hebbian link forms between each motor unit and the auditory code of what it produced. That is the auditory-motor map, learned and not wired. Babbling is biased toward units whose sound resembles recently heard speech, so it drifts toward the language heard.
3. **Copy route:** the auditory echo of what was just heard primes, through the map, the motor units that would produce it. The orchestrator chooses among motor units with this priming added. So a heard word can be said without ever having been an answer.
4. **Speaking in time:** after deciding to speak (the patience gate, unchanged), the brain says one motor unit per tick. The situation at each tick includes the efference copy (`Ear.said`, already there) and self-hearing. It stops when it chooses silence. Long answers take longer, so speech duration joins the fast/slow mechanism naturally.
5. **Learning:**
   - *Imitation:* the adult's answer is heard and segmented by the ear; at each tick the target is the next unit.
   - *Reward:* the curriculum's grader scores the whole utterance (lenient early).
   - The orchestrator's clusters are unchanged: they now predict "what to say next" given the situation and what has been said so far.

## What stays the same

Eye, cortex, orchestrator rules (affinity, growth, sleep, budgets), patience, the eye's command learning, the world. The `word:` outputs are replaced by motor units. Everything else reads the same signals.

## How it will be tested

A controlled pair of lives through curriculum stages B-W2 in one frozen world: today's whole-answer brain vs the speech-motor brain.
- **Babbling drift:** the sound similarity between babble and heard speech, over stage B.
- **Nonword repetition:** "say blick!", with words never heard. Today's brain cannot do this by construction.
- **First words and two words:** stage scores, exact and lenient.
- **Reason items with the answer in the question** (later stages): the copy rate.

## Risks, said plainly

- Choosing bytes one by one is slow to learn at first. Chunks should form, but they may form slowly. Like infants, it may need a long babbling stage.
- With one motor unit per tick, long answers need many ticks. `max_ticks` (12) may need to grow for grade-1 sentences. That changes the timing of everything, so it would be its own controlled comparison.
