# An asynchronous brain: design (queue E 34; started 2026-11-26)

Kris, 2026-11-26: "the final brain architecture i expect it to be async in general with multi stimuli i hope we can
steer that direction as human brains process things". The test it is built toward is his: an app with camera,
microphone, screen and text chat, a conversation that keeps going (queue E 32).

## What is synchronous today

A life is a sequence of moments, and a moment is a closed turn (`organic/speech.py`, `SpeakingBrain.live`):
1. the whole utterance reaches the ear at once (`ear.units(question)`), before anything is seen;
2. then ticks of a fifth of a second: the eye samples the picture (coarse, medium, fine), the gate weighs whether
   to speak (confidence against its learned patience), the serial gaze (behind a switch) moves once a tick;
3. speaking takes a tick a unit; the reward is discounted by the ticks used;
4. the world waits for the reply, judges it, and the next moment starts from a new picture; traces carried between
   moments decay by fixed factors per moment, not per unit of time.

So hearing is not spread in time, the senses never overlap in time, nobody can be interrupted, and silence has no
length. Learning, on the other hand, is already local everywhere: no part waits for a backward pass from another.
That is what makes an asynchronous brain possible here, where a backprop network needs its forward and backward
passes in lock step.

## The design

**One timeline in milliseconds.** The world and the brain share a clock. Streams deliver events at their own
rates; each organ updates when its own events arrive; everything a brain does (a look, a unit said, silence) is an
event in the same queue, and the world answers it in its own time. On this CPU the queue is a discrete-event
scheduler ordered by time: asynchronous in meaning, run slower than real time; tied to the wall clock later
(GPU or cloud), with the camera and microphone as two of its streams.

**Rates and time constants from biology** (all to verify before they are cited):

| stream or organ | rate / time constant | source to check |
|---|---|---|
| cochlea frames | 10 ms | organic/cochlea.py (a standard frame rate) |
| hearing cortex windows | 80 ms | as built (experiments/sound_units_probe.py) |
| syllables | 4–8 a second | Giraud & Poeppel 2012 |
| eye fixations | 3–4 a second (200–300 ms), saccades 20–50 ms | Rayner 1998 |
| video frames | 10 a second (100 ms) | experiments/world/commons_video.py |
| sensory traces | tens of ms; association areas seconds | Murray et al. 2014; Hasson et al. 2008 |
| a reply after the other's turn ends | about 200 ms (planned while listening) | Levinson 2016 |

**Traces decay with time, not with steps.** Every per-tick or per-moment decay becomes exp(−Δt / τ) with τ from the
table: the eye's accumulated view, the ear's echo, the word cells' context, the gate's evidence, the store's fading.

**The gate decides at any time.** Its evidence keeps accumulating as events arrive; when confidence passes patience
it acts: speak, look, keep quiet. It may start before the other has finished (and learn from what that costs), and
it may be interrupted.

## The way there, one switch at a time (today's lives unchanged)

1. **Hearing in time within a moment.** The utterance's units arrive one a tick while the eye looks; the gate may
   answer before the question ends. Measured: how long answers take, how often it speaks over the adult, whether
   the score holds.
2. **No resets.** A picture stays until the world changes it; the adult speaks when the world has something to say;
   silence lasts; traces decay with time.
3. **Real streams.** Sound frames every 10 ms through the cochlea and the hearing cortex (the ear's units found in
   sound: E 32), video frames every 100 ms, fixations on the current frame.
4. **Organs on their own clocks.** Each organ updates only on its own events, with its own τ.
5. **Wall-clock time.** The scheduler tied to real time times a slow-down factor; camera and microphone as streams.

**Measured at every step:** the latency of answers; what each stream adds, by taking it away; and whether answers
come sooner when sight and sound agree (the redundant-signals effect in people: Miller 1982, to verify).

## Organic check

There is no global clock in a brain; areas integrate over a hierarchy of time scales, fast in the senses and slow in
association cortex; speech is parsed at its syllable rhythm; the eyes sample a scene three or four times a second;
people plan their reply while still listening. Nothing in the design is a new learning rule: it changes when the
existing rules are applied, and makes time itself something the learner lives through. What is ours and not a
finding: the rates in the table until each is checked, and the slow-down factor on this CPU.
