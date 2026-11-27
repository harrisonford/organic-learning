# An asynchronous brain: design (queue E 34; started 2026-11-26)

Kris, 2026-11-26: "the final brain architecture i expect it to be async in general with multi stimuli i hope we can
steer that direction as human brains process things". The test it is built toward is his: an app with camera,
microphone, screen and text chat, a conversation that keeps going (queue E 32).

## Kris's clarification (2026-11-26 15:15): neurons, not turns

"yes i want neurons and regions everything to be very async like the brain works like that right? is streams of
stimuli enter and streams of output also go out it can be lets say stream of image the brain moves fixates as it
feels driven to do and talks at the same time everything just happens so it shouldnt really be "turn" every neuron
ticks and send info that oculd be turn but that's the level of stream in stream out we want to achieve".

So the granularity is the neuron's, not the moment's: a small fixed step (10 ms) at which every population updates
its state (leaky integration with its own time constant) and sends on what changed; streams in (sound frames,
video frames, text) and streams out (eye movements, speech) at the same time; every output a population racing to
its own threshold (a saccade starts when its movement population crosses threshold: Hanes & Schall 1996, to verify),
so the eye moves when its drive wins and the mouth speaks when its drive wins, while perception goes on.
Turn-taking is not built in; it has to be learned from what speaking over someone brings. Rhythms (the syllable
rate, three or four fixations a second) coordinate loosely; nothing waits for anything else.
Cost: every synapse every 10 ms is out of reach on this CPU (the brain's 250 million numbers would be 25 billion
operations a simulated second); a brain has 1–2% of its neurons active and only activity travels, so only changes
propagate here too (as spikes do, and as neuromorphic chips work). Slower than real time on the CPU (Kris accepts).
**This replaces the migration below as the main road:** a small stream brain on a 10 ms clock, its organs as
populations from the start (eye, hearing, mouth, word cells), first shown to look, hear and vocalise at once on a
stream of Commons video with sound, before any claim about learning; then the learned pieces moved into it one by
one, each checked. The migration of the turn-based brain stays useful for comparison.

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

## References for the stream brain, checked against Crossref (2026-11-26)

Title, authors, year and venue confirmed by DOI lookup; what each is cited for is our reading and is to be checked
in the paper itself before it is quoted.
- de Boysson-Bardies, Hallé, Sagart & Durand (1989). A crosslinguistic investigation of vowel formants in babbling.
  *J. Child Language* 16, 1–17. doi:10.1017/S0305000900013404 — babbling vowels drift toward the ambient language.
- Hanes & Schall (1996). Neural control of voluntary movement initiation. *Science* 274, 427–430.
  doi:10.1126/science.274.5286.427 — a saccade starts when movement-related activity reaches a threshold.
- Rayner (1998). Eye movements in reading and information processing: 20 years of research. *Psychol. Bull.* 124,
  372–422. doi:10.1037/0033-2909.124.3.372 — fixation durations.
- Houde, Nagarajan, Sekihara & Merzenich (2002). Modulation of the auditory cortex during speech: an MEG study.
  *J. Cogn. Neurosci.* 14, 1125–1138. doi:10.1162/089892902760807140 — one's own speech is suppressed in auditory cortex.
- Klatt (1980). Software for a cascade/parallel formant synthesizer. *JASA* 67, 971–995. doi:10.1121/1.383940 — the
  vocal tract's resonators.
- Giraud & Poeppel (2012). Cortical oscillations and speech processing. *Nat. Neurosci.* 15, 511–517.
  doi:10.1038/nn.3063 — the syllable rhythm in auditory cortex.
- Murray et al. (2014). A hierarchy of intrinsic timescales across primate cortex. *Nat. Neurosci.* 17, 1661–1663.
  doi:10.1038/nn.3862; Hasson, Yang, Vallines, Heeger & Rubin (2008). A hierarchy of temporal receptive windows in
  human cortex. *J. Neurosci.* 28, 2539–2550. doi:10.1523/JNEUROSCI.5487-07.2008 — areas on their own time scales.
- Ladefoged & Broadbent (1957). Information conveyed by vowels. *JASA* 29, 98–104. doi:10.1121/1.1908694; Dean,
  Harper & McAlpine (2005). Neural population coding of sound level adapts to stimulus statistics. *Nat. Neurosci.* 8,
  1684–1689. doi:10.1038/nn1541 — hearing adapts to the voice and to the sound's statistics.
- Cowan (1984). On short and long auditory stores. *Psychol. Bull.* 96, 341–370. doi:10.1037/0033-2909.96.2.341 —
  echoic memory.
- Miller (1982). Divided attention: evidence for coactivation with redundant signals. *Cogn. Psychol.* 14, 247–279.
  doi:10.1016/0010-0285(82)90010-X — answers come sooner when two senses agree.
- Levinson (2016). Turn-taking in human communication. *Trends Cogn. Sci.* 20, 6–14. doi:10.1016/j.tics.2015.10.010 —
  replies about 200 ms after the turn ends, planned while listening.
- Zhang, Bao & Merzenich (2001). Persistent and specific influences of early acoustic environments on primary auditory
  cortex. *Nat. Neurosci.* 4, 1123–1130. doi:10.1038/nn745; Maye, Werker & Gerken (2002). Infant sensitivity to
  distributional information can affect phonetic discrimination. *Cognition* 82, B101–B111.
  doi:10.1016/S0010-0277(01)00157-3 — hearing develops from the sounds heard.

## Measured 2026-11-27 16:04: the stream brain keeps up with the clock; Kris's idea of a live page

**Speed** (CORe50 video at 20 frames a second, 10 s of life, two threads at Nice 15 beside a life and a probe; the
sight population coding every fixated window with the photo cortex and its relay): a 10 ms step takes 5.6 ms with
today's eye (1.8 × real time), 6.9 ms with the motion and pursuit reflexes (1.4 ×), 9.1 ms with hearing running too
(1.1 ×). So step 5 of the way above (wall-clock time, camera and microphone as streams) needs no slow-down factor on
this machine for what the stream brain has today. Not measured: with a vocal tract sounding, on another machine.

**Kris, 2026-11-27 16:00** (a question, not yet an order): could a small app on SpacetimeDB stream a camera, text and
sound to kid-a's brain in real time and let it do whatever it wants, "it doesn't matter if it's bad, I just want to
see hints of something forming"; SpacetimeDB so the page reacts well; hosting open. **Answered: yes for camera and
sound with the stream brain as it is; text has no place in the stream brain yet.** What a first page would show,
all of it already in `organic/stream.py`: where the eye is on the camera's picture (three or four saccades a second,
drawn by what stands out and what moves, following a moving thing between saccades), the babble through the vocal
tract (heard back through its own cochlea), the gestures drifting toward vowels it hears, the sight-and-sound link
filling in. What would form over a session: the map from gesture to sound, the babble's drift, the link. What it
would not do: name, answer, read. **Shape:** the page (TypeScript client) puts 64 × 64 frames at 10-20 a second and
50 ms pieces of sound into tables through reducers and subscribes to the brain's events (gaze, saccade, gesture,
sound out); the brain is a headless client beside the database. Open before building: no Python client was found in
a first search (C#, TypeScript and Rust were), so the brain needs a thin bridge process or a small port; whether
tables are the right carrier for sound at 100 frames a second; where the brain runs (it needs one or two cores and
about 1 GB, no GPU); a camera page must be private at first (kid-a's public page has no login). Nothing is
installed or built until Kris says. The steps that would make it worth watching are already in the queue: sounds
pulling the eye, the stream brain naming what it fixates, a byte ear for typed text as a third stream.
