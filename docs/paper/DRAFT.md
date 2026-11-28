# Organic Learning: Small-Scale Experiments Toward Organic Intelligence, Without Backpropagation

**Draft 12, 2026-11-28, rewritten to Kris's brief of that morning. Not for circulation.** This note is not part of
the paper (the build leaves it out). Kris's words: the draft uses "organic learning" and "organic intelligence";
the line "it was the code, not the rule" is out; "as visual as possible and concise"; the real-time work is in.
What changed against draft 11 (22 pages, 14,500 words, five figures): about a third of the words, nine figures,
one claim a figure; the hearing results and the 10 ms clock are new sections; the staged childhood is one
section; every table, control and setting of draft 11 is kept as `SUPPLEMENT.md` beside this file. The title
keeps Kris's (2026-11-23) and adds the second word; others to choose from: "Organic Learning: An Ear Born With
Nothing Prepared"; "Born, Not Fitted: Small Experiments Toward Organic Intelligence". Every number is from a log
in `out/`. **[pending]** marks a run that is not finished; **[check]** marks a reference written from memory.
References are tracked in REFERENCES.md and each must be opened by hand before anything is sent.

---

## Abstract

*Organic learning* is our name for a way of building a learner: only mechanisms that brains are known to use,
only local learning rules, everything learned in one pass and in the order it arrives, and nothing installed that
could be grown instead. *Organic intelligence* is what we hope such learning adds up to. This paper reports how
far a small one gets on a desktop machine, in three results. **Seeing.** Shown CIFAR-10 once, the learner names
69% of held-out pictures where backpropagation of the same shape names 62%; but when we took the learner apart,
the win belonged to a single decorrelation step, and given that step backpropagation draws level (69% each; on
real photographs 45% against 46%). **Hearing.** An ear that starts from random numbers, hears each spoken
caption once beside its photograph, and is never given text, finds the right caption for an unseen photograph
among 1,000 within its first ten guesses 6.0%, 8.6% and 9.5% of the time for three speakers (chance is 1%).
That is as good as the same ear with its cortex and relay prepared in advance, and it is the ceiling that
written words reach through the same eye. **Real time.** The same ear runs on a 10 ms clock with the same
output, and is live on a web page behind a camera and a microphone. We also report what did nothing (replay, a
second layer, a staged childhood that stalls after picture books), six assumptions of ours that were wrong, and
the limits: one seed so far for the hearing results, a visual cortex that is still prepared in advance, and
scores far below those of trained speech-and-image models.

## Introduction

Brains learn under constraints that deep learning does not accept. A synapse changes according to what its own
two ends are doing, nothing sends an error signal backwards through the layers, no moment of a life comes round
a second time to be studied again, and nothing waits: sound arrives while the eye is still busy with the last
thing it saw. Whether local learning rules can stand in for backpropagation has mostly been asked at scale, and
at scale the answer so far is "not yet".

We asked something smaller and slower. We assembled a learner the way one might assemble a small animal, an
eye, an ear, a relay between cortex and memory, cells that learn words, a mouth that babbles, and admitted a
part only if brains are known to do something like it. We call the method **organic learning**, and since we
would like the name to mean something, here is what it commits us to:

1. **Biology admits the parts.** A mechanism goes in only if brains are known to use something like it.
2. **Local rules, no backward error.** Every synapse changes by what its own two cells are doing.
3. **Once, in order, on a clock.** Each moment is met one time, in the order the world delivers it.
4. **Grown, not installed.** No behaviour is wired in, nothing the learner happens to do is turned into a
   target, and a benchmark is a yardstick and never a goal.
5. **Everything is taken apart again.** Each part is removed to see whether it mattered, and what did nothing
   is reported beside what worked.

**Organic intelligence** is the thing these commitments aim at: one system, born once, that sees, hears and
answers in real time, and whose abilities were grown in it. We have not built one. What we can report is how
far a small one gets, and three places where it surprised us.

The first surprise was unwelcome. Our learner beat backpropagation at learning to see from a single pass, and
when we took it apart to find out why, the credit did not go to the local learning rule we were proud of. It
went to a decorrelation step that we had computed in a batch, the least organic thing in the learner. The
second surprise repaid the first. Having learned that the prepared parts were doing the work, we began taking
the preparation away, expecting to pay for it each time. In hearing we paid almost nothing: an ear with nothing
prepared, every part of it starting from random numbers and learning from the first sound, does as well as the
ear we had carefully prepared. The third is that none of this needed to stand still. The same ear runs on a
10 ms clock.

Figure 1, `fig_organism.pdf`, shows the learner as it stands, with each part coloured by how it comes to be
what it is. The honest reading of that figure is the paper in small: the whole hearing path is now born with
the life, and the seeing path is not yet.

## Seeing in one pass: who earned the win

The eye is one convolutional layer of 256 filters over small colour patches, trained without labels by
winner-take-all Hebbian learning; it passes on 4,096 numbers. Between the eye and whatever reads it sits a
**relay** that decorrelates those numbers a second time (in a brain, learned lateral inhibition would do this;
Földiák 1990). The reader is one layer of plastic synapses, our *word cells*, each learning by the delta rule
to predict its own firing. No error ever travels back through a layer.

We showed this learner every CIFAR-10 training picture exactly once and tested on held-out pictures along the
way, against a backpropagation network of the same shape tuned for one pass. Figure 2, `fig_benchmark.pdf`,
gives the curves. The learner is ahead at every point: by 16 points after 500 pictures and by 7 at the end
(68.7% against 61.6%, three seeds). Backpropagation needs four old pictures replayed for every new one, five
times the updates, to catch up.

Then we removed parts. Figure 3, `fig_credit.pdf`, is the result that changed the project. Without the relay
the order flips: backpropagation is ahead, 61.6% against 59.1%. With the relay in front of *its* readout,
backpropagation reaches 68.8%, level with ours. On real photographs at the eye's own size (33 words, 50,000
photographs seen once) the same thing happens with the sign reversed at the end: level without the relay
(38.9% and 39.2%), and with it backpropagation is two points ahead (46.3% against 44.6%), though ours leads
early in the stream. Three further controls point the same way:

- A readout trained by gradient on the same decorrelated code does exactly as well as our local one (69.4%).
- Filters that were never trained do as well as the Hebbian filters we spent weeks on (69.3%). This is known
  (Coates, Lee & Ng 2011; Thiry et al. 2021) and we rediscovered it the expensive way. It is our rule's
  failure and not Hebbian learning's: the soft competition of Journé et al. (2023), built from their code,
  develops filters that take the same reader to 73%, where backpropagation of that layer's shape reaches 56%
  from a random start and 70% when handed the developed filters.
- A streaming discriminant that keeps second moments beats us by five points on our own code (Hayes & Kanan
  2020). Anyone who only wants accuracy from one pass should use that.

Why should one decorrelation matter so much? A delta rule that meets each example once learns quickly along
the few strong directions of a correlated code and hardly at all along the weak ones, and the weak ones are
where a cat differs from a dog. Decorrelating makes every direction equally easy. That is textbook
optimisation, and others have shown how far it reaches (Prabhu et al. 2024; Dalm et al. 2024). What we add is
the view from the other side: a learner built on purpose from local rules, whose builders credited the rules,
taken apart until the credit sat where that literature says it should.

So the most important part of the learner was the one part that broke our own commitments, a matrix computed in
a batch before learning began. A published circuit of inhibitory interneurons with a purely local rule
(Lipshutz, Pehlevan & Chklovskii 2022; Duong et al. 2023b) does the relay's work to within a point on a fifth
of the stream, and unlike the batch matrix it follows the learner into a new world: a relay fitted on
photographs did not hold on video of handheld objects, and the same circuit, simply left learning, recovered
half of what was lost. That circuit is what made the next section possible.

## An ear born with the life

If prepared parts were earning the results, the organic thing to do was to stop preparing them. We tried it
where the learner was newest: hearing.

**The task.** People reading picture captions aloud (SpokenCOCO; Hsu et al. **[check]**), heard beside the
photograph each caption describes. The ear is never given text; transcripts are used only to measure. One
*life* is one speaker's captions, 8,000 to 9,700 of them, about nine hours of speech, each heard once in one
order. Afterwards we take 1,000 photographs never seen and their captions never heard, and ask, for each
photograph, where its own caption ranks among the 1,000. We report how often it is in the first ten; chance is
1%, and one such number is good to about ±0.8.

**The ear.** Sound enters through a fixed cochlea, 64 frequency bands every 10 ms. Each band is set against
its own recent past (the last half second), as auditory neurons adapt to the level and contrast of what they
hear (Dean, Harper & McAlpine 2005; Rabinowitz et al. 2011). A hearing cortex of 128 detectors looks at 80 ms
patches; a relay of interneurons evens out its code; the stream is cut at the dips of its own loudness into
stretches about a syllable long; and every run of two or three stretches is claimed by one of 2,048 *sound
units* in a winner-take-all competition. A sound unit that fires is tied to whatever the eye holds at that
moment by Hebb's rule, and that is all the "learning to understand" there is.

In our first version three of those parts were prepared before the life: the hearing cortex was developed on
1,000 captions with a learning rate that fell on a schedule, the relay was a matrix computed from them, and
each caption was normalised whole before it was heard. We removed these one at a time, and each time we wrote
down beforehand how much we expected to lose.

Figure 4, `fig_ear.pdf`, shows what happened. Knowing only the past costs little (nothing for one
speaker, a fifth for another), provided the ear adapts fast: over half a second, not thirty. A relay that is a learning circuit costs about a point. And an ear in
which the hearing cortex too is born with the life, its detectors learning by the same winner-take-all rule
with each detector slowing as it is used, and its input evened by the same kind of circuit, ends at 6.0%,
8.6% and 9.5% for the three speakers, against 5.7%, 9.0% and 9.5% for the fully prepared ear. With captions
tied to the wrong photographs the same ears score 0.8% to 1.4%. We had predicted a loss of up to a quarter.
There was none that 1,000 test pairs can show. The schedule, the thousands of rounds and the batch matrix
bought nothing that a local circuit and a rate that falls with use do not give. **[pending: seeds; these are
one seed per speaker.]**

Three more things are worth knowing about this ear. Its detectors come to rest without being told to: after
2,500 captions they no longer move between one reading and the next. It is at a ceiling that is not its own:
the captions' *written* words, pushed through the same eye and the same Hebbian link, reach 8.8%, so what
limits the score is how well this small eye tells photographs apart. And although no text ever reaches it, a
linear readout of the sound units a caption fired finds that caption's own transcript among 1,000 within the
first ten 74% to 85% of the time: the units carry words.

It is also far from good. Speech-and-image models trained by backpropagation retrieve captions many times more
often on harder versions of this test (Harwath, Torralba & Glass 2016 and its successors **[check]**), and an
ear raised on one voice finds almost nothing in a voice it has never heard (between 1% and 3.5%). Which raises the
next question.

## A second voice

What happens when the speaker changes in the middle of a life? We let the ear hear one speaker for 2,500
captions and then another for 2,500, with three pairs of speakers, and compared a relay that keeps learning
with a relay fitted to the first speaker and then fixed. Figure 5, `fig_voices.pdf`, shows the three pairs.

With the learning relay the ear picks up the second speaker about as well as if it had been born to that
speaker (0.73 to 1.02 of what that voice reaches alone) and still finds most of what it had tied for the first
(0.74 to 1.00). With the fixed relay the outcome depends on how alike the two voices are: nearly locked out
when they differ (0.41), unharmed when they are alike (1.04). The sound units show why. Through a relay that
follows the voice, nearly every unit ends up serving both speakers. Through a relay fixed to somebody else,
the new voice lands elsewhere and has to claim units of its own. We expected the opposite, and we expected the
first voice to be forgotten; neither happened. These are one seed per pair, and ratios of numbers this small
are loose, so we draw only the modest conclusion: the part that adapts fastest is what lets the slower parts
be shared. **[pending: seeds.]**

## On the clock

Everything in the last two sections can be computed one caption at a time, which is not how sound arrives. So
we rebuilt the ear to take one 10 ms row of the cochlea at a time and to know nothing but the past.
Figure 6, `fig_clock.pdf`, follows one sentence through it. The loudness dips are known 50 ms after they
happen, a sound unit about 70 ms after its last stretch ends, and each unit that fires leaves a trace that
fades over about a second; while the trace lasts, its synapses onto whatever the eye is holding grow. No
caption boundaries are needed any more, only time.

Three checks. Fed the same captions with a pause between them, the ear on the clock produces exactly the same
sound units as the offline ear, 40 captions out of 40. Inside a brain that also has an eye and a mouth
running on the same clock, with every part of the ear born with it, it reaches 3.3% after 2,500 captions and
6.0% after 5,000, against 3.9% and 6.1% for an ear prepared in a finished life. And it is fast enough: a row
of sound passes through the ear in under 2 ms, and a whole 10 ms step of the brain takes about 4 ms on an idle
desktop processor.

That brain is live on a private web page. A browser streams its camera, its microphone and what is typed
through a database to the brain; the page shows where the eye is looking, what the brain babbles, and which
past view a sound reminds it of. Whoever speaks to it first is the voice its hearing forms on. We have tested
the page with recorded speech played as the microphone, and not yet with a person in a room, where there is
hum, echo and its own voice coming back from the loudspeaker. Typed letters are a third stream on the same
clock: runs of three and four letters, with no words cut out, tied to sight in the same way; on the captions as
written that finds a caption's photograph as well as whole words do (9.3% against 8.8%). Nothing from the camera
or microphone is stored.

## Raising it: a staged childhood

Before any of this, the learner was raised in a small world of real photographs and of speech written by
language models playing adults: babbling, then echo games, then first words, then picture-book pages on which
several things appear and it must look at one. Answers are judged for meaning by a small local model, and
every score is reported above the best blind strategy, on a scale where ignoring the picture scores 0 and an
adult scores 1.

Figure 7, `fig_life.pdf`, follows one life. For 5,000 moments the learner only babbles ("ather cathay");
within a few thousand moments of being asked for names it is giving some; both curves level off near 0.4. The
babble, which nobody rewards, drifts toward the world's words: "can you bus" at 110,000 moments.

Figure 8, `fig_examples.pdf`, shows what it says at the end, on the first six test items of each kind, not
chosen. A mouse close up is a "lion"; a page with a horse and a cow gets "cowbanana".

Figure 9, `fig_ablation.pdf`, removes one part at a time, three lives each, and repeats the benchmark's
verdict in the learner's own world. The relay is the only part whose removal costs anything on single pictures.
Looking is what page naming depends on (0.36 falls to 0.02 without it). And the episodic store, a fast memory
for single moments, gives one-shot memory of an instance and never of its category: named once on a page, a
thing is recalled on another page 85% to 92% of the time if the memory is filed under what the eye was on, and
a different example of the same thing brings back nothing.

Two honest ends to this section. The childhood stalls: in the stage after picture books no life rose above
blind answering (0.00 to 0.02), and earlier stages slid by about a tenth while it tried. And the stages
themselves are ours. The real-time learner of the previous section has none: it is born and it learns from
whatever arrives. We now think of the staged world as a measuring ladder for the offline learner, not as the
way an organic intelligence should be brought up.

## What did nothing, and what we got wrong

A good deal of what we built contributed nothing we could measure: replay of stored moments, awake and in
sleep; a second cortical layer (+0.01); dendritic branches on the word cells; and our own
Hebbian filters on enlarged pictures.

More useful are the assumptions that turned out wrong. The first five made the learner look better or busier
than it was, or a part of it worse; the sixth is the one this paper is built on.

| What we assumed | What happened |
|---|---|
| A behaviour can serve as a gate: keep 80% of the echo score to move on. | Learners that had begun to use words echoed less, as children do, and we held them back for up to 90,000 moments. |
| A "novelty drive" that makes the learner repeat unknown words is harmless help. | It got learners through stages, and it was our rule and not their behaviour. Removed; every life rerun. |
| A lenient judge can only cost precision. | It accepted "bear" for a dog, the adult then repeated the child's word, and 7% of lessons taught the learner its own error. |
| Reward can teach the eye what to trust. | The weight we were pleased with rose just as far when its signal was shuffled. The answer was timing, and it was in the literature. |
| Enlarging a picture to fit the eye is harmless. | Half of what a small window can show is gone; our development "did nothing" only on pictures with half their detail missing. |
| The ear has to be prepared: a developed cortex, a fitted relay, a schedule. | Each piece was removed with the loss written down beforehand. The relay cost about a point, as predicted. The cortex, expected to cost up to a quarter, cost nothing for any of three speakers. |

## Limits

One desktop machine, small codes, small tests. The hearing results rest on three speakers of one corpus and one
seed each, on 1,000 test pairs, and on read speech recorded cleanly. The visual cortex and its relay are still
prepared before the life, without labels, on photographs that may include those later used as tests; "born
with the life" is true of hearing only. We compared against one backpropagation architecture, plainly trained.
The sizes of everything are set by hand and none comes from anatomy. The live page has not met a real room.
In the staged world the stages, the adults and the judge are ours or language models, and no person has yet
labelled the judge's decisions. Nothing here says any of it holds at scale.

## Related work

*Fixed features, one trained layer.* What the seeing section finds belongs to a known family. Random whitened
filters are hard to beat among single-layer learners (Coates, Lee & Ng 2011; Saxe et al. 2011); one layer of
whitened patches with a linear classifier reaches 86.7% on CIFAR-10 and depends on the whitening (Thiry et al.
2021); random features decorrelated and read by class means match continually trained deep networks in one
pass (Prabhu et al. 2024); and frozen features with a closed-form readout are now a standard strong baseline
(McDonnell et al. 2023; Zhuang et al. 2022, 2024). Our streaming-discriminant row is that baseline, and it is
ahead of our learner.

*Why decorrelation helps.* That correlated inputs slow gradient learning is old (Le Cun, Kanter & Solla 1991)
and is now a design principle (Dalm et al. 2024; Offergeld & Ahmad 2024; Jordan 2024). The benefit is not
special to local rules, which is what our like-for-like baseline measures.

*Decorrelation in brains, and the circuit we use.* Populations can whiten their responses online with
interneurons whose gains adapt quickly and whose synapses change slowly (Duong et al. 2023a, 2023b); the
circuit we use is from that line of work (Lipshutz, Pehlevan & Chklovskii 2022), and a relative of it predicts
the wiring of the fly larva's olfactory system (Chapochnikov et al. 2023). Hebbian learning with anti-Hebbian
lateral inhibition goes back to Földiák (1990).

*Hebbian vision.* Deep Hebbian networks reach high accuracy on CIFAR-10 with many epochs of local learning and
a gradient-trained head (Journé et al. 2023; Moraitis et al. 2022). We add the single-pass setting, the
random-filter control and the gradient-readout control.

*Speech grounded in sight.* Learning spoken words from paired images, with no transcripts, was established
with networks trained by backpropagation (Harwath, Torralba & Glass 2016 **[check]**) on corpora such as the
one we use (Hsu et al. **[check]**), and reaches far higher retrieval than ours. Trained offline, a
contrastive model learns word-to-referent mappings from 61 hours of one child's head-camera recordings (Vong
et al. 2024). Ours differs in what it refuses: no backpropagation, no second pass, no text, nothing prepared in
the ear.

*Hebbian models of word learning.* The assembly-calculus models (Mitropolsky & Papadimitriou 2023, 2025) and
the brain-constrained models of Tomasello et al. (2018) learn words with Hebbian rules alone from constructed
input. Ours learns from photographs and recorded speech and is far less principled as a model of any brain
area.

*Fast and slow memory.* Complementary learning systems (McClelland, McNaughton & O'Reilly 1995) predict the
dissociation we measure; our store is a dense associative memory (Krotov & Hopfield 2016).

## What is still unexplained

The developed filters hold something a second-order reader can use and a first-order reader cannot. Nothing we
built generalises from one instance to its category, or from one voice to a voice never heard, and children do
both routinely. We do not know why a first voice survives a second. And the learner was assembled under a
rule, that only what brains do may go in, which did not stop us from building several parts that do nothing.
Biological plausibility told us what was allowed. It did not tell us what mattered; only taking the thing
apart did. The pleasant half of that lesson is the ear: when we finally trusted the organic version of each
part, it was already good enough.

## Statement on the use of AI

The code was written and the experiments were run by an AI assistant (Claude) under the author's direction. The
author set the principles (biological plausibility, no hand-wired behaviour, observations are not targets, no
change made in order to raise a score), took the decisions recorded in the project's log, and corrected the
assistant where it departed from them, including the mistakes described above. The adults in the staged world
and the judge are language models. **[Kris: to be reworded as yours.]**

## Code and data

Everything is public at <https://github.com/harrisonford/organic-learning>: the logs of every run cited here,
the scripts that produce each figure (`experiments/paper_figures.py`, `experiments/paper_figures12.py`), and a
long report with every table, control and setting behind this paper (`docs/paper/SUPPLEMENT.md`). The speech
corpus is licensed for sharing with attribution; we publish no recording of any speaker. **[A snapshot with a
DOI before submission.]**

## Appendix: the terms, and the settings that matter

**The terms.** "Organic learning" has been used before as the title of a United States patent on network
architectures (US 10,846,592). We use it for a method, the five commitments of the introduction, and not for an
architecture. "Organic intelligence" is ours for the aim; it should not be confused with *organoid*
intelligence, the name of a research programme on computing with cultured brain tissue **[check]**.

**Compute.** One desktop machine: an eight-core AMD Ryzen 7 5700G with 15 GB of memory, and an 8 GB graphics
card that serves only the judge. The learner is plain NumPy on the processor.

| Part | Setting | Value |
|---|---|---|
| Visual cortex | filters; patch; code passed on | 256; 5 × 5 colour; 4 × 4 grid, 4,096 numbers |
| Relay (batch form) | pictures it is estimated from; regulariser | 6,000 unlabelled; 0.0005 (chosen among four values) |
| Relay (circuit) | interneurons; rate | as many as the code has numbers; 1e-4 (a tenth of the published rate, see supplement) |
| Cochlea | bands; step | 64; 10 ms |
| Adaptation | each band's running mean and spread fade over | 0.5 s |
| Hearing cortex | detectors; patch; rate of a detector after *n* wins | 128; 80 ms × 8 bands; 1/(1 + *n*/5) |
| Its circuit | interneurons; rate | 64; 1e-3 (the published rate) |
| Sound units | number; competition | 2,048 (half for runs of two stretches, half for three); winner-take-all with a penalty on winning too often |
| Traces | a sound unit's; a typed letter run's | about 1 s; 2 s |
| Backpropagation reference | optimiser; rate; batch (searched: 14 settings) | Adam; 0.003; 32 |
