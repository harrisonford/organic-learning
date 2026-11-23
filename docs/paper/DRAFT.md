# Organic Learning: Small-Scale Architecture Experiments Without Backpropagation

**Second draft, 2026-11-23, rewritten after three independent readings (a style survey of well-known papers, a
hostile referee, a cold reader). Not for circulation.** This note is not part of the paper (the build leaves it
out). The title and the term "organic learning" are Kris's (2026-11-23); "It Was the Code, Not the Rule" stays
as a possible subtitle. Every number is from a log in `out/`. **[pending]** marks a run that is not finished. References are tracked
in REFERENCES.md and each must be opened by hand before anything is sent. Other titles to choose from:
"Raising a Learner Without Backpropagation: What Helped, What Did Nothing, What We Got Wrong";
"We Built a Brain-Like Learner Without Backprop. The Whitening Did the Work.";
"The Relay Did It"; "Most of the Brain We Built Did Nothing".

---

## Abstract

We set out to show that a learner made only of local learning rules, with no backpropagation, can learn from a
single pass through its experience. It can. On CIFAR-10 shown once, one picture at a time, it names 69% of
held-out pictures; a network of the same shape trained end to end by backpropagation names 62%. Then we took the
learner apart to see which part had earned the win, and the credit went somewhere we had not expected. It belongs
to the code and not to the rule. Two fixed decorrelation steps, one before the feature layer and one after it, do
the work: a gradient-trained readout on the same code does exactly as well as our local one, random filters do as
well as the Hebbian filters we had carefully developed, and without the second decorrelation backpropagation wins.
Both steps are computed in batch before learning starts, which is the one place where "local rules only" is not
yet true; a local Hebbian and anti-Hebbian network recovers about four fifths of the effect in a first test. A
small episodic store gives the learner one-shot memory of a picture but not of a category. We also raised the
learner through a staged childhood of photographs and adult speech, and we report what each part did there, what
did nothing, and four mistakes of method that had made earlier versions look better than they were.

## Introduction

Brains learn under constraints that deep learning does not accept. A synapse changes according to what its own two
ends are doing, nothing sends an error signal backwards through the layers, and no moment of a life comes round a
second time to be studied again. For three years the question of whether local learning rules can stand in for
backpropagation has mostly been asked at scale, and at scale the answer so far is "not yet".

We asked something smaller, and we asked it the slow way. We built a learner as one might assemble a small
animal: an ear, a visual cortex, a layer of cells that learn words, a memory for single episodes, a mouth that
babbles before it speaks. A part was admitted only if brains are known to do something like it, and only if it
left behaviour free to emerge; we allowed ourselves no rule that wires a signal directly to a behaviour. The
learner was then raised in a staged world, from babbling through echoing to first words, on real photographs and
on speech written for it.

We call this way of working *organic learning*, and since we would like the name to mean something, here is
what it commits us to. A part goes in only if brains are known to do something like it. No behaviour is wired in
by rule, and nothing the learner does along the way, whether babbling, echoing or a bias towards "no", is ever
turned into a target. Everything learns online, within one life, by local rules, with no error sent back through
layers. Nothing is changed in order to raise a score; a benchmark is a yardstick and not a goal. And every part
is taken out again to see whether it mattered, with what did nothing reported beside what worked. Where the
learner we built falls short of these commitments, and it does in two places, we say so. The phrase has been used
before, as the title of a patent on network architectures; we use it for a method and not for an architecture.
**[reference to add: US patent 10,846,592.]**

It worked better than we had expected, and for a while we thought we knew why. Our account was the obvious one:
local rules, patiently assembled, are simply good at learning from one pass. This paper is about what happened
when we tested that account. We put the same parts on a public benchmark, lined up the baselines a sceptical
reader would ask for, and removed the parts one at a time.

Before looking at the results, it is worth saying what would have vindicated us. If the local learning rule
deserved the credit, three things should hold. The local readout should beat a gradient-trained readout given the
same features. The Hebbian filters we developed should beat filters that were never trained. And the learner's
lead over backpropagation should survive the removal of any one preprocessing step. None of the three holds.

What does hold is narrower and, we think, more useful. Fixed features that are decorrelated twice, and read by a
single layer of plastic synapses, learn more from one pass than the same architecture trained end to end. The
reader can be local or gradient-based; it does not matter. A fast episodic store beside the slow weights adds
one-shot memory of an instance, disturbs nothing already learned, and does not generalise to the category. Most
of the rest of what we built, including the parts we were proudest of, did little or nothing.

We show these things on one small benchmark and in one small world. We conjecture, but do not show, that the same
division of credit would be found in other local learners that report a single-pass advantage. We do not claim an
alternative to backpropagation at scale, a model of a child, or even the best gradient-free learner on these
features; a streaming linear discriminant beats ours by five points, and we say so in the first table.

## The learner

Everything in the learner updates online, one example at a time, and nothing is frozen while it learns. We use a
few home-made names and introduce them here once. An *example* in the staged world is a **moment**: one picture,
one thing said to the learner, one reply expected. A whole training run from random initialisation is a **life**.

**Ear.** Text arrives as bytes. The ear groups bytes that often occur together into units of its own, much as a
sub-word tokeniser would, but online and by use. It hands the rest of the learner two things: the units just
heard, and a decaying, order-preserving trace of the exchange so far.

**Visual cortex.** One convolutional layer of 256 filters over 5 × 5 colour patches. Each patch is first
contrast-normalised and whitened, the textbook stand-in for what centre-surround cells in the retina do to natural
scenes (Atick & Redlich 1992). A filter's response is its drive above the average drive of all filters at that
location, $r_k = \max(0,\, u_k - \bar{u})$, a soft form of lateral inhibition that Coates, Lee & Ng (2011) called
the triangle activation. The filters are trained without labels by winner-take-all Hebbian learning with a
frequency penalty that keeps every filter in use. The cortex passes on the average response of each filter in each
cell of a 4 × 4 grid: a code of 4,096 numbers.

**Relay.** Between the cortex and the word cells, the code is whitened again: $z = (C + \epsilon\lambda_{\max} I)^{-1/2}(s - \mu)$,
with $C$ and $\mu$ estimated from 6,000 unlabelled pictures and $\epsilon = 0.0005$. In a brain, learned lateral
inhibition among the neurons that carry the code would do this (Földiák 1990). **In our learner both whitening
steps, this one and the one on the patches, are eigendecompositions computed in batch before learning begins.**
They are the two places where "local and online" is not yet true of what we built, and since the relay turns out
to be the hero of the paper, the reader should keep that in mind from here on. A later section reports a first
attempt to replace it with a local rule.

**Word cells.** The output layer. Every unit of the ear that has been heard in a reply gets a cell, so in
standard terms the word cells are the output units of a linear readout over sub-word units. Each cell has two
sets of input weights, which we call dendrites. The content dendrite listens to the relayed visual code and to
what was just heard; the context dendrite listens to the ear's trace and predicts which unit comes next. Each
dendrite learns by the delta rule to predict the cell's own firing, $\Delta w = \eta\,(y - \hat{y})\,x$, where $\hat{y}$ is
that dendrite's prediction (in the spirit of Urbanczik & Senn 2014, and identical to the Rescorla–Wagner rule). A
cell fires when its two dendrites agree, and a reply is a sequence of cells. We want to be plain about one thing:
a delta rule on a single layer *is* gradient descent on that layer. What the learner never does is send an error
back through a layer to change the features beneath it.

**Episodic store.** A buffer of 2,000 single moments, written whole whenever the word cells failed to predict the
reply, read by similarity to what is in view now, and overwritten oldest first. The similarity is passed through a
steep nonlinearity so that the store returns a moment and not an average of moments (Krotov & Hopfield 2016). This
is the fast half of a complementary-learning-systems pair (McClelland, McNaughton & O'Reilly 1995).

**Looking.** When a picture contains several things, a gaze driven by colour contrast fixates the regions that
stand out, and the fixated region is then seen at full resolution, as the thing would look alone. A small set of
match units compares what is fixated with a name that was just heard.

**Mouth.** Motor units are formed by babbling, and an auditory-motor map learns what each sounds like. A gating
policy learns from reward whether to echo what was heard, to speak from the word cells, or to speak from the store.

We set the following by hand: the sizes (256 filters, 2,000 word cells, 2,000 stored moments), the two whitening
regularisers, the steepness of the store's similarity, and the order of the stages. One of these was tuned, and we
come back to it under mistakes.

## What it looks like

Before the measurements, here is the thing itself, because a list of parts does not show what was built.

**A life.** Figure 1, `fig_life.pdf`, follows one learner from its first moment to its 115,000th. For the first
5,000 moments it only babbles. Nothing is asked of it; the mouth tries out sounds and the ear listens to the
result, and what comes out is "an?" and "ather cathay". The world then plays echo games with it, and after 10,000
moments begins to show it pictures and ask for names. Within a few thousand moments it is naming some of them.
From 15,000 moments on the world shows it picture-book pages, where several things appear and it has to look at
one. Both curves climb for some 30,000 moments and then level off at about 0.4 on our scale, on which a learner
that ignores the picture scores 0 and an adult scores 1. The babbling is worth following along the bottom of the
figure. Nobody rewards or tests it, yet it drifts towards the world's words: "reend" and "w a chice!" at 30,000
moments, "chairbathe ase?" at 65,000, "can you bus" at 110,000.

**What it says.** Figure 2, `fig_examples.pdf`, shows the learner at the end of that life answering held-out
questions. We did not choose these; they are the first six test items of each kind. It names four of the six
single pictures (over the whole test, about half) and two of the six pages (over the whole test, about a third).
The mistakes are worth reading. A mouse photographed close up is a "lion". A bus is a "train". A page with a horse
and a cow gets "cowbanana", two names run together, which is this learner's characteristic error on pages. On
spoken lessons with no picture to go on, its answers have the same flavour. Asked "what is on your hands when
it's cold?", it says "mittens" where the adult's answer was "gloves", which is why answers are judged for meaning
and not letter by letter. Asked "what's orange and yummy?", it says "carrot". Asked "what does a frog say?", it
says "flies", and asked to make an owl sound it says "mice": wrong, and not random.

**How big it is, and how fast.** After that life the learner holds about 257 million numbers. It runs on a CPU at
about two moments a second, so a life of 115,000 moments takes some sixteen hours, three lives side by side on a
desktop machine.

| part | numbers it holds |
|---|---|
| gating policy (when to answer, and from where) | 128 million |
| word cells (1,897 of them) and the episodic store (2,000 moments) | 61 million |
| ear | 51 million |
| visual cortex and relay (nearly all of it the relay's 4,096 × 4,096 matrix) | 17 million |
| mouth | 0.2 million |

**How long it takes to answer.** The learner decides for itself when to speak. At each tick it looks, and it
answers when its confidence passes a level of patience that it has learned for that kind of situation; waiting
costs it a little reward. Speaking then takes one tick per sound unit.

| kind of question | ticks before it starts to answer | when right / when wrong |
|---|---|---|
| "is there a …?" on a page | 1.0 | 1.0 / 1.1 |
| naming a single picture | 1.3–1.6 | 1.5 / 1.4 |
| naming a thing on a page | 1.5 | 1.6 / 1.4 |
| spoken lessons, no picture to go on | 3.1–4.1 | 3.5 / 3.5 |

So the learner has two speeds. A question about a picture is answered at a glance, and a question in words
alone is answered after a wait three times as long. We would like to call the second one thinking, and we cannot.
Right and wrong answers take the same time, the learner hardly moves its eye (less than one move per question),
and we have not tested whether the wait helps. What emerged is a habit of pausing where the answer is not in
view. Whether anything useful happens in the pause is an open question. These timings are from one learner;
**[pending: three seeds.]**

## Experiment 1: CIFAR-10, seen once

### Setup

The cortex develops its filters on the CIFAR-100 training pictures, without their labels, and the relay is fitted
on 6,000 of them. Then every CIFAR-10 training picture is shown exactly once, in random order. The learner is
asked "what is this?" and is told the class name. At fixed points along the stream we test on 2,000 held-out
pictures. There are three seeds; a seed changes the filters, the order of the stream, and every learner's initial
state. With 2,000 test pictures the sampling error of an accuracy is about 0.010, so we do not interpret
differences smaller than 0.02. A ± is the standard deviation over the three seeds.

Beside the word cells we run, on exactly the same code: a softmax readout trained by gradient descent (17
learning-rate settings run side by side, the best after 10,000 pictures reported); a nearest-class-mean
classifier; streaming linear discriminant analysis computed from exact running sums (Hayes & Kanan 2020); and a
ridge regression refitted on everything seen so far, which is not a stream learner but is the natural ceiling for
a linear readout. The last two choose their regulariser on the test pictures, which is generous to them.

The backpropagation baseline has the same shape as the learner minus the relay: the same whitened patches go in,
256 filters are learned from a random start, the same response and pooling follow, and a linear layer reads the
result with a softmax, trained with Adam. We tuned its learning rate and batch size on the first 10,000 pictures
of seed 0 (14 settings, which ranged from 0.40 to 0.53 there). We also ran it with replay, one or four earlier
pictures revisited for each new one; starting from our developed filters; and with those filters frozen.

### The local learner beats one-pass backpropagation

Figure 3, `fig_benchmark.pdf`, shows the curves, and the table gives three points along them.

| learner | 500 | 5,000 | 50,000 pictures |
|---|---|---|---|
| streaming LDA + relay | 0.375 ± 0.021 | 0.620 ± 0.002 | 0.739 ± 0.001 |
| ridge + relay | 0.395 ± 0.010 | 0.620 ± 0.003 | 0.735 ± 0.001 |
| streaming LDA, no relay (one seed) | 0.397 | 0.623 | 0.722 |
| streaming LDA + relay, random filters | 0.397 ± 0.001 | 0.599 ± 0.001 | 0.719 ± 0.005 |
| gradient softmax readout + relay | | | 0.694 ± 0.007 |
| word cells + relay, random filters | 0.388 ± 0.034 | 0.602 ± 0.003 | 0.693 ± 0.002 |
| **word cells + relay (the learner)** | 0.398 ± 0.027 | 0.595 ± 0.005 | **0.687 ± 0.005** |
| backprop, 4 replays per new picture (one seed) | 0.355 | 0.550 | 0.688 |
| backprop, 1 replay per new picture | 0.277 ± 0.012 | 0.490 ± 0.020 | 0.661 ± 0.011 |
| **backprop, one pass** | 0.242 ± 0.005 | 0.466 ± 0.008 | **0.616 ± 0.023** |
| word cells, no relay | 0.312 ± 0.014 | 0.486 ± 0.014 | 0.591 ± 0.007 |
| nearest class mean + relay | 0.426 ± 0.005 | 0.548 ± 0.004 | 0.569 ± 0.002 |
| backprop readout on frozen developed filters (one seed) | 0.296 | 0.476 | 0.550 |

The learner is ahead of one-pass backpropagation at every point of the stream: by 16 points after 500 pictures and
by 7 at the end. Backpropagation catches up when it is allowed to revisit four old pictures for every new one,
which is five times the number of updates; we ran that on one seed only. So far, so pleasing.

### But the credit belongs to the code

Take the relay away and the order flips. Without it the word cells reach 0.591, and one-pass backpropagation, at
0.616, is ahead of them. Like for like, backpropagation wins. Everything the learner gains over it, and three
points more, comes from one whitening step that the backpropagation baseline was not given. That is the fair way
to read our headline number, and it is why the headline is not in the title. **[pending: the end-to-end baseline
with a whitening layer in front of its readout, which is the comparison this paragraph calls for.]**

Is the local rule at least a good reader of the code? It is exactly as good as gradient descent and no better. A
softmax readout trained by gradient on the same relayed code reaches 0.694, against 0.687 for the word cells,
inside the noise. Without the relay the gradient readout falls to 0.56–0.58, just as the word cells do.

Why should whitening matter so much to a one-pass learner? We first met the effect in the learner's own world,
where the relay was worth eleven points, and we checked the dull explanations there. It is not a matter of
exposure: three passes over the same examples without the relay gained three points. It is not step size: larger
steps gained nothing and then diverged. A delta rule that sees each example once learns quickly along the few
strong directions of a correlated input and hardly at all along the weak ones, and the weak ones are where a cat
differs from a dog. Whitening makes all directions equally easy. This is textbook optimisation, and we claim no
novelty for it. What we had not appreciated is how completely it accounts for the single-pass advantage.

The streaming discriminant shows the same thing from the other side. It keeps the second moments of its input and
inverts them itself, so it should not care whether somebody whitened the code beforehand, and it does not: 0.722
without the relay and 0.741 with it on the seed where we ran both. The relay adds no information. It conditions
the code for a reader too simple to condition it for itself.

### Hebbian development did nothing for the learner

We spent weeks on the visual cortex. With random filters, never trained at all, the learner scores 0.693; with
the developed ones, 0.687. Random whitened patches have been known to be hard to beat for a long time (Coates, Lee
& Ng 2011; Saxe et al. 2011), and we rediscovered it the expensive way. The developed filters are not worthless:
the streaming discriminant gains two points from them (0.739 against 0.719 with random filters), so something in
them is better. Our reader cannot use it. When we looked inside, the developed filters were more selective than
random ones but 658 pairs among them were near copies of each other; random filters had none.

### Second-order readers stay ahead

The streaming discriminant and the ridge regression finish five points above the learner. They keep a covariance
matrix and invert it; the word cells are an online, first-order approximation of what those methods compute. We
are not the best gradient-free learner on our own features, and a reader who only wants accuracy from a single
pass should use a streaming discriminant.

### The relay, learned locally: a first attempt

Because the relay earns the result, its batch computation is the most uncomfortable fact in the paper. As a first
step towards a local form we trained the Hebbian and anti-Hebbian network of Pehlevan & Chklovskii (2014), using
their update rules unchanged, as lateral inhibition among the 256 filters, and used its settled activities as the
code. Over the first 10,000 pictures and with no relay at all, the word cells rise from 0.513 to 0.605 on
average over three seeds (0.500 to 0.595, 0.532 to 0.619, 0.508 to 0.602); the batch relay takes them to 0.627. A
local, online rule therefore recovers roughly four fifths of what the batch step gives. The streaming discriminant is again unmoved
(0.66–0.67 either way), which is what one expects if the network conditions the code without adding to it. This
is a fifth of the stream, the patch whitening is still computed in batch, and our own first formulation of
lateral inhibition ran away and had to be thrown out. **[pending: the full stream.]**

### A fast store remembers a picture, not a category

We held out one class at a time, trained on 5,000 pictures of the other nine, and then showed the held-out class
1, 5 and 20 times, with 20 other pictures after each showing.

| learner | shown pictures given the new name (1 / 5 / 20 showings) | 200 other pictures of the new class | old classes still right |
|---|---|---|---|
| word cells | 0.00 / 0.00 / 0.05–0.08 | 0.00 | 0.60–0.62 |
| word cells + episodic store | 1.00 / 0.96–1.00 / 0.99–1.00 | 0.00 | 0.60–0.63 |
| nearest class mean | 1.00 / 1.00 / 0.95–0.98 | 0.00 / 0.01 / 0.08 | 0.55–0.57 |
| linear discriminant analysis | 1.00 / 1.00 / 1.00 | 0.00 | 0.59–0.62 |

The slow word cells do not remember a picture they have seen once. With the store they do, and nothing they
already knew is disturbed. That is the complementary-learning-systems prediction, and it should not impress
anyone much: a buffer that returns an input it stored twenty steps ago is correct by construction. The more
interesting column is the third. Nobody, the store included, extends the new name to other pictures of the class.
We suspect this says as much about the test as about the learners, since a class mean estimated from 20 pictures
in 4,096 dimensions is far noisier than its rivals estimated from 550, and we draw only the modest conclusion:
one-shot learning in this learner is memory of an instance, and we have not shown it to be more.

## Experiment 2: raising it

The benchmark is where the parts are measured against other people's methods. The learner was not built there. It
was built in a small world of its own, and this section reports what the parts do in that world. The evidence
here is weaker than in Experiment 1, for reasons we give as we go, and we have tried to claim correspondingly less.

### The world, and how answers are scored

The pictures are real photographs from CIFAR, COCO and Wikimedia Commons. The speech is written by language
models playing adults, and only adults: they comment on pictures, play echo games, ask for names, and ask about
picture-book pages on which several things appear. The world has stages: babbling, an echo period, first words,
and then pages. Within a stage 70% of moments belong to that stage and 30% return to earlier ones. Every 5,000
moments we ask 300 held-out questions per stage.

Two things must be said plainly. First, the learner does not see each picture once here. The world's pool of
pictures is finite and is sampled with replacement, so pictures recur over a life of 100,000 moments, as the
things in a child's home recur. Nothing is stored and replayed, but "seen once" is strictly true only of
Experiment 1. **[pending: how often a picture recurs.]** Second, the judge is a model. A local 4-billion-parameter
language model decides whether the learner's answer means what the adult's answer meant, lenient about form and
strict about meaning. It agrees with a larger model on 93% of answers and wrongly accepts between 2 and 16 of 367
wrong answers, depending on how they are batched. No person has labelled its judgements yet.

How strict should such a judge be? Ours is told the child's age, and it grows stricter about form as the child
grows: at first words a babble that sounds like the word will do, and later the word itself is wanted. About
meaning it is strict at every age, so "woof" is not accepted for a dog. That is a decision about upbringing, and
it could be wrong: parents accept a good try from a small child and expect more from an older one. In this
learner the judge's verdict only sets the reward, which reaches the gating policy that decides when to speak and
from where; the word cells are corrected by the adult's own answer whatever the judge says. So strictness can
change how readily the learner speaks, and perhaps how quickly, without changing what its word cells are taught.
**[pending: lives whose reward counts a good try in full, lives where it counts early and less with age, and the
strict lives reported here, all tested by the same strict judge.]**

A raw score would flatter the learner, because many questions can be answered without looking. We therefore
report every score **above a blind baseline**. We drop the questions that can be answered by repeating the
question's last word, and on the rest we compare the learner with the best of three blind strategies: always
"yes", always "no", and always the commonest answer. The score is the fraction of the distance from that baseline
to a perfect answer, (score − baseline) / (1 − baseline). On a scale where a learner that ignores the picture
scores zero and an adult scores one, ours reach about 0.4.

### What three lives look like

Three lives were run from random initialisation, with no engineered exploration bonus and no replay. All three
stop babbling within 5,000 to 10,000 moments and pass through the echo period. In first words the score above the
blind baseline climbs to 0.28–0.40, and it goes on climbing after the world has moved on to pages:

| age | seed 0: first words / pages | seed 1 | seed 2 |
|---|---|---|---|
| on entering the pages stage | 0.28 / 0.19 | 0.36 / 0.15 | 0.40 / 0.23 |
| 50,000 moments | 0.38 / 0.37 | 0.44 / 0.36 | 0.43 / 0.36 |
| 95,000 moments | 0.40 / 0.41 | 0.44 / 0.33 | 0.45 / 0.41 |

The pages score rises for 20,000 to 30,000 moments and then flattens at about 0.4. First words improves while the
learner is in the pages stage, which sounds like transfer but need not be: three moments in ten are still
first-words lessons, so this is at least partly continued training. **[pending: a life with no return to earlier
stages.]**

These lives moved from stage to stage when a score crossed a pass mark that we had chosen with earlier scores in
front of us. We no longer think a mark of ours should decide when a learner is ready, and the lives now running
move on when the curve of the current stage stops rising. Under that rule the three seeds stay in first words
more than twice as long and are still improving at 0.43–0.46. **[pending: those lives; the same lessons in
shuffled order.]**

Some things we observed without asking for them. Echoing fades by itself once words start to work; by the pages
stage the learner repeats the question's last word in under a tenth of its answers. It has a slight bias towards
"no", where two-year-old children are reported to have a bias towards "yes". And on the life's own photographs,
with each naming moment taken once, the word cells name 0.52 of held-out pictures among 16 to 25 names, where the
same architecture trained by backpropagation on the same moments names 0.39 after one pass and 0.50 after five.
That backpropagation run had a single learning rate and optimiser and was not tuned as the CIFAR-10 baseline was.

### Removing one part at a time

We ran each ablation as three lives from random initialisation and read the scores at 50,000 moments.
Figure 4, `fig_ablation.pdf`, shows the seeds; the table gives the means.

| | naming a single picture | naming a thing on a page | "is there a …?" on a page |
|---|---|---|---|
| the whole learner | 0.42 | 0.36 | 0.44 |
| filters never developed | 0.41 | 0.04 | 0.27 |
| no relay | 0.35 | 0.19 | 0.40 |
| no looking | 0.42 | 0.02 | 0.36 |
| no match units | 0.42 | 0.33 | 0.29 |
| no episodic store | 0.44 | 0.45 | 0.44 |

The relay is the only part whose removal costs anything on single pictures, in all three seeds, just as on the
benchmark. Looking is what page naming depends on: without it, 0.36 falls to 0.02. The match units may matter for
"is there a …?", but the evidence is two seeds out of three, and the seeds of the whole learner themselves range
from 0.36 to 0.52 on that question, so we would not lean on it.

Two rows of this table made no sense to us, and chasing them was the most instructive week of the project. Why
would undeveloped filters, which name single pictures perfectly well, be unable to name a thing on a page? And why
would removing the episodic store make page naming *better*?

The two puzzles had one answer. We asked saved learners about test pages one view at a time. Shown only the
fixated region, every learner named the thing about as well, whether its filters were developed (right on
0.48–0.54 of pages) or not (0.39–0.46). The trouble was the store. It filed each moment under the whole view, and
a page seen whole is seen coarsely; coarse pages look alike; so the store kept answering with the name from some
other page. Silencing the store at test time raised the whole learner from 0.36–0.45 to 0.48–0.50 and the
undeveloped one from 0.07–0.15 to 0.33–0.44. Development, it turns out, is worth 0.05 to 0.10 on a fixated thing.
The dramatic first row was the store's doing.

The repair is one that the biology had been suggesting all along: file a moment under what the eye was on, not
under the whole scene. With that change the harm on pages disappears, single pictures are unaffected, and the
store can at last do its own job on a page. When a new thing on a page is named once, the store brings the name
back for the same picture on a different page 85–92% of the time (11–15% with the whole-view key, 0% for the word
cells alone). For a different picture of the same kind of thing it brings back nothing: an instance again, not a
category. We tested this on saved learners and in a probe; no full life has been run with the new key yet.

## What did nothing, and what we got wrong

A fair amount of what we built contributed nothing we could measure. Replay from the episodic store did not
protect earlier stages: the store had filled with small talk (55% of its contents), its surprise signal read 0.84
for everything and so could choose nothing, and old stages made up 3% of what came back. A second cortical layer
added 0.01. Dendritic branches on the word cells added nothing. We looked for the bias children show towards
giving a new word to a thing that has no name yet, and found 0.51 where chance is 0.50. Our first visual cortex
did worse after Hebbian development than before it; the second, as reported above, does no better than random
filters. And no learner got through the stage after pages. Earlier stages slid while it tried, and when we looked,
the lost answers had not been erased so much as displaced by that stage's larger vocabulary for the same pictures
("couch" for a bed). The stage's test also turned out to be answerable blind more than half the time, and we have
since rebuilt it.

Then there are the things we did wrong. We made four mistakes of method, and each of them made the learner look
better, or at least busier, than it was.

*We turned a behaviour into a target.* Echoing was a gate: a learner had to keep 80% of its echo score to move
on. Learners that had begun to use words echoed less, as children do, and we held them back for 20,000 to 90,000
moments for it.

*We wired a drive to a behaviour.* A "novelty drive" raised the probability of repeating an unknown word. It got
learners through stages. It was also a rule of ours and not something the learner did, which is exactly what we
had promised ourselves not to build. We removed it and reran every life.

*Our judge taught the learner its own mistakes.* The first judge accepted "yes" for "no", "bear" for a dog, and
the question repeated back. Worse, when it accepted an answer the adult repeated the learner's word, so about 7%
of lessons taught the learner its own error. The stricter judge described above replaced it.

*The world quietly went back in time.* After 50,000 moments of every life the world reverted to an older version
of itself, through a bug that no score revealed. We found it by reading logs.

Two smaller confessions belong here. The relay's regulariser was chosen from four values by looking at held-out
naming in the learner's world, so one number in this paper was tuned. And the pass marks that moved the first
lives along were set with scores in view; we dropped them for that reason.

## Limits

We only tried one small benchmark, on one machine, with codes of 4,096 numbers. Nothing here says the division of
credit holds at scale. We only compared against one backpropagation architecture, trained with plain Adam and no
augmentation, tuned on a grid of 14 settings, and never given the whitening step that turns out to matter. Both
whitening steps are computed in batch, and the local replacement has been tried on a fifth of the
stream. We tested on 2,000 pictures, with three seeds where five would be better. In the staged world the stages
and their order are ours, the adults and the judge are language models, pictures recur, and no human has yet
labelled the judge's decisions. The staged world supports the benchmark's findings; it could not carry them alone.

## Related work

*Fixed features and one trained layer.* What Experiment 1 finds belongs to a known family: features that the task
does not train, read by a single trained layer. Random whitened filters with a pooled, rectified response were
shown long ago to be hard to beat among single-layer feature learners (Coates, Lee & Ng 2011; Saxe et al. 2011).
Streaming linear discriminant analysis (Hayes & Kanan 2020) and the fruit-fly associative learner of Shen,
Dasgupta & Navlakha (2021) learn in one pass without forgetting, but on features from a network pretrained with
labels by backpropagation; the sparse-distributed-memory learner of Bricken et al. (2023) likewise, and it is
itself trained by backpropagation. Closest in mechanism is a delta rule on a linear readout over fixed random
sparse codes, used for continual learning of text domains (arXiv 2607.26523). In our case no part was ever
trained by backpropagation or with labels, the input is photographs, and the comparison is the same architecture
trained end to end.

*Hebbian vision.* Deep Hebbian networks reach high accuracy on CIFAR-10 with many epochs of local learning and a
gradient-trained supervised head (Journé et al. 2023), and their single-layer precursor reports a one-epoch
advantage over backpropagation (Moraitis et al. 2022). We add the single-pass setting with a label stream, the
random-filter control, the gradient-readout control, and the finding that decorrelation accounts for the
advantage.

*Hebbian models of word learning.* The assembly-calculus models of language (Mitropolsky & Papadimitriou 2023,
2025) and the brain-constrained spiking models of Tomasello et al. (2018) and Constant, Pulvermüller & Tomasello
(2023) learn words with Hebbian rules alone, from constructed input: toy sentences, and pre-formed patterns that
stand in for vision. Ours learns from photographs and from free adult speech, and is far less principled as a
model of any brain area.

*One pass with backpropagation.* A contrastive model of a child's view, trained in one chronological pass, falls
to near chance without replay buffers that hold most of the stream (BabyCL; arXiv 2606.05115). This is the
setting in which our end-to-end baseline also trails.

*Fast and slow memory.* Complementary learning systems (McClelland, McNaughton & O'Reilly 1995) predict the
dissociation we measure. Agents trained by gradient with an external episodic memory show one-shot word learning
(Hill et al. 2021). Our store is a dense associative memory with a steep read-out (Krotov & Hopfield 2016). What
we add is the measurement beside other learners, and the observation that such a store has to be keyed by what
the eye is on and not by the whole view, which fits reports that hippocampal activity follows gaze (Rolls 1999;
Liu et al. 2017).

*Learned lateral inhibition.* Hebbian feedforward learning with anti-Hebbian lateral learning goes back to
Földiák (1990), accounts for the shapes of V1 receptive fields (Zylberberg, Murphy & DeWeese 2011), and has a
derivation from a similarity-matching objective (Pehlevan & Chklovskii 2014), whose update rules we used.

## What is still unexplained

We end with what we cannot yet account for. The developed filters contain something a second-order reader can
use and a first-order reader cannot, and we do not know what it is. A local network recovers most of the relay's
effect when it acts among the filters, and we do not know whether the same rule acting on the pooled code, where
the relay itself sits, would recover the rest. Nothing we built generalises from one instance to its category,
and children do it routinely. And the learner we took apart was assembled under a rule, that only what brains do
may go in, which did not prevent us from building several parts that do nothing. We find that the most
interesting result of all. Biological plausibility told us what was allowed. It did not tell us what mattered,
and only taking the thing apart did.

## Statement on the use of AI

The code was written and the experiments were run by an AI assistant (Claude) under the author's direction. The
author set the principles (biological plausibility, no hand-wired behaviour, observations are not targets, no
change made in order to raise a score), took the decisions recorded in the project's log, and corrected the
assistant where it departed from them, including the mistakes described above. The adults in the staged world and
the judge are language models. **[Kris: to be reworded as yours.]**

## Code and data

The repository holds the logs of every run cited here and the scripts that produce each table and figure
(`experiments/bench_*.py`, `experiments/curriculum_life.py`, `experiments/paper_figures.py`). **[A snapshot with a
DOI before submission.]**
