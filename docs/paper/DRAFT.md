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
well as the Hebbian filters we had carefully developed (on pictures enlarged to fit the learner's eye; at the
pictures' own size the developed filters are two to three points better), and without the second decorrelation
backpropagation wins.
The filters' failure was our rule's and not Hebbian learning's: the soft winner-take-all rule of Journé et al.
(2023), built from its authors' code and read the same way, develops filters that take the learner from 55% to
69% after 10,000 pictures, where our cortex gives it 62%. On that layer the learner ends the stream at 73%;
backpropagation of the layer's shape ends at 56% from a random start and at 70% when it is handed the same
developed filters, so most of that lead was bought with unlabelled pictures.
Both steps are computed in batch before learning starts, which is the one place where "local rules only" is not
yet true; on a fifth of the stream, a published circuit of inhibitory interneurons that learns by a local rule
does the second step's work to within a point. A
small episodic store gives the learner one-shot memory of a picture but not of a category. We also raised the
learner through a staged childhood of photographs and adult speech, and we report what each part did there, what
did nothing, and three mistaken assumptions that had made earlier versions look better than they were.

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
learner we built falls short of these commitments, and it does in two places, we say so.

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

We did not discover that whitened features are strong. Thiry et al. (2021) showed that one layer of whitened
image patches with a linear classifier matches trained networks on CIFAR-10, Prabhu et al. (2024) that a random
code, decorrelated and read by class means, matches continually trained deep networks in a single pass, and
Dalm et al. (2024) that decorrelating its inputs speeds up backpropagation itself. What this paper adds is the
view from the other side: a learner built on purpose from local rules, whose builders credited the rules, taken
apart until the credit sat where that literature says it should.

We show these things on one small benchmark and in one small world. We conjecture, but do not show, that the same
division of credit would be found in other local learners that report a single-pass advantage. We do not claim an
alternative to backpropagation at scale, a model of a child, or even the best gradient-free learner on these
features; a streaming linear discriminant beats ours by five points, and we say so in the first table.

## The learner

Everything in the learner updates online, one example at a time, and nothing is frozen while it learns. We use a
few home-made names and introduce them here once. An *example* in the staged world is a **moment**: one picture,
one thing said to the learner, one reply expected. A whole training run from random initialisation is a **life**.
Figure 1, `fig_architecture.pdf`, shows the parts and how each one comes to be what it is.

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
Brains do both with local circuitry; ours are stand-ins, computed the quick way, and they are the two places
where "local and online" is not yet true of what we built. Since the relay turns out to be the hero of the paper,
the reader should keep that in mind from here on. A later section swaps the relay for a published local rule,
which recovers four fifths of its effect.

**Word cells.** The output layer. Every unit of the ear that has been heard in a reply gets a cell, so in
standard terms the word cells are the output units of a linear readout over sub-word units. Each cell has two
sets of input weights, which we call dendrites. The content dendrite listens to the relayed visual code and to
what was just heard; the context dendrite listens to the ear's trace and predicts which unit comes next. Each
dendrite learns by the delta rule to predict the cell's own firing, $\Delta w = \eta\,(y - \hat{y})\,x$, where $\hat{y}$ is
that dendrite's prediction (in the spirit of Urbanczik & Senn 2014, and identical to the Rescorla–Wagner rule).
The step is not the same for every synapse: each one consolidates with its own use, $\eta_i = 1/(1 + u_i/5)$ with
$u_i$ the running sum of that input's squared activity, so a synapse that has carried much signal moves little.
Readers who know AdaGrad will recognise it; it uses only what the synapse itself has seen. A
cell fires when its two dendrites agree, and a reply is a sequence of cells. We want to be plain about one thing:
a delta rule on a single layer *is* gradient descent on that layer. What the learner never does is send an error
back through a layer to change the features beneath it.

**Episodic store.** A buffer of 2,000 single moments, written whole whenever the word cells failed to predict the
reply, read by similarity to what is in view now, and overwritten oldest first. The similarity is passed through a
steep nonlinearity so that the store returns a moment and not an average of moments (Krotov & Hopfield 2016). This
is the fast half of a complementary-learning-systems pair (McClelland, McNaughton & O'Reilly 1995).

**Looking.** When a picture contains several things, a gaze driven by colour contrast fixates the regions that
stand out, and the fixated region is then seen at full resolution, as the thing would look alone. A small set of
match units compares what is fixated with a name that was just heard. In the lives reported here every fixated
region reaches the word cells at once and for nothing; a later section rebuilds looking as one look at a time.

**Mouth.** Motor units are formed by babbling, and an auditory-motor map learns what each sounds like. A gating
policy learns from reward whether to echo what was heard, to speak from the word cells, or to speak from the store.

We set the following by hand: the sizes (256 filters, 2,000 word cells, 2,000 stored moments), the two whitening
regularisers, the steepness of the store's similarity, and the order of the stages. No size comes from anatomy.
The cortex follows the single-layer recipe of Coates, Lee & Ng (2011), cut down to what one desktop machine can
run three lives of at once, and using the same filters at every position is an engineering shortcut that a brain
does not have: it has to learn its detectors place by place. One of these was tuned, and we
come back to it under mistakes.

## What it looks like

Before the measurements, here is the thing itself, because a list of parts does not show what was built.

**A life.** Figure 2, `fig_life.pdf`, follows one learner from its first moment to its 115,000th. For the first
5,000 moments it only babbles. Nothing is asked of it; the mouth tries out sounds and the ear listens to the
result, and what comes out is "an?" and "ather cathay". The world then plays echo games with it, and after 10,000
moments begins to show it pictures and ask for names. Within a few thousand moments it is naming some of them.
From 15,000 moments on the world shows it picture-book pages, where several things appear and it has to look at
one. Both curves climb for some 30,000 moments and then level off at about 0.4 on our scale, on which a learner
that ignores the picture scores 0 and an adult scores 1. The babbling is worth following along the bottom of the
figure. Nobody rewards or tests it, yet it drifts towards the world's words: "reend" and "w a chice!" at 30,000
moments, "chairbathe ase?" at 65,000, "can you bus" at 110,000.

**What it says.** Figure 3, `fig_examples.pdf`, shows the learner at the end of that life answering held-out
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
on 6,000 of them. The learner's eye is 64 pixels across, as it is in the world of Experiment 2, so every 32-pixel
CIFAR picture is enlarged to 64 by bilinear interpolation before the cortex sees it; the backpropagation baseline
of our shape sees the same enlarged pictures. (The soft layer of Journé et al., below, sees the pictures at their
own size, as its authors' code does.) Then every CIFAR-10 training picture is shown exactly once, in random order. The learner is
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

Figure 4, `fig_benchmark.pdf`, shows the curves, and the table gives three points along them.

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
to read our headline number, and it is why the headline is not in the title. With the relay in front of its readout, backpropagation reaches 0.688 ± 0.005 on the whole stream (three
seeds: 0.682, 0.690, 0.691), against 0.627 without it (one seed) and 0.687 ± 0.005 for the word cells. Given the same
decorrelated code, the two are level. The lead was the whitening step, as the literature on decorrelation would
predict. The baseline's best learning rate was the lowest of the first search, so the search was widened to lower rates and smaller batches;
nothing did better. With the relay settled again every 5,000 pictures in place of the fixed one, backpropagation reaches
0.684 (0.671, 0.689, 0.693).

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

### Our Hebbian development did nothing for the learner

We spent weeks on the visual cortex. With random filters, never trained at all, the learner scores 0.693; with
the developed ones, 0.687. Random whitened patches have been known to be hard to beat for a long time (Coates, Lee
& Ng 2011; Saxe et al. 2011; Thiry et al. 2021), and we rediscovered it the expensive way. The result is about
our rule, hard winner-take-all, and not about Hebbian learning in general: Journé et al. (2023) report the same
thing for hard winner-take-all on one layer (62.7% against 63.9% for random weights) and then a clear gain from a
soft version of the competition (71.1%). We built their layer from their own code, changing nothing, and read
it as we read everything else, on three seeds. Their finding holds in their measure (a linear readout of the
whole layer from 5,000 labelled pictures names 0.614 ± 0.002 against 0.517 ± 0.003 for the same layer never
developed), and it holds in ours: after 10,000 pictures of the stream the word cells name 0.689 ± 0.005 on the
soft layer, 0.548 ± 0.002 on the same layer as born, and 0.620 ± 0.010 on our hard-rule cortex; ridge regression
on the same three codes reaches 0.700 ± 0.001, 0.605 ± 0.002 and 0.664 ± 0.007. A competition in which the
losers unlearn develops filters worth having; ours did not.

That verdict has a condition we found late, and it is about the pictures and not the rule. The learner's eye is
64 pixels across, and CIFAR's pictures, at 32, were enlarged to fit it. Enlarging adds pixels and no detail: of
the 75 directions in which a 5 × 5 colour window can vary, an enlarged CIFAR picture uses 22 above the floor of
our decorrelation, where the same picture at its own size uses 47 and a photograph that really is 64 pixels
across uses 48. So we ran the same cortex, the same code and the same stream at CIFAR's own size, on three
seeds and 10,000 pictures:

| our cortex, after 10,000 pictures | word cells + relay | ridge + relay | near-copy pairs among the 256 filters |
|---|---|---|---|
| enlarged to 64, filters as born | 0.643 ± 0.010 | 0.656 ± 0.008 | 0 |
| enlarged to 64, developed | 0.621 ± 0.005 | 0.665 ± 0.007 | 257 to 282 |
| own size, filters as born | 0.626 ± 0.002 | 0.632 ± 0.003 | 0 |
| own size, developed | 0.649 ± 0.008 | 0.666 ± 0.001 | 96 to 111 |

On enlarged pictures development costs the word cells two points and gives ridge one. At the pictures' own size
it gives the word cells two points and ridge three, on every seed, and the same rule ends with a third as many
near copies. "Did nothing" is true of our rule on pictures with half their detail missing. It also means the
comparison with the soft layer just above is not like for like: the soft layer saw the pictures at their own
size (0.689) and our cortex, in that sentence, saw them enlarged (0.620). At their own size ours reaches 0.649.
The soft rule is still ahead, by four points and not by seven.

We then took their rule apart, one part at a time, on the same three seeds (word cells with the relay after
10,000 pictures; 0.689 for the rule as published, 0.548 for the layer as born). If only the winner learns, the
layer ends under where it was born, 0.502 ± 0.009, with about 900 pairs of near-copy filters among the 256: our
own failure, reproduced in their layer. If the losers learn with the winner's sign instead of against it, it is
worse still, 0.433 ± 0.011, and some 19,000 of the 32,640 pairs are near copies. The unlearning by the losers is
what does the work. One step of the rule is not local, though: in the authors' code every change is divided by
the largest change anywhere in the layer, a number no synapse has. We tried the two obvious local stand-ins and
neither works. If each filter divides by its own largest change, the weights grow to hundreds of times their length at
birth and the code is destroyed (0.238 ± 0.017). If the factor is held at its early value the layer hardly develops (0.578 ± 0.020),
and held at its median over the run the weights overflow. We did not design a third. So the best Hebbian
filters we measured come from a rule that is local except for one shared step, and a fully local rule that
does as well is, for us, still open.

The nearest we came uses another published layer. SAILnet (Zylberberg, Murphy & DeWeese 2011) is Földiák's
three local rules on spiking cells: a Hebbian rule for the weights into a cell, a threshold that holds the
cell to its rate, and inhibition between two cells that grows when they fire together. No step needs a number
from elsewhere in the layer. We built it from its authors' code, with their constants, and let it develop the
256 filters of our own cortex on CIFAR at its own size; only its feed-forward weights stand in for ours,
brought to unit length as ours are, and the layer's own spikes are not read. Its cells settle at their target
rate, between 192 and 201 of the 256 filters end different from one another, and after 10,000 pictures the
word cells read 0.643 ± 0.008 from them behind the relay, against 0.626 ± 0.002 from the filters as born and
0.649 ± 0.008 from our hard rule (ridge: 0.657, 0.632 and 0.666). A fully local rule, then, develops filters as
good as our winner-take-all's. It does not reach the soft rule's. On pictures enlarged to twice their size the
same layer had collapsed to 98 different filters, which is how we came to look at the enlargement at all.

The relay still earns its place on the better
layer: without it the word cells fall from 0.689 to 0.591 ± 0.013. The whole stream on the soft layer, beside
backpropagation of that layer's shape, is in the next section. Our own developed filters are not worthless:
the streaming discriminant gains two points from them (0.739 against 0.719 with random filters), so something in
them is better. Our reader cannot use it. When we looked inside, the developed filters were more selective than
random ones but 658 pairs among them were near copies of each other; random filters had none.

### The same contest on the better layer

Everything above compares learners on our cortex, whose development did little. So we ran the whole stream
again on the soft layer, and built for it the reference we had built for ours: a network of the soft layer's
shape (the same standardised colours go in, 256 filters on 5 × 5 windows, the same response and pooling, a linear
layer read by a softmax), trained end to end with Adam. Its learning rate, its batch size and the exponent of its
response were tuned from a random start on the first 10,000 pictures of seed 0: twelve settings without the relay
(0.38 to 0.48 there) and twelve with it (0.45 to 0.53). Where it has the relay, the relay is settled once on the
filters it starts from and kept. Every row is three seeds.

| learner, on the soft layer | 500 | 5,000 | 50,000 pictures |
|---|---|---|---|
| streaming LDA + relay | 0.439 ± 0.014 | 0.617 ± 0.008 | 0.748 ± 0.005 |
| streaming LDA, no relay | 0.455 ± 0.013 | 0.666 ± 0.006 | 0.747 ± 0.006 |
| ridge + relay | 0.456 ± 0.005 | 0.663 ± 0.006 | 0.744 ± 0.006 |
| ridge, no relay | 0.471 ± 0.001 | 0.662 ± 0.005 | 0.744 ± 0.003 |
| gradient softmax readout + relay, soft filters frozen | 0.432 ± 0.001 | 0.635 ± 0.001 | 0.739 ± 0.003 |
| **word cells + relay** | 0.416 ± 0.055 | 0.654 ± 0.010 | **0.730 ± 0.004** |
| **backprop + relay, started from the soft filters** | 0.276 ± 0.013 | 0.544 ± 0.003 | **0.703 ± 0.004** |
| word cells, no relay | 0.384 ± 0.041 | 0.574 ± 0.003 | 0.671 ± 0.005 |
| word cells + relay, the layer as born | 0.354 ± 0.020 | 0.530 ± 0.006 | 0.630 ± 0.004 |
| backprop, no relay, started from the soft filters | 0.297 ± 0.045 | 0.471 ± 0.049 | 0.620 ± 0.017 |
| gradient softmax readout, no relay, soft filters frozen | 0.270 ± 0.026 | 0.429 ± 0.057 | 0.607 ± 0.021 |
| backprop + relay, from a random start | 0.304 ± 0.009 | 0.475 ± 0.007 | 0.606 ± 0.008 |
| backprop, no relay, from a random start | 0.271 ± 0.007 | 0.419 ± 0.026 | 0.562 ± 0.013 |
| word cells, no relay, the layer as born | 0.270 ± 0.009 | 0.386 ± 0.015 | 0.503 ± 0.010 |

On the better layer the word cells end at 0.730, four points above where our cortex leaves them, and twelve to
seventeen above backpropagation of the same shape from a random start. It would be easy to stop there and call it
the headline. It is not a like-for-like contest. The soft layer developed its filters on 50,000 other pictures,
without labels, before the stream began, so the local learner has seen twice the pictures the backpropagation
network has. Using pictures nobody labelled is what a local rule is for, and we think that is a fair thing for it
to be good at; but that number cannot tell better filters from more pictures.

Two rows can. The first is the layer as it is born, which has seen no pictures at all. There the contest goes as
it went on our cortex: with the relay the word cells on filters nobody developed reach 0.630 and backpropagation,
learning its filters behind the same relay, 0.606, two points, at the edge of what we interpret; without the
relay the order flips, 0.503 against 0.562. The second is backpropagation handed the developed filters to start
from. It ends at 0.703 with the relay, ten points above its random start. Most of the gap, then, was the filters,
which is to say the unlabelled pictures. What is left is 2.7 points to the word cells (0.730, and ahead on every
seed), and one more thing: the same filters left alone, with only a readout trained by gradient, give 0.739. One
pass of backpropagation through good filters leaves them worse than not touching them. Its settings were tuned
from a random start and not again for this one, and its relay is not settled again as the filters move (on our
cortex settling it again changed nothing), so we would not lean on the 2.7 points.

Three more things. First, the reader: the gradient readout on the frozen filters behind the relay (0.739) and
the word cells (0.730) differ by less than we interpret. As on our cortex, the local reader is as good as the
gradient one and no better. Without the relay the gradient readout falls to 0.607 and the word cells to 0.671.
Second, ridge regression and the streaming discriminant are ahead again, at 0.744 and 0.748, and the relay is
worth nothing to them where it is worth six points to the word cells. On the layer as born they reach 0.666 and
0.663 behind the relay, so the development is worth eight points to them and ten to the word cells. Third,
backpropagation of this shape from a random start does worse than backpropagation of ours (0.562 against 0.616).
Our shape whitens each patch before the filters and the soft layer's does not. That fits the account of the last
section, and it is not a test of it.

### Second-order readers stay ahead

The streaming discriminant and the ridge regression finish five points above the learner. They keep a covariance
matrix and invert it; the word cells are an online, first-order approximation of what those methods compute. We
are not the best gradient-free learner on our own features, and a reader who only wants accuracy from a single
pass should use a streaming discriminant.

### The relay, learned locally: two attempts

Because the relay earns the result, its batch computation is the most uncomfortable fact in the paper. As a first
step towards a local form we trained the Hebbian and anti-Hebbian network of Pehlevan & Chklovskii (2014), using
their update rules unchanged, as lateral inhibition among the 256 filters, and used its settled activities as the
code. Over the first 10,000 pictures and with no relay at all, the word cells rise from 0.513 to 0.605 on
average over three seeds (0.500 to 0.595, 0.532 to 0.619, 0.508 to 0.602); the batch relay takes them to 0.627. A
local, online rule therefore recovers roughly four fifths of what the batch step gives. The streaming discriminant is again unmoved
(0.66–0.67 either way), which is what one expects if the network conditions the code without adding to it. This
is a fifth of the stream, the patch whitening is still computed in batch, and our own first formulation of
lateral inhibition ran away and had to be thrown out. **[pending: the full stream.]**

The second attempt replaces the relay itself. Lipshutz, Pehlevan & Chklovskii (2022) describe a circuit in which
every principal cell excites interneurons that inhibit it back, and the synapse between a cell and an interneuron
grows with the product of their two activities and decays. With the principal cells' own leak (Duong et al.
2023b), the circuit comes to rest where every direction of the code that varies more than average has been
brought down to the average and the others are left alone. Nothing in that resting point is set by hand, where
our batch relay has its constant $\epsilon$. We gave the circuit as many interneurons as the code has numbers
(4,096) and let it learn from the same 6,000 unlabelled pictures the batch relay was computed from, in blocks of
50. After 10,000 pictures of the stream, over three seeds, the word cells name 0.620 ± 0.010 behind the batch
relay; 0.609 ± 0.009 behind the rule's resting point computed directly; 0.602 ± 0.013 behind the circuit
developed on the unlabelled pictures and then held still; and 0.614 ± 0.007 behind the circuit developed the
same way and then left learning as the stream goes by (0.000, 0.008 and 0.010 under the batch relay, seed by
seed). A circuit that starts from nothing and learns only along the stream ends at 0.608 ± 0.007. So a local
rule does the relay's work to within about a point, and the word cells are not hurt by a code that keeps moving
slowly under them. Three cautions. This is a fifth of the stream. We compute the circuit's settled activity
with a matrix solve and do not run its dynamics. And the published learning rate did not carry over: a picture
changes the synapses along its own activity in proportion to the rate times the squared length of that
activity, which is about 350 for our 4,096 cells and far smaller in the papers' small examples. At their rate
the circuit never came to rest, and we lowered it tenfold. The circuit is not yet in the lives.

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

### Giving the cortex what a cortex has

The random-filter result left us two ways to go on. We could hunt for whatever raises the number. Or we could
stop letting the number decide, and ask what an early visual system is known to have that ours does not. We took
the second, and gave ourselves a rule for it: a mechanism that is well established in biology goes in whether
or not the score moves, and it is still measured, first on what it does to the code and then on the stream, and
reported either way. Four mechanisms have been through this so far. Each was read on 10,000 pictures of the
stream, where the cortex as described above gives the word cells 0.620 ± 0.010 and the ridge 0.664 ± 0.007. For
scale, after as many pictures of the full stream one-pass backpropagation of the same shape has 0.510 ± 0.025,
or 0.604 ± 0.005 when it is given the relay, and the streaming discriminant has 0.672 ± 0.007.

| added to the cortex | what it does to the code | word cells + relay | ridge + relay |
|---|---|---|---|
| nothing | | 0.620 ± 0.010 | 0.664 ± 0.007 |
| divisive normalisation | filters answering at a place: sparseness 0.71 → 0.83; the response now grows with contrast and saturates | 0.630 ± 0.006 | 0.650 ± 0.004 |
| a blurred start | filters a little coarser (0.28 against 0.30 cycles per pixel) and more coloured | 0.620 ± 0.005 | 0.659 ± 0.005 |
| a second stage of filters | the same picture two pixels aside looks less alike: 0.85 → 0.80 | 0.638 ± 0.012 | 0.645 ± 0.010 |
| the same stage, learning from the eye's drift | about half of that tolerance comes back: 0.83 | 0.634 ± 0.010 | 0.644 ± 0.007 |
| a wide sparse layer after the relay (ten times the cells, seven inputs each) | two different pictures look less than half as alike; a picture seen once is remembered better | 0.578 ± 0.009 | 0.606 ± 0.006 |
| the same layer, a hundred inputs each | two different pictures a quarter as alike | 0.609 ± 0.005 | 0.616 ± 0.003 |

*Divisive normalisation.* In cortex a neuron's response is divided by the pooled activity of its neighbours
(Carandini & Heeger 2012). Our filters' responses were only thresholded. With each response divided by the
summed activity of all filters at its place, the code changes exactly as the physiology says it should, and
identically on every seed: fewer than half as many filters answer strongly at any place, and the response,
which before did not know contrast at all, grows with it and levels off. On the stream the effect is small and
goes both ways. The word cells gain a point (and a point without the relay), below what we read as a
difference; the ridge loses a point and a half on every seed. Division conditions the code a little for a
reader that sees each picture once and discards a little of what a reader with a covariance matrix could have
used. Pooling over the surrounding places as well did nothing more for the word cells and cost the ridge another point. We keep the
normalisation at the place, on its ground, and claim nothing for it.

*A blurred start.* A newborn sees at about a fortieth of adult acuity, in weak colour, and before the eyes open
the retina rehearses with waves of its own activity. Networks whose input matures the same way are reported to
lean more on shape and to cope better with degraded pictures (Lu et al. 2025; Ligeralde et al. 2023). We let the
filters develop on waves, then on blurred, pale pictures that sharpen and gain colour. They end up a little
coarser and more coloured, every time. And nothing else moves: not the stream, and not the stream with test
pictures blurred by a pixel (0.580 against 0.583 for the word cells). We think we know why. One layer of small
filters that keeps learning through the sharp end of the schedule has, by the end, relearned the sharp world.
The reports are for deep networks, where what is learned early shapes what later stages build on. The schedule
is the eye's own and costs nothing, so it stays for a deeper cortex, with no gain claimed.

*A second stage that learns from the eye's drift.* Eyes are never still. During a fixation the image slides
over the retina by a few receptors, so a cortex gets, for nothing, several slightly shifted views of the same
thing, and a rule that asks a cell to answer the next view as it answered the last one (Földiák 1991; Halvagal
& Zenke 2023) should make it tolerant of the shift. We added a second stage of filters above the first, let it
develop only after the first had settled, and gave it four drifting views of each fixation. The rule does
what it says, on every seed: the first stage answers a picture and the same picture two pixels aside 0.85
alike, a second stage developed without the rule only 0.80, and one developed with it 0.83. The rule gives back
about half of the tolerance that stacking a stage costs, and no more than the first stage had to begin with.
On the stream it changes nothing. The second stage itself helps the word cells a little on every seed (0.638
and 0.634 against 0.620 for one stage) and costs the ridge two points, and the two second stages cannot be told
apart (on the first seed the plain one looked better by a point and a half; the next two seeds took that
back). A two-pixel shift is simply not what makes one picture of a ship hard to recognise from another. The
rule stays, for the ground it stands on and for the day the eye moves over larger things.

*A wide sparse layer.* The fly's mushroom body, the cerebellum's granule layer and the dentate gyrus all put
the same thing in front of cells that learn associations: many more cells than inputs, each wired to a few
inputs at random, and few of them active at once (Caron et al. 2013; Litwin-Kumar et al. 2017). We wired the
relay's 4,096 numbers onto up to ten times as many cells, seven inputs each as in the fly, and let one
inhibitory loop keep 5% of them on. The code does what such layers are said to do. Two different pictures look
less than half as alike as before, and the code spreads over three times as many directions. That is the
opposite of what a namer of new pictures needs, because two pictures of the same class become less alike too:
the word cells fall from 0.620 to 0.578, and to 0.509 when the layer is no wider than its input. What improves
is memory for the single picture. A picture seen once is named right twenty pictures later 85% of the time on
the relay's code and 95% on the wide one, on every seed. The same wiring without the inhibition changes
nothing, as a linear map should.

Seven inputs is the fly's number, but a fly's cell samples a seventh of its 50 input channels and ours a
six-hundredth of 4,096. With a hundred inputs a cell, which we had planned as a second wiring before seeing any
result, most of the price disappears. At ten times the cells the word cells name new pictures at 0.609 ± 0.005
against 0.620 ± 0.010 on the relay's code, a difference we do not read, and they remember a picture seen once
98% of the time against 85%. Two costs remain. The wide code starts slowly (0.26 against 0.39 after 500
pictures), and the ridge, which could use everything the relay's code held, loses five points on it. So the
layer buys a memory for single moments out of slow cells, at the price of a slow start and of information a
better reader could have used. Biology puts these layers in front of memories, the hippocampus's among them,
and our numbers agree that memory is what they are good for. A first direct test of the layer as the key of
the episodic store was too easy to say more, since with either code the store found a stored picture from a
blurred view of it every time. We have not put the layer into the learner.

None of the four makes the learner a better namer of pictures by an amount we would defend, and one, wired
as thinly as the fly wires it, makes it worse. That is what working this way costs, and we would rather pay it
than tune. The gap to the baselines is where it was.

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

Two things must be said plainly. First, "seen once" is strictly true only of Experiment 1. The world draws its
lessons from finite pools with replacement, so things recur, as the things in a child's home recur. For
photographs the life is still close to a single pass: over the 35,000 moments one learner spent in first words, a
CIFAR photograph was met 0.8 times on average, a COCO cut-out 0.1 times and a Commons photograph 1.9 times. The
spoken lessons are another matter. There are about a thousand of them and each was met five or six times, and the
ninety social routines some sixty times each. Nothing is stored and replayed; the repetition is the world's. Second, the judge is a model. A local 4-billion-parameter
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
move on when the curve of the current stage stops rising. Under that rule the three seeds left first
words after 45,000, 50,000 and 70,000 moments there, at 0.49, 0.49 and 0.53 above blind answering, and pages after
30,000 to 40,000 moments at 0.31 to 0.36. In the stage after pages none of them rose above blind answering in the
10,000 to 30,000 moments it had there (0.00 to 0.02), and first words and pages fell by about a tenth meanwhile:
the rule moves the learner on, it does not make the next stage learnable. The old mark had let
the earlier lives go after 5,000 to 10,000 moments at 0.27–0.29: it was sending the learner on with little more
than half of what it could get from the stage.
**[pending: those lives; the same lessons in shuffled order.]**

Some things we observed without asking for them. Echoing fades by itself once words start to work; by the pages
stage the learner repeats the question's last word in under a tenth of its answers. It has a slight bias towards
"no", where two-year-old children are reported to have a bias towards "yes". And on the life's own photographs,
with each naming moment taken once, the word cells name 0.52 of held-out pictures among 16 to 25 names, where the
same architecture trained by backpropagation on the same moments names 0.39 after one pass and 0.50 after five.
That backpropagation run had a single learning rate and optimiser and was not tuned as the CIFAR-10 baseline was.

### Removing one part at a time

We ran each ablation as three lives from random initialisation and read the scores at 50,000 moments.
Figure 5, `fig_ablation.pdf`, shows the seeds; the table gives the means.

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

### One look at a time

Looking, as the lives above have it, is generous. The word cells are handed the whole view and every region the
gaze reflex picks, all at once and at no cost. No eye works like that, and there was nothing in it to learn. So we
rebuilt it: the word cells first get the whole view, coarse, and each look then brings them one region in detail.

Before building anything we asked the word cells a question: do they already know where to look? We raised them
as usual, then took looks one at a time and ranked the candidate regions by how strongly a blurred view of each
drove the cells of the words in the question. On pages where the order of looks could matter, the named thing
was in the fovea on the first look 81–85% of the time (three seeds), against 46% when the regions were ranked by
contrast and 38–41% at random. Nothing new was learned for this. It is the word cells' own picture of what a
word's thing looks like, the "template" of the attention literature (Bichot et al., 2015), and it is how a heard
word draws a listener's eye (Tanenhaus et al., 1995), in infants too (Bergelson & Swingley, 2012). The learner
had it all along and had never used it to move its eye.

What remained was how the eye should weigh what stands out against what was asked for, and we got that wrong
twice before getting it from the literature (see the next section). What is established is a matter of timing:
contrast moves the eye only in the first quarter of a second after a scene appears, and later looks follow what
is sought (van Zoest, Donk & Theeuwes, 2004; Donk & van Zoest, 2008; Schütz et al., 2012). A look drawn by a
heard word always comes later than that. So in the learner what stands out is a transient that fades from the
moment the scene appears, and the pull of the heard words is sustained. Nothing about the balance is learned,
and no weight of ours sets it. Before the word cells know a word, its pull is nothing and the reflex decides,
as in a newborn.

We tested this in small learners, not in lives: 6,000 moments each, starting in first words with an ear that had
already heard speech, graded by the letter, and raised twice on exactly the same moments.

| at 6,000 moments (three seeds) | every region at once | one look at a time |
|---|---|---|
| pages | 0.56 / 0.51 / 0.59 | 0.50 / 0.61 / 0.59 |
| "is there a …?" right | 0.67 / 0.74 / 0.69 | 0.68 / 0.73 / 0.72 |
| detailed views the word cells get before answering | all candidates, free | 0.86 / 0.88 / 0.88 looks |
| first look lands on the named thing | – | 0.70 / 0.70 / 0.68 |

A learner that takes about one look, at the region its words pull, answers as well as one that is given every
region for nothing. Its first look finds the named thing more often as its word cells learn (51–62% at 3,000
moments, 68–70% at 6,000; the reflex alone manages 49%). The price is about one tick more per moment. This is a
small learner and a hundred test items per seed, so "as well as" means "not told apart". Counting is untested,
and there every region must be seen. The eye movements that the gating policy could learn from reward die out in
both arms: reward and a cost on time, by themselves, teach an eye not to look.

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

Then there are the things we got wrong. Four of our assumptions were mistaken, and each of them made the learner
look better, or at least busier, than it was.

*We turned a behaviour into a target.* Echoing was a gate: a learner had to keep 80% of its echo score to move
on. Learners that had begun to use words echoed less, as children do, and we held them back for 20,000 to 90,000
moments for it.

*We wired a drive to a behaviour.* A "novelty drive" raised the probability of repeating an unknown word. It got
learners through stages. It was also a rule of ours and not something the learner did, which is exactly what we
had promised ourselves not to build. We removed it and reran every life.

*We assumed a kind judge could do no harm.* We wanted answers judged as a caring adult would judge a child, and
took it that leniency could only cost us precision. The first judge accepted "yes" for "no", "bear" for a dog, and
the question repeated back; and because an adult who accepts an answer repeats the child's word, about 7% of
lessons taught the learner its own error. Kindness about the form of an answer is harmless. Kindness about its
meaning is a training signal. The judge described above is strict on meaning for that reason, and how strict the
reward should be at each age is now an experiment of its own.

*We assumed reward could teach the eye what to trust.* When looks became serial, something had to weigh what
stands out against what was asked for. We first added the two with a weight we made up, which drowned the word.
Then we let reward learn the two weights, and got a result we liked: the word's weight rose from nothing to 0.75
and the first look began to find the named thing. The control undid it. With the word's signal shuffled across
the regions, so that it said nothing, its weight still rose to 0.53. The rule always took the best-scoring
region and then credited whatever had scored it. With that bias removed and a rate slow enough for so weak a
signal, the real weight reached 0.010 in 10,000 moments and the shuffled one 0.006. Where the first look lands
changes too few answers for reward to teach it in any span we could run. The answer was in the literature we had
already collected (the timing described above), and we had passed over it because it needed a clock.

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
The modern statements are stronger and closer to ours. Thiry et al. (2021) reach 86.7% on CIFAR-10 with one layer
of whitened patches sampled from the data and a linear classifier, find that training the patches changes
nothing, and that whitening is what the result depends on. In the online setting, Prabhu et al. (2024) show that
random features, decorrelated and read by class means in one pass, match or beat deep networks trained
continually, and that taking the decorrelation away costs more than taking the random features away. Frozen
features with a decorrelated or closed-form readout are now a standard strong baseline in continual learning
(McDonnell et al. 2023; Zhuang et al. 2022, 2024; Panos et al. 2023). Our streaming discriminant row is that
baseline, and it is ahead of our learner.
Streaming linear discriminant analysis (Hayes & Kanan 2020) and the fruit-fly associative learner of Shen,
Dasgupta & Navlakha (2021) learn in one pass without forgetting, but on features from a network pretrained with
labels by backpropagation; the sparse-distributed-memory learner of Bricken et al. (2023) likewise, and it is
itself trained by backpropagation. Closest in mechanism is a delta rule on a linear readout over fixed random
sparse codes, used for continual learning of text domains (arXiv 2607.26523). In our case no part was ever
trained by backpropagation or with labels, the input is photographs, and the comparison is the same architecture
trained end to end.

*Why decorrelation helps a learner.* That correlated inputs slow gradient learning is old (Le Cun, Kanter & Solla
1991). Recent work makes it a design principle: a local decorrelating rule at every layer more than halves the
training time of deep networks (Dalm et al. 2024), input correlations are identified as the data's share of the
natural gradient (Offergeld & Ahmad 2024), a frozen patch-whitening layer is the single most useful ingredient in
the fastest CIFAR-10 training recipes (Jordan 2024), and with decorrelated inputs even crude credit assignment
such as node perturbation comes close to backpropagation (Ahmad et al. 2022; Dalm et al. 2023). So the benefit
is not special to local rules, which is what our like-for-like baseline is there to measure. One caution cuts the
other way: full whitening can discard information a learner needs to generalise (Wadia et al. 2021); our relay is
a regularised whitening, and its regulariser mattered.

*Whitening in brains.* The relay stands in for circuitry that is being worked out now. Populations can whiten
their responses online with fixed synapses and interneurons that adjust gains (Duong et al. 2023a), with slow
synaptic plasticity setting the structure the gains work within (Duong et al. 2023b); a similarity-matching
circuit of this kind predicts the wiring of the fly larva's olfactory system (Chapochnikov et al. 2023); and
recordings place a decorrelation step in visual cortex and not before it (Elsayed et al. 2026). The picture is
not uniform: under natural gaze shifts the retina's output is more redundant than efficient-coding theory
predicts (Karamanlis et al. 2025).

*Hebbian vision.* Deep Hebbian networks reach high accuracy on CIFAR-10 with many epochs of local learning and a
gradient-trained supervised head (Journé et al. 2023), and their single-layer precursor reports a one-epoch
advantage over backpropagation (Moraitis et al. 2022). Hard winner-take-all with lateral inhibition has since
matched its backpropagation twin at 75% (Jiménez Nimmo & Mondragón 2025), in a pipeline that whitens its input
and reports no random-filter control, so it cannot say which of the two did the work. With depth the choice of
local rule does matter (Wu et al. 2026). We add the single-pass setting with a label stream, the random-filter
control, the gradient-readout control, and the measurement that, for one layer and our rule, decorrelation
accounts for the advantage.

*Hebbian models of word learning.* The assembly-calculus models of language (Mitropolsky & Papadimitriou 2023,
2025) and the brain-constrained spiking models of Tomasello et al. (2018) and Constant, Pulvermüller & Tomasello
(2023) learn words with Hebbian rules alone, from constructed input: toy sentences, and pre-formed patterns that
stand in for vision. Ours learns from photographs and from free adult speech, and is far less principled as a
model of any brain area.

*One pass with backpropagation, and a child's view.* A contrastive model of a child's view, trained in one
chronological pass, falls to near chance without replay buffers that hold most of the stream (BabyCL; arXiv
2606.05115). This is the setting in which our end-to-end baseline also trails. Trained offline, a contrastive
model learns word-to-referent mappings from 61 hours of one child's head-camera recordings (Vong et al. 2024).
Whether the order of a childhood helps a learner is unsettled: curricula mostly failed to help language models
trained on child-sized corpora (Warstadt et al. 2023), while developmental order helped visual learning from
infant video (Sheybani et al. 2023). Our stages are an assumption we have not yet tested against shuffled order.

*A model as the judge.* Large language models agree with human graders often but not reliably, small ones less
(Zheng et al. 2023; Thakur et al. 2025; Bavaresco et al. 2025), which is why we report our judge's measured
errors and still owe the reader a person's labels.

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

Everything is public at <https://github.com/harrisonford/organic-learning>. The repository holds the logs of every run cited here and the scripts that produce each table and figure
(`experiments/bench_*.py`, `experiments/curriculum_life.py`, `experiments/paper_figures.py`). **[A snapshot with a
DOI before submission.]**

## Appendix: settings, compute, and the judge's instruction

**Settings.** Nothing below was searched over except where the table says so.

| Part | Setting | Value |
|---|---|---|
| Patch whitening | patch; regulariser (share of the largest eigenvalue) | 5 × 5 colour; 0.001 |
| Visual cortex | filters; Hebbian step; penalty on a filter that wins too often | 256; 0.02; 0.3 × (its wins / mean wins − 1) |
| Visual cortex | patches learned from, per picture | 32, among those with at least 0.3 of the picture's strongest contrast |
| Summary | average of each filter over a grid | 4 × 4, giving 4,096 numbers |
| Relay | pictures it is estimated from; regulariser | 6,000 unlabelled; 0.0005 (chosen among four values, see mistakes) |
| Word cells | most cells; step of synapse $i$ | 2,000; $1/(1 + u_i/5)$, shared among the inputs active in that moment |
| Episodic store | moments held; steepness of the comparison | 2,000; 12 |
| Speaking | cost of each tick of waiting; most ticks; step of the patience | 0.02 of the reward; 12; 0.05 |
| Tests | questions per stage; how often | 300 held out; every 5,000 moments |
| Moving on (lives with no pass mark) | tests averaged on each side of the comparison | 3 |
| Backpropagation reference | optimiser; rate; batch (searched: 14 settings on 10,000 pictures of seed 0) | Adam; 0.003; 32 |

**Compute.** Everything ran on one desktop machine: an eight-core AMD Ryzen 7 5700G with 15 GB of memory, and an
8 GB graphics card (RTX 3070) that serves only the judge. The learner is plain NumPy on the processor. A moment
takes about half a second, so a life of 115,000 moments takes some sixteen hours, and three lives run side by
side. One pass of the benchmark stream takes minutes for the readers on a fixed code and about a hundred minutes
for the backpropagation reference.

**The judge.** A language model of about four billion parameters, served on the same machine and asked for the
probability of "yes". For questions the world settles (what is in a picture, how many, yes or no) its instruction
is, word for word: "The adult's own answer is the truth about what is there. Did the child mean the same as the
adult? Another way of saying the same word counts (baby talk, a mispronunciation, an extra word). A different
thing, the opposite of yes or no, a different number, or only repeating the adult's question does not count.
Several different words run together into one (the right word stuck to the names of other things) do not count
either." For open exchanges, where many replies are fine, it is asked instead whether a caring adult would accept
the child's answer as a right answer for a child of that age. Every test in this paper is scored under these two
instructions and no other.

**The term.** "Organic learning" has been used before as the title of a United States patent on network
architectures (US 10,846,592). We use it for a method, the five commitments of the introduction, and not for an
architecture.
