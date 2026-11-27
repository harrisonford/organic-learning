# Is there a paper here? (assessment, 2026-11-19)

Kris asked whether what exists is relevant enough to publish on arXiv or in a journal. This is my assessment after
two rounds of literature search (reports/Backprop free developmental learning survey.md; research_notes/Relatives
and publishability 2/: closest relatives read in full, exemplar papers and venues, the literature on judging
answers). It is a judgement, kept up to date by the loop (docs/LOOP.md).

## Short answer

Yes, there is a paper, and it is not ready. What exists today would be accepted as a poster or a two-page abstract
at a developmental or NeuroAI venue. For arXiv plus a journal (TMLR is the fit), three to five weeks of machine time
on controls are missing, most of all one controlled experiment on a public benchmark. Nothing found in either
search does what this brain does: name held-out real photographs with no backprop anywhere, each moment seen once.

## What is ours to claim (and how firm it is)

| result | number | firm? |
|---|---|---|
| Single pass, local rules, real photographs: held-out naming among 16–25 names | 0.52 (3 seeds), chance 0.05–0.09 | firm on our own pictures; no public benchmark yet |
| The same shape trained by backprop on the same moments | 0.39 after one pass, 0.50 after five | lightly tuned (one learning rate); must be tuned for one pass, and a replay version added |
| What the word cells could not use: decorrelation between cortex and word cells | 0.39 → 0.52 (3 seeds); a ridge on the same code 0.56 | firm; the decorrelation is computed in batch (a debt) |
| Fast and slow memory: naming a picture seen once | 0.9 with the episodic store, 0.1 without; another picture of the kind 0.0 | firm (3 seeds); expected by theory; needs the same store on raw and backprop features |
| One brain through babbling, echo, first words and two words | 3 of 3 seeds | weak as evidence: the stages are ours, the passes were made with a hand-wired drive and a judge that taught mistakes; three clean lives started 2026-11-19 |
| Honest negatives | old cortex's development worse than random filters; a second layer +0.01; dendritic branches; replay did not protect; no new-word bias; text answers no better than a parrot; preschool not passed | firm, and an asset |
| Method | a retention gate on a behaviour that was not a target; a model judge that taught the learner its mistakes; a world that silently reverted | worth a section: nobody reports these |

## Where it sits

- **Closest in spirit**: the assembly-calculus language models (Hebbian, no backprop; toy input, pre-formed
  "vision") and the Berlin brain-constrained Hebbian models (fast mapping after a babbling stage; six words as random
  patterns). Both are proofs of concept on made-up input. Ours is on real photographs and adult-written text.
- **The comparison a reviewer will reach for**: BabyCL (June 2026), the child's-view contrastive model trained in
  one chronological pass: 27.5 on a four-way choice (chance 25) naively, 43.4 with replay buffers holding 60% of the
  stream, 57.3 offline with 400 epochs. It says one-pass learning is hard for backprop, which is our niche.
- **The warning**: "Alignment Without Binding" (workshop, 2026): strictly local credit fitted image–caption pairs
  and barely transferred; a frozen random encoder did better. We need the random-filter row in every table.
- **What the field accepts** (twelve exemplars read): 3–5 seeds with spread, a same-shape backprop baseline, an
  ablation table, at least one public benchmark, stated failures. Small teams and single authors do get in (TMLR
  2026 has a single-author controlled delta-rule study). A small Hebbian forgetting paper from a university team
  was rejected by TMLR the same year.

## What the paper is

Not "an alternative to backprop" (the scaling evidence says no) and not "a model of the child" (our input is not a
child's). It is: **what each part buys in a learner that sees each moment once and uses only local rules**, in two
halves.

1. **A controlled core on a public benchmark.** One stream, one pass, accuracy against samples seen, five seeds:
   - the learning rule: ours, against online backprop of the same shape tuned for one pass, with and without replay;
   - the readout on the same features: word cells, nearest class mean, streaming discriminant analysis, ridge;
   - the features: developed filters, random filters, with and without decorrelation;
   - the episodic store: on and off, also on raw and on backprop features (does the dissociation need our cortex?).
2. **A staged life with the same parts**: three to five clean seeds through first words and two words, scored above
   the blind floor, with the observations (echo fading, yes bias, the new-word bias absent) and the failures
   (earlier stages lost at a stage change; text at the parrot's level; preschool).

## What is missing, in order

1. The public-benchmark experiment (CIFAR-10 and CIFAR-100 as one-pass streams; all offline; days, not weeks).
2. The backprop reference tuned for one pass (learning rate, batch size), and with replay at matched updates.
3. Gradient-free readouts on our features (nearest class mean, streaming discriminant analysis).
4. An ablation per organ in lives (relay, episodic store, looking, match neurons, growth, chooser).
5. The same lessons in shuffled order.
6. Five seeds for every headline number.
7. Forgetting reported the standard way (backward transfer), with a small-rehearsal baseline.
8. The decorrelation learned online by a local rule (or stated plainly as computed in batch).
9. A person's labels on the judge's audit set.

## Where and when

- **Now, as it stands**: nothing archival. A two-page CCN abstract (due 1 April 2027) or an ICLR 2027 workshop
  paper (lists out 29 November 2026, papers due about 1 February) would likely be accepted and are compatible with a
  later journal paper.
- **When 1–6 exist**: Zenodo snapshot of code and data; arXiv in cs.NE (a personal endorsement is the gate: write
  individually to a few authors of the papers we build on, with the draft and the code); TMLR (median about four
  months to publication; near even odds in my and the searcher's judgement).
- **Not compatible with TMLR**: ICDL full papers and CCN eight-page papers are archival. CogSci's six pages
  (due 1 February 2027) is a grey area.
- **Disclosure**: the paper needs a section saying that an AI assistant wrote the code and ran the experiments,
  what Kris decided and verified, and that every reference was opened by hand. arXiv's section chair has announced a
  one-year ban for hallucinated references; the notes mark which numbers were verified.

## My recommendation

Do the controlled core first (it is cheap and it decides whether the efficiency claim survives a tuned baseline),
keep the clean lives running beside it, and decide about venue when the table exists. If the tuned one-pass
backprop baseline catches the brain, the paper becomes the fast/slow dissociation plus the staged life and the
negatives, which is still a workshop paper and an honest preprint, not a journal paper.

## Update, 2026-11-20: what the first benchmark seed does to the claims

(CIFAR-10 as a one-pass stream, seed 0: docs/VISION.md, last section. One seed; the tuned backprop runs are not in.)

- **Gone as a claim: "our learner is the best without gradients".** On the same code streaming LDA and ridge are 2
  to 5 points above the word cells (0.74 against 0.69 after 50,000 pictures). The word cells are an online, local
  approximation of what those compute with a matrix inverse; that is the honest sentence.
- **Gone as a selling point: the cortex's development.** Random filters on the same decorrelated windows do as
  well (0.691 against 0.686). What matters in the visual path is the decorrelation (windows, and the relay: +10
  points for the word cells), not the Hebbian development of the kinds. The paper must say so and carry the
  random-filter row everywhere.
- **Standing, for now: single-pass efficiency against backprop of the same shape** (0.62 against 0.40–0.48 after
  10,000 pictures in the first three settings). It waits for the rest of the tuning grid and for replay.
- **Standing: the decorrelating relay** as the one architectural finding that transfers to a public set.
- **To be measured on the benchmark:** fast and slow memory with a class held back (`bench_newclass.py`).

So the paper's centre of gravity moves: from "a brain that learns without backprop" to **"what a single-pass
local learner needs: decorrelation at two places and a fast store beside slow weights; what it does not need:
developed filters; and what it still cannot do"**, with the staged life as the second half. It is a smaller claim
and a more defensible one.

## Update, 2026-11-20 (later): the aim is an arXiv preprint; and what the benchmark is for

Kris: "it could be just an arxiv publication right?" Yes. A preprint needs no reviewer; it needs (1) a personal
endorsement from an established arXiv author in the category (the one real gate for an unaffiliated first
submission), (2) to read as a finished research article (moderators decline drafts; versions are permanent),
(3) the AI-use statement, (4) every reference opened by hand. For it to be worth an endorser's name it should carry
three seeds on its headline numbers, the same-shape backprop reference with replay, the random-filter rows and the
negatives. That is what the running queue and the two-words lives produce; no further scale is needed.

Kris, the same day: "NEVER FORGET we dont want to hack performance for the sake of it we want to build slow organic
wins." For the paper this settles what the benchmark is: a ruler. Its numbers are reported as they fall; nothing
in the brain is changed to move them (no more filter kinds, no tuning to CIFAR). What the paper offers is how an
organic learner behaves and what each part does, measured honestly, not a score.

## Update, 2026-11-21: three seeds, and the control that moves the claim once more

- Three seeds hold (VISION.md): word cells through the relay 0.687 ± 0.005; one-pass backprop of the same shape
  0.616 ± 0.023 (0.688 with four replayed pictures per new one); random filters 0.693; streaming LDA 0.739.
- **A gradient softmax readout on the same relayed code does as well as the word cells** (0.70 against 0.69, two
  seeds so far). So the efficiency claim is not "a local rule beats backprop": it is **"fixed, decorrelated features
  read by one layer of local synapses learn more from one pass than the same shape trained end to end"**, and a
  delta rule on one layer is gradient descent on that layer. Written into the draft's abstract and first claim.
  This is a known direction (random features with a linear readout); what is ours is the measurement inside a
  developmental learner, the two places of decorrelation, the fast store beside it, and the staged life.
- The draft is docs/paper/DRAFT.md. Pending in it: the readout control's third seed, the ablation lives, the
  related-work section with every reference opened by hand.

## Update, 2026-11-23: where publishing stands

- **The paper**: docs/paper/DRAFT.md, second draft, rewritten for readers after three independent readings; title
  "Organic Learning: Small-Scale Architecture Experiments Without Backpropagation" (Kris's; the term defined in the
  introduction; the phrase is the title of US patent 10,846,592, to be cited). Built to PDF by docs/paper/build.py
  (15 pages, five figures, an appendix; 26 recent papers cited after the survey of 2026-11-23, docs/RESEARCH_RECENT.md). An explainer page for Kris: docs/paper/explainer.html.
- **What a hostile referee found and the logs confirmed** (VISION.md, 2026-11-23 18:00): both whitening steps are
  batch; like for like (no relay) backprop wins 0.616 to 0.591; streaming LDA needs no relay; the life repeats its
  lessons; one number was tuned. The draft says each. **The claim is now: a decorrelated code, and not the local
  rule or the developed filters, carries the single-pass advantage; shown with the controls that could have
  refuted it.**
- **Owed before sending** (docs/LOOP.md, queue D): backprop with the relay step (running); the local lateral
  network on the whole stream; all 10,000 test pictures; a life with no going back; three seeds of the answering
  times; a person's labels on 200 judgements; picture licences for the examples figure; every reference opened by
  hand (docs/paper/REFERENCES.md); the author line and the AI-use statement in Kris's words.
- **arXiv**: Kris's account needs an endorsement for cs.NE (docs/paper/ENDORSEMENT.md; the code is kept out of this
  public repository). A former coauthor with computer science papers on arXiv is the likeliest endorser. Licence:
  CC BY 4.0.
- **Zenodo**: Kris's account is linked to ORCID; a DOI is to be reserved in a draft record, nothing published until
  the paper is ready; then two linked records (the PDF; the code and logs from a GitHub release).

## Update, 2026-11-24: where publishing stands

- **The paper**: 16 pages. New since the last update: the like-for-like comparison with backpropagation is closed
  (with the relay step in front of both, backprop 0.688 ± 0.005 and the word cells 0.687 ± 0.005: level; a relay
  that tracks the stream and a wider search of its settings found nothing better); a subsection on looking one
  place at a time (three seeds: a learner that looks where the heard words pull answers as well as one handed
  every place at once); a fourth mistaken assumption (that reward could teach the eye what to trust; the
  shuffled control undid a result we had already written down). **Draft 7 (17 pages) sent to Kris by email on 2026-11-24 21:06**, with the section
  "Giving the cortex what a cortex has" (normalisation, blurred start and the sparse layer on three seeds; the
  drift stage on one, marked pending).
- **Owed for that section**: three seeds each of divisive normalisation, of a second stage that learns from the
  eye's drift, and of a blurred start (seed 0 of each is read, VISION.md 2026-11-24; seeds 1–2 running); each
  beside the same-shape backprop and the streaming baselines; then the sparse expansion (queue E 29f, running).
- **Still owed before sending** (unchanged): the local lateral network on the whole stream; all 10,000 test
  pictures; a life with no going back; three seeds of the answering times; a person's labels on 200 judgements;
  picture licences for the examples figure; every reference opened by hand (docs/paper/REFERENCES.md, six more
  rows since yesterday, none opened); the author line and the AI-use statement in Kris's words.
- **arXiv and Zenodo**: as on 2026-11-23; nothing uploaded.

## Update, 2026-11-25: where publishing stands

- **The paper**: the drift stage's row and paragraph are filled with three seeds (the rule gives back half of the
  tolerance a second stage loses and changes nothing on the stream); its pending mark is gone. Draft 7 is the
  last one sent; draft 8 goes when one of the two items below changes what the paper says.
- **Two items that may change the paper's words** (both running, VISION.md 2026-11-25):
  (1) *the relay by a local circuit* (a published interneuron rule): its resting point is as good as the batch
  relay with no hand-set constant; whether the online rule gets there in a stream is being read on three seeds.
  If it does, "two batch stand-ins" becomes one. (2) *the published soft winner-take-all rule, unchanged*: in its
  authors' measure in small it replicates (0.616 developed against 0.515 as born); its reading in our measure is
  running. **The paper's sentence "developed filters do no better than random" is about our hard rule and must
  say so**, whatever the stream shows; if the soft rule is ahead in our measure too, the section is rewritten.
- **References to add as unopened when cited**: Lipshutz, Pehlevan & Chklovskii 2022; Duong et al. 2023a, 2023b
  (the relay circuit); Journé et al. 2023 and Moraitis et al. 2022 (the soft rule); Tobler, Fiorillo & Schultz
  2005 and Louie & Glimcher (the reward's error against its usual size).
- **Still owed before sending**: as on 2026-11-24, less the drift stage's seeds.
- **arXiv and Zenodo**: as on 2026-11-23; nothing uploaded.

### Update, 2026-11-25 04:11
- **Draft 8 (18 pages) sent to Kris.** Both items of the last update changed the paper:
  (1) the soft rule on three seeds: the abstract has a sentence, and the section is now "Our Hebbian development
  did nothing for the learner" with the three seeds at 10,000 pictures; (2) the relay circuit on three seeds:
  the abstract's sentence on the batch steps and the section "The relay, learned locally: two attempts".
- **New pending mark:** the whole stream on the soft layer with backprop of that layer's shape
  (`organic-softbase1`, about eight hours). When it is in: the benchmark table gains the soft layer's rows, and
  the headline comparison (abstract, "The local learner beats one-pass backpropagation", "What we cannot claim")
  is rewritten on whichever layer is the paper's learner. **Until then the headline is on our cortex and says so.**
- **Reference added, unopened:** Lipshutz, Pehlevan & Chklovskii 2022 (title and venue to be checked on the
  page). Journé 2023, Moraitis 2022 and Duong 2023a/b were already in the list. Not yet cited, so not yet added:
  Tobler, Fiorillo & Schultz 2005; Louie & Glimcher.
- **A caution for the reader of the draft:** the circuit's numbers are from blocks of 50 pictures and a matrix
  solve for the settled activity; the paper says both.

### Update, 2026-11-25 10:50
- **In the draft since draft 8:** a paragraph on the soft rule taken apart on three seeds (the losers' unlearning
  does the work; the division by the layer's largest change is not local and neither local stand-in works). No
  new reference is cited by it. Draft 9 goes to Kris when the soft layer's whole-stream rows are in.
- **The pending mark stands** (seed 0 complete, seed 1 begun, about 16:00 for the rest). What seed 0 says, so that
  the rewrite is not a surprise: end-to-end backprop of the soft layer's shape 0.547 (0.597 with the fixed
  relay) against word cells + relay 0.729; **but a softmax readout trained by gradient on the frozen developed
  filters through the relay reaches 0.737, over the word cells**, as streaming LDA does (0.743). On our cortex
  that same row was level with the word cells (0.694 against 0.693). The headline has to be said as it is:
  local filters with any one-pass reader beat one-pass backprop of the same shape; our own reader is not the
  best of the readers.
- **Not in the paper, and not to be until they are more than one seed:** the BCM layer (independent cells end as
  ten filters; under the layer as born in the stream) and SAILnet (one filter on raw windows; 98 different
  filters of 256 on our cortex's windows, stream reading pending). If either enters the paper, its references
  are: Intrator & Cooper 1992 and Blais et al. 1998 (opened); Bienenstock, Cooper & Munro 1982 (reference
  verified, text not opened); the Plasticity simulator as software, with the date read; Zylberberg, Murphy &
  DeWeese 2011 (preprint opened, **journal pages to be compared**) and its code; Földiák 1990 (opened).
- **The photo world** stood still for a day (a free model withdrawn); mended (docs/WORLD.md). Nothing in the
  paper depends on it, but the lives of the next generation will start in a world whose photo questions are
  partly written by a different adult: one more reason for the frozen-world question put to Kris.
- **arXiv and Zenodo:** nothing uploaded.
- **2026-11-25 12:59: a caution on two of the paper's sentences, pending three seeds.** The benchmark shows the cortex
  CIFAR blown up from 32 to 64 pixels, and a 5 × 5 window of a blown-up picture holds 22 directions where a
  real picture's holds 47–48 (`experiments/upsample_probe.py`). On seed 0, at CIFAR's own size, our hard rule's
  development is worth three points (0.656 against 0.627), where the paper says it "did nothing". If seeds 1–2
  agree: (1) that section says the finding was on blown-up pictures and gives the native-size rows; (2) the
  sentence comparing our cortex (0.620) with the soft layer (0.689) says the two looked at different sizes,
  or gives our cortex's native-size number beside it; (3) the methods say plainly that the benchmark's
  pictures are enlarged to the eye's size. **A decision for Kris then:** whether the paper's benchmark stays
  "the learner as it lives" (64-pixel eye, CIFAR enlarged) or moves to the pictures' own size.

- **2026-11-25 14:55: the soft layer's table is in the draft** (new section "The same contest on the better layer",
  ten rows, three seeds each: word cells + relay 0.730 ± 0.004; backprop of the layer's shape 0.562 ± 0.013, with
  the relay 0.606 ± 0.008; gradient readout on the frozen soft filters + relay 0.739 ± 0.003). The old pending
  mark is gone; a new one names three rows still running (the layer as born on the whole stream; backprop
  started from the soft filters; the frozen readout without relay at the layer's own exponent). **The abstract
  and the title sentence are unchanged**: the caveat (the soft layer developed on 50,000 other unlabelled
  pictures) cannot be resolved before the row "backprop started from the soft filters" is in. Draft 9 goes to
  Kris when the as-born rows are in (this evening); the abstract is reworded after softbase2.
- **2026-11-25 15:55:** the Setup of Experiment 1 now states that the eye is 64 pixels across and CIFAR pictures are
  enlarged to it by bilinear interpolation (item 3 of the 12:59 caution; it was true whatever the native-size
  seeds say, and the draft had not said it).
- **2026-11-25 16:55: draft 9 sent to Kris** (20 pages): second benchmark table (soft layer; as-born rows in on three
  seeds: word cells + relay 0.630 ± 0.004, without 0.503 ± 0.010), the three-seed dissection, the eye's size in
  the Setup. Pending in it: backprop started from the soft filters; frozen readout at the layer's exponent
  (softbase2). Abstract to be reworded after softbase2 and the native-size control.
- **2026-11-25 18:55: the native-size control has three seeds and they agree** (development by our hard rule: +0.023
  for the word cells behind the relay, +0.034 for ridge, at CIFAR's own size, 10,000 pictures). The section
  "Our Hebbian development did nothing for the learner" is true only of enlarged pictures and will be reworded
  with both sizes side by side when the enlarged rows on the same 10,000 pictures are in (tonight). The
  abstract's "random filters do as well as the Hebbian filters we had carefully developed" needs the same
  qualification. Draft 9 (sent 16:55) still has the old wording; the email said so.
- **2026-11-25 21:35: the paper's section on our development now shows both sizes** (three seeds: enlarged −0.022 /
  +0.009, own size +0.023 / +0.034 for word cells / ridge behind the relay), the soft-layer comparison is
  corrected (0.689 against 0.649 at the same size) and the abstract's sentence is qualified. **Still to do,
  following Kris's rule 10 (real pictures of all kinds):** the photo stream becomes the vision evidence beside
  CIFAR-10 when its table and its backprop reference are in; the title of the section ("did nothing") and the
  list of mistaken assumptions (the enlargement is a fourth) are reworked then; the "658 near-copy pairs"
  sentence is checked against the 257–282 measured today. Draft 10 after that.
- **2026-11-26 01:40: the soft layer's section is complete** (14 rows on three seeds; backprop handed the developed
  filters 0.703 ± 0.004 against the word cells' 0.730 ± 0.004 and 0.739 with the filters frozen; the abstract
  says 73% / 56% / 70% and that most of the lead was bought with unlabelled pictures). Also in since draft 9:
  the two-size table, the SAILnet paragraph (three seeds), the Setup's sentence on the eye. **To come before
  draft 10:** a section for the photo stream (rule 10), with its backprop reference; then the title of "did
  nothing", the list of mistaken assumptions and the 658-pairs sentence are revisited together.
- **2026-11-26 09:15: a section on real photographs is in the draft** (33 words, COCO crops and Commons at the eye's
  size; three seeds of ours, one seed of backprop: word cells + relay 0.446 ± 0.006 against backprop + relay 0.462;
  level without the relay, 0.389 against 0.387; development helps ridge and streaming LDA by 2.5 points and the word
  cells by nothing). Pending in it: backprop's seeds 1–2 and its widened tuning. Draft 10 after those, with the
  development section's title and the list of mistaken assumptions revisited.
- **2026-11-26 19:20: the photo section has its explanation and most of its reference.** The nearest class mean
  (three seeds) loses two points from development where ridge and streaming LDA gain two: the developed filters'
  worth lies in the code's second moments, which a first-order one-pass reader cannot take (written into the
  section, with the two explanations ruled out: the relay's sample size and its floor). Backprop's widened tuning
  found nothing better; seed 1: 0.464 with the relay, 0.390 without (seed 0: 0.462 / 0.387); seed 2 running. The
  658-pairs sentence now names its threshold (0.8; the table counts 0.9). Draft 10 when backprop's seed 2 is in,
  with the "did nothing" section's title and the list of mistaken assumptions revisited.
- **Not for this paper** (a second paper, or a later version, if they hold): the two senses linked by Hebb's rule on
  Commons video (true sound 60–64% against another video's on three seeds; the claim that sight and hearing together
  name a video better than sight did not hold on the third seed), the ear that hears sound (111 word-like units on one seed since 2026-11-26 23:10, common words only), the stream brain with a vocal tract
  (babbling learns an auditory-motor map; heard vowels draw near imitations, three seeds) and the order-vs-shuffled
  video probe on CORe50 (done 2026-11-27: order teaches neither second stage anything, three seeds; the whole-frame
  code is ruled by the scene; an eye on the held object lifts the same object across sessions from 0.12 to 0.165 on
  three seeds, but the word cells do not take the gain where the ridge does. Found why on 2026-11-27, three
  seeds: not the photo section's first-order reader (the means take the gain), but the relay, which had been evened
  out on COCO; settled again on the videos' frames the cells' rule takes the gain in full, and a relay circuit that
  keeps learning along the stream gives half of it. This bears on the paper's relay paragraph: the batch relay is a
  stand-in that only holds in the world it was settled in; one sentence there, citing the control on the photo
  benchmark, when the next draft is made. Also not for this paper: the stream brain's eye with pursuit, 0.137
  against the whole frame's 0.118 on three seeds; read beside the whole view, that eye's fixation adds three points
  of naming with relay circuits and nothing batch, 0.387 / 0.386 / 0.394 against 0.357 / 0.352 / 0.361, said
  before it ran, 2026-11-27 15:19). **A negative to keep on record for that later paper** (2026-11-27 14:24–15:19,
  seed 0): the text-free ear's units do not find a caption's photograph (recall at chance); the eye's code and the
  Hebb link do from written words (R@10 0.11–0.15, a measure only); the ear holds the common words only, and two
  thirds of a word's trace is lost between voices. **And its other half** (16:41; **three readers agree, 17:33:** R@10 0.048 / 0.065 / 0.065, 72–92% of written words through the same link): an
  ear that grows up in one voice finds the photographs from its own sound kinds (R@10 0.048 against 0.067 from
  written words through the same Hebb link, shuffled 0.005) and hears rare words. If three readers agree, this is
  the opening result of the second paper (speech grounded in sight with no text and no backprop, starting as an
  infant does, in one voice), with the paradigm's visually grounded speech as its reference row. Kris may present the work to neuroscientists in about two weeks; the artifact
  (https://claude.ai/artifact/NLZYiDC19SLy3UHpsvRHwL) carries the parts, their stand-ins and checked references.
- **2026-11-27 15:26: draft 11 emailed to Kris** (22 pages). One change: a paragraph at the end of "The relay,
  learned locally" (a batch relay fitted on photographs does not hold on CORe50's videos: 131–137 effective
  directions against 511–534 at home; the word cells miss a better code that ridge takes; fitted again the cells'
  rule takes the gain in full, the learning circuit half of it; the photo benchmark is the control; three seeds),
  a clause in Limits, and Lomonaco & Maltoni 2017 in REFERENCES.md (to open by hand). Pending marks unchanged.
- **2026-11-26 21:55: draft 10 emailed to Kris.** The photo section is complete on three seeds (backprop + relay 0.463
  ± 0.001 against the word cells' 0.446 ± 0.006; level without the relay, 0.392 against 0.389; the word cells ahead
  early), with the nearest-class-mean rows and the second-moment reading; the development section retitled "What our
  Hebbian development is worth" with both sizes; "What we got wrong" now has five assumptions (the enlargement added;
  the abstract had said three while the section listed four).

