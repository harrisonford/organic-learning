# The arXiv endorsement for cs.NE

2026-11-23: arXiv asks Kris's account for an endorsement before it accepts a paper in cs.NE. arXiv sent Kris an
endorsement code by email. **The code is not written in this repository (it is public); it goes only into the note
Kris sends to a chosen endorser.** Kris sends the notes; nobody writes to authors on Kris's behalf.

## The rule (arXiv's own words, 2026-11-23)

An endorser "must have submitted 3 papers to any of [the cs categories] earlier than three months ago and less than
five years ago". Whether a person qualifies: open one of their papers on arXiv while logged in and click "Which of
the authors of this article can endorse?" at the bottom of the abstract. Endorsement says the paper belongs in the
category; it is not a review. It is "necessary but not sufficient": moderators still look at the submission.

## Before any note is sent

- [ ] The paper compiles to a PDF Kris has read from start to end and can answer questions about.
- [ ] Every reference opened by hand (REFERENCES.md).
- [ ] The author line and the AI-use statement are Kris's own.
- [ ] The person's endorser status checked with the link above.

## Whom to ask (authors the paper cites; closest work first)

Ask one or two at a time and wait a week before the next. Strangers often do not answer; that says nothing about
the paper. Never the same note to a list of people at once.

| Whom | Their paper we cite | Why them | Abstract page (for the endorser link) |
|---|---|---|---|
| T. Moraitis, A. Journé | Hebbian deep learning without feedback (SoftHebb) | the nearest claim: Hebbian learning without backpropagation and an advantage when each example is seen once; our benchmark is a direct check of it | https://arxiv.org/abs/2209.11883 |
| D. Mitropolsky, C. Papadimitriou | The architecture of a biologically plausible language organ | the nearest aim: language acquisition from local rules; we use real photographs and judged speech where they use toy input | https://arxiv.org/abs/2306.15364 |
| C. Pehlevan, D. Chklovskii | A Hebbian/anti-Hebbian network … | we use their network's rules unchanged and report what it does to a one-pass learner | https://arxiv.org/abs/1503.00680 |
| D. Krotov | Dense associative memory for pattern recognition | the store's steep comparison; also an author of the sparse-distributed-memory continual learner we compare with | https://arxiv.org/abs/1606.01164 |
| Y. Shen, S. Dasgupta, S. Navlakha | Algorithmic insights on continual learning from fruit flies | a local-rule continual learner on fixed features: the same finding from the other side | https://arxiv.org/abs/2107.07617 |
| T. Hayes, C. Kanan | Deep streaming linear discriminant analysis | our strongest baseline is their method; we report that it stays ahead | https://arxiv.org/abs/1909.01520 |
| M. Ren, B. Lake (with Jiang, Yang, Norman) | BabyCL | one chronological pass of a child's experience; the nearest setting for the staged life | https://arxiv.org/abs/2606.05115 |

(The older papers on the list are outside the five-year window themselves; what counts is the author's recent
papers, which the endorser link checks.)

## The note (a draft for Kris to make their own)

Subject: Request for an arXiv endorsement (cs.NE): a preprint that builds on your work

Dear Dr. [name],

I am an independent researcher and I am writing to ask whether you would consider endorsing my first arXiv
submission in cs.NE. I ask you because the paper [uses / tests / compares with] your work: [one sentence that is
true for this person, from the table above].

The paper describes a small learner made only of local learning rules, with no backpropagation, that sees each
moment once. On CIFAR-10 as a single-pass stream it names 0.687 of held-out pictures against 0.616 for
backpropagation of the same shape, and we show that the lead belongs to a decorrelated code and not to the learning
rule (a gradient readout on the same code does as well; a streaming LDA stays ahead). We also report what did not
work, and three assumptions of ours that turned out wrong.

I should say plainly how it was made: I directed the work and I am responsible for it; an AI assistant wrote most of
the code and the first draft under my direction, which the paper states. The PDF is attached, and the code and logs
are public: [repository link].

If you are willing: arXiv's endorsement page is https://arxiv.org/auth/endorse and my endorsement code is [code].
Endorsing says only that the paper belongs in the category. If you would rather not, I understand, and thank you
for your papers either way.

With thanks,
[name]
[contact address]
