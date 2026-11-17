# The craft: how this brain is worked on (kept up to date; started 2026-11-16)

Kris asked for the process to be documented, not only the results. This file is the method and the running
ledger of what was tried in each cycle and why. The detailed numbers live in the files it points to.

## The two questions, in this order

1. **Is it how living brains and beings work?** A mechanism is proposed with its biology first (who showed it, in
   what animal or child, what the rule is). A performance trick with no counterpart in life is not built, however
   well it would score. If the biology is contested, that is said.
2. **Is the answer more or better experience instead?** Before changing the architecture: is there enough to see
   and hear, is it of the quality a child gets (things in view while they are talked about, replies to what the
   child itself said, things seen whole and many times), has it lived long enough, are there enough layers for
   what is asked? Composite moments, seeing and hearing together, are the normal case in life and should be here.

Only then: does it help, measured.

## Working rules

- **Measure the ceiling before building.** For any pathway, first ask what a simple reference learner (a count, a
  nearest neighbour, a linear or kernel readout fitted in a batch) gets from the same input. References are not
  brains; they locate the loss. Three times this cycle the reference said "the loss is not where the plan
  assumed": the last-word parrot, the unanswerable photo questions, the ear's cuts.
- **Offline first, three seeds, then a life.** A probe on cached data takes minutes; a life takes a day. A claim
  needs three seeds (three orders of moments) and 300-item tests. A single seed is called a single seed.
- **One change per life**, so that a life answers one question. When two things changed, the confound is written
  down.
- **Negative results are kept**, with the code, and with what was learned from them. A design that fails is
  usually the fastest way to the next one (the lexicon ear → the carving ear → the edge ear).
- **Hand-set numbers are debts.** Each is named where it stands, and removed when the brain can measure the
  quantity itself (the copy route's 0.75 recency; the ear's two thresholds are still owed).
- **Health before any report** (`experiments/health.py`), long jobs under `systemd-run`, probes capped in memory so
  that a life is never the process that dies.
- **Everything is written twice**: here in `docs/` in full, and one sentence per decision or finding in memoize.

## Ledger

### Cycle of 2026-11-10 to 11-16 (after "L3 only repeats the last word")

| what | organic basis | data question | result | where |
|---|---|---|---|---|
| word cells in place of the hub (unit cells, two dendrites, company links) | dendritic prediction (Urbanczik & Senn 2014); fast and slow learning systems (McClelland et al. 1995) | – | pictures 0.29 → 0.44 offline; lives pass first words; stop at two words | AREAS.md |
| the copy route's hand-set recency removed; novelty from the ear's own counts, the heard words competing | repetition suppression; echoing as the default when meaning is silent (dual stream) | echo games reworded by the adults so the word is not always last | echo with the word anywhere learned; last word said in 6–17% of replies | CONVERSATION.md §6 |
| one stream instead of three routes | one motor plan fed by inner and outer speech | – | negative: the word cells always have something to say | CONVERSATION.md §6 |
| the parrot beside every score, and as the gate | – (a yardstick) | the lessons' own last-word habit is real in speech to infants and stays | a stage is no longer passed on what a parrot answers | CONVERSATION.md §6 |
| heard words bringing back what was seen (feedback synapses, bursts) | cell assemblies (Pulvermüller); labels shape perception (Lupyan) | – | no effect; capped by the eyes and by the ear's units | CONVERSATION.md §6 |
| sound of what was heard as an input to the word cells | combination-sensitive auditory neurons | – | +0.05 same answer across wordings, little else; not adopted | CONVERSATION.md §6 |
| a lexicon ear (cheapest explanation by known units) | known-word segmentation (Brent & Cartwright 1996) | – | negative: glues collocations, naming 0.21 | EAR_CARVING.md |
| words heard alone carve longer speech | free forms; isolated words (Brent & Siskind 2001); possible-word constraint (Norris et al. 1997) | adults' one-word answers are the isolated words | names consistent across settings; text unchanged; kept as the teacher | EAR_CARVING.md |
| the ear learns what word edges sound like | infants learn language-specific edge cues from known words (Jusczyk et al. 1999; Mattys & Jusczyk 2001) | – | offline: parrot-proof text +0.10 in 3 seeds. In lives: text +0.08, pictures −0.05, first words passed later | EAR_CARVING.md |
| a word is the same word whatever mark follows it | pauses and marks learned by the ear itself | – | offline: text +0.02 more, pictures unchanged | EAR_CARVING.md |
| word cells with their own firing level | a neuron's threshold between its own two levels of input | – | more often right when they speak (pictures 0.35 → 0.48), silent on half of new-word echoes instead of a quarter, but fewer answers overall | CONVERSATION.md §7 |
| replaying the day in sleep | hippocampal replay (Wilson & McNaughton 1994) | it is "more training" on the same experience | pictures +0.02 in 3 seeds, text nothing; not adopted | CONVERSATION.md §7 |
| leisure talk: a parent answers what the child said, nothing expected | contingent replies (Goldstein & Schwade 2008); recasts (Brown & Bellugi 1964) | a new kind of experience, not a mechanism | built; not yet measured in a life | LEISURE.md |
| reference ceilings for the eyes | – (a measurement) | naming rises with pictures seen, slowly beyond 8,000 with this cortex | the code holds 0.55; the two-words photo questions are not answerable from it | VISION.md |
| a bigger picture book (95,000 things cut from COCO photographs) | a child sees a dog thousands of times | yes: this is the data answer | built (world 3) | VISION.md |
| things in view when they are talked about (world 3) | joint attention; seeing while hearing is the normal case | yes | built; not yet measured | CONVERSATION.md §7 |
| pruning in sleep | overproduce and remove (Huttenlocher 1979); least used goes (Navlakha et al. 2015); sleep scales down sparing the most used (de Vivo et al. 2017) | – | literature read; first probe running | PRUNING.md |
| the cortex taught by the words; more to see for the cortex | feedback-gated plasticity (Roelfsema & van Ooyen 2005); perceptual learning by category (Sigala & Logothetis 2002) | the cortex developed on 4,000 pictures per layer; its top neurons see 17 of 64 pixels | running | VISION.md |

### What the cycle taught about the method

- The tests were paying for the wrong thing twice (the last word; photo questions nobody could answer). Checking
  the yardstick comes before reading it.
- "The limit is X" was wrong whenever it was inferred from a score and right whenever it was measured with a
  reference on the same input.
- A change that helps offline can lose in a life (the new ear and picture naming). The life is the only test that
  counts, and it is slow; so the offline probe must look like the life (the echo stage first, the same budgets).

### 2026-11-17: a life must show the world it lives in

**A bug of mine, found in the leisure talk.** `experiments/curriculum_life.py` reloads the world every 50,000 moments
(so that new items written by the adults join in). The line rebuilt the world with its default, world 1. Every life
born in world 2 or 3 therefore lived in world 1 from 50,000 moments on: no reworded echo games, no social routines,
no picture book, no thing in view when talked about. The tests were not touched (they are built once).
What caught it was not a score: the log of the leisure talk showed that talk about a picture stops at exactly
50,000 moments in all three lives.

What this changes in the record:

| lives | what they really lived | what still stands |
|---|---|---|
| words_s0..2 (world 1) | world 1 throughout | everything |
| conv_s0..2, edge_s0..2 (world 2) | world 2 to 50,000, then world 1 | passing first words at 60–85,000 was earned almost wholly in world 2. The comparison of the two ears stands (both had the same fault). "Stops at two words at 0.38–0.39" was measured in world 1's lessons |
| world3_s0..2 | world 3 to 50,000, then world 1 with leisure talk without pictures | only the checkpoints up to 50,000. Stopped on 2026-11-17 at 125–135,000 |

Fixed (the reloaded world is the world the life was born into, with its dice kept). Each checkpoint now also records
`lived`: how many moments of each stage and source the life was given since the last one, and how many came with a
picture. Three fresh lives, `world3b_s0..2`, started on 2026-11-17.

**What the first 50,000 moments of world 3 say** (valid; means of three seeds, first-words test, against conv at the
same age):

| at 40,000 / 50,000 moments | conv (world 2) | world3 |
|---|---|---|
| CIFAR pictures named | 0.54 / 0.56 | 0.43 / 0.44 |
| Commons pictures named | 0.55 / 0.56 | 0.60 / 0.64 |
| COCO things named (new) | – | 0.34 / 0.30 |
| first-words text | 0.29 / 0.32 | 0.34 / 0.36 |
| same answer however worded | 0.76 / 0.79 | 0.66 / 0.64 |
| says the last word | 0.08 / 0.06 | 0.04 / 0.04 |

- The picture book did not raise the naming of the older pictures as a whole: Commons up 0.06, CIFAR down 0.11, in
  every seed. I do not know yet which part of world 3 costs the CIFAR pictures (the COCO lessons sharing the same
  names with a different look, the picture put in view during text lessons, or the parent's whole sentences heard
  while looking at a picture). One life cannot tell these apart; an offline probe with each part alone can.
- Text is a little better (+0.045) and the last-word habit lower. Seeing the thing while it is talked about is the
  likely reason, not shown.
- The answer changes more with the wording (0.78 → 0.65).
- When the COCO lessons stopped at 50,000 the COCO naming fell from 0.30–0.34 to 0.19–0.25 within 10–35,000 moments.
  That is forgetting, measured by accident.
- Two words ended at 0.38–0.41, the same as conv. Both lived world 1's two-words lessons, so this says nothing new;
  the reference learners already said the photo questions are not answerable with these eyes (VISION.md).

**Leisure talk, read** (`out/world3_s1.chat.jsonl`, 26,000 exchanges). Nothing is graded, so this is only reading.
The child is never silent. Its whole reply is found inside what the parent just said in 22% of exchanges early and
35% late. Late in life it picks a word from the middle as often as the end ("you're safe in your cozy bed" → "cozy";
"did you hear that big truck outside" → "trucraoutside"), answers some questions with yes or no ("do you feel the
rain outside" → "no"), and blends ("you did a really funny dance there" → "yodadance"). While it was shown pictures,
it said the name of the thing in view in 12% of exchanges before 20,000 moments and 39% between 20,000 and 50,000.

**Method.**
- A life is long and unattended, so its record must say what it was given, not only what it scored. The fault lived
  from 2026-11-12 to 11-17 across nine lives because nothing in a checkpoint depended on the world.
- I reported three probes as "still running" on 2026-11-16 at 16:10. The kernel had killed all three between 15:37
  and 15:58. A unit's state is checked, not assumed, before it is reported.

### 2026-11-17, afternoon and evening: the day loop (docs/DAYLOOP.md, docs/SNAPSHOT_2026-11-17.md)

Kris: implement the three vision steps, take a snapshot, then a day of research without input; memory may be a
missing piece. What was tried, in the ledger's form:

| what | organic basis | data, training or size? | result | where |
|---|---|---|---|---|
| a reflex gaze and match neurons (looking; "is there…?") | superior colliculus; match enhancement (Miller & Desimone 1994) | architecture, no size | mechanism works with ideal eyes (1.00 / 0.89); with the old eyes it did not carry | VISION.md |
| the cortex's first layer taught by the words, slower | feedback-gated plasticity | training | +0.04; overtaken | VISION.md |
| a second cortex: decorrelated windows, graded response, 256 kinds, averaged summary | retina's decorrelation; lateral inhibition; V1 overcompleteness | size (kinds) and training | linear reference 0.28 → 0.38 (COCO), 0.49 → 0.55; the old cortex's development was worse than random filters | VISION.md, RESEARCH_VISION.md |
| a relay that decorrelates the summary before the word cells | lateral anti-Hebbian inhibition (Földiák 1990) | training (the learner's input) | word cells 0.40 → 0.51; **the word cells, not the eyes, were the limit** | VISION.md |
| corrections at full size (plasticity not scaled by the cell's own error) | Rescorla-Wagner has one rate; the error already stops a cell that is right | training | pictures +0.03 (with the relay), text beyond the parrot +0.08 | VISION.md |
| the relay as a circuit with local correlation synapses | the same, in its local form | – | 0.49 of the whitening's 0.53; not in lives yet | RESEARCH_MEMORY.md |
| a second cortex layer with local fan-in | dendrites reach a neighbourhood | size | +0.01, three times the time; not adopted | VISION.md |
| graded ("soft") competition; a centre-surround retina for the whitening; a fourth old layer | – | – | negative, negative, nothing | VISION.md |
| dendritic branches in the word cells against forgetting | branch-specific plasticity (Cichon & Gan 2015) | size | negative: all branches fill with ordinary variation | RESEARCH_MEMORY.md |
| the ear's echo decorrelated before the context dendrites | (by analogy with the relay) | training | negative: −0.08; shared words are what carries one question to another | RESEARCH_MEMORY.md |
| an episodic store beside the word cells | hippocampus; complementary learning systems | architecture, small | fast mapping 0.1 → 0.9 on the same picture (three seeds); nothing for another of the kind | RESEARCH_MEMORY.md |
| pages twice the view, coarse whole and detailed fixations; a gaze only to what stands apart | the fovea; object-based attention | training (the world) and the eye | a fixated thing named 0.47–0.53 (0.05 whole view); singles no longer pay | VISION.md |
| counting out of the two-words stage | one-knowers at 24–47 months (Sarnecka 2014) | training program | done in world 4 | RESEARCH_TWO_WORDS.md |
| an intake limit for leisure talk | toddlers keep the end of a long utterance (Echols & Newport 1992) | training | run-ons gone; a third of a small cost (0.018) back | LEISURE.md |
| lives born with the new cortex, relay and full-size corrections (eyes2) | – | – | at 5,000 moments into first words 0.39–0.40 against 0.35; running | DAYLOOP.md |

**What the day taught about the method.**

- **Find out which organ is the limit before improving any.** For three weeks "the eyes are the wall" was read from
  the word cells' naming. A linear readout fitted on the very moments the word cells had lived was the missing
  measurement: it showed in an afternoon that the cells used two thirds of what the code held, and that a better
  code did not reach them. The reference must sit at the same place in the chain as the learner it judges, with
  the same data.
- **A literature survey before building** paid three times in one day (the cortex's known failure; what two-year-olds
  count; what they take in). Each survey named the experiment that would decide, and the experiments took minutes.
- **Offline probes with kept codes** (minutes per arm) made a dozen negative results cheap. Recorded with their
  numbers, they are worth as much as the positives: branches, soft competition, the echo relay, bursts, the second
  layer.
- **"More of the same" was the wrong answer every time it was tried**: more pictures (+0.02), more layers (nothing),
  more passes (+0.03), larger steps (nothing, then divergence). What moved things was the shape of what a learner
  is given (decorrelated), not its amount.
- **Check the developmental fact before fighting a score.** Counting at the two-words stage was a wall a child does
  not climb either.
- **Memory and CPU are the machine's limits, and I hit both again**: probes killed by my own caps (four times),
  lives slowed by a third by probes beside them. Estimate a probe's memory from its largest array before starting
  it; keep the machine to the lives when they are the long pole.
