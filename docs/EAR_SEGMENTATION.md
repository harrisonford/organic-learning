# How the ear finds words: status and design (2026-11-04, draft)

The user asked for a slow, thorough design, with no shortcuts and no fixed rules: the
ear's units must come out of what it hears. This file holds the measurements, the
biology, and a plan in small steps, each measured before the next. It is updated as
the steps run.

## 1. What the ear does now (`organic/ear.py`)

- **Statistical learning:** Hebbian counts of which byte follows the last 0–4 bytes,
  backing off to a shorter context when a context is new.
- **Segmentation:** at each byte, the entropy of the ear's guess for the next byte. A
  unit starts where the entropy jumps by more than `rise = 1.2` bits, or where it is
  above `unknown = 0.85 × log2(257)`, i.e. the ear has no idea at all. These two
  thresholds are **fixed constants**, chosen by hand in 2026-10. They are the first
  thing this design removes.
- **Sound codes:** a unit's sound is fixed random sparse patterns for its bytes and
  byte n-grams, so units that share sounds overlap.
- **Downstream:** the question code is an order-aware echo of the units plus their gist.
  Every word, cluster and output of the brain is built on these units.

## 2. Measurements

`experiments/segmentation_probe.py`, on 1,500 held-out utterances of stages B–K. The
scores are the standard ones from infant word-segmentation modelling (Brent 1999;
Goldwater, Griffiths & Johnson 2009):

- **boundary P/R:** of the ear's cuts, the share that fall at a word edge; of the word
  edges, the share it cuts.
- **word P/R:** units that are exactly one word, plus its trailing space or punctuation.
- **unit kinds:** one word / several words fused / a piece of a word.

| ear | boundary P | boundary R | word P | word R | one word | fused words | piece of a word |
|---|---|---|---|---|---|---|---|
| fresh, 100 utterances heard | 0.23 | 0.22 | 0.08 | 0.07 | 0.08 | 0.30 | 0.63 |
| fresh, 1,000 | 0.50 | 0.52 | 0.31 | 0.32 | 0.31 | 0.21 | 0.48 |
| fresh, 5,000 (the ear of L1) | 0.57 | 0.61 | 0.38 | 0.40 | 0.38 | 0.19 | 0.43 |
| fresh, 20,000 | 0.64 | 0.65 | 0.42 | 0.43 | 0.42 | 0.19 | 0.38 |
| fresh, 40,000 | 0.65 | 0.67 | 0.44 | 0.45 | 0.44 | 0.19 | 0.38 |
| kid_s3 now (age 315k) | pending | | | | | | |

The curve flattens at about 44% whole words. Two failures persist:

1. **Fused frames** (19% of units): "is this?", "how many ", "is there a ", "color is
   the ", "people are ". Frequent frames become one unit. Toddlers do the same
   ("whatsthat", Peters 1983). It is not wrong in itself. What is missing is the later
   step where the child **takes the frame apart**.
2. **Rare words broken into pieces** (38%): "bi|gge|st", "batt|er's u|niform",
   "a|irplane". A rare word is unpredictable inside, so entropy keeps jumping in the
   middle of it. Infants are also worse on rare words, but they still hear a new word as
   one thing between known neighbours.

Both failures cut the frame-vs-word test: a frame fused onto the word makes the same
word in another frame sound different, and a fragmented word has no stable sound.

A correction: the prediction-error (C) probe earlier today used this same young ear
(from L1, at 5k heard), not kid_s3's current ear. It is rerun on the real ear once the
life writes it (`out/kid_s3.ear.pkl`, saved at each checkpoint from 2026-11-04).

## 3. What infants do (the biology to follow)

| finding | source | what the ear has |
|---|---|---|
| Boundaries fall where the transitional probability **drops** relative to its neighbours | Saffran, Aslin & Newport 1996; Aslin, Saffran & Newport 1998 | a jump past a fixed threshold, not a relative dip |
| Infants also use **backward** transitional probabilities (how well the next sound predicts the previous one) | Pelucchi, Hay & Saffran 2009 | forward only |
| **Known words carve out their neighbours**: a familiar word ("mommy", the child's name) helps segment the word next to it | Bortfeld, Morgan, Golinkoff & Rathbun 2005; Brent & Cartwright 1996 | no: units do not compete |
| Words at **utterance edges** are segmented first | Seidl & Johnson 2006 | partly: silence is heard |
| Chunks form by repeated co-occurrence and **decay or are absorbed** when they don't recur; memory and attention shape the units, not thresholds | Perruchet & Vinter 1998 (PARSER) | units are never forgotten or merged |
| Frozen formulas are later analysed by **slot and frame**: "is there a dog / is there a cat" splits into "is there a" + the slot | Peters 1983; Tomasello 2003; Mintz 2003 (frequent frames) | no analysis step |
| **Sleep** helps infants abstract the patterns from what they heard | Gómez, Bootzin & Nadel 2006; Friedrich et al. 2015 | sleep consolidates the orchestrator only, never the ear |

The text world has spaces, which speech does not. The ear treats a space as just a
byte, with no special meaning, and that stays: spaces are a weaker cue than a pause,
and here the ear has to learn their predictability like any other byte.

## 4. Plan: one step at a time, each measured by §2's scores and the frame-vs-word probe

1. **Measure kid_s3's real ear** at age 315k+ (the row marked pending), and rerun the
   C probe on it.
2. **Relative dips instead of fixed thresholds.** Cut where the ear's predictability is
   at a local low compared with its neighbours, as in the original Saffran analysis.
   This removes both `rise` and `unknown`: a cut is relative to the ear's own
   experience, never to a constant. Measured on its own.
3. **Both directions.** The ear also learns backward predictability from the same
   Hebbian counts, read in reverse, and a boundary is wherever either direction dips.
   Measured on its own and with step 2.
4. **Units that compete** (known-word bootstrapping, PARSER). The ear's stored units are
   used to hear: an utterance is explained by the units already known, and the part they
   leave over becomes a candidate new unit (the "mommy effect"). Units that keep being
   used gain strength; units that don't decay away; a unit that is always heard inside
   a bigger one is absorbed. There is no size limit and no frequency threshold: use
   and decay decide, the same use-it-or-lose-it rule the orchestrator's pruning uses.
5. **Taking frames apart in sleep** (slot and frame). During sleep the ear replays what
   it heard that day. When a fused unit keeps turning up with one part changing, the
   stable part becomes its own unit and the changing part becomes the slot. This is the
   step toddlers do between 18 and 30 months.
6. **Downstream check, offline:** frame-vs-word probe and question-code tests with the
   new ear.
7. **Controlled small lives** (like the cluster-size runs): old ear vs new ear, same
   seed and budget, BB → W1. Compared on W1 score and on how babbling and first words
   develop.
8. **Only then a new lineage.** Every unit, cluster and output of a living brain is built
   on its ear's units, so swapping the ear of kid_s3 would be a transplant, not growth.
   A new ear means a new life from BB (kid_s4). Its levels are released in order, as
   always.

Every step is recorded in `docs/LOG.md`, negative results included.
