# The ear learns what the edges of words sound like (2026-11-15)

Proposal 1 of [CONVERSATION.md](CONVERSATION.md), steps 4–5 of [EAR_SEGMENTATION.md](EAR_SEGMENTATION.md).
Code: `organic/ear_carving.py` (`CarvingEar`, `EdgeEar`), `organic/ear_lexicon.py` (kept, negative).
Probes: `experiments/ear_probe.py` (a fresh ear on a life's diet: 130,000 utterances, questions, answers, unasked
comments and overheard sentences), then `EAR=out/tmp/ear_<kind>.pkl experiments/imagery_probe.py` for what the
word cells make of its units.

## A correction first

The offline tables of 2026-11-10/11 (docs/AREAS.md, CONVERSATION.md §6) used kid_s3's ear, the worst ear the
project has (20% whole words, after 305,000 moments in one stage). An ear grown on what the small lives hear is
better (56% whole words), and with it the word cells' offline numbers are a little higher (pictures 0.49, same
answer across wordings 0.74). The comparisons made there still hold (same ear in every arm), the absolute
numbers were pessimistic for pictures. The reference "cut at the spaces" comparison is redone below with the
life-grown ear.

## What matters for text: every word one whole unit

Diagnostics with reference ears (seed 0, one pass; not brains, only to locate the loss):

| ear | pictures | same answer ×3 | first-words text | of it, parrot-proof | two-words text | colour | two words |
|---|---|---|---|---|---|---|---|
| first ear, life-grown | 0.489 | 0.740 | 0.255 | 0.213 | 0.274 | 0.190 | 0.430 |
| cut at spaces, marks dropped | 0.510 | 0.767 | 0.350 | 0.250 | 0.351 | 0.265 | 0.462 |
| cut at spaces, marks kept on their words ("spoon?") | 0.472 | 0.747 | 0.359 | 0.389 | 0.340 | 0.263 | 0.469 |

Whether "cat" and "cat?" are one cell is not what costs. What costs is that in the first ear 8% of word tokens
are in pieces and 21% are fused with a neighbour ("do you ", "color is ", "show me your ").

## Three designs

**1. A lexicon that explains the utterance most cheaply** (`LexiconEar`: known units cost the surprise of
meeting them, a new stretch costs the ear's byte-by-byte surprise). Negative. It hears new words whole ("tulip",
"alcohol") but glues what goes together ("the dog", "a rocket", " dog?"): 60% of its units are fused words, the
known weakness of such models without word-to-word context (Goldwater, Griffiths & Johnson 2009). Downstream
picture naming fell to 0.21.

**2. Words heard alone carve longer speech** (`CarvingEar`). A stretch heard alone, between two silences, is a
candidate word (Bloomfield's free form; Brent & Siskind 2001). It pops out when heard again inside longer
speech if what it leaves of the first-hearing units it cuts is more familiar than those units (the possible-word
constraint, Norris et al. 1997; parts more familiar than the whole make a phrase, Hay 2001), never from the
middle of one unit, and only if it is itself not such a phrase. A name is then the same unit in six different
settings for 60–96% of names (first ear 36%). Downstream: same answer across wordings 0.74 → 0.78, pictures
0.49 → 0.50, text unchanged (0.25). What went wrong on the way: judging what is left over by counts that the
carving itself feeds ("w|ha|t "), and words inside words ("t|ouch", "b|all").

**3. Learning what word edges sound like** (`EdgeEar`), which is what infants do next (Jusczyk, Houston &
Newsome 1999; Mattys & Jusczyk 2001; Thiessen & Saffran 2003). The carving is only the teacher: each time a known
word pops out of longer speech, the positions at its ends are edges and the positions inside it are not. For
every byte the ear counts how often it stood just left, and just right, of each. Hearing: a position is a cut
when an edge is more likely than not from the bytes on its two sides; where the ear has no such experience the
first ear's cut stands. Bytes heard more often at the edge of known words than inside them are pauses or marks
and stay with the unit before them.

What it learned by itself after 130,000 utterances: an edge follows a space with probability 0.94 and comes
before one with 0.89; before "?" 0.96, before "," 0.96; after "a" 0.08, after "e" 0.38. Nobody told it about
spaces.

| ear | cuts that are word edges | word edges cut | units that are one word | fused words |
|---|---|---|---|---|
| first ear | 0.74 | 0.81 | 0.56 | 0.12 |
| edge ear (marks as units of their own, for this score) | 0.81 | 1.00 | 0.69 | 0.00 |

"can you say tulip for me" → can | you | say | tulip | for | me; "please say alcohol again" → please | say |
alcohol | again; "look, a rocket!" → look, | a | rocket! (the first ear: "tul|ip", "alco|hol", "ro|ck|et!").

Downstream, seed 0, one pass:

| ear | pictures | same answer ×3 | first-words text | of it, parrot-proof | two-words text | yes/no | two words |
|---|---|---|---|---|---|---|---|
| first ear | 0.489 | 0.740 | 0.255 | 0.213 | 0.274 | 0.473 | 0.430 |
| edge ear, marks on their words | 0.456 | 0.760 | 0.330 | 0.329 | 0.314 | 0.553 | 0.456 |
| edge ear, marks as units | 0.491 | 0.773 | 0.251 | 0.227 | 0.312 | 0.560 | 0.461 |

Three orders of moments (the same two grown ears; mean ± spread):

| | pictures | same answer ×3 | first-words text | of it, parrot-proof | two-words text | colour | yes/no | two words |
|---|---|---|---|---|---|---|---|---|
| first ear | 0.473 ± 0.015 | 0.735 ± 0.003 | 0.288 ± 0.029 | 0.245 ± 0.038 | 0.281 ± 0.009 | 0.217 ± 0.020 | 0.464 ± 0.012 | 0.435 ± 0.005 |
| edge ear | 0.461 ± 0.009 | 0.729 ± 0.033 | 0.341 ± 0.014 | **0.341 ± 0.025** | 0.310 ± 0.003 | 0.203 ± 0.005 | 0.511 ± 0.039 | 0.448 ± 0.007 |

Text on the parrot-proof items rises in every seed (+0.12, +0.10, +0.08), two-words text in every seed (+0.03 to
+0.04); pictures and the same answer across wordings do not change. The edge ear goes into the next lives
(`EAR=edge`; a new ear is a new lineage: EAR_SEGMENTATION.md step 8).

Open: the ear still breaks a few words by a word inside them while it is the teacher ("t|ouch", "s|now"); the
marks stay on their words ("dog?" and "dog " are two cells; separating them lost the text gain on seed 0, why is
not understood); and the carving has two rules that are choices of mine (no popping out of the middle of a unit;
remnants judged by first-hearing counts).
