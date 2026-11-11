# Why a level answers text with the last word it heard (2026-11-11)

Kris talked to the first-words level (L3 of the word-cells lineage, 15,000 moments) with text only:
"how are you?" → "you?", "do you have a cat?" → "cat?", "do you say anything else than the last
word?" → "word?". His question: is the word architecture wrongly designed, and what does biology say
about getting past this. This file is the diagnosis (measured), the reading, and the proposals.
Scripts: `experiments/echo_diagnosis.py`; numbers are seed 0 of the lives unless said otherwise.

## 1. What was measured

**Which pathway speaks.** Held-out text-only items (eyes closed), 64–150 per source:

| brain | items | score | a parrot of the last word would score | copy route / own habits / word cells | says exactly the last word |
|---|---|---|---|---|---|
| L3 (15,000 moments) | echo (new words) | 0.89 | 1.00 | 0.86 / 0.07 / 0.07 | 0.82 |
| | first-words text lessons | 0.36 | 0.33 | 0.77 / 0.02 / 0.22 | 0.74 |
| | two-words text lessons | 0.34 | 0.43 | 0.73 / 0.01 / 0.25 | 0.73 |
| | preschool conversation (dailydialog, chat) | 0.01–0.05 | 0.01–0.06 | 0.83–0.95 copy | 0.80–0.90 |
| same brain at 115,000 | echo (new words) | 0.73 | 1.00 | 0.62 / 0.06 / 0.32 | 0.59 |
| | first-words text lessons | 0.26 | 0.33 | 0.23 / 0.02 / 0.76 | 0.30 |
| | two-words text lessons | 0.43 | 0.43 | 0.14 / 0.00 / 0.86 | 0.36 |
| | preschool conversation | 0.01–0.08 | 0.01–0.06 | 0.37–0.46 copy | 0.36–0.47 |

**The world's side.** Where the adult's reply sits in what was just heard (training lessons):

| lessons with eyes closed | reply = the last word(s) heard | reply not in what was heard |
|---|---|---|
| echo ("say bread" → "bread", "tulip" → "tulip") | 1.00 | 0.00 |
| first words ("where's granny?" → "granny", "what do you say when you want more?" → "more") | 0.35 | 0.62 |
| two words ("is snow hot or cold?" → "cold", "is the car big?" → "big") | 0.18 | 0.70 |
| preschool conversation | 0.01 | 0.97–0.98 |

**The hand-set part.** The copy route starts at a heard word chosen with a weight of 0.75 per word
back from the end (`SpeakingBrain.recency`, written in milestone 3 with the comment "toddlers echo
the last word"). With the weight removed at test time (1.0), L3 says the last word on 0.35 of
first-words text items instead of 0.73, and its score there falls from 0.34 to 0.23 (two words:
0.33 → 0.19). At 115,000: last word 0.33 → 0.17, score unchanged (0.25 → 0.24, 0.43 → 0.41).

**What the word cells would have said**, and how sure they were (confidence 0–1):

| heard (eyes closed) | L3 said | L3 word cells wanted | 115,000 said |
|---|---|---|---|
| how are you? | you? (copy) | "mou?ar" 0.05 | you? (copy; the cells say nothing) |
| do you have a cat? | cat? (copy) | "hamcatpasta" 0.03 | yes (cells) |
| what does the cow say? | say? (copy) | "bray" 0.08 | bray (cells) |
| ball is what i want you to say | say (copy) | "ball" 0.25 | ball (cells) |
| where is mommy? | mommy? (copy) | "mommy" 0.13 | here (cells) |
| do you want more milk? | milk? (copy) | "milkes" 0.02 | milk (cells) |

The cells' confidence is 0.09–0.15 when they are right and 0.05–0.09 when they are wrong: it hardly
tells the two apart, and it is what the route choice leans on.

## 2. Reading

1. **It is not the word cells converging to the last word.** At L3 the copy route speaks in 73–97%
   of text utterances. The same brain 100,000 moments later answers 76–86% of the same items from its
   word cells and says the last word about as often as the lessons do (0.30–0.36 against 0.33–0.43).
2. **At L3 the last word is what the world taught, plus a rule of mine.** Every text exchange the
   brain had lived by then was an echo (reply = last word, always) or a first-words text lesson
   (a third of them). About half of the habit is the hand-set recency weight. That weight also earned
   score: +0.11 on first-words text items at the moment L3 passed. A designed rule that fits how the
   adults write lessons was inside a passed stage. It should not be there.
3. **The yardstick pays a parrot.** Saying the last word scores 0.33 on first-words text items and
   0.43 on two-words text items. The lessons the free models wrote put the answer at the end too often
   ("is snow hot or cold?" → "cold").
4. **Text is where the design is weak, and that is the real finding.** Eyes closed, the word cells are
   no better than the parrot (0.26 vs 0.33; 0.44 vs 0.43) even after 115,000 moments. Their gains
   (docs/AREAS.md) were about pictures. Four structural reasons:
   - *A heard word means only itself.* The content dendrite gets heard units as identities, in a bag.
     Hearing "cat" brings back nothing of what cat pictures were like, so nothing learned by seeing is
     there when the eyes are closed, and nothing compares what is asked about with what is seen (this
     is also why yes/no and "is there a…" sit at base rates, where all three lives stopped).
   - *There is no message.* A reply is an association from the words just heard to the words adults
     said next. With the eyes closed there is nothing the brain is talking *about*, and it has no
     states of its own (nothing it wants), so it has nothing to say that was not asked.
   - *Three routes, chosen by a number that carries no information* (the confidence above). Recall
     also erodes echo (0.89 → 0.73), as the hub's route did.
   - *It has never had a conversation.* Conversation is a preschool stage, behind the two-words gate
     (colours, counting and yes/no about photographs), which all three lives failed (0.41–0.43 against
     0.5 after 100,000 moments). A child greets and takes turns long before it counts.

## 3. What biology says

| observation | source | what it means here |
|---|---|---|
| Repeating speech (sound → articulation, dorsal stream) and understanding it (sound → meaning, ventral stream) are separate pathways running in parallel. | Hickok & Poeppel 2004, 2007 | The copy route is the dorsal stream. It is right to have one. |
| When the meaning pathway is cut off and repetition is spared (transcortical sensory aphasia), patients echo what they hear without understanding it. | Boatman et al. 2000; the classical "isolation of the speech area" | L3 with the eyes closed is exactly that brain. Echo is not a wrong design, it is what is left when meaning has nothing. The cure is on the meaning side, not a rule against echoing. |
| Typical 1–3-year-olds echo immediately, most with utterances they do not understand, and it fades as understanding grows; frontal damage brings imitation back. | developmental echolalia literature; Lhermitte 1986 (contested in detail) | Nobody chooses between "copy" and "answer": the copy is the default and a prepared reply out-competes it. Strength, not a route decision. |
| Children's imitations keep the stressed content words in order and drop the function words ("telegraphic"); the ends of utterances are salient because the sound trace fades. | Brown & Bellugi 1964; Slobin 1973 | Recency is real, but it comes from a fading trace and from words heard everywhere evoking less, not from a per-word constant. |
| Speaking starts from a message; a meaning pathway activates words and a separate sequencing pathway orders them. Both are learned by predicting the next word while listening. | Levelt 1989; Dell 1986; Chang, Dell & Bock 2006 ("Becoming syntactic") | The word cells' two dendrites are already these two pathways. What is missing is a message when the eyes are closed. This is also the difference from a language model: there the text predicts the text; here words must come from something that is not words. |
| Understanding a word re-activates the perceptual and motor areas it was learned with (word cells are assemblies reaching into visual cortex). | Pulvermüller 2005, Pulvermüller & Fadiga 2010; Barsalou 1999; Patterson, Nestor & Rogers 2007 | Hearing "cat" should bring back, faintly, what cats look like, through the same synapses that learned to name them. |
| Labels change what is perceived (hearing "dog" makes dogs easier to see). | Lupyan 2012; Lupyan & Ward 2013 | Top-down from word cells to visual neurons; where it meets bottom-up input the neuron fires strongly. That coincidence is a basis for "is there a dog?" other than the base rate. |
| First words are mostly social acts (greeting, requesting, refusing, taking turns), not names; babies learn fastest from replies that are contingent on their own sounds. | Ninio & Snow 1996; Bruner 1983 (formats); Goldstein & Schwade 2008 | Conversation belongs at the start of the curriculum; replies should depend on what the child said. |
| Speech acts differ by what controls them: echoing, naming what is seen, asking for what is wanted, replying to words with other words. | Skinner 1957 | The brain has the first two and a little of the fourth. It has no third, because it has no wants. |
| Listeners plan their reply while still listening. | Levinson & Torreira 2015 | Later: the reply forms during hearing, not after a route is chosen. |

## 4. Proposals, in the order I would do them

A. **The world first** (no architecture). Report every text score beside what a last-word parrot gets on
   the same items; a stage is not passed on items the parrot answers. Rebalance the adults' lessons so
   the answer is not at the end more often than chance. Bring greetings and turn-taking formats
   ("hi", "bye", "how are you?", "thank you", "want more?") into first words, and stop holding
   conversation behind photograph questions.

B. **Heard words bring back what was seen** (the meaning side; one mechanism, already in the tissue).
   Each word cell's visual synapses also work downwards: a heard unit faintly drives the visual neurons
   its own dendrite learned. Eyes closed, that imagined pattern drives the other word cells as a picture
   would, so text gets a meaning beyond its own sound. Eyes open, a visual neuron driven from both
   sides fires strongly (the same coincidence of two compartments the word cells already use), and
   "yes", "no" and "there" can learn from those neurons. Offline first, 3 seeds. Tests that would count
   as generalisation: (1) "is there a …?" above its base rate (0.62–0.65 now); (2) questions about things
   learned only with pictures, asked with the eyes closed; (3) first-words text above the parrot.
   This is also the open vision step of docs/VISION.md (labels shaping perception).

C. **One stream instead of three routes, and no hand-set recency.** The recalled reply is heard
   inwardly and the mouth follows what is heard, from outside or inside, whichever is stronger; the
   outer echo stays as what happens when the inside is silent. Where copying starts comes from the
   ear's fading trace and from over-heard units answering less (the word cells already keep each cue's
   use). Removes a route choice, the confidence number and the 0.75.

D. **Something to say** (later, larger). A few inner states that vary and that adults talk about
   (rested/tired, fed/hungry, surprised) as inputs to the content dendrite, so that "how are you?" has
   something to be about and asking for things becomes possible. Without this, conversation can only be
   words answering words.

Not proposed: more data for the same text pathway. The parrot comparison says the pathway, not the
amount, is the limit for text.

## 5. Also in this session

- The three lives stopped at two words: 0.41, 0.41, 0.43 against 0.5 after 100,000 moments there
  (first words held at 0.41–0.43; echo fell to 0.68–0.91). Offline the word cells alone reach 0.51 on
  the same stage; in a life, with the routes around them, 0.41–0.43.
- The offline queue of 2026-11-10 was killed by its own memory cap at 15:59: the two 30,000-moment
  curiosity runs ended at 20,000 (curious W1 0.23 / W2 0.45; uniform 0.23 / 0.47: still no order of
  stages and no gain), and the third seed of the mixed-stages table never ran. It was restarted on
  2026-11-11. The health check did not report this: it watches lives, not probes.
- The 115,000-moment brain is on the preview as "L4 two words (still learning)", marked as not passed.

## 6. What was built and measured after Kris's answer (2026-11-11, afternoon)

Kris: the new models are better than kid_s3, replace the public hosting with them; the 0.75 recency "sounds
like not emergent but forced"; onwards with all proposals.

**Hosting.** The public kid-a now serves the word-cells lineage (words_s0: L1–L3 as passed, L4 "two words (still
learning)" at 115,000 moments, marked as not passed). kid_s3's levels are kept in `kid-a/lineages/kid_s3/`.

### C. The mouth without the hand-set recency, and without a route choice

- `SpeakingBrain._novelty`: where copying starts is weighted by how much each heard word still stands out: the
  ear's own count of each unit with the tissue's law of consolidation, 1 / (1 + times heard / 5). A new word is
  repeated by its sound wherever it stands; words heard everywhere are not. `SpeakingBrain.recency` (0.75) remains
  only for the brains made before (the served levels).
- `one_stream`: no route is chosen. What was heard outside and the reply the word cells recalled (heard
  inwardly) prime the mouth together with the orchestrator's vote; the mouth follows whichever resonates more.
- Test-time check on the existing brains (not trained this way; seed 0):

| | pictures named | echo, new words | first-words text | says the last word (text) |
|---|---|---|---|---|
| L3, routes + recency (as served) | 0.53 | 0.89 | 0.36 | 0.71 |
| L3, one stream, inner voice at full strength | 0.53 | 0.68 | 0.28 | 0.24 |
| L3, one stream, inner voice as loud as the word cells' activity | 0.11 | 0.85 | 0.23 | 0.41 |
| 115,000, routes + recency | 0.50 | 0.72 | 0.31 | 0.29 |
| 115,000, one stream, inner voice at full strength | 0.50 | 0.67 | 0.31 | 0.19 |

  Making the inner voice as loud as the cells' activity failed: the two strengths (novelty of a heard word,
  activity of a word cell) are not on one scale, and naming collapsed. Kept: a cell that won fired, and what is
  heard inwardly is heard. What the one-stream mouth loses at test time is new-word echo (the cells always say
  something, even to a word they have never heard, because their content drive is taken relative to the
  strongest cell). Whether a brain raised this way learns to stay silent there is what the lives test.
- Two lives from babbling, one seed each to choose the arm before spending three seeds: `stream_s0`
  (one stream) and `routes_s0` (the route choice kept, novelty instead of recency), `experiments/conv_pipeline.sh`.

### A. The world

- **Echo games worded by the adults** (`experiments/world/echo_frames.py`, 84 written, 20 kept): "can you say
  WORD for me", "please say WORD again", "say WORD if you can"… The word is last in 10, in the middle in 8, alone
  in 2. The judging adult kept none with the word first ("WORD! can you say it?"). Worded afresh at every moment.
- **Social routines** (`experiments/world/social_formats.py`, 98 kept: how are you / comfort 35, turn-taking 28,
  greetings 20, thanks and sorry 15; the "offers" batch did not come back): first words, source "social".
  Many are "are you hungry?" → "yes" or "no", both accepted: the world cannot know, and neither can a brain
  with no inner states (proposal D).
- **The parrot beside every score** (`curriculum_life.py`): what repeating the last word would score on the same
  items, how often the brain says exactly the last word, and the score on the items the parrot gets wrong.
  With `PARROT=1` a stage from first words on is passed and held on that last score.
- `WORLD=2` turns the first two on; every earlier life and every offline table is `WORLD=1`.
- Not removed: the last-word habit of the adults' text lessons. Speech to infants really does put the word that
  matters last (Fernald & Mazzie 1991), so the bias belongs to the world; it must not be in the brain by hand, and
  the yardstick must not pay for it.

### B. Heard words bringing back what was seen: built, and mostly negative

`Words(imagery=True)`: feedback synapses from the ear's unit cells onto the visual neurons learn, by the same
delta rule, what each heard unit predicts will be seen. Eyes open, a visual neuron driven from below and above
bursts, and the content dendrite reads the bursts through its own synapses; eyes closed, the feedback is passed
on as a faint picture. `experiments/imagery_probe.py` (first and two words mixed, one unasked comment per moment).

| seed 0, one pass | pictures named | first-words text beyond the parrot | "is there a …?" | says yes when it is there minus when it is not |
|---|---|---|---|---|
| word cells | 0.472 | 0.189 | 0.627 | 0.10 |
| + feedback and bursts | 0.468 | 0.181 | 0.627 | 0.10 |

Nothing moved. Two measurements say why:

1. **The signal exists but is small with these eyes.** A name's visual pattern learned from picture-book pictures
   does tell whether the thing is in a photograph it never named (AUC 0.69 over all questions; 0.63 within one
   asked thing). But the best linear readout of it, fitted in a batch, lifts "is there a …?" from 0.637 (each
   thing's base rate) to 0.67–0.68. At most +0.04 is there to be had, and an online rule gets less.
2. **The ear's units block the transfer.** The feedback synapses learned strong pictures for units like
   'that a turtle', 'ooh a cat again', 'ahh a boat' (whole comments cut as one unit). In "is there a dog?" the
   unit heard is 'dog?': a different cell, which has only ever been heard in such questions.

**The same finding from the other side: what the ear's cuts cost.** A reference ear that cuts at spaces and drops
punctuation, everything else equal (seed 0, one pass):

| | pictures named | same answer ×3 | first-words text | beyond the parrot | two-words text | colour |
|---|---|---|---|---|---|---|
| the ear's own units (662 unit cells, 4,245 cues) | 0.472 | 0.647 | 0.263 | 0.189 | 0.279 | 0.115 |
| cut at the spaces (379 unit cells, 2,570 cues) | 0.509 | 0.753 | 0.358 | 0.255 | 0.362 | 0.204 |

**Letting the content dendrite hear sound** (`Words(sound=True)`, `Ear.combinations`: the ear's
combination-sensitive neurons answering to every 2- and 3-byte sequence wherever the units were cut, so that
"dog", "dog?" and "ooh a dog" overlap; 3 seeds, one pass):

| | pictures named | same answer ×3 | first-words text beyond the parrot | two words |
|---|---|---|---|---|
| word cells | 0.450 ± 0.016 | 0.645 ± 0.011 | 0.239 ± 0.043 | 0.433 ± 0.007 |
| + sound | 0.453 ± 0.018 | 0.698 ± 0.007 | 0.264 ± 0.028 | 0.433 ± 0.009 |

The same answer across wordings rises in every seed (+0.05); text beyond the parrot rises in every seed but by
little (+0.04, +0.03, +0.00); after three passes (seed 0) only the consistency gain is left (0.797 against
0.750). On seed 0 "is there a …?" rose to 0.667; seeds 1 and 2 did not repeat it (0.620, 0.607). Not put in
the lives.

With sound, the feedback onto the visual neurons can come from the sound neurons instead of the unit cells
(so "dog?" and "ooh a dog" share it). Seed 0, one pass: "is there a …?" 0.680 against 0.667 for sound alone,
everything else equal. One seed, and sound alone did not repeat on the other two; stopped there.

**How far the text pathway is from what the lessons hold.** Reference learners on the same held-out first-words
text items, parrot-proof ones only: a count of which reply goes with which words (naive Bayes) 0.45; the nearest
lesson by shared rare words 0.49; the word cells 0.19–0.29 (0.24 after three passes; 0.26 with the reference
ear). The content dendrite alone scores what the whole recall scores (0.196 against 0.189), so the context
dendrite is not what loses it. For two-words text the references are no better than the parrot (0.41, 0.32
against 0.43): there the lessons themselves do not hold the answers.

### What this says about the next step

The text pathway is short of what its own lessons hold by about a half, for two reasons that are now measured:
the ear's cuts (phrases as single units, the same word in several cells), and the slowness of the cue
synapses (each lesson is met once among forty others, and the delta rule shares each correction between all
inputs). Imagery is right in principle and capped by the eyes. So the order of work is the ear's units first
(docs/EAR_SEGMENTATION.md steps 2–5, open since October), not more machinery on top of them.

### D. Something to say: not built

No inner states were added. It needs a decision about what the brain's body is (what varies, what an adult can
see of it) that should not be improvised; the social routines above are the first place where it would show.

### The two trial lives, and the arm chosen (2026-11-11, evening)

Both trial lives stuck at the echo stage, which every life of 2026-11-10 passed at its first checkpoint (0.95):

| moments | 10,000 | 15,000 | 20,000 | 25,000 |
|---|---|---|---|---|
| one stream (`stream_s0`) | 0.64 | 0.63 | 0.63 | 0.65 |
| routes + novelty (`routes_s0`) | 0.74 | 0.61 | 0.52 | 0.60 |

Two causes, both measured on the brains at 25,000 moments:

1. **The novelty was too faint.** The ear cuts a new word into pieces it knows ("ro|ck|et"), each heard often,
   so the word's novelty was 0.05 where "tuli|p" got 1.0. The order was always right (the word to repeat was the
   most novel word in 200 of 200 held-out echo utterances); its size was not. The ear's byte-level surprise was
   tried as the measure instead and is worse (right word in 0.72: the first word of any utterance is surprising).
   Fix: the heard words compete, the most novel one counts as 1 (divisive normalisation within the utterance).
   With that, when the copy route speaks it repeats the right word in 0.95 of utterances wherever the word stands
   ("can you say WORD for me" included).
2. **The word cells always say something**, also about a word they have never heard ("method", "bottle", "k!"):
   their content drive is taken relative to the strongest cell, so "nothing to say" does not exist. In the
   one-stream brain that inner voice spoke in 55% of echo utterances and nothing can learn to keep it out.
   In the routes brain the orchestrator's learned choice can.

So: **one stream is a negative result for now** (it needs word cells that can stay silent), and the arm kept is
the learned route choice with the normalised novelty in place of the recency. Three seeds from babbling:
`conv_s0..2` (units organic-conv-s0..2, world 2, parrot gate). The trial lives were stopped.

### The three seeds of the kept arm at 20,000 moments (`conv_s0..2`, 2026-11-11 19:20)

| seed | echo passed (pass mark 0.8; the test now has the word mid-utterance) | first words on parrot-proof items (pass 0.4) | says the last word at first words (parrot would score 0.13) | new-word echo once first words began |
|---|---|---|---|---|
| 0 | 15,000 (0.83) | 0.37 after 5,000 moments | 0.10 | 0.81 → 0.61 |
| 1 | 10,000 (0.82) | 0.43, 0.41 | 0.08, 0.09 | 0.82 → 0.49, 0.58 |
| 2 | 10,000 (0.86) | 0.42, 0.42 | 0.06, 0.06 | 0.86 → 0.54, 0.53 |

By source at first words: pictures 0.46–0.58, text lessons 0.21–0.27, social routines 0.17–0.33 (8 held-out
items), familiar-word echo 0.92–0.97.

Read: the last-word habit is gone (6–10% of replies, below what the lessons themselves would justify), echo
with the word anywhere is learned in 10–15 thousand moments without the recency, and first words clear their
mark on the parrot-proof items in two of three seeds. They are **not promoted**, because new-word echo falls
below 80% of its passing score as soon as the word cells start to answer: the recall route speaks in about half
of the echo utterances with a word never heard ("please say alcohol again" → "fish"). The same cause as in the
one-stream life: a word cell always has something to say. That is the next thing to solve in the word cells
(a content drive that can be "nothing"), not in the mouth. The lives keep running.
