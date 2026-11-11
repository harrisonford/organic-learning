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
