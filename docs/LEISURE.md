# Leisure talk: not every moment is a task (2026-11-15)

Kris, on proposal 4 of [CONVERSATION.md](CONVERSATION.md) ("something to say"): *"not everything is a task in a human
brain in the sense of something is expected; talking and vibing should be fine; maybe we should just keep on
expanding the dataset and also sneak some leisure talk."* So no body and no inner states were designed. Instead
the world gets moments in which nothing is expected.

## What a leisure moment is

1. A parent says something in passing ("are you comfortable in your bed"). Half of these are the adults' written
   social routines (`data/world/social_formats.jsonl`), half are made up on the spot by the parent model about an
   everyday topic.
2. The brain says whatever it says (it explores, as in a lesson; before words it babbles). Nothing is right or
   wrong: no answer is given, no grade, no reward.
3. The parent answers **what the child said**: takes the word up and adds a little, or just answers kindly.
4. The brain hears that as a listener does: the ear hears it, the mouth imitates it covertly, and the word cells
   learn that after what was just said (by the brain itself) this is what people say. That is role reversal
   (Tomasello 1999): what the parent says back to "doggy" is what the child may one day say back to "doggy".

Why this and not more lessons: replies contingent on the infant's own sounds are what infants learn fastest from
(Goldstein & Schwade 2008), and parents' answers are mostly recasts and expansions of what the child just said
(Brown & Bellugi 1964). Until now every exchange in this world was a question with a right answer.

## The parent

`organic/parent.py`, systemd unit `organic-parent`: Llama 3.2 3B (Q4) served by llama.cpp on the GPU beside the
judge, 127.0.0.1:8791, about 0.1 s per reply, no cost. The hosted free models have 500 requests a day and the judge
is a decision model that cannot write. The GPU is now nearly full (7.6 of 8 GB). This is the environment, not the
brain. If the parent does not answer in time the moment is skipped.

## In a life

`CHAT=0.2` (leisure moments per lesson) in `experiments/curriculum_life.py`; `SpeakingBrain.chat`. Every exchange is
written to `out/<life>.chat.jsonl` so that it can be read. From a smoke test, a brain 9 to 450 moments into echo:

    parent: let's clean up the playroom together      child: thogether     parent: that's almost right, together is the word
    parent: fill the tub up                           child: up            parent: you want me to fill it up
    parent: are you tired now little one              child: d             parent: you're so sleepy little one, time for nap.
    parent: i see you want that toy please            child: mor           parent: you want it, i can get it for you

Not measured yet: what leisure talk does to a life (the next lives have it; the three `edge` lives do not, so
that the new ear is measured alone).

## Still open

- The parent does not see the picture (leisure talk is with the eyes closed for now).
- The parent model sometimes praises babble as a word ("oh wow you said spoon!" to "spowon"). Parents do too.
- Whether the brain should have states of its own to talk about is left open, as Kris preferred.

## Does leisure talk cost the answers to lessons? (2026-11-17, `experiments/chat_cost_probe.py`, three seeds)

In a life with leisure talk some text answers are run-on, parent-like strings ("you're looking you want to happy
so"). Word cells alone on the text lessons of first words and two words (three passes, full-size corrections), with
one logged leisure exchange after every fourth lesson (what the child said → what the parent said back):

| | held-out text score | units in a reply | replies of four or more units |
|---|---|---|---|
| lessons only | 0.456 0.462 0.467 | 1.19 | 0.01 |
| with leisure talk | 0.441 0.444 0.445 | 1.42 | 0.04 |
| leisure talk, the parent's reply cut to its first unit | 0.463 0.461 0.462 | 1.19 | 0.01 |

- Leisure talk costs a little, the same in every seed (−0.018), and it is the length of the parent's replies that
  costs, not hearing them: the child learns that what follows an utterance is long.
- It is not the main cause of the bad text answers in lives (text alone gives replies of the right length).
- A toddler hears long sentences all day and answers in one or two words: what it takes in from a long utterance is
  limited (a few units, the stressed and the last ones). The word cells take in every unit of the reply. Limiting
  what is taken in from a reply that was not an answer to a lesson is the organic direction; not built.

**An intake limit** (same probe, three seeds): only the last two units of the parent's reply learned from (what a
one-word speaker takes in: its own usual length plus one, `SpeakingBrain.intake`, life env INTAKE=1): held-out text
0.454 0.456 0.441 (lessons only 0.462, whole replies 0.443), replies of four or more units 0.01. The run-ons go; a
third of the small cost comes back. It goes into the next lives.
