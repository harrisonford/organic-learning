# A curriculum: teaching in order, from babbling to fourth grade (design, 2026-10-25)

So far the brain has lived in a world where every kind of question arrives at once, from "is there a person?" to WinoGrande, with one-word answers. A child is not taught that way. Capabilities are built in layers: sounds, then words, then two-word phrases, then sentences, then school. Each layer rests on the one before.

This design makes the world teach in order. It also makes the brain's limits measurable: **it moves to the next stage only when it has mastered the current one.** If it cannot get past a stage, we stop there and know why. We do not want a structure that cannot handle second grade but scores on fifth-grade items by luck.

## Principles

1. **Stages in order, gated by mastery.** A stage is passed when the brain reaches its pass mark on held-out items of that stage, *and* still holds every earlier stage (at least 80% of the pass mark it reached there).
2. **Spaced review.** While at stage *n*: 70% of moments come from stage *n*, 30% from earlier stages, weighted toward the most recent. School works this way, and it guards against forgetting.
3. **Leniency that shrinks.**
   - Early on, anything in the right direction is rewarded: a babble that sounds a bit like "dog" counts.
   - Each stage asks for more: the right word, then the right words, then the right order.
   - Leniency only changes the reward. What the adult *says* (the model to imitate) is always the full correct answer.
4. **Stop where it stops.** If a stage is not passed after a long time (a budget of moments per stage, e.g. 500k), the run ends with a verdict: "this structure, with this much experience, cannot pass stage X". That verdict is the result, and it is what should drive an architecture change. It tells us which missing piece to add (below), instead of guessing.
5. **The brain is never told the stage.** As with deadlines, the environment knows the stage and grades by it. The brain only lives through moments and rewards.

## The stages

Ages are the human reference, not a claim. "Answer form" is what earns full reward at that stage.

| stage | human reference | what is taught | answer form | leniency (what earns partial reward) | sources we have / will have |
|---|---|---|---|---|---|
| **0 babbling** | 6-12 months | hearing speech; producing sounds; echoing back | any sound | any vocalization; more for sounding like what was just heard | ear on all text; self-hearing |
| **1 first words** | 12-18 months | naming things seen ("what's this?" -> "dog"); hi, bye, more, no | one word | sound similarity to the word (byte n-gram overlap, measured by the ear's own code) | CIFAR-100 names, COCO objects, adults' naming items |
| **2 two words** | 18-30 months | colours, big/small, yes/no, counting to 3, "red ball", "dog run" | one or two words | right content word in any form | VQA colour, yes/no; COCO presence and counts; adults |
| **3 preschool** | 3-5 years | where (in/on/under), counting to 10, feelings, short replies, who-did-what in one-sentence stories | 2-5 word phrase | right content words, any order | COCO where; DailyDialog short replies; text world chat, mind, story; adults |
| **K kindergarten** | 5-6 | letters and sight words on photos, rhymes, opposites, counting to 20, shapes, more/less | short sentence | content words right; small grammar slips allowed | reading on photos; wordplay; logic; adults |
| **1 first grade** | 6-7 | addition/subtraction within 20; reading short sentences; questions about a 2-3 sentence story; time to the hour; describing a picture in a sentence | sentence | content words + order of the key parts | bAbI tasks 1, 6-9; COCO captions (human sentences); adults |
| **2 second grade** | 7-8 | within 100, place value, measuring; longer stories; why/because; following instructions on a grid ("right 2, up 1: where are you?") | sentence with because / then | full meaning; wording may differ (judged) | bAbI 2-3, 15-18; text world why/order/analogy; adults |
| **3 third grade** | 8-9 | multiplication facts, simple fractions; main idea of a paragraph; multi-step problems; comparing and explaining | two sentences | judged by an adult model | CommonsenseQA, ARC-Easy; adults |
| **4 fourth grade** | 9-10 | multi-digit arithmetic; WinoGrande-style reasoning; **simple coding**: predict what a 3-line program prints, write the moves for a robot, fix one wrong step | sentences, short code | judged; code checked by running it | WinoGrande; adults; generated programs (run to check) |

Higher stages are not planned until the brain reaches stage 3.

## Grading without spending the budget

- Stages 0-2 are graded locally: the ear's sound similarity, and exact or content-word match.
- Stages 3-K are graded locally against **several acceptable answers per item**, written by the adults when the item is made (not at grading time).
- From grade 1, free answers are compared to the acceptable answers by content-word overlap and order. Only a sample of tests is sent to an adult model, to check that local grading agrees with it.
- Coding answers are checked by running them.

## What each stage needs from the brain (the language proposals, re-ordered)

[LANGUAGE_BIOLOGY.md](LANGUAGE_BIOLOGY.md) listed what is missing. A curriculum turns that list into a development order. Each piece is added when the brain gets stuck at the stage that needs it, not before.

| stage | needs | biological piece | today |
|---|---|---|---|
| 0 | to make sounds and hear itself | a speech motor area whose units are linked, by babbling, to the sounds they make (auditory-motor map) | missing: answers are whole word-forms |
| 1 | to say a word it has heard, and tie words to things seen | the copy route (hearing -> saying) and cross-situational word-object links | missing |
| 2 | to say two units in a row | sequencing: say one unit per tick, hear itself (efference copy, `Ear.said` exists), stop at silence | missing |
| 3 / K | phrases; listening to a one-sentence story | listening over ticks; fast episodic binding of who + where (the repo's `hippocampus.py`) | partly: an 8-unit echo |
| 1 | sentences; questions about a short story; conversation | working memory that is rehearsed (phonological loop); memory of the previous turn | missing |
| 2+ | following instructions step by step | the same sequencing loop applied to actions (grid moves), reusing the gaze motor map idea | not yet |

The current brain would likely get stuck at **stage 0-1**. It cannot babble or repeat, and that is honest: it is an image-question answerer, not a speaker. The first architecture step would therefore be the speech motor area with babbling, tested on stage 0-1 alone. The cortex comparison showed how: a controlled pair of lives in one frozen world.

## What can start now, before any architecture change

- **Curriculum data.** The adults write items tagged by stage and subject, each with several acceptable answers and the expected answer form (`experiments/world/curriculum_world.py`, `data/world/curriculum.jsonl`). It shares the text budget with the text world.
- **Mapping the existing sources onto stages** (table above), so nothing collected so far is wasted.
- **Grading functions per stage** (local, as above), testable on the adults' own answers.
- **A curriculum world:** serves moments by stage with spaced review, and tracks promotion. It can be run with today's brain to get the baseline verdict: where does the current structure stop?
