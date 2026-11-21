# Judging answers like an adult would (2026-11-09, study and design)

The user: the brain should have freedom in how it answers ("what is this?" → "a dog",
"doggie", "that's my dog" should all be right), judged intelligently, the way people judge
each other; and this has to work at scale without costing money. Maybe it is also an
ingredient for more human-like interaction.

## A free judge that speaks probabilities

`parallel/decisions-fast:free` (OpenRouter Decisions API, `/api/alpha/decisions`, released
2026-11-06, $0) answers typed questions about a state with probabilities, many questions per
request evaluated in parallel. One request judged a toddler's eight answers to "what is
this?" (a dog): doggie 0.96, dog dog! 0.92, a dog 0.90, puppy 0.89, that's my dog 0.55,
woof woof 0.37, the ball 0.03, cat 0.02.

## Study: our grader vs the judge on a brain's real answers

`experiments/judge_study.py`: kid_s3 (its final state, age 745k) answered 80 held-out items
each of W1, W2, P; the curriculum's local grader and the judge graded every answer (20 answers
per request, 12 requests).

| stage | local grade | judge acceptance (mean p) | judge accepts, grader < 0.5 | grader ≥ 0.5, judge rejects |
|---|---|---|---|---|
| W1 first words | 0.20 | 0.25 | 0.00 | 0.00 |
| W2 two words | 0.34 | 0.39 | 0.03 | 0.02 |
| P preschool | 0.08 | 0.20 | 0.07 | 0.00 |

- With one-word answers, the grader is not the bottleneck: kid_s3 hardly uses the freedom
  it would be given. The gap opens with longer, freer answers (preschool: 0.08 vs 0.20; "hold my
  place in line okay" → "okay" accepted 0.90). It will matter more at every later stage.
- The judge catches the grader's false positives: echoing the question ("what is the
  toddler kissing?" → "kissing?") is graded 1.0 locally and rejected (0.27) by the judge.
- The judge is noisy near 0.5 ("where is the person?" → "yes" 0.72; "how many microwaves?" →
  "yes" 0.53). Its probability is best used as a graded signal, not a hard yes/no.

## Design (proposed, not built yet)

1. **Graded acceptance as the reward.** The reward is the probability that an adult accepts the
   answer, not right/wrong. Caregivers respond in degrees.
2. **Adults correct only what they do not accept.** When an answer is accepted, the adult does
   not impose their own wording (no imitation target): the brain's own form stands, and that
   is its freedom. When it is not accepted, the adult says theirs, as now. Child-directed
   speech works this way: adults reformulate erroneous utterances and let accepted ones
   stand (Chouinard & Clark 2003; Saxton 1997).
3. **Tiered judging, at no cost:**
   - the local grader, instantly;
   - a cache of judged (question, answer) pairs, so a brain's habitual answers, which repeat a
     lot, are judged once (like a caregiver learning the child's idiolect);
   - new pairs queued and judged in batches by the free decisions model (~20 per request).
     Until a pair is judged, the local grade stands.
4. **Budget:** judging needs a share of the 500 requests/day (the user's half-of-limit cap);
   at 20 answers per request, 100 requests are 2,000 new judgments a day.

## Other models (2026-11-09 scan)

- `liquid/lfm-3-1.7b:free`: well-formed but unreliable items ("is the cat sleeping?" →
  "yes, it sleeps" / "no, it wakes up" / "it has soft ears"). Not a knowledge writer. A
  candidate for simple paraphrase jobs, always with a judge.
- `parallel/decisions-fast:free`: a decisions model (typed judgments, not chat). The judge above,
  and a candidate to replace Jev (typesafe, paid, capped at $1.50) for fast decisions.

## A local judge: open System One models on the GPU (2026-11-09)

llama.cpp (prebuilt CUDA 13.5 build b12229 in `~/tools/llama.cpp/llama-b12229`, no system
install) serves open-weight decision models with `llama serve -hf <repo>` at a local
`/v1/systemone` endpoint, in the same typed-decisions format as the hosted API. On the RTX
3070 (8 GB, unused by training), the same 240 answers were judged by each
(`experiments/local_judge_compare.py`; flags `-ngl 99 -b 2048 -ub 2048 -c 8192 -np 4`):

| local judge | per request | agrees with decisions-fast | + exact-match override | correlation | judgments/s |
|---|---|---|---|---|---|
| **lev 4B (Apache 2.0, ggml-org/lev-GGUF Q8_0)** | 20 | 0.92 | **0.93** | 0.86 | 12.6 |
| lev 4B | 1 | 0.89 | 0.92 | 0.84 | 18.8 |
| Liquid d1-3B (Liquid licence, sees images) | 20 | 0.83 | 0.91 | 0.88 | 20.6 |
| Liquid d1-3B | 1 | 0.86 | 0.88 | 0.89 | 38.8 |
| Laya 421M (Apache 2.0) | 1 | 0.84 | 0.86 | 0.83 | 82 |
| Laya 421M | 20 | 0.42 | 0.44 | 0.70 | 11 (lost in a 20-item state) |
| Kev-4B (Apache 2.0) | 1 / 20 | 0.78 / 0.83 | 0.82 / 0.88 | 0.83 / 0.88 | 19 / 6 |

- The common local error is rejecting an answer that exactly matches the adult's ("how
  many people?" → "zero" when it is zero); an exact match is accepted outright and the judge
  decides the rest (the hosted judge accepts 97% of exact matches anyway).
- **lev 4B with the exact-match override agrees with the hosted judge 93% of the time**, with
  the hosted judge's own noise near 0.5 included, at 12.6 judgments/s, eight times a
  life's pace, free and with no request limits.
- Laya is the fast option (82/s) if speed ever matters more; it accepts echoes of the
  question ("what's this one?" → "one?").

## Built (2026-11-09)

- `organic-judge` (systemd user service, `experiments/systemd/organic-judge.service`): llama.cpp
  serving lev 4B on the GPU at 127.0.0.1:8790/v1/systemone (local only).
- `organic/judge.py`: the probability an adult accepts an answer. An exact match counts as accepted
  (content words: "a dog" = "dog"); every judged answer is cached in data/judgments.jsonl; if the
  judge is down, the local grade stands.
- `JUDGE=1` in curriculum_life: the training reward is the judged acceptance; an accepted answer's
  own form stands (`brain.own_forms`: the imitation target and the hub's link become what the
  brain said, as an adult echoing "doggie!"; only rejected answers get the adult's wording);
  every checkpoint records `judged` next to the local `score` per stage and per source, plus the
  judge's calls, cache hits and fallbacks. **The promotion gate stays on the local score** until
  the two have been compared over a life and the pass marks recalibrated.
- Smoke test (small hub brain, 12k moments): no fallbacks; B 0.83 local / 0.73 judged (stricter
  on echoes), W1 0.33 / 0.31; W2, not yet trained, 0.00 local / 0.31 judged, which needs a look
  (toddler shorthand like "dog!" for "is there a dog?", or too lenient).

## 2026-11-10: the local judge is too kind to "yes"

Found in the first lives with the judge on: "where is the dog?" → "yes" was accepted at 0.99 (also
"where is the man?", "where is the cup?"; "no" was rejected). With accepted own forms standing, a
life would have learned to answer where-questions with "yes". `organic/judge.py` now accepts a bare
"yes" or "no" only when yes or no is among the adult's answers. A stricter instruction alone fixed
two of three cases, not all. Other spot checks were right ("doggie" for dog 0.94, "cat" for dog 0.01,
"white" for blue 0.01, "zero" for three 0.00). "there" to a where-question is accepted (0.9): left,
a toddler pointing is an answer.

## 2026-11-19: the judge audited answer by answer; a design for "acceptable, not absolute"

Kris, 2026-11-19: yes/no questions "should not be like that, so rigid, we should be more lenient with what the
question is; it has to feel acceptable instead of asking for absolutes; we can use Jev or a similar model, but we
need to be able to allow lenient scoring somehow, or maybe not always, but we need to design that properly."

### Where things stood
Since 2026-11-09 the local judge (lev 4B) gives the reward in every life, and an answer it accepts is not
corrected: the word cells are taught what the child said. Stages are still passed on the rigid grade. The two had
never been compared answer by answer. `experiments/judge_audit.py` does it on the three brains as they passed two
words (out/judge_audit_*.jsonl: 548 held-out answers each).

### What the audit found
1. **On names and yes/no the two agree on the score** (first-words names 0.44–0.71 rigid, 0.42–0.74 judged; a thing
   on a page 0.41–0.45 both). The judge is kinder on text (first-words text 0.30–0.46 rigid, 0.48–0.62 judged).
2. **The judge is kind in the wrong places.** It cannot see the picture, and it is not told that the adult's
   answer is the truth. Accepted at 0.93–0.99: "bear" for a dog, "baby" for a bear, "train" for a bus, "lion" for a
   tiger; **"yes" to "can you see a bed?" when there is none** (every such answer); the question said back
   ("who is holding you?" → "holding you?", "what is under your pants?" → "pants?"). Of 333 answers to closed
   questions that were wrong in meaning, it accepted 41.
3. **It is also right where the rigid grade is wrong.** It rejects blends the grade half-accepts ("elephantbear"
   0.03 against 0.58) and overlaps of words that do not answer ("do you see the big car?" → "big": grade 0.80,
   judge 0.03); it accepts a child's form of the right word ("caair", "yog", "here" for "there").
4. **So the lives were taught some of their own mistakes.** 6–8% of all answers are accepted by the judge and
   wrong by the grade; for those the word cells were taught the child's word on the adult's picture. The wrong name
   said most often by these brains is "bear" (27 of the wrong names in two audits; next "cup" 18), and "bear" is
   what the judge accepts for dogs and cats. Not proven to be cause and effect; it is what such a loop would do.
5. **What a child who neither looks nor listens gets:**

   | source | best blind answer | rigid | judged |
   |---|---|---|---|
   | names (first words, pages) | the commonest name | 0.11–0.22 | 0.09–0.20 |
   | "is there a …?" | always "yes" | 0.53 | **0.64** |
   | two-words text | the question's last word | 0.49 | 0.52 |

   Leniency raises the floor of the yes/no source from 0.53 to 0.64: the present judge makes that problem worse.
   The two-words text source as a whole is at the level of saying the last word back (0.44–0.46 against 0.49); the
   gate already counts only the items a parrot gets wrong, where the brains score 0.50–0.56.
6. **Two words above the blind floor** (each source's score less its best blind answer, as a share of what is left;
   rigid grade): seed 1 0.22, seed 0 0.27, seed 2 0.32. Page naming 0.34–0.38 and "is there" 0.32–0.57 carry it;
   text carries nothing in this reading.
7. **A judge that is lenient on form and strict on meaning is within reach of the same local model.** Told that
   the adult's answer is the truth and that another way of saying the same word counts while a different thing,
   the opposite of yes or no, a different number or the question repeated does not, it accepts 2–7 of 257 distinct
   wrong answers and 218–230 of 231 right ones. What it still lets through are run-ons that contain the right word.

### Design (proposed; nothing below is built)
1. **One question for the adult: "did the child mean what I meant?"** Lenient about form, strict about meaning.
   "Doggie", "a dog!", "caair" are right; "bear" for a dog is not, however gently it is said.
2. **Closed and open questions.** Where the world settles the answer (the name of a thing in view, yes/no about
   what is on the page, how many, what colour), the adult's answer is the truth and the judge only decides whether
   the child's words mean it. Where it does not (talk, routines, stories, "what do you want?"), many answers are
   acceptable and the judge decides as it does now. Each source says which it is; that belongs to the adults, not
   to the brain. This is the "maybe not always".
3. **Teaching.** The adult lets the child's own form stand only when it means the same; otherwise the adult says
   theirs. A wrong meaning is never confirmed. (Children's own forms are kept; their mistakes are not.)
4. **Measuring.** Every stage score stands beside what a blind child gets from the same judge (always yes, the
   commonest answer, the last word), and a stage is passed on the part above that. This generalises "beyond the
   parrot", takes the yes/no floor out, and makes a lenient judge safe to gate on.
5. **An audit set that stays.** Some 500 real answers (those of this audit), labelled once by a stronger judge
   (Jev, a few cents of its $1.50) or by Kris; the local judge's agreement with it is measured again whenever the
   judge or its instruction changes, and written here. The gate moves from the rigid grade to the judge only when
   that agreement is known.
6. **Pass marks are set again, in the open**, after the existing levels have been scored the new way. They are
   not to move silently in either direction.

### Built (2026-11-19, evening; Kris: "fix and improve")
- **The judge** (`organic/judge.py`): closed questions are asked "did the child mean the same as the adult?", with
  the adult's answer given as the truth; open sources (talk, routines, stories, echo games) as before; words of the
  question said back that are in no accepted answer are not an answer. On the audited answers: of 367 distinct
  wrong answers to closed questions it accepts 2 (twenty at a time) to 16 (one at a time), 14 of the 16 being the
  right word run together with another; no right answer is refused. It now agrees with the rigid grade on names
  and yes/no within 0.03 and is stricter on text (0.37–0.43 against 0.44–0.46).
- **Teaching** (`SpeakingBrain.recast`, on in every life from now; RECAST=0 for the old way): the adult says
  their own word after every answer; acceptance decides the reward, not what is taught. The literature search of
  the same evening supports exactly this (research_notes/Relatives and publishability 2/judging_answers_literature.md):
  parents confirm meaning and model the form in one turn, recast accepted utterances too, and children's own
  uncorrected errors breed more errors.
- **Measuring**: every checkpoint records, by source, the judged score, the blind floor (the best of always yes,
  always no, the commonest answer, the question's last word, under the same judge) and the score above it, and a
  stage score `above` (the mean over sources with 20 items or more). Also, as an observation, the yes bias on
  yes/no sources (two-year-olds: +0.24 to +0.34, Okanda & Itakura 2008; eyes2w_s2 at its pass: +0.10).
- **The gate** (`GATE=above`, used by the lives started tonight): from first words on, a stage is passed when the
  score above the floor reaches PASS_ABOVE, and earlier stages (not babbling or echo) are held at 80% of theirs.
  **PASS_ABOVE = 0.25 is provisional**: a quarter of the way from the blind child to an adult, one number for every
  stage. Where the existing brains stand on it:

  | brain as it passed | first words above the floor | two words above the floor |
  |---|---|---|
  | seed 1: first words (20,000) / two words (55,000) | 0.28 | 0.22 |
  | seed 2: first words (40,000) / two words (80,000) | 0.40 | 0.32 |
  | seed 0: first words (110,000) / two words (180,000) | 0.34 | 0.27 |

  So at 0.25 all three pass first words and two of three pass two words (seed 1 misses by 0.03). **To be set with
  Kris**; every score is in the records, so any mark can be applied afterwards.
- **What the new reading shows.** Text carries nothing above the floor at either stage (first-words text 0.00–0.12,
  two-words text 0.00): the brain's text answers are, as a whole, no better than saying the question's last word.
  What it has learned above a blind child is in the pictures. And preschool's test is mostly answerable blind
  (a blind child gets 0.75 on "how many" photo questions, 0.5 on presence and where): that stage needs rebuilding
  before any brain is asked to pass it.
- **Not built, from the literature's list:** a person labels the audit set (about 200 answers, loaded with wrong
  ones) and reported scores are corrected by the judge's measured error rates; a picture-blanked brain as one more
  blind baseline; a near-miss policy (is "lion" for a tiger half right?); silence counted as its own outcome.

### 2026-11-20: the floor is read on the questions a parrot cannot answer
The first version of `above` compared a whole source with its best blind child, the parrot among them. On the
adults' text lessons that read as "nothing above the floor", and I wrote so. Looking at the answers item by item
(three brains at their two-words pass) it was wrong in an instructive way:

| text lessons | share of items | the brains are right |
|---|---|---|
| first words: the wanted answer is in the question ("touch your knee" → "knee") | 38% | 0.71 |
| first words: the answer is not in the question ("what does a frog say?" → "croak") | 62% | 0.22 (40% of the answers are question words said back) |
| two words: the answer is in the question | 37% | 0.52 |
| two words: the answer is not, or only partly, in the question | 63% | 0.32–0.33 |

A question whose answer is one of its own words shows nothing: a parrot gets it, and a brain that gets it may be a
parrot. So such questions are left out (those the last word said back answers), and on the rest the brain's judged
score is read beside the best of always-yes, always-no and the commonest answer. `above` is how far that takes it
from the blind child to an adult. This joins the old "beyond the parrot" and the blind floor in one number.

| brain as it passed | first words above | by source: CIFAR, COCO, Commons names, text | two words above | text, a thing on a page, "is there" |
|---|---|---|---|---|
| seed 1 | 0.29 | 0.44, 0.15, 0.52, 0.05 | 0.23 | 0.11, 0.33, 0.24 |
| seed 2 | 0.46 | 0.44, 0.42, 0.74, 0.23 | 0.35 | 0.21, 0.38, 0.47 |
| seed 0 | 0.43 | 0.42, 0.29, 0.64, 0.37 | 0.32 | 0.25, 0.33, 0.38 |
| seed 2 after 100,000 moments of preschool | 0.23 | 0.19, 0.16, 0.50, 0.06 | 0.14 | 0.00, 0.28, 0.13 |

**Correction of what I told Kris on 2026-11-19**: the brains do know some of what the text lessons teach (0.11 to
0.33 of the questions a parrot cannot answer, against a floor of 0.06–0.11); "no better than a parrot" was true of
the source as a whole and false of the knowledge in it. In `curriculum_life.py` from 2026-11-20 (the lives started
on 2026-11-19 were relaunched with it after a checkpoint); each source's record holds `asked` (how many questions
count), `known` (the judged score on them), `floor` and `above`.

## 2026-11-21: no pass mark (GATE=plateau), and the stage is called "pages"

Kris asked what the "two words pass mark" meant ("we're forcing the model to say a certain amount of words?"). No:
"two words" was only the stage's name, and the mark was the score at which the world moved on. But the mark was a
number of ours (0.25, then 0.35 proposed), and the name misled. Kris agreed to both changes:

- **The stage W2 is called "pages"** (`organic/curriculum.py`, NAMES; the key stays W2 so that logs and levels still
  read). It is about picture-book pages: a thing on a page named, "is there a …?", and the adults' short lessons.
  Older docs say "two words" for the same stage. kid-a's `release.py` still says "two words" (no such level is
  released; to be reworded when one is).
- **No mark: the world moves on when the stage's own curve has stopped rising** (`GATE=plateau` in
  `experiments/curriculum_life.py`). At every test the stage's score above the blind floor joins its curve. When the
  mean of the last three tests is no higher than the mean of the three before by more than their own chance error,
  the stage is left. The level it leaves with (the mean of the last three) is recorded, never asked for; earlier
  stages are still to be held at 0.8 of the level they were left with. Babbling is unchanged (its own old mark,
  0.17, observed and never held), echo stays a period.
- Set by hand, and said so: three tests on each side (`PLATEAU_SPAN`), and "more than the chance error" (one
  standard error of the difference, from the tests' own items; about 0.04).
- **What it would have done** (replayed on the lives held in the stage, `cleanw_s0..2`): leave after 45,000 / 45,000
  / 40,000 moments in the stage at 0.34 / 0.33 / 0.38, with first words at 0.36 / 0.43 / 0.44. The mark of 0.25 let
  them leave after 10,000–15,000.
- A consequence to know: a stage in which nothing is learned is also left, after six tests, with a level near zero
  on record. The record says what was learned; the world does not wait. Whether a stage that slides while the next
  is lived should hold the brain back (the 0.8 rule) is the next thing of this kind to look at.
