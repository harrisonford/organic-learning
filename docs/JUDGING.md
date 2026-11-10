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
