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
