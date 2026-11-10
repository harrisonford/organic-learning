"""Judging answers the way an adult would (docs/JUDGING.md).

The brain is free in how it answers ("what is this?" -> "a dog", "doggie", "that's my dog"):
what matters is whether a caring adult would accept the answer from a child of that age.
A local open-weight decision model (lev 4B, served by llama.cpp on the GPU at
127.0.0.1:8790/v1/systemone, systemd unit organic-judge) gives the probability of acceptance;
it agrees with the hosted decisions-fast judge 93% of the time.

- An answer that is exactly one of the adult's (same content words: "a dog" = "dog") is accepted outright.
- A bare "yes" or "no" is accepted only when yes or no is among the adult's answers (the judge model is too kind to it).
- Every judged (stage, question, answers, said) is cached on disk: a brain's habitual
  answers, which repeat a lot, are judged once (as a caregiver learns a child's idiolect).
- If the judge is unreachable, the curriculum's local grade stands (a life never waits on it).

This is the environment (the adults), not the brain: it may use any model.
"""

import hashlib
import json
import os
import time
import urllib.request

URL = os.environ.get("JUDGE_URL", "http://127.0.0.1:8790/v1/systemone")
CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "judgments.jsonl")
AGE = {"B": "a 12-month-old", "W1": "an 18-month-old", "W2": "a two-year-old", "P": "a preschooler (3-5 years)",
       "K": "a kindergarten child", "G1": "a first grader", "G2": "a second grader", "G3": "a third grader", "G4": "a fourth grader"}
ASK = ("For item{i}: would a caring adult accept what the child answered as a right answer for a child of that age "
       "(meaning matters, not the exact words; a picture the judge cannot see is described by the adult's own answer)?")


def _tidy(s):
    from .curriculum import tidy
    return tidy(s)


def _core(s):
    """The content words of an answer ("a dog" -> "dog"): the grader's own stopwords dropped."""
    from .curriculum import STOP
    words = [w for w in _tidy(s).replace("!", " ").replace("?", " ").replace(".", " ").replace(",", " ").split() if w not in STOP]
    return " ".join(words)


class Judge:
    def __init__(self, cache=CACHE):
        self.cache_path = cache
        self.cache = {}
        self.down_until = 0.0
        self.calls = self.hits = self.fallbacks = 0
        if os.path.exists(cache):
            for line in open(cache):
                try:
                    d = json.loads(line)
                    self.cache[d["k"]] = d["p"]
                except ValueError:
                    pass

    @staticmethod
    def key(stage, question, answers, said):
        raw = json.dumps([stage, question, sorted(_tidy(a) for a in answers), _tidy(said)])
        return hashlib.sha1(raw.encode()).hexdigest()

    def _ask(self, items):
        state = {f"item{i}": {"child": AGE.get(st, "a young child"), "adult asked": q, "adult's own answer": a[0],
                              "other answers the adult would accept": list(a[1:]), "the child answered": said or "(nothing)"}
                 for i, (st, q, a, said) in enumerate(items)}
        questions = {f"item{i}": {"type": "noul", "instructions": ASK.format(i=i)} for i in range(len(items))}
        req = urllib.request.Request(URL, data=json.dumps({"state": state, "questions": questions}).encode(),
                                     headers={"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=120))
        return [float(d["answers"][f"item{i}"]["noul"]) for i in range(len(items))]

    def accept(self, items, fallback):
        """items: [(stage, question, answers, said)]; fallback(i) -> local grade if the judge is down.
        Returns the probability an adult accepts each answer."""
        out, todo = [None] * len(items), []
        for i, (st, q, a, said) in enumerate(items):
            if said and (_tidy(said) in [_tidy(x) for x in a] or (_core(said) and _core(said) in [_core(x) for x in a])):
                out[i] = 1.0
                continue
            if not (said or "").strip():
                out[i] = 0.0
                continue
            # the local judge accepts a bare "yes" to what/where/who questions (0.99 for "where is the dog?" -> "yes",
            # found 2026-11-10); an adult does not, unless yes or no is among the answers
            if _core(said) in ("yes", "no") and not any(w in ("yes", "no") for x in a for w in _core(x).split()):
                out[i] = 0.0
                continue
            k = self.key(st, q, a, said)
            if k in self.cache:
                out[i] = self.cache[k]
                self.hits += 1
            else:
                todo.append((i, k))
        for j in range(0, len(todo), 20):  # batches of 20 (agreement 0.93 with the hosted judge at this size)
            chunk = todo[j : j + 20]
            probs = None
            if time.time() >= self.down_until:
                try:
                    probs = self._ask([items[i] for i, _ in chunk])
                    self.calls += 1
                except Exception:  # noqa: BLE001  the judge is down: local grades for a minute
                    self.down_until = time.time() + 60
            if probs is None:
                for i, _ in chunk:
                    out[i] = float(fallback(i))
                    self.fallbacks += 1
                continue
            with open(self.cache_path, "a") as f:
                for (i, k), p in zip(chunk, probs):
                    out[i] = p
                    self.cache[k] = p
                    f.write(json.dumps({"k": k, "p": round(p, 4)}) + "\n")
        return out
