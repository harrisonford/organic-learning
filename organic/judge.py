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

2026-11-19 (docs/JUDGING.md, the audit): asked only "would a caring adult accept this?", the judge accepted "bear"
for a dog, "yes" where the answer was "no", and the question said back; and since an accepted answer was not
corrected, lives were taught some of their own mistakes. Now:
- Lenient about form, strict about meaning. Where the world settles the answer (a name, yes or no about what is
  there, how many, the adults' own lessons) the adult's answer is told to the judge as the truth, and the question
  is whether the child meant the same. Measured on 367 wrong answers of three brains: 2-16 accepted (nearly all of
  them the right word run together with another), none of the right ones refused.
- Where many replies are fine (talk, routines, stories: OPEN) the judge is asked as before.
- Words of the adult's question said back that are not in any accepted answer are not an answer ("who is holding
  you?" -> "holding you?").
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
# where the world settles the answer: the adult's answer is the truth; only the form is free
MEANT = ("For item{i}: the adult's own answer is the truth about what is there. Did the child mean the same as the adult? "
         "Another way of saying the same word counts (baby talk, a mispronunciation, an extra word). A different thing, "
         "the opposite of yes or no, a different number, or only repeating the adult's question does not count. "
         "Several different words run together into one (the right word stuck to the names of other things) do not count either.")
OPEN = ("social", "text:chat", "text:dialogue", "dailydialog", "text:story", "text:mind", "echo", "echo:wide")  # sources where many replies are fine
VERSION = "2"  # of the questions above: an answer judged under another version is judged again


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
    def key(stage, question, answers, said, closed=None):
        raw = json.dumps([stage, question, sorted(_tidy(a) for a in answers), _tidy(said)] + ([] if closed is None else [VERSION, bool(closed)]))
        return hashlib.sha1(raw.encode()).hexdigest()

    def _ask(self, items, closed=False):
        own = "adult's own answer (the truth)" if closed else "adult's own answer"
        state = {f"item{i}": {"child": AGE.get(st, "a young child"), "adult asked": q, own: a[0],
                              "other answers the adult would accept": list(a[1:]), "the child answered": said or "(nothing)"}
                 for i, (st, q, a, said) in enumerate(items)}
        questions = {f"item{i}": {"type": "noul", "instructions": (MEANT if closed else ASK).format(i=i)} for i in range(len(items))}
        req = urllib.request.Request(URL, data=json.dumps({"state": state, "questions": questions}).encode(),
                                     headers={"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=120))
        return [float(d["answers"][f"item{i}"]["noul"]) for i in range(len(items))]

    def accept(self, items, fallback, sources=None):
        """items: [(stage, question, answers, said)]; fallback(i) -> local grade if the judge is down; sources: where
        each item is from (None: as before 2026-11-19, every question open). Returns the probability an adult
        accepts each answer."""
        out, todo = [None] * len(items), []
        closed = [sources is not None and sources[i] not in OPEN for i in range(len(items))]
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
            if sources is not None:  # words of the question said back that are in no accepted answer are not an answer
                asked, wanted = set(_core(q).split()), {w for x in a for w in _core(x).split()}
                mine = set(_core(said).split())
                if mine and mine <= asked and not mine & wanted:
                    out[i] = 0.0
                    continue
            k = self.key(st, q, a, said, closed[i] if sources is not None else None)
            if k in self.cache:
                out[i] = self.cache[k]
                self.hits += 1
            else:
                todo.append((i, k))
        chunks = []  # batches of 20 (agreement 0.93 with the hosted judge at this size); the open and the closed questions apart
        for kind in (False, True):
            part = [t for t in todo if closed[t[0]] == kind]
            chunks += [part[j : j + 20] for j in range(0, len(part), 20)]
        for chunk in chunks:
            probs = None
            if time.time() >= self.down_until:
                try:
                    probs = self._ask([items[i] for i, _ in chunk], closed[chunk[0][0]])
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
