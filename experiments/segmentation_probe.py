"""How well does the ear find words? (offline measurement, no change to the ear)

Standard word-segmentation scores (Brent 1999; Goldwater et al. 2009), against the
written word boundaries (spaces and punctuation are kept inside the ear's input,
so a unit is counted as a word if it is a word plus its trailing space/punctuation):
  boundary precision / recall: of the ear's cuts, how many are at a word edge,
                               and of the word edges, how many it cuts
  token precision / recall:    units that are exactly one word
  unit kinds:                  one word / several words fused / a piece of a word / other
Measured on kid_s3's ear and on fresh ears grown from 0 to N utterances (development).
    python experiments/segmentation_probe.py
"""
import collections, copy, pickle, re, sys
sys.path.insert(0, ".")
import numpy as np
from organic.curriculum import Curriculum
from organic.ear import Ear

cur = Curriculum(seed=0)
STAGES = ("B", "W1", "W2", "P", "K")
held = [cur.moment(s)["question"] for _ in range(300) for s in STAGES]
cur.rng.seed(99)
life = [cur.moment(s)["question"] for _ in range(8000) for s in STAGES]


def words(text):
    """Character spans of words, each with its trailing spaces/punctuation."""
    b = text.encode()
    return [(m.start(), m.end()) for m in re.finditer(rb"[A-Za-z0-9']+[^A-Za-z0-9']*|[^A-Za-z0-9']+", b)]


def score(ear, texts):
    bp = bt = br = 0
    tp = tu = tw = 0
    kinds = collections.Counter()
    fused = collections.Counter()
    for t in texts:
        units = ear.units(t, learn=False)
        cuts, pos = set(), 0
        spans = []
        for u in units:
            spans.append((pos, pos + len(u))); pos += len(u); cuts.add(pos)
        ws = words(t)
        edges = {e for _, e in ws}
        cuts.discard(pos); edges.discard(pos)
        bt += len(cuts); br += len(edges); bp += len(cuts & edges)
        wset = set(ws)
        tu += len(spans); tw += len(ws); tp += len(set(spans) & wset)
        for s, e in spans:
            n_in = sum(1 for a, b in ws if a >= s and b <= e)
            if (s, e) in wset: kinds["one word"] += 1
            elif n_in >= 2 or (n_in >= 1 and any(a < e < b or a < s < b for a, b in ws)) and n_in >= 1: kinds["fused words"] += 1; fused[t.encode()[s:e]] += 1
            elif n_in == 0: kinds["piece of a word"] += 1
            else: kinds["word + piece"] += 1
    n = sum(kinds.values())
    return {"boundary P": bp / max(bt, 1), "boundary R": bp / max(br, 1), "word P": tp / max(tu, 1), "word R": tp / max(tw, 1),
            **{k: kinds[k] / n for k in ("one word", "fused words", "word + piece", "piece of a word")},
            "units/word": tu / tw, "fused": fused.most_common(8)}


def show(label, r):
    print(f"{label:24} " + " ".join(f"{k} {r[k]:.2f}" for k in r if k != "fused"), flush=True)


ear3 = pickle.load(open("out/tmp/ear_kid_s3.pkl", "rb"))
r = score(ear3, held)
show(f"kid_s3 ({ear3.utterances} heard)", r)
print("   most fused:", [(u.decode(errors='replace'), n) for u, n in r["fused"]], flush=True)
for t in held[:12:2]:
    print("   ", " | ".join(u.decode(errors="replace") for u in ear3.units(t, learn=False)))

ear = Ear(seed=7)
done = 0
for n in (100, 1000, 5000, 20000, 40000):
    for t in life[done:n]:
        ear.units(t)
    done = n
    show(f"fresh, {n} heard", score(ear, held))
