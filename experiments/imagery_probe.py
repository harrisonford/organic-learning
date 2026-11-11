"""Heard words bring back what was seen with them (offline; docs/CONVERSATION.md, proposal B).

The same moments in the same order (first words and two words mixed, with one unasked comment per moment, as a
life at two words lives them) are lived by the word cells
  plain      organic/words.py as in the lives of 2026-11-10
  imagery    the same with feedback synapses from the ear's unit cells onto the visual neurons: eyes open, a
             visual neuron driven from below and above bursts and the word cells read the bursts too; eyes
             closed, the feedback is passed on as a faint picture
Measured on held-out items after one pass and after PASSES passes:
  first words: pictures named right; text lessons (eyes closed), beside what a parrot of the last word gets,
               and on the items the parrot gets wrong
  two words by kind; for "is there a ...?" also how much the answer depends on the photograph:
               P(says yes | it is there) - P(says yes | it is not)
    SEEDS=0,1,2 PASSES=3 python experiments/imagery_probe.py
"""
import json, os, pickle, random, sys, time, zlib
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.words import Words  # noqa: E402
from organic.curriculum import Curriculum, grade, tidy, words as split  # noqa: E402
from organic.world import WORLD_DIR  # noqa: E402

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
N = int(os.environ.get("N", 9000))
PASSES = int(os.environ.get("PASSES", 3))
ARMS = os.environ.get("ARMS", "plain,imagery").split(",")
cur = Curriculum(seed=0)
ear = pickle.load(open(os.environ.get("EAR", "out/kid_s3.ear.pkl"), "rb"))
cortex = pickle.load(open("data/visual_cortex.pkl", "rb"))
FRAMES = json.load(open(os.path.join(WORLD_DIR, "naming_frames.json")))["frames"]
V = pickle.load(open("out/tmp/vis_cache.pkl", "rb")) if os.path.exists("out/tmp/vis_cache.pkl") else {}


def vis(ref):
    if ref not in V:
        img = cur.image(ref)
        V[ref] = None
        if img.max() > 0:
            v = np.concatenate(cortex.code(img, grid=4))
            V[ref] = (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)
    return V[ref]


# HEAR=words: a reference ear that cuts at the spaces and drops punctuation (to see what the ear's own cuts cost)
if os.environ.get("HEAR") == "words":
    hear = lambda q: [w.encode() + b" " for w in split(q)]  # noqa: E731
else:
    hear = lambda q: ear.units(q, learn=False)  # noqa: E731
w1 = [it for items in cur.train["W1"].values() for it in items]
w2 = [it for items in cur.train["W2"].values() for it in items]
test_pics = [it for src in ("cifar:name", "commons:name") for it in cur.test["W1"][src][:150]]
test_text = cur.test["W1"].get("curriculum", [])[:150]
test_w2 = {src: items[:150] for src, items in sorted(cur.test["W2"].items())}
named = {it[2][0] for it in w1 if it[4] in ("cifar:name", "commons:name")}
parrot = lambda it: (split(tidy(it[1])) or [""])[-1]  # noqa: E731
dim = 7168
results = {}


def measure(words):
    f = lambda q, v: words.recall(hear(q), v)[0] or ""  # noqa: E731
    row = {}
    ok, same = [], 0
    for it in test_pics:
        asks = [FRAMES[(zlib.crc32(repr(it[0]).encode()) + 17 * j + 5) % len(FRAMES)] for j in range(3)]
        said = [f(q, vis(it[0])) for q in asks]
        ok += [tidy(s) in [tidy(a) for a in it[2]] for s in said]
        same += len(set(said)) == 1
    row["W1 pictures"], row["same x3"] = float(np.mean(ok)), same / len(test_pics)
    g = [grade("W1", f(it[1], None), it[2]) for it in test_text]
    p = [grade("W1", parrot(it), it[2]) for it in test_text]
    row["W1 text"], row["parrot"] = float(np.mean(g)), float(np.mean(p))
    row["W1 text beyond parrot"] = float(np.mean([a for a, b in zip(g, p) if b < 0.5]))

    def content_only(it):  # the unit the content dendrite drives most, whatever the context expects
        act, drive, ctx = words.expect(hear(it[1]), None)
        return words.units[int(np.argmax(drive[1:])) + 1].decode("utf-8", "replace") if len(drive) > 1 else ""
    row["beyond parrot, content dendrite alone"] = float(np.mean([grade("W1", content_only(it), it[2]) for it, b in zip(test_text, p) if b < 0.5]))
    for src, items in test_w2.items():
        said = [f(it[1], vis(it[0])) for it in items]
        row["W2 " + src] = float(np.mean([grade("W2", s, it[2]) for s, it in zip(said, items)]))
        if src == "coco:presence":
            yes = lambda keep: float(np.mean([tidy(s) == "yes" for s, it in zip(said, items) if keep(it)] or [np.nan]))  # noqa: E731
            row["there: yes|yes - yes|no"] = yes(lambda it: it[2][0] == "yes") - yes(lambda it: it[2][0] == "no")
            thing = lambda it: tidy(it[1]).split()[-1]  # noqa: E731
            for label, keep in (("named in first words", lambda it: thing(it) in named), ("never named", lambda it: thing(it) not in named)):
                sub = [(s, it) for s, it in zip(said, items) if keep(it)]
                row[f"there ({label}, n {len(sub)})"] = float(np.mean([grade("W2", s, it[2]) for s, it in sub] or [np.nan]))
    row["W2"] = float(np.mean([row["W2 " + src] for src in test_w2]))
    return row


for seed in SEEDS:
    rng = np.random.default_rng(seed)
    order = [w1[i] for i in rng.choice(len(w1), min(N, len(w1)), replace=False)] + [w2[i] for i in rng.choice(len(w2), min(N, len(w2)), replace=False)]
    order = [order[i] for i in rng.permutation(len(order))]
    for arm in ARMS:
        words = Words(dim, ear, imagery="imagery" in arm, sound="sound" in arm)
        wr, t0 = random.Random(seed), time.time()
        cur.orng = random.Random(seed + 2)
        for p in range(PASSES):
            for it in order:
                words.learn(hear(cur.wording(it, wr)), vis(it[0]), hear(it[2][0]))
                s, src = ("W1", ("cifar:name", "commons:name")[cur.orng.randrange(2)])  # an adult names a picture, unasked
                ob = cur.orng.choice(cur.train[s][src])
                w = ob[2][0]
                says = cur.orng.choice(cur.comments).replace("a THING", ("an " if w[0] in "aeiou" else "a ") + w).replace("THING", w)
                words.learn([], vis(ob[0]), hear(says))
            if p in (0, PASSES - 1):
                row = measure(words)
                results.setdefault(f"{arm}, x{p + 1}", []).append(row)
                print(f"seed {seed} {arm:8} x{p + 1} ({time.time() - t0:.0f}s, {words.cells()}): " + " | ".join(f"{k} {v:.3f}" for k, v in row.items()), flush=True)
        if arm == "imagery" and seed == SEEDS[0]:  # (per-cue feedback only)  # what the mind's eye shows: does a heard name bring back its own thing?
            n = len(words.units)
            names = [w for w in sorted(named) if w.encode() in words.index and w.encode() in words.cue_index]
            idx = [words.index[w.encode()] for w in names]
            hit = [names[int(np.argmax(words.Wv[idx] @ words.G[words.cue_index[w.encode()]]))] == w for w in names]
            print(f"  mind's eye: hearing a name, the picture it brings back drives its own name cell most in {np.mean(hit):.2f} of {len(names)} names", flush=True)
            for q in ("is there a dog?", "is there a person?", "what does the cow say?", "what color is the banana?"):
                cues = [c for c in (words._cue(u, False) for u in set(hear(q))) if c is not None]
                print(f"    {q!r}: heard as {[u.decode() for u in hear(q)]}; size of what each unit predicts will be seen: "
                      + ", ".join(f"{words.cues[c].decode()!r} {np.linalg.norm(words.G[c]):.2f}" for c in cues), flush=True)

print(f"\nmean over seeds {SEEDS} (spread):")
for name, rows in results.items():
    print(f"  {name:12} " + " | ".join(f"{k} {np.nanmean([r[k] for r in rows]):.3f}±{np.nanstd([r[k] for r in rows]):.3f}" for k in rows[0]))
json.dump(results, open(f"out/imagery_probe_{'_'.join(ARMS)}_{'_'.join(map(str, SEEDS))}.json", "w"))
