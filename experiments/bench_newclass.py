"""A new kind of thing, shown once: fast and slow memory on the public benchmark (docs/PUBLISH.md, 2026-11-19).

In the lives a name heard once with a new kind of thing is gone a moment later from the word cells and there with
the episodic store (0.1 against 0.9 on the same picture; nothing for another picture of the kind). Here the same
question on CIFAR-10, where other learners can be asked it too. For each class held back in turn:
    the stream     BEFORE pictures of the other nine classes, each once
    then           SHOTS pictures of the held-back class (1, then up to 5, then up to 20), each once, with 20 pictures
                   of the other classes after each
    asked          the very pictures shown (is the new name said of them?), 200 held-out pictures of the new class
                   (does it reach the kind?), 500 held-out pictures of the old classes (are they still named right,
                   and how often is the new name said of them?)
Learners, on the codes of experiments/bench_codes.py through the relay: the word cells, the word cells with the
episodic store, nearest class mean, linear discriminant analysis from the sums.
    SEEDS=0 HELD=0,2,4,6,8 ARMS=cells,cells+store,ncm,slda python experiments/bench_newclass.py
"""
import os, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("ARMS", "cells,cells+store,ncm,slda")
ARMS_ = os.environ["ARMS"].split(",")
os.environ["ARMS"] = "ncm"  # (bench_stream runs its own arms when imported: one cheap arm on a short stream)
os.environ["N"], os.environ["TESTS"] = "500", "10"
import experiments.bench_stream as B  # noqa: E402  (its codes, relay and learners)

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
HELD = [int(x) for x in os.environ.get("HELD", "0,2,4,6,8").split(",")]
BEFORE, SHOTS = int(os.environ.get("BEFORE", 5000)), (1, 5, 20)
see = lambda X, i: B.seen(X, i, True)  # noqa: E731


def named(learner, xs):
    if isinstance(learner, B.Moments):
        learner.ready()
        W, b, m = learner.readouts[1]  # the middle shrinkage (no picking on the test pictures here)
        return np.argmax((np.stack(xs) - m) @ W.T + b, axis=1)
    if isinstance(learner, B.NCM):
        learner.ready()
    return np.array([learner.name(x) for x in xs])


for seed in SEEDS:
    rng = np.random.default_rng(seed)
    for arm in ARMS_:
        rows = {k: {"shown": [], "kind": [], "old right": [], "new name on old": []} for k in SHOTS}
        t0 = time.time()
        for c in HELD:
            others = rng.permutation(np.nonzero(B.ytr != c)[0])
            mine = rng.permutation(np.nonzero(B.ytr == c)[0])[: max(SHOTS)]
            new_te = np.nonzero(B.yte == c)[0][:200]
            old_te = rng.permutation(np.nonzero(B.yte != c)[0])[:500]
            learner = B.make(arm.replace("store", "relay+store") if "store" in arm else arm)
            at = 0
            for i in others[:BEFORE]:
                learner.learn(see(B.Xtr, i), int(B.ytr[i]))
            at = BEFORE
            for k, j in enumerate(mine, 1):
                learner.learn(see(B.Xtr, j), c)
                for i in others[at : at + 20]:
                    learner.learn(see(B.Xtr, i), int(B.ytr[i]))
                at += 20
                if k in SHOTS:
                    r = rows[k]
                    r["shown"].append(float(np.mean(named(learner, [see(B.Xtr, q) for q in mine[:k]]) == c)))
                    r["kind"].append(float(np.mean(named(learner, [see(B.Xte, q) for q in new_te]) == c)))
                    old = named(learner, [see(B.Xte, q) for q in old_te])
                    r["old right"].append(float(np.mean(old == B.yte[old_te])))
                    r["new name on old"].append(float(np.mean(old == c)))
        for k in SHOTS:
            print(f"seed {seed} {arm:12} shown {k:>2}x ({time.time() - t0:.0f}s; classes held back {HELD}): " + " | ".join(f"{name} {np.mean(v):.3f}" for name, v in rows[k].items()), flush=True)
