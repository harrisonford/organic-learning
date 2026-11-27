"""The trace where a whole view is seen: an IT-like stage on CORe50, in order against shuffled (queue E 31; VISION.md
2026-11-26 22:25).

The 3 x 3 second stage learned nothing from time (core50_trace_probe.py, seed 0): a turning object changes the whole
view, not a patch of it. Inferotemporal cells see the whole object, and that is where temporal contiguity is known
to shape them (Li & DiCarlo 2008; Wallis & Rolls 1997 put the trace at the top). Here a population of KINDS cells
reads the first stage's whole-view code through its relay (the photo cortex's 4,096 numbers, already computed for
every tenth CORe50 frame by core50_trace_probe.py: half a second apart, in each video's order) and develops by the
eye's own rule (winner-take-all Hebbian with a conscience), one pass, three arms on the same frames:
    order      videos in a random order, each video's frames in their order; a cell that won a frame before still
               carries a trace (0.7 a frame) and learns the present frame in proportion (Földiák 1991)
    shuffled   the same frames in a random order, the same trace: time the only difference
    plain      in order, no trace
The stage's answer to a frame: each cell's match above the average cell's (the filters' response), at unit length.
Read on the held-out sessions: the nearest frame from another session, same object (chance 0.02) and same kind
(chance 0.10); beside it the first stage's code alone.

    SEEDS=0,1,2 python experiments/core50_it_probe.py
"""
import os, pickle, sys
import numpy as np

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
KINDS, KEEP = int(os.environ.get("KINDS", 256)), 0.7
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # the pickled cortex needs the organic package
videos = [json.loads(line) for line in open(os.path.join(os.path.expanduser("~/datasets/core50"), "frames.jsonl"))]
TEST_SESSIONS = {3, 7, 10}


def rows(vs, every=10):
    return np.array([(v["start"] + i, v["object"], (v["object"] - 1) // 5, v["session"], k) for k, v in enumerate(vs) for i in range(0, v["frames"], every)])


train_v = [v for v in videos if v["session"] not in TEST_SESSIONS]
test_v = [v for v in videos if v["session"] in TEST_SESSIONS]
TR, TE = rows(train_v), rows(test_v)


def unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


def elsewhere(Z, meta):
    obj, kind, sess = meta[:, 1], meta[:, 2], meta[:, 3]
    so = sk = 0
    for c in range(0, len(Z), 1024):
        S = Z[c: c + 1024] @ Z.T
        S[sess[c: c + 1024, None] == sess[None, :]] = -np.inf
        j = np.argmax(S, axis=1)
        so += int((obj[j] == obj[c: c + 1024]).sum())
        sk += int((kind[j] == kind[c: c + 1024]).sum())
    return so / len(Z), sk / len(Z)


for seed in SEEDS:
    cortex_seed = seed if os.path.exists(f"out/bench/core50_c50first_s{seed}_train.npy") else 0
    relay = pickle.load(open(f"out/bench/cortex_c50first_s{cortex_seed}.pkl", "rb")).relay
    Xtr = unit((np.load(f"out/bench/core50_c50first_s{cortex_seed}_train.npy").astype(np.float32) - relay.mu) @ relay.Z)
    Xte = unit((np.load(f"out/bench/core50_c50first_s{cortex_seed}_test.npy").astype(np.float32) - relay.mu) @ relay.Z)
    assert len(Xtr) == len(TR) and len(Xte) == len(TE), "the saved codes were made with another EVERY"
    rng = np.random.default_rng(seed + 31)
    order = np.concatenate([np.nonzero(TR[:, 4] == k)[0] for k in rng.permutation(len(train_v))])  # videos at random, frames in order
    shuffled = rng.permutation(order)
    o, k = elsewhere(Xte, TE)
    print(f"seed {seed} (first stage of seed {cortex_seed}) first stage alone: same object {o:.3f}, same kind {k:.3f}", flush=True)
    for arm, seq, trace in (("order", order, True), ("shuffled", shuffled, True), ("plain", order, False)):
        r = np.random.default_rng(seed)
        U = unit(r.standard_normal((KINDS, Xtr.shape[1])).astype(np.float32))
        wins, tr, last_video = np.zeros(KINDS), np.zeros(KINDS, np.float32), -1
        for n, i in enumerate(seq):
            x = Xtr[i]
            if arm != "shuffled" and TR[i, 4] != last_video:  # a new video: nothing of the last one carries over
                tr[:] = 0
            last_video = TR[i, 4]
            c = int(np.argmax(U @ x - 0.3 * (wins / (wins.mean() + 1e-8) - 1)))
            wins[c] += 1
            tr = KEEP * tr if trace else 0 * tr
            tr[c] = 1.0
            rate = 0.05 * (1 - n / len(seq)) + 0.005
            for kk in np.nonzero(tr > 0.1)[0]:
                w = U[kk] + rate * tr[kk] * (x - U[kk])
                U[kk] = w / np.linalg.norm(w)
        R = Xte @ U.T
        R = unit(np.maximum(R - R.mean(axis=1, keepdims=True), 0))
        o, k = elsewhere(R, TE)
        print(f"seed {seed} IT stage, {arm:8}: same object {o:.3f}, same kind {k:.3f} | wins min/max {int(wins.min())}/{int(wins.max())}", flush=True)
