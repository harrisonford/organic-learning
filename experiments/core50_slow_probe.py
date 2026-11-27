"""The trace on top of the whole code: slow directions of the relayed view, in order against shuffled (queue E 31;
docs/VISION.md 2026-11-27 02:25).

core50_it_probe.py put 256 winner-take-all cells on the first stage's whole-view code and they threw most of it away
(same object from another session 0.03-0.04 against the code's 0.117), so its comparison of orders said nothing.
Here the stage keeps what it reads: KINDS linear cells read the relayed code (whitened by the relay: every direction
of it varies about as much as any other) and learn by Hebb's rule with a trace of their own answer, x as it comes and
the cell's answer averaged over the last frames (Foldiak 1991), with the subspace rule's own decay keeping the cells
apart and at unit length (Oja 1989: dW = rate * (y_trace x^T - y_trace y^T W)). On whitened input, Hebb on a trace
finds the directions that change slowly (slow feature analysis, Wiskott & Sejnowski 2002; derived from spike-timing
plasticity, Sprekeler, Michaelis & Wiskott 2007); without an order there is nothing slow to find and the cells settle
on any directions (the input has no main ones). Three arms on the same frames, one pass:
    order      videos in a random order, each video's frames in their order (half a second apart); the trace keeps
               KEEP of itself a frame and is cleared at a new video
    shuffled   the same frames in a random order, the same trace
    plain      in order, no trace (plain Hebb with the subspace rule's decay)
Measured first, what the trace is for, on the held-out sessions: how alike a frame's answer is to the next frame of
the same video (0.5 s later), to the same video 5 s later, to the same object in another session, to another object of
its kind and to another kind (mean cosine). Then whether it carries: the nearest frame from another session, same
object (chance 0.02) and same kind (chance 0.10). Beside every arm, the code it reads.

    SEEDS=0,1,2 python experiments/core50_slow_probe.py
"""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # the pickled cortex needs the organic package

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
KINDS, KEEP, RATE = int(os.environ.get("KINDS", 256)), float(os.environ.get("KEEP", 0.7)), float(os.environ.get("RATE", 0.5))
RELAY, CODE = os.environ.get("RELAY", "photos"), os.environ.get("CODE", "first")  # CODE: centre or salient for core50_fovea_probe.py's codes  # photos: the relay of 6,000 COCO thumbnails (as the trace probe); world: see below
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


def alike(Z, meta, rng):
    """Mean cosine of a frame's answer with: the next frame of its video, the frame 10 later, the same object in another
    session, another object of its kind, another kind (a random partner of each sort for every frame)."""
    obj, kind, sess, vid = meta[:, 1], meta[:, 2], meta[:, 3], meta[:, 4]
    out = {}
    for lag, name in ((1, "next 0.5 s"), (10, "5 s later")):
        ok = np.nonzero((vid[lag:] == vid[:-lag]))[0]
        out[name] = float(np.mean(np.sum(Z[ok] * Z[ok + lag], axis=1)))
    for name, cond in (("same object, other session", lambda i: (obj == obj[i]) & (sess != sess[i])),
                       ("other object, same kind", lambda i: (kind == kind[i]) & (obj != obj[i])),
                       ("other kind", lambda i: kind != kind[i]),
                       ("other kind, same session", lambda i: (kind != kind[i]) & (sess == sess[i]))):  # the background and the light alone
        idx = rng.choice(len(Z), 1500, replace=False)
        partners = [rng.choice(np.nonzero(cond(i))[0]) for i in idx]
        out[name] = float(np.mean(np.sum(Z[idx] * Z[partners], axis=1)))
    return out


def show(seed, label, Z):
    o, k = elsewhere(Z, TE)
    a = alike(Z, TE, np.random.default_rng(seed + 7))
    print(f"seed {seed} {label:22}: elsewhere same object {o:.3f}, same kind {k:.3f} | alike: " + ", ".join(f"{n} {v:.3f}" for n, v in a.items()), flush=True)


for seed in SEEDS:
    cortex_seed = seed if os.path.exists(f"out/bench/core50_c50{CODE}_s{seed}_train.npy") else 0
    relay = pickle.load(open(f"out/bench/cortex_c50{CODE}_s{cortex_seed}.pkl", "rb")).relay
    if RELAY == "world":  # the relay adapted to the world it is in: settled on the training sessions' frames (never the held-out ones)
        from organic.cortex2 import Relay
        relay = Relay(relay.shrink).settle(np.load(f"out/bench/core50_c50{CODE}_s{cortex_seed}_train.npy").astype(np.float32))
    Xtr = unit((np.load(f"out/bench/core50_c50{CODE}_s{cortex_seed}_train.npy").astype(np.float32) - relay.mu) @ relay.Z)
    Xte = unit((np.load(f"out/bench/core50_c50{CODE}_s{cortex_seed}_test.npy").astype(np.float32) - relay.mu) @ relay.Z)
    del relay
    assert len(Xtr) == len(TR) and len(Xte) == len(TE), "the saved codes were made with another EVERY"
    rng = np.random.default_rng(seed + 31)
    order = np.concatenate([np.nonzero(TR[:, 4] == k)[0] for k in rng.permutation(len(train_v))])  # videos at random, frames in order
    shuffled = rng.permutation(order)
    show(seed, f"the code it reads ({Xtr.shape[1]})", Xte)
    for arm, seq, trace in (("order", order, True), ("shuffled", shuffled, True), ("plain", order, False)):
        r = np.random.default_rng(seed)
        W = unit(r.standard_normal((KINDS, Xtr.shape[1])).astype(np.float32))
        W0 = W.copy()
        tr, last_video = np.zeros(KINDS, np.float32), -1
        for n, i in enumerate(seq):
            x = Xtr[i]  # at unit length: the rule is stable for rate * |x|^2 below about 2
            if arm != "shuffled" and TR[i, 4] != last_video:  # a new video: nothing of the last one carries over
                tr[:] = 0
            last_video = TR[i, 4]
            y = W @ x
            tr = KEEP * tr + (1 - KEEP) * y if trace else y
            rate = RATE * (1 - n / len(seq)) + RATE / 10
            W += rate * np.outer(tr, x - y @ W)  # Hebb on the trace, against what the cells together already give back
        norms = np.linalg.norm(W, axis=1)
        G = unit(W) @ unit(W).T
        off = np.abs(G[~np.eye(KINDS, dtype=bool)])
        moved = float(np.mean(np.abs(np.sum(unit(W) * W0, axis=1))))
        print(f"seed {seed} {arm:8} cells: length {norms.min():.2f}-{norms.max():.2f}, overlap between cells mean {off.mean():.3f} max {off.max():.3f}, still like where they began {moved:.3f}", flush=True)
        show(seed, f"slow stage, {arm}", unit(Xte @ W.T))
