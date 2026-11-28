"""The link in a stream: a trace of the kinds heard, and whatever is in view (docs/VISION.md 2026-11-27 22:28).

The one-pass ear ties a caption's bag of kinds to its photograph. A stream has neither: kinds are heard one at a
time, each leaves a trace that fades over TAU seconds, and Hebb's rule ties the traces to what is in view for as
long as both last. Here a kept one-pass ear (cortex, relay and kinds as its life ended, held still) hears its
reader's captions again on the clock (organic/hear_stream.py), which gives every kind and the moment it was heard;
then the life is laid out as one stream, caption after caption with GAP seconds between, each photograph in view
from its caption's first sound to the next caption's, and the link is what Hebb's rule would have summed:

    for each photograph, X = the time each kind's trace spent in view with it (its own caption's kinds, and what
    was still fading from the captions before), and M += x (X) V

with x the probe's (X less its running mean, at unit length) or X itself. Read on the held-out pairs as the
probe reads (a test caption heard alone, its traces summed while it lasts), beside the caption-bag link.

    EAR=out/bench/grounded_online_ear_s0_rtop2_live2.pkl READER=top2 python experiments/grounded_link_check.py
"""
import json, os, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.hear_stream import StreamEar, unit  # noqa: E402

DIR = os.path.expanduser("~/datasets/spokencoco")
READER, SEED, TEST, FIRST, MAXU = os.environ.get("READER", "top2"), int(os.environ.get("SEED", 0)), int(os.environ.get("TEST", 1000)), int(os.environ.get("FIRST", 1000)), int(os.environ.get("MAXU", 0))
EAR = os.environ.get("EAR", f"out/bench/grounded_online_ear_s{SEED}_r{READER}_live2.pkl")
TAUS = [float(x) for x in os.environ.get("TAUS", "0.5,1,2,4,8").split(",")]
GAPS = [float(x) for x in os.environ.get("GAPS", "0,1").split(",")]
K23, t0 = 1024, time.time()
say = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)  # noqa: E731

rng = np.random.default_rng(SEED)
utts = [u for u in map(json.loads, open(os.path.join(DIR, "cochlea.jsonl"))) if u["split"] == "train" and u["rows"] >= 50]
count = {}
for u in utts:
    count[u["speaker"]] = count.get(u["speaker"], 0) + 1
who = sorted(count, key=count.get, reverse=True)[int(READER[3:] or 1) - 1] if READER.startswith("top") else READER
utts = [u for u in utts if u["speaker"] == who]
grams = np.memmap(os.path.join(DIR, "cochlea_64.f16"), np.float16, "r").reshape(-1, 64)
by_image = {}
for u in utts:
    by_image.setdefault(u["image"], []).append(u)
images = sorted(by_image)
images = [images[i] for i in rng.permutation(len(images))]
test_u = [by_image[i][int(rng.integers(len(by_image[i])))] for i in images[:TEST]]
life_u = [by_image[i][int(rng.integers(len(by_image[i])))] for i in images[TEST:]][: MAXU or None]
CACHE = EAR.replace(".pkl", f"_heard{'_' + str(MAXU) if MAXU else ''}.npz")

if os.path.exists(CACHE):
    z = np.load(CACHE)
    when, kind, start, rows, V_life, V_test = z["when"], z["kind"], z["start"], z["rows"], z["V_life"], z["V_test"]
else:
    kept = pickle.load(open(EAR, "rb"))
    ear = StreamEar(kept["layer"], kept["relay"], tau=2.0, learning=False)
    for organ, name in ((ear.pairs, "pairs"), (ear.triples, "triples")):
        organ.U, organ.wins, organ.founded = kept[name]
    when, kind, start, rows, held = [], [], [0], [], None
    for n, u in enumerate(life_u + test_u):
        if n == min(FIRST, len(life_u)):
            held = (ear.mu.copy(), ear.var.copy(), ear.top)  # a test caption is heard by an ear that has just heard the first captions
        if n >= len(life_u):
            ear.mu, ear.var, ear.top = held[0].copy(), held[1].copy(), held[2]
        g = np.asarray(grams[u["start"]: u["start"] + u["rows"]], np.float32)
        for t, row in enumerate(g):
            for what, k in ear.step(row):
                when.append(t)
                kind.append(k + (K23 if what == "triple" else 0))
        for what, k in ear.pause():
            when.append(len(g))
            kind.append(k + (K23 if what == "triple" else 0))
        start.append(len(when))
        rows.append(len(g))
        if n % 2000 == 0:
            say(f"heard {n} captions")
    when, kind, start, rows = np.array(when, np.int32), np.array(kind, np.int32), np.array(start, np.int64), np.array(rows, np.int32)
    cortex = pickle.load(open(f"out/bench/cortex_ph_d_s{SEED}.pkl", "rb"))
    vrelay, cortex.relay = cortex.relay, None
    row_of = {i: r for r, i in enumerate(json.load(open(os.path.expanduser("~/datasets/coco_thumbs/train2017_ids.json"))))}
    thumbs = np.load(os.path.expanduser("~/datasets/coco_thumbs/train2017_64.npy"), mmap_mode="r")

    def see(us):
        out = np.empty((len(us), vrelay.Z.shape[1]), np.float32)
        for a in range(0, len(us), 500):
            X = np.stack([cortex.whole(np.asarray(thumbs[row_of[u["image"]]], np.float32) / 255, 4) for u in us[a: a + 500]])
            out[a: a + 500] = unit((X - vrelay.mu) @ vrelay.Z)
        return out

    V_life, V_test = see(life_u), see(test_u)
    del cortex
    np.savez(CACHE, when=when, kind=kind, start=start, rows=rows, V_life=V_life, V_test=V_test)
    say("heard and seen; kept in", CACHE)

L, T = len(V_life), len(V_test)
rec = lambda r: " ".join(f"R@{k} {np.mean(r < k):.3f}" for k in (1, 5, 10))  # noqa: E731


def read(X, Xt, x_of, name):
    """The link from the life's rows X, read on the test rows Xt; x_of: how a row enters Hebb's rule."""
    mean, A = np.zeros(X.shape[1], np.float64), np.empty_like(X)
    for n in range(len(X)):
        mean += (X[n] - mean) / (n + 1)
        A[n] = x_of(X[n], mean)
    M = A.T @ V_life
    S = unit(unit(Xt - mean.astype(np.float32)) @ M) @ V_test.T
    r_img, r_cap = (S > np.diag(S)[None, :]).sum(0), (S > np.diag(S)[:, None]).sum(1)
    say(f"{name}: photograph -> its caption among {T}: {rec(r_img)}; caption -> its photograph: {rec(r_cap)}; median rank {np.median(r_img) + 1:.0f} / {np.median(r_cap) + 1:.0f}")


normed = lambda x, mean: unit(x - mean.astype(np.float32))  # noqa: E731
plain = lambda x, mean: x  # noqa: E731


def traces(n0, n1, tau, gap, alone):
    """For captions n0..n1: the time (in units of tau) each kind's trace spends in view with each caption's photograph."""
    X, left = np.zeros((n1 - n0, 2 * K23), np.float32), np.zeros(2 * K23, np.float64)
    for i, n in enumerate(range(n0, n1)):
        dur = rows[n] / 100 + gap  # the photograph is in view from the caption's first sound to the next caption's
        e = slice(start[n], start[n + 1])
        rest = np.exp(-(dur - when[e] / 100) / tau)  # what is left of each kind's trace when the photograph goes
        own, spill = np.zeros(2 * K23), np.zeros(2 * K23)
        np.add.at(own, kind[e], 1 - rest)
        np.add.at(spill, kind[e], rest)
        X[i] = own + (0 if alone else left * (1 - np.exp(-dur / tau)))
        left = left * np.exp(-dur / tau) + spill
    return X


bags = np.zeros((L + T, 2 * K23), np.float32)
for n in range(L + T):
    np.add.at(bags[n], kind[start[n]: start[n + 1]], 1)
say(f"reader {READER}: {L} captions of the life and {T} held out, {bags[:L].sum(1).mean():.1f} kinds a caption, {rows[:L].mean() / 100:.1f} s a caption")
read(bags[:L], bags[L:], normed, "a caption's bag, as the probe ties it (kinds held as they ended)")
read(bags[:L], bags[L:], plain, "a caption's bag, as it is (no mean taken, no length set)")
for gap in GAPS:
    for tau in TAUS:
        X, Xt = traces(0, L, tau, gap, False), traces(L, L + T, tau, gap, True)
        read(X, Xt, normed, f"traces fading over {tau:g} s, {gap:g} s between captions, set as the probe sets a bag")
        read(X, Xt, plain, f"traces fading over {tau:g} s, {gap:g} s between captions, as they are")
