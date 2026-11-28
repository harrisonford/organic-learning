"""Is the ear on the clock (organic/hear_stream.py) the probe's ear? (2026-11-27 22:30)

One reader's first N captions, in order, heard twice with the same hearing cortex and relay (a kept one-pass ear's)
and kinds that start empty and learn: once as experiments/grounded_online_probe.py hears them with LIVE (each
caption whole, its bands against the sound so far), once row by row through StreamEar with a pause after each
caption. The same ear gives the same stretches, the same kinds and the same bags.

    READER=top2 N=40 python experiments/hear_stream_check.py
"""
import json, os, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.hear_stream import Kinds, StreamEar, unit  # noqa: E402
from scipy.signal import lfilter  # noqa: E402

DIR = os.path.expanduser("~/datasets/spokencoco")
READER, N, TAU = os.environ.get("READER", "top2"), int(os.environ.get("N", 40)), float(os.environ.get("TAU", 2))
KINDS, WINDOW, BANDS, HALF, SMOOTH, K23 = 128, 8, 8, 5, 5, 1024
ear = pickle.load(open(os.environ.get("EAR", f"out/bench/grounded_online_ear_s0_r{READER}.pkl"), "rb"))
layer, relay = ear["layer"], ear["relay"]
utts = [u for u in map(json.loads, open(os.path.join(DIR, "cochlea.jsonl"))) if u["split"] == "train" and u["rows"] >= 50]
count = {}
for u in utts:
    count[u["speaker"]] = count.get(u["speaker"], 0) + 1
who = sorted(count, key=count.get, reverse=True)[int(READER[3:] or 1) - 1]
utts = [u for u in utts if u["speaker"] == who][:N]
grams = np.memmap(os.path.join(DIR, "cochlea_64.f16"), np.float16, "r").reshape(-1, 64)
raw = lambda u: np.asarray(grams[u["start"]: u["start"] + u["rows"]], np.float32)  # noqa: E731
k = 0.01 / TAU

# --- as the probe hears (LIVE): a caption whole
pairs, triples = Kinds(K23, 4 * KINDS * BANDS), Kinds(K23, 6 * KINDS * BANDS)
state, whole, n_whole, t0 = None, [], 0, time.time()
for u in utts:
    g = raw(u)
    mu0, var0, top = (g[0], np.ones(64, np.float32), 0.0) if state is None else state
    mu = lfilter([k], [1, k - 1], g, axis=0, zi=((1 - k) * mu0)[None])[0]
    var = lfilter([k], [1, k - 1], (g - mu) ** 2, axis=0, zi=((1 - k) * var0)[None])[0]
    z = ((g - mu) / (np.sqrt(var) + 1e-3)).astype(np.float32)
    r = layer.respond(z[..., None], pooled=False)
    b = r.shape[1] // BANDS
    c = r[:, : BANDS * b].reshape(r.shape[0], BANDS, b, KINDS).mean(axis=2).reshape(r.shape[0], -1)
    loud = np.convolve(g.sum(axis=1), np.ones(WINDOW) / WINDOW, "valid")[: len(c)]
    C = unit((unit(c) - relay.mu) @ relay.Z)
    env = np.convolve(loud, np.ones(SMOOTH) / SMOOTH, "same")
    ref = np.empty_like(env)
    for t, e in enumerate(env):
        top = max(float(e), top * (1 - k))
        ref[t] = top
    state = (mu[-1].astype(np.float32), var[-1].astype(np.float32), top)
    quiet = env < 0.1 * ref
    troughs = [t for t in range(1, len(env) - 1) if env[t] == env[max(0, t - HALF): t + HALF + 1].min()]
    cuts = sorted(set(troughs) | {t for t in range(1, len(env)) if quiet[t] != quiet[t - 1]} | {0, len(env)})
    out = []
    for a, b2 in zip(cuts[:-1], cuts[1:]):
        if b2 - a < 5 or quiet[a:b2].mean() > 0.5:
            if out:
                out[-1][1] = True
            continue
        h = (a + b2) // 2
        out.append([np.concatenate([unit(C[a:h].mean(0)), unit(C[h:b2].mean(0))]) / np.sqrt(2), False])
    v, gap = [s for s, _ in out], [x for _, x in out]
    bag = np.zeros(2 * K23, np.float32)
    for i in range(len(v) - 1):
        if not gap[i]:
            bag[pairs(np.concatenate(v[i: i + 2]) / np.sqrt(2), True)] += 1
    for i in range(len(v) - 2):
        if not (gap[i] or gap[i + 1]):
            bag[K23 + triples(np.concatenate(v[i: i + 3]) / np.sqrt(3), True)] += 1
    whole.append(bag)
    n_whole += len(v)
t_whole = time.time() - t0

# --- on the clock: a row at a time, a pause after each caption
live, stream, rows, t0 = StreamEar(layer, relay, tau=TAU), [], 0, time.time()
for u in utts:
    bag, events = np.zeros(2 * K23, np.float32), []
    for row in raw(u):
        events += live.step(row)
        rows += 1
    for kind, i in events + live.pause():
        bag[i + (K23 if kind == "triple" else 0)] += 1
    stream.append(bag)
t_stream = time.time() - t0
whole, stream = np.array(whole), np.array(stream)
same = [(a == b).all() for a, b in zip(whole, stream)]
cos = [float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)) for a, b in zip(whole, stream)]
print(f"reader {READER}, {len(utts)} captions, {rows / 100:.0f} s of sound: kinds heard a caption {whole.sum(1).mean():.1f} whole / {stream.sum(1).mean():.1f} on the clock; "
      f"captions with the very same bag {int(np.sum(same))} of {len(same)}; likeness of the two bags, median {np.median(cos):.3f}, lowest {np.min(cos):.3f}; "
      f"kinds founded {pairs.founded}+{triples.founded} / {live.pairs.founded}+{live.triples.founded}; the kinds themselves differ by at most {np.abs(pairs.U - live.pairs.U).max():.2g}; "
      f"band state differs by {np.abs(state[0] - live.mu).max():.2g}; loudest so far {state[2]:.4f} / {live.top:.4f}; "
      f"time: whole {t_whole:.1f} s, on the clock {t_stream:.1f} s ({1000 * t_stream / rows:.2f} ms a 10 ms row)")
