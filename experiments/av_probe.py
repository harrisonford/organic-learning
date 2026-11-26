"""Two senses teaching each other, the first probe: does a frame of a video predict its own sound? (queue E 32;
docs/VISION.md 2026-11-26 11:10, with what I said before it ran)

Material: the Commons videos kept by experiments/world/commons_video.py (64 x 64 frames, 10 a second) and their
sound through organic/cochlea.py (experiments/world/video_audio.py: 64 bands, a row every 10 ms, ten rows under each
frame). Videos without sound and titles that name an animation, a render, a time-lapse, a game or an interview are
left out. Videos are split, not frames: a fifth of them (by SEED) are never seen until the test.

Every EVERY-th frame of a video is one moment: the picture, and the second of sound around it (100 rows x 64 bands).
    sight      the photo benchmark's developed cortex (out/bench/cortex_ph_d_s{SEED}.pkl) and its own relay: 4,096
    hearing    the eye's own layer (organic/cortex2.Layer2) on the second of sound read as a picture: windows of 8 rows
               (80 ms) x 8 bands, decorrelated, 128 kinds, the response averaged over 2 halves of the second x 4
               stretches of the bands (1,024 numbers); "born" (random kinds) or "developed" (winner-take-all Hebbian
               learning with a conscience on the training videos' sound, the eye's schedule); "cochlea": the bands'
               mean in each half second (128 numbers). Each through its own relay (Relay, settled on training sound).
    teaching   one sense's code predicts the other's, learned on the training videos' moments seen once in order:
               "hebb" (the running mean of the product of the two codes: Hebb's rule; after the relays it is close
               to regression), "delta" (the word cells' delta rule and step, 1/(1 + n/5)), and "ridge" (least
               squares on everything at once, the best of three strengths on the test: an optimistic reference).
Tests, on the held-out videos only:
    which sound   the true second of sound against another video's (chance 0.5): which is nearer to what the picture
                  predicts; and the other way (which picture, from the sound)
    when          the true second against the same video's sound at least 5 s away (chance 0.5)
    what          for videos fetched under a word: the word read by the nearest class mean (balanced over words):
                  from sight, from hearing, from both, and from the sound that sight predicts

    SEED=0 python experiments/av_probe.py            -> out/bench/av_probe_s0.log
    SEED=0 MAXV=40 EVERY=40 python experiments/av_probe.py   (a smoke test)
"""
import json
import os
import pickle
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.cortex2 import Layer2, Relay  # noqa: E402

DIR = os.path.expanduser("~/datasets/commons_video")
SEED, EVERY, MAXV = int(os.environ.get("SEED", 0)), int(os.environ.get("EVERY", 20)), int(os.environ.get("MAXV", 0))
KINDS, WINDOW, SPAN, HELD, SHRINK = 128, 8, 100, 0.2, 0.0005
NOT_REAL = re.compile(r"anim|cgi|cartoon|render|\b3d\b|simulat|game|blender|clip ?art|infographic|slideshow|timelapse|"
                      r"time-lapse|stop.motion|screencast|tutorial|lecture|interview|music video|lyric|karaoke|visuali[sz]", re.I)
t0 = time.time()


def say(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


# ---- the moments ----
heard = {e["page"]: e for e in map(json.loads, open(os.path.join(DIR, "audio.jsonl"))) if e["sound"]}
videos = [v for v in map(json.loads, open(os.path.join(DIR, "videos.jsonl")))
          if v["page"] in heard and not NOT_REAL.search(v["title"]) and v["frames"] >= 30]
PAGES = "out/bench/av_pages.json"  # the videos of the first full run, so that every seed hears the same material
if not MAXV:
    if os.path.exists(PAGES):
        keep = set(json.load(open(PAGES)))
        videos = [v for v in videos if v["page"] in keep]
    else:
        json.dump([v["page"] for v in videos], open(PAGES, "w"))
rng = np.random.default_rng(SEED)
videos = [videos[i] for i in rng.permutation(len(videos))][: MAXV or None]
held = set(range(int(len(videos) * (1 - HELD)), len(videos)))
frames = np.memmap(os.path.join(DIR, "frames_64.u8"), np.uint8, "r").reshape(-1, 64, 64, 3)
sound = np.memmap(os.path.join(DIR, "audio_64.f16"), np.float16, "r").reshape(-1, 64)
moments = []  # (video index, frame index in the video, row of the frame, first row of its second of sound)
for k, v in enumerate(videos):
    a, per = heard[v["page"]], heard[v["page"]]["per_frame"]
    for i in range(EVERY // 2, v["frames"], EVERY):
        s = i * per + per // 2 - SPAN // 2
        if s >= 0 and s + SPAN <= a["rows"]:
            moments.append((k, i, v["start"] + i, a["start"] + s))
moments = np.array(moments)
test = np.isin(moments[:, 0], list(held))
say(f"seed {SEED}: {len(videos)} videos with sound ({len(held)} held out), {len(moments)} moments ({test.sum()} held out); one frame in {EVERY}")


def clip(m):
    return np.asarray(sound[m[3]: m[3] + SPAN], np.float32)[..., None]  # (100 rows, 64 bands, 1): a picture of sound


# ---- hearing: the eye's layer on sound ----
def hearing_code(layer, x):
    r = layer.respond(x, pooled=False)  # (time, bands, kinds)
    t, b = r.shape[0] // 2, r.shape[1] // 4
    return r[: 2 * t, : 4 * b].reshape(2, t, 4, b, -1).mean(axis=(1, 3)).ravel()


def cochlea_code(x):
    return x[:, :, 0].reshape(2, SPAN // 2, 64).mean(axis=1).ravel()


CODES = f"out/bench/av_codes_s{SEED}{'_' + str(MAXV) if MAXV else ''}_{EVERY}.npz"  # the coded moments, kept so the readers can run again
if os.path.exists(CODES):
    z = np.load(CODES)
    H, V = {k: z[k] for k in ("born", "developed", "cochlea")}, z["V"]
    say(f"codes read from {CODES}")
else:
    train_idx = np.nonzero(~test)[0]
    born = Layer2(1, KINDS, WINDOW, 1, seed=SEED)
    born.settle((clip(moments[i]) for i in rng.choice(train_idx, min(800, len(train_idx)), replace=False)), rng)
    developed = pickle.loads(pickle.dumps(born))
    N = int(os.environ.get("DEV", 16000))  # seconds of sound the hearing develops on (the eye's 16,000 pictures)
    for n in range(N):
        developed.learn(clip(moments[rng.choice(train_idx)]), rng, rate=0.05 * (1 - n / N) + 0.005)
    wins = developed.wins[0]
    say(f"hearing developed on {N} seconds of the training videos' sound; wins min/max {int(wins.min())}/{int(wins.max())}")

    H = {"born": np.zeros((len(moments), 8 * KINDS), np.float32), "developed": np.zeros((len(moments), 8 * KINDS), np.float32),
         "cochlea": np.zeros((len(moments), 128), np.float32)}
    for j, m in enumerate(moments):
        x = clip(m)
        H["born"][j], H["developed"][j], H["cochlea"][j] = hearing_code(born, x), hearing_code(developed, x), cochlea_code(x)
    for k in H:
        H[k] = unit(H[k])
        relay = Relay(SHRINK).settle(H[k][~test])
        for c in range(0, len(H[k]), 4096):  # in place, a block at a time (memory)
            H[k][c: c + 4096] = unit((H[k][c: c + 4096] - relay.mu) @ relay.Z)
    say("hearing coded")

    # ---- sight: the photo benchmark's developed cortex and its relay ----
    cortex = pickle.load(open(f"out/bench/cortex_{os.environ.get('CORTEX', 'ph_d')}_s{SEED}.pkl", "rb"))
    relay, cortex.relay = cortex.relay, None
    V = np.zeros((len(moments), cortex.layers[0].kinds * 16), np.float32)
    for j, m in enumerate(moments):
        V[j] = cortex.whole(np.asarray(frames[m[2]], np.float32) / 255, 4)
    for c in range(0, len(V), 4096):
        V[c: c + 4096] = unit((V[c: c + 4096] - relay.mu) @ relay.Z)
    del cortex, relay
    say(f"sight coded ({V.shape[1]} numbers)")
    np.savez(CODES, V=V, **H)

# ---- teaching, and the tests ----
rng = np.random.default_rng(SEED + 100)  # the order and the negatives, the same whether the codes were made or read
order = np.array([i for k in rng.permutation(len(videos)) if k not in held for i in np.nonzero(moments[:, 0] == k)[0]])
te = np.nonzero(test)[0]
other = np.array([rng.choice(te[moments[te, 0] != moments[i, 0]]) for i in te])
later = []
for i in te:
    same = te[(moments[te, 0] == moments[i, 0]) & (np.abs(moments[te, 1] - moments[i, 1]) >= 50)]
    later.append(rng.choice(same) if len(same) else -1)
later = np.array(later)


def readers(X, Y):
    """Maps predicting Y from X, learned on the training moments in order: hebb, delta, ridge (three strengths)."""
    YX = np.zeros((Y.shape[1], X.shape[1]), np.float32)
    G = np.zeros((X.shape[1], X.shape[1]), np.float32)
    for c in range(0, len(order), 2048):  # a block at a time, no copy of the whole training set (memory)
        Xc, Yc = X[order[c: c + 2048]], Y[order[c: c + 2048]]
        YX += Yc.T @ Xc
        G += Xc.T @ Xc
    out = {"hebb": YX / len(order)}
    W = np.zeros((Y.shape[1], X.shape[1]), np.float32)
    for n, i in enumerate(order):
        x, y = X[i], Y[i]
        W += (1.0 / (1 + n / 5)) * np.outer(y - W @ x, x)
    out["delta"] = W
    before = 0.0
    for lam in (0.01, 0.1, 1.0):
        G[np.diag_indices_from(G)] += lam - before  # G + lam I, raised in place
        before = lam
        out[f"ridge {lam:g}"] = np.linalg.solve(G, YX.T).T
    return out


def which(X, M, Y, negatives):
    """Share of held-out moments whose own target is nearer (cosine) to the prediction X M^T than the negative's
    (a block at a time: memory)."""
    ok, hits = np.nonzero(negatives >= 0)[0], 0
    for c in range(0, len(ok), 2048):
        k = ok[c: c + 2048]
        P = unit(X[te[k]] @ M.T)
        hits += int(((P * Y[te[k]]).sum(1) > (P * Y[negatives[k]]).sum(1)).sum())
    return hits / max(len(ok), 1)


results = {}
if os.environ.get("WHAT_ONLY") == "1":  # the codes are cached: only Hebb's map for the developed hearing, then the word test
    YX = sum(H["developed"][order[c: c + 2048]].T @ V[order[c: c + 2048]] for c in range(0, len(order), 2048)) / len(order)
    H["developed from sight"] = np.vstack([unit(V[c: c + 4096] @ YX.T) for c in range(0, len(V), 4096)])
for name, A in ([] if os.environ.get("WHAT_ONLY") == "1" else list(H.items())):
    for direction, (X, Y) in (("sight->sound", (V, A)), ("sound->sight", (A, V))):
        maps = readers(X, Y)
        ridge = max((k for k in maps if k.startswith("ridge")), key=lambda k: which(X, maps[k], Y, other))
        for r in ("hebb", "delta", ridge):
            results[(name, direction, r.split()[0])] = (which(X, maps[r], Y, other), which(X, maps[r], Y, later))
        if direction == "sight->sound" and name == "developed":  # only this one is read by word below
            H[name + " from sight"] = np.vstack([unit(V[c: c + 4096] @ maps["hebb"].T) for c in range(0, len(V), 4096)])
        del maps
        say(f"hearing {name:9s} {direction}: " + " | ".join(f"{r.split()[0]}{' (' + r.split()[1] + ')' if ' ' in r else ''} which {results[(name, direction, r.split()[0])][0]:.3f} when {results[(name, direction, r.split()[0])][1]:.3f}" for r in ("hebb", "delta", ridge)))

# what: the word of the videos fetched under a word
word = np.array([videos[k].get("word") or "" for k in moments[:, 0]])
words = sorted(w for w in set(word) if w and len({k for k in moments[word == w, 0] if k in held}) >= 1
               and len({k for k in moments[word == w, 0] if k not in held}) >= 3)
mask_tr, mask_te = ~test & np.isin(word, words), test & np.isin(word, words)


def what(rows):
    """Nearest class mean, balanced over words; `rows(idx)` gives the codes of those moments (a block at a time: memory)."""
    def mean(idx):
        return sum(rows(idx[c: c + 4096]).sum(0) for c in range(0, len(idx), 4096)) / len(idx)
    M = unit(np.stack([mean(np.nonzero(mask_tr & (word == w))[0]) for w in words]))
    te_idx = np.nonzero(mask_te)[0]
    guess = np.array(words)[np.concatenate([np.argmax(rows(te_idx[c: c + 4096]) @ M.T, axis=1) for c in range(0, len(te_idx), 4096)])]
    truth = word[te_idx]
    return float(np.mean([np.mean(guess[truth == w] == w) for w in words]))


arms = {"sight": lambda i: V[i], "hearing (developed)": lambda i: H["developed"][i], "hearing (born)": lambda i: H["born"][i],
        "hearing (cochlea)": lambda i: H["cochlea"][i], "sight + hearing": lambda i: np.hstack([V[i], H["developed"][i]]) / np.sqrt(2),
        "the sound sight predicts (hebb, developed)": lambda i: H["developed from sight"][i]}
if not words:
    say("what: no word has videos on both sides of the split")
    sys.exit(0)
held_videos = {w: len({k for k in moments[mask_te & (word == w), 0]}) for w in words}
say(f"what: {len(words)} words with held-out videos ({sum(held_videos.values())} videos, {mask_te.sum()} moments; chance {1 / len(words):.3f}): "
    + ", ".join(f"{w} {n}" for w, n in held_videos.items()))
for a, X in arms.items():
    say(f"what, from {a}: {what(X):.3f} (balanced over words)")
