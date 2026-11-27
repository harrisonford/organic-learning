"""Can the ear's first steps be done as the sound comes in? (2026-11-27; a measure before a run, seconds, no cortex.)

The text-free ear (experiments/grounded_speech_probe.py) knows each caption whole before it hears it: every band
is set against the caption's own mean and spread ("adapted to the voice"), and a stretch counts as quiet under a
tenth of the caption's loudest moment. A live ear has only the past. Here one reader's captions are played one after
another as one stream and the same two steps are done with what was heard so far: a running mean and spread a band
and a running loudest moment, each fading with one time constant TAU (seconds). Measured against the whole-caption
ear, over the captions after the first WARM:

  * the bands as the cortex gets them: correlation of the two versions, a caption;
  * the cuts: how many of the whole-caption ear's cuts have a cut of the live ear within 30 ms, and the reverse;
  * the stretches kept (not quiet, at least 50 ms): how many, and how many are the same stretch (both ends within
    30 ms).

Usage: READER=top N=300 TAUS=0.5,2,8,32 .venv/bin/python experiments/ear_causal_check.py
"""
import json, os
import numpy as np

DIR = os.path.expanduser("~/datasets/spokencoco")
READER, N, WARM, SEED = os.environ.get("READER", "top"), int(os.environ.get("N", 300)), int(os.environ.get("WARM", 5)), int(os.environ.get("SEED", 0))
TAUS = [float(x) for x in os.environ.get("TAUS", "0.5,2,8,32").split(",")]
WINDOW, HALF, SMOOTH, DT = 8, 5, 5, 0.01

rng = np.random.default_rng(SEED)
utts = [u for u in map(json.loads, open(os.path.join(DIR, "cochlea.jsonl"))) if u["split"] == "train" and u["rows"] >= 50]
count = {}
for u in utts:
    count[u["speaker"]] = count.get(u["speaker"], 0) + 1
who = sorted(count, key=count.get, reverse=True)[int(READER[3:] or 1) - 1] if READER.startswith("top") else READER
utts = [u for u in utts if u["speaker"] == who]
utts = [utts[i] for i in rng.permutation(len(utts))[:N]]
grams = np.memmap(os.path.join(DIR, "cochlea_64.f16"), np.float16, "r").reshape(-1, 64)
raws = [np.asarray(grams[u["start"]: u["start"] + u["rows"]], np.float32) for u in utts]
print(f"reader {READER}: {len(raws)} captions as one stream, {sum(map(len, raws)) * DT / 60:.1f} minutes; measured after the first {WARM}")


def cuts_of(loud, quiet_under):
    """The probe's cuts: troughs of the smoothed loudness and the edges of quiet. quiet_under: a number or one a frame."""
    env = np.convolve(loud, np.ones(SMOOTH) / SMOOTH, "same")
    quiet = env < quiet_under
    troughs = [t for t in range(1, len(env) - 1) if env[t] == env[max(0, t - HALF): t + HALF + 1].min()]
    cuts = sorted(set(troughs) | {t for t in range(1, len(env)) if quiet[t] != quiet[t - 1]} | {0, len(env)})
    kept = [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if not (b - a < 5 or quiet[a:b].mean() > 0.5)]
    return cuts, kept


def loudness(g):
    return np.convolve(g.sum(axis=1), np.ones(WINDOW) / WINDOW, "valid")


whole = []
for g in raws:
    loud = loudness(g)
    env = np.convolve(loud, np.ones(SMOOTH) / SMOOTH, "same")
    whole.append(((g - g.mean(0)) / (g.std(0) + 1e-3),) + cuts_of(loud, 0.1 * env.max()))


def near(a, b, within=3):
    b = np.asarray(sorted(b))
    return float(np.mean([np.abs(b - x).min() <= within for x in a])) if len(a) and len(b) else 0.0


for tau in TAUS:
    k = DT / tau
    mu, var, top = None, None, 0.0
    corr, found, extra, n_whole, n_live, same = [], [], [], 0, 0, 0
    for i, g in enumerate(raws):
        live = np.empty_like(g)
        for t, row in enumerate(g):  # each band against what was heard so far, fading over tau
            if mu is None:
                mu, var = row.copy(), np.ones_like(row)
            mu += k * (row - mu)
            var += k * ((row - mu) ** 2 - var)
            live[t] = (row - mu) / (np.sqrt(var) + 1e-3)
        loud = loudness(g)
        env = np.convolve(loud, np.ones(SMOOTH) / SMOOTH, "same")
        ref = np.empty_like(env)
        for t, e in enumerate(env):  # the loudest moment heard, fading over tau
            top = max(e, top * (1 - k))
            ref[t] = top
        cuts, kept = cuts_of(loud, 0.1 * ref)
        if i < WARM:
            continue
        gw, cw, kw = whole[i]
        corr.append(np.corrcoef(gw.ravel(), live.ravel())[0, 1])
        found.append(near(cw[1:-1], cuts[1:-1]))
        extra.append(near(cuts[1:-1], cw[1:-1]))
        n_whole, n_live = n_whole + len(kw), n_live + len(kept)
        same += sum(any(abs(a - a2) <= 3 and abs(b - b2) <= 3 for a2, b2 in kept) for a, b in kw)
    print(f"fading over {tau:g} s: the bands as the cortex gets them, correlation with the whole-caption ear {np.median(corr):.3f} "
          f"(a tenth of the captions under {np.quantile(corr, 0.1):.3f}); its cuts found within 30 ms {np.mean(found):.3f}, "
          f"the live ear's cuts that are its cuts {np.mean(extra):.3f}; stretches kept {n_whole} / {n_live}, the same stretch {same / n_whole:.3f}")
