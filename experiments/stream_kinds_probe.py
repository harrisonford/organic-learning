"""The kinds' ear and its link inside the stream brain, measured on the captions (docs/VISION.md 2026-11-28 00:38).

organic/stream.py with `give_ear`: the text-free ear on the clock (organic/hear_stream.py), each kind heard leaving
a trace that fades over TRACE seconds, tied by Hebb's rule to what the eye holds. Here the brain's own code for
that organ is driven with one reader's captions, 10 ms at a time, in the one-pass probe's order: the sight
population holds a caption's photograph (the probe's code of it) from its first sound until PAUSE seconds after
its last; kinds start empty and learn as they are heard; the hearing cortex and relay are a kept ear's. Read at
CHECKS on the 1,000 held-out pairs, nothing learning: a test caption is heard alone, its traces summed while it
lasts, `recall` gives what the eye held with those kinds, and that is set against the held-out photographs.

    EAR=out/bench/grounded_online_ear_s0_rtop2_live0.5.pkl READER=top2 MAXU=5000 python experiments/stream_kinds_probe.py
"""
import json, os, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.hear_stream import StreamEar, unit  # noqa: E402
from organic.stream import StreamBrain  # noqa: E402

DIR = os.path.expanduser("~/datasets/spokencoco")
READER, SEED, TEST, MAXU = os.environ.get("READER", "top2"), int(os.environ.get("SEED", 0)), int(os.environ.get("TEST", 1000)), int(os.environ.get("MAXU", 5000))
LIVE, TRACE, PAUSE = float(os.environ.get("LIVE", 0.5)), float(os.environ.get("TRACE", 1.0)), float(os.environ.get("PAUSE", 0.3))
EAR = os.environ.get("EAR", f"out/bench/grounded_online_ear_s{SEED}_r{READER}_live{LIVE:g}.pkl")
CHECKS = [int(x) for x in os.environ.get("CHECKS", "2500").split(",")]
t0 = time.time()
say = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)  # noqa: E731

rng = np.random.default_rng(SEED)
utts = [u for u in map(json.loads, open(os.path.join(DIR, "cochlea.jsonl"))) if u["split"] == "train" and u["rows"] >= 50]
count = {}
for u in utts:
    count[u["speaker"]] = count.get(u["speaker"], 0) + 1
grams = np.memmap(os.path.join(DIR, "cochlea_64.f16"), np.float16, "r").reshape(-1, 64)
all_utts = utts


def people(reader, n, taken=()):
    """One reader's held-out captions and life, a caption a photograph; no photograph in `taken`."""
    who = sorted(count, key=count.get, reverse=True)[int(reader[3:] or 1) - 1] if reader.startswith("top") else reader
    by_image = {}
    for u in all_utts:
        if u["speaker"] == who and u["image"] not in taken:
            by_image.setdefault(u["image"], []).append(u)
    images = sorted(by_image)
    images = [images[i] for i in rng.permutation(len(images))]
    test = [by_image[i][int(rng.integers(len(by_image[i])))] for i in images[:TEST]]
    return test, [by_image[i][int(rng.integers(len(by_image[i])))] for i in images[TEST:]][:n]


test_u, life_u = people(READER, MAXU)
# (2026-11-28 04:34) THEN=<reader>: after the first reader's life a second voice, THEN_N captions with their photographs
# (none of them seen with the first voice); both readers' held-out pairs are read at the change and at the end.
# FIRST_HEARD=0: the first life is not heard at all (the second voice alone, on the same captions and tests).
THEN, THEN_N, FIRST_HEARD = os.environ.get("THEN", ""), int(os.environ.get("THEN_N", 2500)), os.environ.get("FIRST_HEARD", "1") != "0"
if THEN:
    test2_u, life2_u = people(THEN, THEN_N, {u["image"] for u in test_u + life_u})
raw = lambda u: np.asarray(grams[u["start"]: u["start"] + u["rows"]], np.float32)  # noqa: E731

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
if THEN:
    V_life2, V_test2 = see(life2_u), see(test2_u)
cortex.relay = vrelay
say(f"reader {READER}: {len(life_u)} captions with their photographs, {len(test_u)} held out; ear fading over {LIVE:g} s, traces over {TRACE:g} s, {PAUSE:g} s after each caption")

kept = pickle.load(open(EAR, "rb"))
brain = StreamBrain(cortex, None, seed=SEED)
# (2026-11-28 03:36) RELAY=learns: the kept relay, a circuit, goes on learning on the clock (hear_stream.py
# `relay_learns`); RELAY=born: a circuit born random with this brain, nothing of it kept from before
RELAY = os.environ.get("RELAY", "")
if RELAY == "born":
    from organic.cortex2 import RelayCircuit  # noqa: E402
    kept["relay"] = RelayCircuit(kept["layer"].K * 8 if hasattr(kept["layer"], "K") else 1024, rate=1e-4, block=50, seed=SEED)
# (2026-11-28 07:40) LAYER=born: the hearing cortex too is born with this brain and learns on the clock (hear_stream.py
# HearingLayer, `layer_learns`); with RELAY=born nothing of the ear is kept from before (VISION.md 07:40)
LAYER = os.environ.get("LAYER", "")
if LAYER == "born":
    from organic.hear_stream import HearingLayer  # noqa: E402
    kept["layer"] = HearingLayer(seed=SEED)
brain.give_ear(StreamEar(kept["layer"], kept["relay"], tau=LIVE), kind_tau=TRACE)
brain.kinds_ear.relay_learns, brain.kinds_ear.layer_learns = bool(RELAY), LAYER == "born"
if RELAY:
    say(f"the relay is a circuit that learns along the life ({RELAY})")
if LAYER:
    say("the hearing cortex is born with the brain and learns along the life")
brain.frame = np.zeros((2, 2, 3), np.float32)  # something is in view: the sight population is set by hand below
ear, quiet_steps = brain.kinds_ear, int(round(PAUSE * 100))
rec = lambda r: " ".join(f"R@{k} {np.mean(r < k):.3f}" for k in (1, 5, 10))  # noqa: E731


def hear(u, v):
    """A caption with its photograph in view; then the pause, the photograph still there."""
    brain.sight.x = v
    events = []
    for row in raw(u):
        brain._hear_kinds(row, True, events)
    for what, k in ear.pause():
        brain.kind_trace[k + (ear.k23 if what == "triple" else 0)] += 1.0
        events.append(1)
    for _ in range(quiet_steps):
        brain._hear_kinds(None, False, events)
    return len(events)


def read(name, test_u=None, V_test=None):
    test_u, V_test = (globals()["test_u"], globals()["V_test"]) if test_u is None else (test_u, V_test)
    held = (ear.mu.copy(), ear.var.copy(), ear.top, brain.kind_trace.copy())
    ear.learning = brain.kind_learn = False
    R, none = np.zeros((len(test_u), V_test.shape[1]), np.float32), 0
    used = np.zeros(len(brain.kind_trace))  # how often each kind is heard in these held-out captions
    for n, u in enumerate(test_u):
        ear.mu, ear.var, ear.top = held[0].copy(), held[1].copy(), held[2]
        brain.kind_trace[:] = 0
        total, events = np.zeros_like(brain.kind_trace), []
        for row in raw(u):
            brain._hear_kinds(row, True, events)
            total += brain.kind_trace
        for what, k in ear.pause():
            brain.kind_trace[k + (ear.k23 if what == "triple" else 0)] += 1.0
            events.append(("kind", k + (ear.k23 if what == "triple" else 0)))
        for _, i in events:
            used[i] += 1
        total += brain.kind_trace * quiet_steps / 2
        r = brain.recall(total)
        if r is None:
            none += 1
        else:
            R[n] = r
    ear.mu, ear.var, ear.top = held[0], held[1], held[2]
    brain.kind_trace[:] = held[3]
    ear.learning = brain.kind_learn = True
    S = unit(R) @ unit(V_test - (brain.sight_sum / brain.sight_time).astype(np.float32)).T
    r_cap, r_img = (S > np.diag(S)[:, None]).sum(1), (S > np.diag(S)[None, :]).sum(0)
    say(f"{name}: photograph -> its caption among {len(test_u)}: {rec(r_img)}; caption -> its photograph: {rec(r_cap)}; median rank {np.median(r_img) + 1:.0f} / {np.median(r_cap) + 1:.0f}"
        f" (captions that called up nothing: {none}; kinds founded {ear.pairs.founded} + {ear.triples.founded})")
    return used


kinds = 0
for n, u in enumerate(life_u if FIRST_HEARD else [], 1):
    kinds += hear(u, V_life[n - 1])
    if n in CHECKS or n == len(life_u):
        read(f"after {n} captions ({kinds / n:.1f} kinds a caption)")
if THEN:
    if FIRST_HEARD:
        read(f"the second voice ({THEN}) before any of it is heard", test2_u, V_test2)
        # (2026-11-28 05:28) a measure only: where the kinds are at the change, to see what the second voice does to them
        U0 = (ear.pairs.U.copy(), ear.triples.U.copy())
        wins0 = np.concatenate([ear.pairs.wins, ear.triples.wins]).copy()
        W0 = ear.layer.W[0].copy() if LAYER == "born" else None  # (2026-11-28 08:58) and where the hearing cortex's kinds are
    kinds = 0
    for n, u in enumerate(life2_u, 1):
        kinds += hear(u, V_life2[n - 1])
        if n in CHECKS or n == len(life2_u):
            read(f"after {n} captions of the second voice ({kinds / n:.1f} kinds a caption), the second voice", test2_u, V_test2)
            if FIRST_HEARD:
                u1 = read(f"after {n} captions of the second voice, the first voice ({READER})")
                if n == len(life2_u):
                    u2 = read("the second voice once more, for its kinds", test2_u, V_test2)
                    wins = np.concatenate([ear.pairs.wins, ear.triples.wins])
                    share = (wins - wins0) / np.maximum(wins, 1)  # of a kind's wins in the whole life, those won in the second voice
                    moved = np.concatenate([(a * b).sum(1) for a, b in zip(U0, (ear.pairs.U, ear.triples.U))])
                    p1, p2 = u1 / u1.sum(), u2 / u2.sum()
                    say(f"the kinds: of a kind's wins, the share won in the second voice: quartiles {np.percentile(share, 25):.2f} / {np.median(share):.2f} / {np.percentile(share, 75):.2f}; "
                        f"kinds won nine tenths in one voice: {np.mean(share < 0.1):.2f} the first, {np.mean(share > 0.9):.2f} the second; "
                        f"a kind against itself at the change (cosine): quartiles {np.percentile(moved, 25):.2f} / {np.median(moved):.2f} / {np.percentile(moved, 75):.2f}; "
                        f"held-out use of the kinds by the two voices: overlap {np.minimum(p1, p2).sum():.2f} (1 the same kinds as often, 0 no kind shared), "
                        f"kinds carrying half of each voice's use {int(np.searchsorted(np.cumsum(np.sort(p1)[::-1]), 0.5)) + 1} / {int(np.searchsorted(np.cumsum(np.sort(p2)[::-1]), 0.5)) + 1} of {len(p1)}")
                    if W0 is not None:
                        cw = np.sort((W0 * ear.layer.W[0]).sum(1))
                        say(f"the hearing cortex's kinds against themselves at the change (cosine): lowest tenth {cw[len(cw) // 10]:.2f}, median {np.median(cw):.2f}")
