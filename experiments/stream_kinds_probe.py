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
who = sorted(count, key=count.get, reverse=True)[int(READER[3:] or 1) - 1] if READER.startswith("top") else READER
utts = [u for u in utts if u["speaker"] == who]
grams = np.memmap(os.path.join(DIR, "cochlea_64.f16"), np.float16, "r").reshape(-1, 64)
by_image = {}
for u in utts:
    by_image.setdefault(u["image"], []).append(u)
images = sorted(by_image)
images = [images[i] for i in rng.permutation(len(images))]
test_u = [by_image[i][int(rng.integers(len(by_image[i])))] for i in images[:TEST]]
life_u = [by_image[i][int(rng.integers(len(by_image[i])))] for i in images[TEST:]][:MAXU]
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
cortex.relay = vrelay
say(f"reader {READER}: {len(life_u)} captions with their photographs, {len(test_u)} held out; ear fading over {LIVE:g} s, traces over {TRACE:g} s, {PAUSE:g} s after each caption")

kept = pickle.load(open(EAR, "rb"))
brain = StreamBrain(cortex, None, seed=SEED)
brain.give_ear(StreamEar(kept["layer"], kept["relay"], tau=LIVE), kind_tau=TRACE)
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


def read(name):
    held = (ear.mu.copy(), ear.var.copy(), ear.top, brain.kind_trace.copy())
    ear.learning = brain.kind_learn = False
    R, none = np.zeros((len(test_u), V_test.shape[1]), np.float32), 0
    for n, u in enumerate(test_u):
        ear.mu, ear.var, ear.top = held[0].copy(), held[1].copy(), held[2]
        brain.kind_trace[:] = 0
        total, events = np.zeros_like(brain.kind_trace), []
        for row in raw(u):
            brain._hear_kinds(row, True, events)
            total += brain.kind_trace
        for what, k in ear.pause():
            brain.kind_trace[k + (ear.k23 if what == "triple" else 0)] += 1.0
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


kinds = 0
for n, u in enumerate(life_u, 1):
    kinds += hear(u, V_life[n - 1])
    if n in CHECKS or n == len(life_u):
        read(f"after {n} captions ({kinds / n:.1f} kinds a caption)")
