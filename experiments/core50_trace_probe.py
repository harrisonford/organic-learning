"""Real time for the eye: a second stage that learns from video in its order, against the same frames shuffled
(queue E 31 (b); docs/VISION.md 2026-11-26 13:40, with what I said before it ran).

Every temporal attempt so far learned from views made out of a still (a picture sliding a pixel, a zoom): such
sequences can teach tolerance to position and size, never to an object turning, the hand moving it, the light.
CORe50 (Lomonaco & Maltoni 2017; CC BY 4.0) films 50 handheld objects of 10 kinds, each for about 15 s at 20 frames
a second, in 11 sessions with different backgrounds and light (experiments/world/core50_frames.py: 64 x 64, in order).

The first stage is the photo benchmark's developed cortex of the seed (out/bench/cortex_ph_d_s{SEED}.pkl), as it is.
A second stage (256 kinds in 16 groups, each listening to 32 kinds of the first over 3 x 3 places: as in
experiments/cortex_trace_probe.py) develops on FIX fixations of VIEWS frames, STEP frames apart (100 ms at STEP 2),
from the training sessions, by the first stage's own rule, in three arms that see exactly the same frames:
    order      the frames of a fixation follow each other in the video; a cell that won a frame before at a place
               still carries a trace of it (0.7 a frame) and learns the present frame in proportion (Földiák 1991)
    shuffled   the same frames, the same trace, but the order of all frames shuffled: time is the only difference
    plain      the frames in order, no trace: only the winner of each frame learns it
Read on the sessions held out (3, 7 and 10, the dataset's own test sessions: other backgrounds, other light):
    same thing elsewhere   for every held-out frame, the nearest held-out frame of ANOTHER session: is it the same
                           object (chance 1/50) and the same kind (chance 1/10); through each code's relay
    the stream             codes kept for experiments/bench_stream.py (SET=core50, VARIANT=c50{first,order,shuffled,
                           plain}): every tenth frame of the training sessions in a stream of 10 kinds, every tenth
                           of the held-out sessions as the test
Each relay is settled on 6,000 COCO thumbnails (the photo benchmark's unlabelled rows), never on CORe50.

    SEEDS=0 python experiments/core50_trace_probe.py
    SEEDS=0 FIX=40 EVERY=60 python experiments/core50_trace_probe.py      (a smoke test; writes c50*_s0 files too)
"""
import copy, json, os, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.cortex2 import Layer2, Relay  # noqa: E402

DIR = os.path.expanduser("~/datasets/core50")
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
ARMS = os.environ.get("ARMS", "order,shuffled,plain").split(",")
FIX, VIEWS, STEP, KEEP = int(os.environ.get("FIX", 4000)), int(os.environ.get("VIEWS", 4)), int(os.environ.get("STEP", 2)), 0.7
EVERY, NRELAY, GRID, SHRINK = int(os.environ.get("EVERY", 10)), int(os.environ.get("NRELAY", 6000)), 4, 0.0005
TEST_SESSIONS = {3, 7, 10}
WORDS = ["adapter", "phone", "scissors", "bulb", "can", "glasses", "ball", "marker", "cup", "remote"]  # objects 1-5, 6-10, ...

frames = np.memmap(os.path.join(DIR, "frames_64.u8"), np.uint8, "r").reshape(-1, 64, 64, 3)
videos = [json.loads(line) for line in open(os.path.join(DIR, "frames.jsonl"))]
train_v = [v for v in videos if v["session"] not in TEST_SESSIONS]
test_v = [v for v in videos if v["session"] in TEST_SESSIONS]


def rows(vs):
    """Every EVERY-th frame of the videos: (frame rows, object, kind, session)."""
    out = [(v["start"] + i, v["object"], (v["object"] - 1) // 5, v["session"]) for v in vs for i in range(0, v["frames"], EVERY)]
    return np.array(out)


TR, TE = rows(train_v), rows(test_v)
pic = lambda r: np.asarray(frames[r], np.float32) / 255  # noqa: E731
thumbs = np.load(os.path.expanduser("~/datasets/coco_thumbs/train2017_64.npy"), mmap_mode="r")
dev = np.load("out/bench/photos_dev_rows.npy")


def learn(layer, xs, rng, trace, samples=32, rate=0.02):
    """experiments/cortex_trace_probe.py's rule: at places sampled once for the fixation, the winner of each frame
    moves towards it; with `trace`, cells that won an earlier frame at the place move too, as far as their trace lasts."""
    seen = [layer._drive(x)[2] for x in xs]
    for g in range(layer.groups):
        e = seen[0][g][1].ravel()
        strong = np.nonzero(e > 0.3 * e.max())[0] if e.max() > 1e-6 else []
        if not len(strong):
            continue
        W, wins = layer.W[g], layer.wins[g]
        V = [s[g][0].reshape(-1, s[g][0].shape[2]) for s in seen]
        for i in rng.choice(strong, min(samples, len(strong)), replace=False):
            tr = np.zeros(len(W), np.float32)
            for v in V:
                c = int(np.argmax(W @ v[i] - 0.3 * (wins / (wins.mean() + 1e-8) - 1)))
                wins[c] += 1
                tr = KEEP * tr if trace else 0 * tr
                tr[c] = 1.0
                for k in np.nonzero(tr > 0.1)[0]:
                    w = W[k] + rate * tr[k] * (v[i] - W[k])
                    W[k] = w / np.linalg.norm(w)


def elsewhere(Z, meta):
    """For each held-out frame, its nearest held-out frame from another session: same object, same kind."""
    obj, kind, sess = meta[:, 1], meta[:, 2], meta[:, 3]
    same_o = same_k = 0
    for c in range(0, len(Z), 1024):
        S = Z[c: c + 1024] @ Z.T
        S[sess[c: c + 1024, None] == sess[None, :]] = -np.inf
        j = np.argmax(S, axis=1)
        same_o += int((obj[j] == obj[c: c + 1024]).sum())
        same_k += int((kind[j] == kind[c: c + 1024]).sum())
    return same_o / len(Z), same_k / len(Z)


json.dump(WORDS, open("out/bench/core50_names.json", "w"))
np.save("out/bench/core50_train_labels.npy", TR[:, 2])
np.save("out/bench/core50_test_labels.npy", TE[:, 2])
print(f"{len(train_v)} training videos, {len(test_v)} held out (sessions {sorted(TEST_SESSIONS)}); stream {len(TR)} frames, test {len(TE)}", flush=True)

for seed in SEEDS:
    base = pickle.load(open(f"out/bench/cortex_ph_d_s{seed}.pkl", "rb"))
    base_relay, base.relay = base.relay, None
    first = base.layers[0]
    rng0 = np.random.default_rng(seed + 7)
    settle_rows = np.sort(rng0.choice(dev, NRELAY, replace=False))
    groups = []  # the fixations in their order: VIEWS frames STEP apart within one training video
    for _ in range(FIX):
        v = train_v[rng0.integers(len(train_v))]
        s = int(rng0.integers(0, max(1, v["frames"] - STEP * (VIEWS - 1))))
        groups.append([v["start"] + s + STEP * k for k in range(VIEWS)])
    groups = np.array(groups)
    shuffled = rng0.permutation(groups.ravel()).reshape(-1, VIEWS)  # the same frames, their order lost

    variants = {"first": None}
    variants.update({a: a for a in ARMS})
    for name, arm in variants.items():
        t0 = time.time()
        if arm is None:  # the first stage alone, with its own relay: the photo benchmark's code
            code = lambda im: base.code(im, GRID)[0]  # noqa: E731
            relay = base_relay
        else:
            rng = np.random.default_rng(seed + 11)
            second = Layer2(first.kinds, 256, 3, 1, groups=16, fan_in=32, seed=seed + 1, scaled=False)
            second.settle((first.respond(pic(r)) for r in groups[:200].ravel()), rng)
            for n, g in enumerate(shuffled if arm == "shuffled" else groups):
                learn(second, [first.respond(pic(r)) for r in g], rng, arm != "plain", rate=0.05 * (1 - n / FIX) + 0.005)
            cortex = copy.deepcopy(base)
            cortex.layers = [first, second]
            code = lambda im, c=cortex: c.code(im, GRID)[1]  # noqa: E731
            relay = Relay(SHRINK).settle([code(np.asarray(thumbs[r], np.float32) / 255) for r in settle_rows])
            pickle.dump(second, open(f"out/bench/second_c50{arm}_s{seed}.pkl", "wb"))
        for part, R in (("train", TR), ("test", TE)):
            out = np.zeros((len(R), GRID * GRID * 256), np.float16)
            for i, r in enumerate(R[:, 0]):
                out[i] = code(pic(r))
            np.save(f"out/bench/core50_c50{name}_s{seed}_{part}.npy", out)
        keep = copy.deepcopy(base)  # the stream reads only the relay from this file
        keep.relay = relay
        pickle.dump(keep, open(f"out/bench/cortex_c50{name}_s{seed}.pkl", "wb"))
        T = np.load(f"out/bench/core50_c50{name}_s{seed}_test.npy").astype(np.float32)
        Z = (T - relay.mu) @ relay.Z
        Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
        o, k = elsewhere(Z, TE)
        print(f"seed {seed} {name:8}: nearest frame of another session: same object {o:.3f} (chance 0.020), same kind {k:.3f} (chance 0.100) ({time.time() - t0:.0f}s)", flush=True)
