"""The eye on the thing: CORe50 seen through a fixation, not the whole frame (queue E 31 and E 34; docs/VISION.md
2026-11-27 03:30).

core50_slow_probe.py found the first stage's whole-view code ruled by the scene: a frame of another kind of object
filmed in the same session is more alike (cosine 0.15) than the same object filmed in another session (0.03). Within a
video the scene never changes, so anything slow learns it. A brain does not see a held object as a whole frame: the
eyes fixate it and the fovea gives it most of the cortex. Here the same cortex (the photo benchmark's developed one of
the seed, unchanged) reads the fixated half of each frame enlarged to fill the view (as organic/stream.py's eye does),
with a relay settled on COCO thumbnails seen the same way (as the trace probe settles its own):
    centre    the middle place (the operator holds the object before the camera; it stands in for pursuit)
    salient   the place that stands out most from its surround (organic/stream.py's places(): the reflex's pick)
    moving    the place that changed most in the last 100 ms (VIEWS=moving; added 03:55 when the salient pick missed)
    w+...     the whole frame's code and a fixation's, side by side (the periphery and the fovea)
Read on the held-out sessions as before: the nearest frame from another session (same object, chance 0.02; same kind,
chance 0.10) and how alike a frame's code is to its video's next frame, to the same object elsewhere, and to another
kind in the same session (the scene alone). Codes saved as core50_c50{centre,salient}_s{S}_{train,test}.npy with
cortex_c50{...}_s{S}.pkl (its relay) for core50_slow_probe.py (CODE=...) and bench_stream.py.

    SEEDS=0,1,2 python experiments/core50_fovea_probe.py
"""
import copy, json, os, pickle, sys, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.cortex2 import Relay  # noqa: E402
from organic.stream import places  # noqa: E402

DIR = os.path.expanduser("~/datasets/core50")
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
VIEWS = os.environ.get("VIEWS", "centre,salient").split(",")
EVERY, NRELAY, GRID, SHRINK = 10, int(os.environ.get("NRELAY", 6000)), 4, 0.0005  # as core50_trace_probe.py
TEST_SESSIONS = {3, 7, 10}
frames = np.memmap(os.path.join(DIR, "frames_64.u8"), np.uint8, "r").reshape(-1, 64, 64, 3)
videos = [json.loads(line) for line in open(os.path.join(DIR, "frames.jsonl"))]
train_v = [v for v in videos if v["session"] not in TEST_SESSIONS]
test_v = [v for v in videos if v["session"] in TEST_SESSIONS]


def rows(vs):
    return np.array([(v["start"] + i, v["object"], (v["object"] - 1) // 5, v["session"], k) for k, v in enumerate(vs) for i in range(0, v["frames"], EVERY)])


TR, TE = rows(train_v), rows(test_v)
thumbs = np.load(os.path.expanduser("~/datasets/coco_thumbs/train2017_64.npy"), mmap_mode="r")
dev = np.load("out/bench/photos_dev_rows.npy")


def unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


def fixate(img, how, before=None):
    """The fixated half of the view, enlarged to fill it (bilinear, as the stream brain's eye). `before`: the frame
    100 ms earlier, for the eye drawn by what moves."""
    side = img.shape[0] // 2
    if how == "centre":
        r = c = side // 2
    elif how == "moving":  # the place that changed most in the last 100 ms (motion draws the eye: the colliculus)
        r, c, _ = max(places(np.abs(img - before) + 0.5), key=lambda p: p[2]) if before is not None else (side // 2, side // 2, 0)
    else:
        r, c, _ = max(places(img), key=lambda p: p[2])
    w = img[r: r + side, c: c + side]
    return np.asarray(Image.fromarray((np.clip(w, 0, 1) * 255).astype(np.uint8)).resize(img.shape[:2][::-1], Image.BILINEAR), np.float32) / 255


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
    obj, kind, sess, vid = meta[:, 1], meta[:, 2], meta[:, 3], meta[:, 4]
    ok = np.nonzero(vid[1:] == vid[:-1])[0]
    out = {"next 0.5 s": float(np.mean(np.sum(Z[ok] * Z[ok + 1], axis=1)))}
    for name, cond in (("same object, other session", lambda i: (obj == obj[i]) & (sess != sess[i])),
                       ("other kind, same session", lambda i: (kind != kind[i]) & (sess == sess[i])),
                       ("other kind, other session", lambda i: (kind != kind[i]) & (sess != sess[i]))):
        idx = rng.choice(len(Z), 1500, replace=False)
        partners = [rng.choice(np.nonzero(cond(i))[0]) for i in idx]
        out[name] = float(np.mean(np.sum(Z[idx] * Z[partners], axis=1)))
    return out


def show(seed, label, Z):
    o, k = elsewhere(Z, TE)
    a = alike(Z, TE, np.random.default_rng(seed + 7))
    print(f"seed {seed} {label:8}: elsewhere same object {o:.3f}, same kind {k:.3f} | alike: " + ", ".join(f"{n} {v:.3f}" for n, v in a.items()), flush=True)


pic = lambda r: np.asarray(frames[r], np.float32) / 255  # noqa: E731
for seed in SEEDS:
    base = pickle.load(open(f"out/bench/cortex_ph_d_s{seed}.pkl", "rb"))
    base.relay = None
    settle_rows = np.sort(np.random.default_rng(seed + 7).choice(dev, NRELAY, replace=False))
    tests = {}
    whole_relay = pickle.load(open(f"out/bench/cortex_c50first_s{seed}.pkl", "rb")).relay
    tests["whole"] = unit((np.load(f"out/bench/core50_c50first_s{seed}_test.npy").astype(np.float32) - whole_relay.mu) @ whole_relay.Z)
    show(seed, "whole", tests["whole"])
    for how in VIEWS:
        t0 = time.time()
        code = lambda im, before=None: base.code(fixate(im, how, before), GRID)[0]  # noqa: E731
        settle_how = "salient" if how == "moving" else how  # a photograph has no motion: the relay settles on the reflex's pick
        relay = Relay(SHRINK).settle([base.code(fixate(np.asarray(thumbs[r], np.float32) / 255, settle_how), GRID)[0] for r in settle_rows])
        for part, R in (("train", TR), ("test", TE)):
            out = np.zeros((len(R), GRID * GRID * 256), np.float16)
            for i, r in enumerate(R[:, 0]):
                first_of_video = i == 0 or R[i, 4] != R[i - 1, 4]
                out[i] = code(pic(r), pic(r + 2) if first_of_video else pic(r - 2))  # 100 ms apart (20 frames a second)
            np.save(f"out/bench/core50_c50{how}_s{seed}_{part}.npy", out)
        keep = copy.deepcopy(base)
        keep.relay = relay
        pickle.dump(keep, open(f"out/bench/cortex_c50{how}_s{seed}.pkl", "wb"))
        tests[how] = unit((np.load(f"out/bench/core50_c50{how}_s{seed}_test.npy").astype(np.float32) - relay.mu) @ relay.Z)
        show(seed, how, tests[how])
        print(f"   ({time.time() - t0:.0f}s)", flush=True)
    for how in VIEWS:
        show(seed, f"w+{how}", unit(np.concatenate([tests["whole"], tests[how]], axis=1)))
