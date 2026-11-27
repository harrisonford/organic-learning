"""The stream brain's own eye on CORe50: where it lands and what it then recognises, with and without the two
reflexes of 2026-11-27 (docs/VISION.md 04:57, said before it runs).

The eye is organic/stream.py's StreamBrain, stepped at its 10 ms clock through every frame of the held-out sessions'
videos (20 frames a second: a frame every five steps), one brain a video (a new scene, nothing carried over). Arms:
    today      the eye as it is: saccades three or four a second among nine places, drawn by what stands out
    motion     what moves against the view's own flow draws saccades too (`motion_drive`)
    pursuit    between saccades the eye drifts after the motion in its own field, tau 100 ms (`pursuit_tau`)
    both       the two together
    middle     no eye: the middle place always (where CORe50's crop puts the object), the same relay: the reference
At every tenth frame (the frames the other CORe50 probes code) the fixated half is taken where the eye is, enlarged to
fill the view, and coded as experiments/core50_fovea_probe.py codes it (the photo benchmark's cortex of the seed, the
first layer's code through a relay settled on COCO thumbnails seen through the reflex's pick: cortex_c50salient_s{S}
.pkl, the same relay for every arm). The sight population's own code is not computed while stepping (the probe reads
where the eye is, and codes those frames itself); everything else in the brain runs as it is.
Measured: how far the fixated half's corner is from the middle's (16, 16) and the share within 4 and 8 px; saccades a
second; then the code read as before: the nearest frame from another session (same object, chance 0.02; same kind,
0.10) and how alike a frame is to its video's next coded frame, the same object elsewhere, another kind in the same
session (the scene).

    SEEDS=0,1,2 python experiments/core50_stream_eye_probe.py
"""
import json, os, pickle, sys, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.stream import StreamBrain  # noqa: E402

DIR = os.path.expanduser("~/datasets/core50")
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
ARMS = os.environ.get("ARMS", "today,motion,pursuit,both,middle").split(",")
EVERY, GRID, STEPS_A_FRAME = 10, 4, 5  # coded frames as the other probes; 20 frames a second at a 10 ms clock
MID = np.array([16.0, 16.0])
frames = np.memmap(os.path.join(DIR, "frames_64.u8"), np.uint8, "r").reshape(-1, 64, 64, 3)
videos = [json.loads(line) for line in open(os.path.join(DIR, "frames.jsonl"))]
test_v = [v for v in videos if v["session"] in {3, 7, 10}]
test_v = test_v[:: int(os.environ.get("THIN", 1))]  # a smoke test: every THIN-th video
TE = np.array([(v["start"] + i, v["object"], (v["object"] - 1) // 5, v["session"], k) for k, v in enumerate(test_v) for i in range(0, v["frames"], EVERY)])
SETTINGS = {"today": {}, "motion": {"motion_drive": True}, "pursuit": {"pursuit_tau": 0.1}, "both": {"motion_drive": True, "pursuit_tau": 0.1}}


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
    obj, kind, sess, vid = meta[:, 1], meta[:, 2], meta[:, 3], meta[:, 4]
    ok = np.nonzero(vid[1:] == vid[:-1])[0]
    out = {"next 0.5 s": float(np.mean(np.sum(Z[ok] * Z[ok + 1], axis=1)))}
    for name, cond in (("same object, other session", lambda i: (obj == obj[i]) & (sess != sess[i])),
                       ("other kind, same session", lambda i: (kind != kind[i]) & (sess == sess[i]))):
        idx = rng.choice(len(Z), min(1500, len(Z)), replace=False)
        cands = [(i, np.nonzero(cond(i))[0]) for i in idx]  # (in the full set every frame has partners of both sorts)
        out[name] = float(np.mean([Z[i] @ Z[rng.choice(c)] for i, c in cands if len(c)]))
    return out


def window(img, r, c):
    side = img.shape[0] // 2
    w = img[r: r + side, c: c + side]
    return np.asarray(Image.fromarray((np.clip(w, 0, 1) * 255).astype(np.uint8)).resize(img.shape[:2][::-1], Image.BILINEAR), np.float32) / 255


def where_eye(brain):
    if brain.pursuit_tau and brain.pos is not None:
        return float(brain.pos[0]), float(brain.pos[1])
    r, c, _ = brain.places[brain.where]
    return float(r), float(c)


for seed in SEEDS:
    cortex = pickle.load(open(f"out/bench/cortex_c50salient_s{seed}.pkl", "rb"))
    relay = cortex.relay
    for arm in ARMS:
        t0 = time.time()
        corners, saccades, seconds = [], 0, 0.0
        for k, v in enumerate(test_v):
            f = np.asarray(frames[v["start"]: v["start"] + v["frames"]], np.float32) / 255
            if arm == "middle":
                corners += [(16.0, 16.0)] * len(range(0, v["frames"], EVERY))
                continue
            brain = StreamBrain(cortex, None, seed=seed * 1000 + k, **SETTINGS[arm])
            zero = np.zeros(relay.Z.shape[1], np.float32)

            def no_code(b=brain, z=zero):  # where the eye is matters here, not the sight population's code
                if b.pursuit_tau:
                    b.pos_seen = b.pos.copy()
                return z
            brain._see = no_code
            for i in range(v["frames"]):
                for s in range(STEPS_A_FRAME):
                    ev = brain.step(f[i] if s == 0 else None, None)
                    if s == 0 and i % EVERY == 0:
                        corners.append(where_eye(brain))
                    saccades += sum(e[0] == "saccade" for e in ev)
            seconds += v["frames"] * STEPS_A_FRAME * 0.010
        corners = np.array(corners)
        assert len(corners) == len(TE)
        d = np.hypot(*(corners - MID).T)
        X = np.stack([cortex.code(window(np.asarray(frames[r], np.float32) / 255, int(round(a)), int(round(b))), GRID)[0] for r, (a, b) in zip(TE[:, 0], corners)])
        Z = unit((X - relay.mu) @ relay.Z)
        o, kd = elsewhere(Z, TE)
        a = alike(Z, TE, np.random.default_rng(seed + 7))
        rate = f"{saccades / seconds:.1f} saccades a second; " if seconds else ""
        print(f"seed {seed} {arm:8}: {rate}from the middle {d.mean():4.1f} px, within 4 px {np.mean(d <= 4):.2f}, within 8 px {np.mean(d <= 8):.2f} | "
              f"elsewhere same object {o:.3f}, same kind {kd:.3f} | alike: " + ", ".join(f"{n} {x:.3f}" for n, x in a.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
