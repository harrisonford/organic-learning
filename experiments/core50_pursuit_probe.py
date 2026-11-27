"""Where does an eye that follows land? Pursuit on CORe50, measured before any code is read (queue E 34; docs/VISION.md
2026-11-27 02:55).

core50_fovea_probe.py: an eye on the held object (the frame's middle, where CORe50's crop puts it) recognises the
object in another session far better than the whole frame (0.164 against 0.117-0.120, two seeds), but the reflexes we
have jump every frame and miss it (the place that stands out most: 0.075; the place that changed most: 0.093). Smooth
pursuit keeps the eye on a moving thing: it is driven by the motion of what is already on and near the fovea (retinal
slip), not by every change in the view, and it catches up by a saccade only when the thing is lost (Lisberger 2010; to
verify). Here, at CORe50's 20 frames a second, on the held-out sessions' videos:
    jump        every frame, the place that changed most since the frame before (the motion reflex of the fovea probe)
    pursuit     the eye starts on the place that stands out most; each frame it moves toward the centre of the motion
                within its own field (the fixated half and a margin of a quarter of it), by DT / TAU of the way
                (TAU 100 ms); motion is each pixel's change less the frame's median change
    pursuit+    the same, with a catch-up saccade to the place that changed most when the motion in the eye's own half
                falls under the average place's for 300 ms (the thing is lost)
Measured: how far the fixated half's corner is from the middle place's (in pixels of the 64 x 64 frame; the middle is
16, 16), the share of frames within 4 and 8 pixels of it, and how far the eye moves from one frame to the next.

    python experiments/core50_pursuit_probe.py
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.stream import places  # noqa: E402

DIR = os.path.expanduser("~/datasets/core50")
TAU, DT, LOST = float(os.environ.get("TAU", 0.1)), 0.05, 0.3
MOTION = os.environ.get("MOTION", "plain")  # plain: every change; relative: what moves against the view's own flow
frames = np.memmap(os.path.join(DIR, "frames_64.u8"), np.uint8, "r").reshape(-1, 64, 64, 3)
videos = [json.loads(line) for line in open(os.path.join(DIR, "frames.jsonl"))]
test_v = [v for v in videos if v["session"] in {3, 7, 10}]
SIDE, MID = 32, np.array([16.0, 16.0])


def motion(a, b):
    """Each pixel's change less the frame's median change. With MOTION=relative, the frame before is first shifted by
    the whole-pixel step (up to 3 in each direction) that matches it best to this one: the view's own flow, made by
    the head and camera moving, is taken out and what still changes moves by itself (the visual system discounts
    self-motion: the optokinetic system and MST; to verify)."""
    a, b = a.astype(np.float32), b.astype(np.float32)
    if MOTION == "relative":
        best, m = None, None
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                d = np.abs(a[3:61, 3:61] - b[3 - dy: 61 - dy, 3 - dx: 61 - dx]).mean(axis=2)
                if best is None or d.mean() < best:
                    best, m = d.mean(), d
        m = np.pad(m, 3) / 255
    else:
        m = np.abs(a - b).mean(axis=2) / 255
    return np.maximum(m - np.median(m), 0)


def best_moving(m):
    r, c, _ = max(places(np.repeat(m[..., None], 3, axis=2) + 0.5), key=lambda p: p[2])
    return np.array([r, c], float)


KINDS = ("jump", "pursuit", "pursuit+")
dist, steps = {k: [] for k in KINDS}, {k: [] for k in KINDS}


def run(kind, f, maps):
    r, c, _ = max(places(f[0].astype(np.float32) / 255), key=lambda p: p[2])
    pos, lost_for = np.array([r, c], float), 0.0
    for t in range(1, len(f)):
        m = maps[t - 1]
        before = pos.copy()
        if kind == "jump":
            pos = best_moving(m)
        else:
            margin = SIDE // 4
            r0, c0 = int(max(pos[0] - margin, 0)), int(max(pos[1] - margin, 0))
            r1, c1 = int(min(pos[0] + SIDE + margin, 64)), int(min(pos[1] + SIDE + margin, 64))
            field = m[r0:r1, c0:c1]
            if field.sum() > 1e-6:
                rr, cc = np.mgrid[r0:r1, c0:c1]
                centre = np.array([(rr * field).sum(), (cc * field).sum()]) / field.sum()
                pos += (DT / TAU) * (centre - SIDE / 2 - pos)
            pos = np.clip(pos, 0, 64 - SIDE)
            if kind == "pursuit+":
                inside = m[int(pos[0]): int(pos[0]) + SIDE, int(pos[1]): int(pos[1]) + SIDE].mean()
                lost_for = lost_for + DT if inside < m.mean() else 0.0
                if lost_for >= LOST:
                    pos, lost_for = best_moving(m), 0.0
        dist[kind].append(np.hypot(*(pos - MID)))
        steps[kind].append(np.hypot(*(pos - before)))


print(f"{len(test_v)} held-out videos, {sum(v['frames'] for v in test_v)} frames; motion {MOTION}; the eye's places are 0, 16 or 32 px from the edge (the middle: 16, 16)", flush=True)
for v in test_v:
    f = np.asarray(frames[v["start"]: v["start"] + v["frames"]])
    maps = [motion(f[t], f[t - 1]) for t in range(1, len(f))]
    for kind in KINDS:
        run(kind, f, maps)
for kind in KINDS:
    d = np.array(dist[kind])
    print(f"{kind:9}: from the middle {d.mean():5.1f} px (median {np.median(d):4.1f}); within 4 px {np.mean(d <= 4):.2f}, within 8 px {np.mean(d <= 8):.2f}; "
          f"moves {np.mean(steps[kind]):.2f} px a frame", flush=True)
