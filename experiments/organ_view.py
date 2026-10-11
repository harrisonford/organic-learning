"""Look inside a raised progressive eye.

    python experiments/organ_view.py out/organ_settle_s0.pkl [tag]

Writes out/organ_<tag>_neighbours.png (test photos and the reference photos the
organ finds most similar), out/organ_<tag>_gaze.png (where it looks: heat map
over the 5x5 gaze grid and example scan paths) and out/organ_<tag>_predict.png
(what the coarse view led it to expect around the gaze point, vs what came).
"""

import os
import pickle
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic import cifar  # noqa: E402

eye = pickle.load(open(sys.argv[1], "rb"))
tag = sys.argv[2] if len(sys.argv) > 2 else "view"
OUT = os.path.join(os.path.dirname(__file__), "..", "out")
xtr, ytr = cifar.load("train")
xte, yte = cifar.load("test")
S = 3


def pic(a, scale=S):
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((a.shape[1] * scale, a.shape[0] * scale), Image.NEAREST)


def final(img):
    outs, path, _ = eye.look(img, learn=False, explore=False)
    return outs[-1], path


# neighbours
ref = np.stack([final(x)[0] for x in xtr[:3000]])
ref /= np.linalg.norm(ref, axis=1, keepdims=True) + 1e-8
rng = np.random.default_rng(1)
queries = rng.choice(len(xte), 10, replace=False)
cell = 32 * S + 6
sheet = Image.new("RGB", (cell * 7 + 10, cell * len(queries) + 20), "white")
d = ImageDraw.Draw(sheet)
d.text((4, 2), "test photo  |  the 6 reference photos the organ finds most similar (final tick)", fill=(0, 0, 0))
for r, q in enumerate(queries):
    v, _ = final(xte[q])
    v /= np.linalg.norm(v) + 1e-8
    nn = np.argsort(-(ref @ v))[:6]
    y = 20 + r * cell
    sheet.paste(pic(xte[q]), (2, y))
    d.text((2, y + 32 * S - 10), cifar.CLASSES[yte[q]][:8], fill=(255, 255, 255))
    for j, n in enumerate(nn):
        sheet.paste(pic(xtr[n]), (cell * (j + 1) + 10, y))
        ok = ytr[n] == yte[q]
        d.rectangle([cell * (j + 1) + 10, y, cell * (j + 1) + 10 + 32 * S - 1, y + 32 * S - 1], outline=(0, 160, 0) if ok else (200, 0, 0), width=2)
sheet.save(os.path.join(OUT, f"organ_{tag}_neighbours.png"))

# gaze
counts = np.zeros(25)
paths = []
for x in xte[:1000]:
    _, path = final(x)
    paths.append(path)
    for t in path[1:]:
        counts[t] += 1
heat = counts.reshape(5, 5) / max(counts.sum(), 1)
g = Image.new("RGB", (5 * 40 + 8 * (32 * S + 6), 5 * 40 + 20), "white")
dg = ImageDraw.Draw(g)
for i in range(5):
    for j in range(5):
        v = heat[i, j] / (heat.max() + 1e-8)
        col = (int(255 - 200 * v), int(255 - 120 * v), 255)
        dg.rectangle([j * 40, 20 + i * 40, j * 40 + 39, 20 + i * 40 + 39], fill=col, outline=(200, 200, 200))
        dg.text((j * 40 + 6, 20 + i * 40 + 14), f"{heat[i, j]:.2f}", fill=(0, 0, 0))
dg.text((2, 2), "where saccades land (fixations 2-4)        example scan paths", fill=(0, 0, 0))
for k in range(8):
    im = pic(xte[k]).copy()
    di = ImageDraw.Draw(im)
    pts = [(eye.sensor.targets[t][0] * S, eye.sensor.targets[t][1] * S) for t in paths[k]]
    for a, b in zip(pts, pts[1:]):
        di.line([a, b], fill=(255, 255, 0), width=2)
    for n, p in enumerate(pts):
        di.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], outline=(255, 255, 0), width=2)
        di.text((p[0] + 4, p[1] - 10), str(n + 1), fill=(255, 255, 0))
    g.paste(im, (5 * 40 + 8 + k * (32 * S + 6), 20))
g.save(os.path.join(OUT, f"organ_{tag}_gaze.png"))

# predictions: coarse view -> expected medium view at the gaze target
p = Image.new("RGB", (6 * (3 * 8 * 6 + 20), 8 * 6 + 40), "white")
dp = ImageDraw.Draw(p)
dp.text((2, 2), "per example: medium view expected from the coarse glance  |  medium view that came", fill=(0, 0, 0))
for k in range(6):
    img = xte[20 + k]
    states = [np.zeros(0, np.float32) for _ in range(4)]
    views = eye.sensor.views(img, 7)
    states[0] = eye._perceive(eye.groups[0], states[0], views[0], False)
    pred = eye.predict_mid.predict(states[0], 7)
    x0 = k * (3 * 8 * 6 + 20)
    for j, v in enumerate([pred, views[1]]):
        if v is None:
            continue
        a = v.reshape(8, 8, 3)
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
        p.paste(pic(a, 6), (x0 + j * (8 * 6 + 4), 30))
p.save(os.path.join(OUT, f"organ_{tag}_predict.png"))
print("saved", tag)
