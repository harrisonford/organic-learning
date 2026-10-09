"""Draw the curriculum log as two stacked panels sharing the age axis.

    python experiments/plot_curriculum.py out/curriculum2.jsonl out/curriculum.png
"""

import json
import sys

from PIL import Image, ImageDraw

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"

recs = [json.loads(line) for line in open(sys.argv[1])]
out = sys.argv[2] if len(sys.argv) > 2 else "out/curriculum.png"
onset = next((r["age"] for r in recs if r["phase"] != "base"), None)
onset = onset - 250 if onset else None

W, H = 900, 640
L, R = 70, 120
img = Image.new("RGB", (W, H), SURFACE)
d = ImageDraw.Draw(img)
ages = [r["age"] for r in recs]
x0, x1 = 0, max(ages)


def X(a):
    return L + (a - x0) / (x1 - x0) * (W - L - R)


def panel(top, bottom, title, series, ymax, yfmt, yticks):
    def Y(v):
        return bottom - v / ymax * (bottom - top)

    d.text((L, top - 26), title, fill=INK)
    for t in yticks:
        d.line([(L, Y(t)), (W - R, Y(t))], fill=GRID, width=1)
        d.text((L - 8 - 6 * len(yfmt(t)), Y(t) - 6), yfmt(t), fill=MUTED)
    d.line([(L, bottom), (W - R, bottom)], fill=AXIS, width=1)
    if onset:
        d.line([(X(onset), top), (X(onset), bottom)], fill=MUTED, width=1)
        d.text((X(onset) + 6, top + 2), "logic questions start", fill=INK2)
    for label, key, color in series:
        rows = [r for r in recs if key != "logic_acc" or r["phase"] != "base"]
        pts = [(X(r["age"]), Y(r[key])) for r in rows]
        d.line(pts, fill=color, width=2, joint="curve")
        for p in pts:
            d.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=color, outline=SURFACE, width=2)
        d.text((pts[-1][0] + 10, pts[-1][1] - 6), label, fill=INK2)
    return Y


# top: neurogenesis per checkpoint, as bars (first bar is birth itself, capped)
top, bottom, cap = 60, 290, 160
d.text((L, top - 26), "New speech-area clusters born per 250 episodes", fill=INK)
for t in [0, 50, 100, 150]:
    y = bottom - t / cap * (bottom - top)
    d.line([(L, y), (W - R, y)], fill=GRID, width=1)
    d.text((L - 8 - 6 * len(str(t)), y - 6), str(t), fill=MUTED)
d.line([(L, bottom), (W - R, bottom)], fill=AXIS, width=1)
bw = (X(250) - X(0)) * 0.6
for r in recs:
    v = min(r["new_clusters"], cap)
    xc = X(r["age"]) - (X(250) - X(0)) / 2
    color = BLUE
    d.rounded_rectangle([xc - bw / 2, bottom - v / cap * (bottom - top), xc + bw / 2, bottom], radius=4, fill=color)
    if r["new_clusters"] > cap:
        d.text((xc - 12, top - 12), str(r["new_clusters"]), fill=INK2)
if onset:
    d.line([(X(onset), top), (X(onset), bottom)], fill=MUTED, width=1)
    d.text((X(onset) + 6, top + 2), "logic questions start", fill=INK2)
panel(360, 590, "Accuracy on new scenes", [("base tasks", "base_acc", BLUE), ("logic", "logic_acc", ORANGE)],
      1.0, lambda v: f"{v:.0%}", [0, 0.25, 0.5, 0.75, 1.0])
for a in range(0, x1 + 1, 1000):
    d.text((X(a) - 10, 598), f"{a}", fill=MUTED)
d.text((W // 2 - 60, 616), "age (episodes lived)", fill=INK2)
# legend for the two-series panel
d.rectangle([W - R - 230, 335, W - R - 222, 343], fill=BLUE)
d.text((W - R - 216, 331), "base tasks", fill=INK2)
d.rectangle([W - R - 130, 335, W - R - 122, 343], fill=ORANGE)
d.text((W - R - 116, 331), "logic", fill=INK2)
img.save(out)
print("saved", out)
