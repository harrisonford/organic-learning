"""Watch the raw eye: saccades, what the retina sees, and the V1 cells it grows.

    python examples/eye_view.py

Writes out/eye_saccades.png (scan paths and retinal views) and out/eye_v1.png
(the developed simple cells).
"""

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.natural import natural_views  # noqa: E402
from organic.retina import Retina, develop_v1, retinal_wave  # noqa: E402
from organic.scenes import LOGIC_INTENTS, RELATION_INTENTS, SINGLE_INTENTS, episode  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
S = 6  # display scale


def to_img(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def retinal_image(retina, image, fx, fy):
    """Paint what the receptors report back onto the image plane (blurry periphery)."""
    _, pyr = retina._pyramid(image)
    canvas = Image.new("RGB", (40 * S, 40 * S), (0, 0, 0))
    d = ImageDraw.Draw(canvas)
    xs, ys = retina.receptor_positions(fx, fy)
    for i in reversed(range(retina.rings)):
        c = retina._sample(pyr[np.round(retina.rf[i], 1)], xs[i], ys[i])
        rad = max(1.5, retina.rf[i] * 1.6) * S
        for j in range(retina.angles):
            l, m, s_ = c[j]
            # approximate inverse of the cone mixing, for display only
            r = np.clip(1.86 * l - 0.86 * m, 0, 1)
            g = np.clip(-1.0 * l + 2.0 * m, 0, 1)
            b = np.clip((s_ - 0.1 * g) / 0.9, 0, 1)
            col = tuple(int(255 * v) for v in (r, g, b))
            d.ellipse([xs[i, j] * S - rad, ys[i, j] * S - rad, xs[i, j] * S + rad, ys[i, j] * S + rad], fill=col)
    return canvas


def logpolar(m, scale=10):
    """A (rings, angles) map as an image: rows = eccentricity, columns = angle."""
    m = m / (m.max() + 1e-8)
    return to_img(np.repeat(np.repeat(m, scale, 0), scale, 1)[..., None].repeat(3, -1))


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(1)
    retina = Retina()
    print("developing V1 (retinal waves, then natural images)...")
    experience = natural_views(rng, 600)
    window = int(os.environ.get("V1_WINDOW", 3))
    v1 = develop_v1(retina, experience, waves=1500, rng=rng, window=window)

    rows = []
    for _ in range(4):
        e = episode(rng, "test", SINGLE_INTENTS + RELATION_INTENTS + LOGIC_INTENTS)
        stream = retina.view(e["image"], ticks=12)
        rows.append((e, stream))
    W = 40 * S
    sheet = Image.new("RGB", (W * 5 + 60, len(rows) * (W + 30)), "white")
    d = ImageDraw.Draw(sheet)
    for r, (e, stream) in enumerate(rows):
        y = r * (W + 30)
        scene = to_img(e["image"]).resize((W, W), Image.NEAREST)
        sd = ImageDraw.Draw(scene)
        fixes = []
        for t in stream:
            if not fixes or fixes[-1] != t.fixation:
                fixes.append(t.fixation)
        for k, (fx, fy) in enumerate(fixes):
            if k:
                sd.line([fixes[k - 1][0] * S, fixes[k - 1][1] * S, fx * S, fy * S], fill=(255, 255, 0), width=2)
            sd.ellipse([fx * S - 6, fy * S - 6, fx * S + 6, fy * S + 6], outline=(255, 255, 0), width=2)
            sd.text((fx * S + 8, fy * S - 8), str(k + 1), fill=(255, 255, 0))
        sheet.paste(scene, (0, y))
        for k, (fx, fy) in enumerate(fixes[:4]):
            sheet.paste(retinal_image(retina, e["image"], fx, fy), ((k + 1) * (W + 15), y))
            d.text(((k + 1) * (W + 15), y + W + 4), f"fixation {k + 1}: what the retina reports", fill=(0, 0, 0))
        d.text((0, y + W + 4), "scan path: " + ", ".join(f"{o['color']} {o['shape']}" for o in e["objects"]), fill=(0, 0, 0))
    sheet.save(os.path.join(OUT, "eye_saccades.png"))

    # V1 cells in a grid: each shown as its window (rings down, angles across)
    # for the three opponent channels (ON minus OFF; gray = 0)
    K = v1.kinds
    n = v1.window
    Wk = v1.W.reshape(K, n, n, 3)
    cell, cols = 48, 4
    tile_w, tile_h = 3 * cell + 24, cell + 38
    img = Image.new("RGB", (cols * tile_w, ((K + cols - 1) // cols) * tile_h), "white")
    dd = ImageDraw.Draw(img)
    for k in range(K):
        x0, y0 = (k % cols) * tile_w, (k // cols) * tile_h
        scale = np.abs(Wk[k]).max() + 1e-8
        for c, name in enumerate(["lum", "L-M", "S-LM"]):
            v = Wk[k, :, :, c] / scale
            gray = (v * 0.5 + 0.5)[..., None].repeat(3, -1)
            img.paste(to_img(gray).resize((cell, cell), Image.NEAREST), (x0 + c * cell, y0))
            dd.text((x0 + c * cell + 2, y0 + cell + 2), name, fill=(0, 0, 0))
        dd.text((x0, y0 + cell + 18), f"cell {k}: won {int(v1.wins[k])}", fill=(0, 0, 0))
    img.save(os.path.join(OUT, f"eye_v1_w{n}.png"))

    # a retinal wave, for the record
    to_img(retinal_wave(np.random.default_rng(3))).resize((W, W), Image.NEAREST).save(os.path.join(OUT, "eye_wave.png"))
    print("wins per V1 cell:", v1.wins.astype(int))
    print("saved out/eye_saccades.png, out/eye_v1.png, out/eye_wave.png")


if __name__ == "__main__":
    main()
