"""How a code's variance is spread, and how much of it the relay can even out (docs/VISION.md, 2026-11-26 08:50).

The relay whitens the 4,096 summary numbers as Z = E (d + shrink * d_max)^(-1/2) E^T (organic.cortex2.Relay): a
direction whose variance d is far below shrink * d_max is not raised to the others' level. Settling the relay on
20,000 pictures in place of 6,000 hardly changed how many directions the developed photo code spreads over after it
(386 -> 396), so the number of pictures is not what limits it. This reads, for codes as born and developed, the
spectrum of the code itself: its participation ratio, the largest variance over the average, and how many directions
lie above shrink * d_max (the ones the relay can bring up to level), for the relay's own shrink 0.0005.

    PAIRS=photos:ph_r:ph_d,cifar10:nat_r:nat_d,cifar10:big_r:big_d SEED=0 MANY=4000 python experiments/spectrum_probe.py
"""
import os
import numpy as np

SEED, MANY, SHRINK = int(os.environ.get("SEED", 0)), int(os.environ.get("MANY", 4000)), float(os.environ.get("SHRINK", 0.0005))
PAIRS = [p.split(":") for p in os.environ.get("PAIRS", "photos:ph_r:ph_d,cifar10:nat_r:nat_d,cifar10:big_r:big_d").split(",")]

for SET, *variants in PAIRS:
    for v in variants:
        X = np.asarray(np.load(f"out/bench/{SET}_{v}_s{SEED}_train.npy", mmap_mode="r")[:MANY], np.float64)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12  # the relay settles on summaries as the cortex sends them
        X -= X.mean(axis=0)
        d = np.clip(np.linalg.eigvalsh(X.T @ X / (len(X) - 1)), 0, None)[::-1]
        pr = d.sum() ** 2 / (d * d).sum()
        above = int((d > SHRINK * d[0]).sum())
        gain = (d / (d + SHRINK * d[0]))  # how far each direction is brought to level by the relay (1 = fully)
        out = gain * 1.0
        pr_after = (out.sum() ** 2) / (out * out).sum()
        top = d[:10].sum() / d.sum()
        print(f"{SET:8} {v:6} seed {SEED}: participation ratio {pr:6.1f} | largest / average variance {d[0] / d.mean():7.1f} | "
              f"top 10 directions hold {top:.3f} | directions above shrink x largest: {above:4d} of {len(d)} | "
              f"spread after the relay (participation ratio of the levelled spectrum): {pr_after:6.1f}", flush=True)
