"""A public benchmark as a one-pass stream: the cortex's codes (docs/PUBLISH.md, 2026-11-19).

The paper's controlled core needs a yardstick that is not ours. CIFAR-10 as a stream: the cortex develops without
labels on other pictures (CIFAR-100's training pictures, no labels used), then every labelled CIFAR-10 picture is
seen once. This script makes the cortices and keeps their codes, so that every learner and readout can be run on
the same numbers (experiments/bench_stream.py):
    developed   256 kinds, winner-take-all Hebbian with a conscience on decorrelated windows (organic/cortex2.py)
    random      the same layer never developed (random filters on the same decorrelated windows)
each with its relay settled on 6,000 of the unlabelled pictures. Kept: the summary before the relay (unit length,
half precision), the relay, the labels.
    SEEDS=0 VARIANTS=developed,random python experiments/bench_codes.py
"""
import os, pickle, sys, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.cortex2 import VisualCortex2  # noqa: E402

DATA = os.path.expanduser("~/datasets")
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
VARIANTS = os.environ.get("VARIANTS", "developed,random").split(",")
GRID, SHRINK = 4, 0.0005


def load(path, key=b"labels"):
    d = pickle.load(open(path, "rb"), encoding="bytes")
    return d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), np.array(d[key])


def big(x):
    """A 32x32 picture as the eye gets it (64x64, values in [0, 1]); as the world shows CIFAR-100 to the lives."""
    return np.asarray(Image.fromarray(x).resize((64, 64), Image.BILINEAR), np.float32) / 255


unlabelled, _ = load(f"{DATA}/cifar-100-python/train", b"fine_labels")
train = [load(f"{DATA}/cifar-10-batches-py/data_batch_{i}") for i in range(1, 6)]
Xtr, ytr = np.concatenate([a for a, _ in train]), np.concatenate([b for _, b in train])
Xte, yte = load(f"{DATA}/cifar-10-batches-py/test_batch")
np.save("out/bench/cifar10_train_labels.npy", ytr)
np.save("out/bench/cifar10_test_labels.npy", yte)


class Pictures:
    """The unlabelled pictures, enlarged when looked at."""
    def __len__(self):
        return len(unlabelled)

    def __getitem__(self, i):
        return big(unlabelled[int(i)])


for seed in SEEDS:
    for variant in VARIANTS:
        t0, rng = time.time(), np.random.default_rng(seed)
        cortex = VisualCortex2(seed=seed, kinds=(256,), scaled=False)
        pics = Pictures()
        if variant == "developed":
            cortex.develop(pics, rng, log=lambda s: print("   " + s, flush=True))
        else:  # the decorrelation of the windows only; the kinds stay as they were born
            cortex.layers[0].settle((pics[i] for i in rng.choice(len(pics), 800, replace=False)), rng)
        cortex.settle_relay([pics[i] for i in rng.choice(len(pics), 6000, replace=False)], GRID, SHRINK)
        cortex.grid = GRID
        relay, cortex.relay = cortex.relay, None
        for name, X in (("train", Xtr), ("test", Xte)):
            out = np.zeros((len(X), 16 * 256), np.float16)
            for i, x in enumerate(X):
                out[i] = cortex.whole(big(x), GRID)
            np.save(f"out/bench/cifar10_{variant}_s{seed}_{name}.npy", out)
        cortex.relay = relay
        pickle.dump(cortex, open(f"out/bench/cortex_{variant}_s{seed}.pkl", "wb"))
        print(f"seed {seed} {variant}: cortex and codes kept ({time.time() - t0:.0f}s)", flush=True)
