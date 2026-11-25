"""The benchmark's cortex on CIFAR at its own size (docs/VISION.md, 2026-11-25 12:40).

experiments/bench_codes.py shows CIFAR's 32 x 32 pictures to the cortex blown up to 64 x 64, as the lives' world
shows them. A 5 x 5 window of a blown-up picture holds less than half the directions of a window of a real picture
(experiments/upsample_probe.py: 22 against 47 above the decorrelation's floor). Here the same cortex, the same code
and stream, nothing new, the pictures at 32 x 32: the windows' decorrelation settled on them, the kinds as born
(nat_r) or developed by our hard rule (nat_d: organic.cortex2.VisualCortex2.develop, unchanged), the relay settled
on 6,000 unlabelled pictures; codes of 10,000 CIFAR-10 training pictures and 2,000 test pictures for
experiments/bench_stream.py (VARIANT=nat_r N=10000).
    SEEDS=0 VARIANTS=nat_r,nat_d python experiments/cortex_native_probe.py
    SIZE=64 SEEDS=0 VARIANTS=big_r,big_d ...   the control of the control: the same script, the pictures blown up
"""
import os, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import softhebb_probe as P  # noqa: E402
from bcm_probe import near_copies  # noqa: E402
from organic.cortex2 import VisualCortex2  # noqa: E402

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
VARIANTS = os.environ.get("VARIANTS", "nat_r,nat_d").split(",")
GRID, SHRINK = 4, 0.0005
unlabelled = P.load(f"{P.DATA}/cifar-100-python/train")
Xtr, Xte = P.load(f"{P.DATA}/cifar-10-batches-py/data_batch_1"), P.load(f"{P.DATA}/cifar-10-batches-py/test_batch")[:2000]
yte = np.load("out/bench/cifar10_test_labels.npy")[:2000]
SIZE = int(os.environ.get("SIZE", 32))  # 64: the same script on the pictures blown up, as the benchmark shows them (variants big_r, big_d)
if SIZE == 32:
    small = lambda x: x.astype(np.float32) / 255  # noqa: E731
else:
    from PIL import Image
    small = lambda x: np.asarray(Image.fromarray(x).resize((SIZE, SIZE), Image.BILINEAR), np.float32) / 255  # noqa: E731


class Pictures:
    def __len__(self):
        return len(unlabelled)

    def __getitem__(self, i):
        return small(unlabelled[int(i)])


for seed in SEEDS:
    for variant in VARIANTS:
        t0, rng = time.time(), np.random.default_rng(seed)
        cortex = VisualCortex2(seed=seed, kinds=(256,), scaled=False)
        pics = Pictures()
        if variant.endswith("_d"):
            cortex.develop(pics, rng, log=lambda s: print("   " + s, flush=True))
        else:
            cortex.layers[0].settle((pics[i] for i in rng.choice(len(pics), 800, replace=False)), rng)
        alike, pairs, groups = near_copies(cortex.layers[0].W[0])
        cortex.settle_relay([pics[i] for i in rng.choice(len(pics), 6000, replace=False)], GRID, SHRINK)
        cortex.grid = GRID
        relay, cortex.relay = cortex.relay, None
        for name, X in (("train", Xtr), ("test", Xte)):
            S = np.zeros((len(X), GRID * GRID * 256), np.float16)
            for i, x in enumerate(X):
                S[i] = cortex.whole(small(x), GRID)
            np.save(f"out/bench/cifar10_{variant}_s{seed}_{name}.npy", S)
        S = S.astype(np.float32)
        through = (S - relay.mu) @ relay.Z
        cortex.relay = relay
        pickle.dump(cortex, open(f"out/bench/cortex_{variant}_s{seed}.pkl", "wb"))
        print(f"seed {seed} {variant}: two kinds' weights alike {alike:.3f}, pairs above 0.9: {pairs}, different filters {groups} | "
              + P.read_code("summary", S[:1500], yte[:1500]) + " | " + P.read_code("through the relay", through[:1500], yte[:1500]) + f" ({time.time() - t0:.0f}s)", flush=True)
