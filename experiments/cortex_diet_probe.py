"""A blurred start: the cortex's filters develop on what a maturing eye sees (queue E 29b; docs/RESEARCH_ARCH.md).

Our filters develop on sharp, coloured pictures from the first step. No eye starts so. Before it opens, the retina
drives the cortex with its own slow waves of activity; a newborn then resolves about a fortieth of the detail an
adult does and little colour, and acuity rises over months (roughly evenly in the logarithm). Models: filters like
V1's come from such spontaneous patterns alone (Albert, Schnabel & Field 2008); a start without fine detail gives
more reliance on shape and more robustness (Vogelsang et al. 2018; Lu et al. 2025; Ligeralde et al. 2023).
Here the same rule (winner-take-all Hebbian with a conscience), the same number of steps, the same decorrelation
of the windows and the same first weights as the present cortex of this seed; only what is seen while developing
differs:
    now    sharp coloured pictures from the first step (the present cortex, out/bench/cortex_developed_s<seed>.pkl)
    diet   first waves (blobs of slowly varying noise with soft edges, no colour), then pictures whose blur falls
           evenly in the logarithm from a fortieth of full detail to full detail, colour coming in with it
Stand-ins, not from biology: the share of steps given to waves (an eighth), and that the whole schedule fits the
development's 16,000 steps. The order (waves, coarse, fine) is what is established.
Printed: the filters' mean spatial frequency and how much of their pattern is colour. Kept for
experiments/bench_stream.py (N=10000): codes as `diet` (sharp test pictures) and, for both cortices, with the test
pictures blurred (`dietb`, `norm0b`: what a learner raised on sharp pictures does when the world is less sharp).
    SEEDS=0 python experiments/cortex_diet_probe.py
"""
import copy, os, pickle, sys, time
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.cortex2 import VisualCortex2, Relay  # noqa: E402

DATA = os.path.expanduser("~/datasets")
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
STEPS, WAVES = int(os.environ.get("STEPS", 16000)), float(os.environ.get("WAVES", 0.125))
NTRAIN, NTEST, NRELAY = int(os.environ.get("NTRAIN", 10000)), int(os.environ.get("NTEST", 2000)), int(os.environ.get("NRELAY", 6000))
GRID, SHRINK, TEST_BLUR = 4, 0.0005, 1.0


def load(path):
    d = pickle.load(open(path, "rb"), encoding="bytes")
    return d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)


big = lambda x: np.asarray(Image.fromarray(x).resize((64, 64), Image.BILINEAR), np.float32) / 255  # noqa: E731
blur = lambda im, s: gaussian_filter(im, (s, s, 0), mode="nearest") if s > 0.05 else im  # noqa: E731
unlabelled = load(f"{DATA}/cifar-100-python/train")
Xtr, Xte = load(f"{DATA}/cifar-10-batches-py/data_batch_1")[:NTRAIN], load(f"{DATA}/cifar-10-batches-py/test_batch")[:NTEST]


def wave(rng):
    """A spontaneous pattern: slowly varying noise cut into blobs with soft edges, without colour."""
    f = gaussian_filter(rng.standard_normal((64, 64)).astype(np.float32), 6, mode="wrap")
    b = gaussian_filter((f > np.quantile(f, rng.uniform(0.3, 0.7))).astype(np.float32), 1.5, mode="wrap")
    return np.repeat(b[..., None], 3, axis=2)


def seen(n, rng):
    """What the developing eye sees at step n."""
    if n < WAVES * STEPS:
        return wave(rng)
    age = (n - WAVES * STEPS) / ((1 - WAVES) * STEPS)  # 0: eyes just open, 1: grown
    im = big(unlabelled[rng.integers(len(unlabelled))])
    sigma = 0.5 * 40 ** (1 - age) - 0.5  # a fortieth of full detail at first (about 20 pixels), none at the end
    im = blur(im, sigma)
    return im.mean(axis=2, keepdims=True) + age * (im - im.mean(axis=2, keepdims=True))  # colour comes in with detail


def filters(layer):
    """(mean spatial frequency in cycles a pixel, share of the pattern that is colour) of the filters as patterns of light."""
    P = (layer.W[0] @ layer.Z[0].T).reshape(-1, 5, 5, 3)
    fy, fx = np.meshgrid(np.fft.fftfreq(5), np.fft.fftfreq(5), indexing="ij")
    power = (np.abs(np.fft.fft2(P, axes=(1, 2))) ** 2).sum(axis=3)
    freq = (power * np.hypot(fy, fx)).sum(axis=(1, 2)) / power.sum(axis=(1, 2))
    grey = P.mean(axis=3, keepdims=True)
    return float(freq.mean()), float((((P - grey) ** 2).sum(axis=(1, 2, 3)) / (P ** 2).sum(axis=(1, 2, 3))).mean())


def keep(name, seed, cortex, relay, test_blur):
    for part, X in (("train", Xtr), ("test", Xte)):
        path = f"out/bench/cifar10_{name}_s{seed}_{part}.npy"
        if part == "train" and test_blur and os.path.exists(f"out/bench/cifar10_{name[:-1]}_s{seed}_train.npy"):
            if os.path.lexists(path):
                os.remove(path)
            os.symlink(f"cifar10_{name[:-1]}_s{seed}_train.npy", path)  # the same lessons; only the test differs
            continue
        out = np.zeros((len(X), GRID * GRID * 256), np.float16)
        for i, x in enumerate(X):
            out[i] = cortex.whole(blur(big(x), TEST_BLUR if (part == "test" and test_blur) else 0), GRID)
        np.save(path, out)
    c = copy.deepcopy(cortex)
    c.relay = relay
    pickle.dump(c, open(f"out/bench/cortex_{name}_s{seed}.pkl", "wb"))


for seed in SEEDS:
    t0 = time.time()
    base = pickle.load(open(f"out/bench/cortex_developed_s{seed}.pkl", "rb"))
    cortex = VisualCortex2(seed=seed, kinds=(256,), scaled=False)  # the same first weights as the present cortex
    layer = cortex.layers[0]
    layer.Z = [z.copy() for z in base.layers[0].Z]  # and the same decorrelation of the windows
    rng = np.random.default_rng(seed + 3)
    for n in range(STEPS):
        layer.learn(seen(n, rng), rng, rate=0.05 * (1 - n / STEPS) + 0.005)
    f0, c0 = filters(base.layers[0])
    f1, c1 = filters(layer)
    print(f"seed {seed}: developed on the diet ({time.time() - t0:.0f}s; wins min/max {int(layer.wins[0].min())}/{int(layer.wins[0].max())}) | "
          f"filters' mean spatial frequency: now {f0:.3f}, diet {f1:.3f} cycles a pixel | share of the pattern that is colour: now {c0:.3f}, diet {c1:.3f}", flush=True)
    rs = np.random.default_rng(seed + 7)
    S = np.stack([cortex.whole(big(unlabelled[i]), GRID) for i in rs.choice(len(unlabelled), NRELAY, replace=False)])
    relay = Relay(SHRINK).settle(S)
    del S
    cortex.grid = GRID
    keep("diet", seed, cortex, relay, False)
    keep("dietb", seed, cortex, relay, True)
    now_relay, base.relay = pickle.load(open(f"out/bench/cortex_norm0_s{seed}.pkl", "rb")).relay, None
    keep("norm0b", seed, base, now_relay, True)
    pickle.dump(cortex, open(f"out/bench/cortex_dietfull_s{seed}.pkl", "wb"))
    print(f"seed {seed}: codes kept as diet, dietb, norm0b ({time.time() - t0:.0f}s)", flush=True)
