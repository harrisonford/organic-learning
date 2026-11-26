"""SAILnet in our cortex's place (queue E 27; docs/VISION.md, 2026-11-25 09:45).

experiments/sailnet_probe.py ran the published layer (Zylberberg, Murphy & DeWeese 2011; the authors' code, every
number theirs) on the soft layer's raw colour windows: all cells ended as one filter, because the rule's numbers
are in the units of a whitened, mean-free patch at unit deviation. Our cortex's first layer gives its filters
exactly that kind of input (organic.cortex2.Layer2._drive: the window's mean removed, decorrelated by the settled Z,
unit length). Here the same layer, unchanged, develops on those windows and its feed-forward weights stand where
our hard rule's filters stand; the benchmark's own pipeline codes the pictures.
    input        the cortex's decorrelated window (75 numbers, unit length) times sqrt(75): a component then has
                 unit deviation as a pixel of the code's patches has (a change of units, said)
    windows      as our own development takes them: of an unlabelled CIFAR-100 picture at 64 x 64, among the
                 places whose contrast is above 0.3 of the picture's largest; 10 windows from each of 10 pictures
                 make a batch of 100
    everything else   sailnet_probe.activities and its three rules, 25,000 batches at the code's rates and 10,000
                 at the paper's last
    sailc / sailcu   Q as developed / the same at unit length (as every filter of our cortex is)
The base is out/bench/cortex_random_s{seed}.pkl (filters as born, the decorrelation settled); the codes, the relay
(6,000 unlabelled pictures) and the stream are the benchmark's (experiments/cortex_norm_probe.py, bench_stream.py).
    SIZE=32      (2026-11-25 18:50) the same on CIFAR at its own size, where a window holds 47 directions and not
                 the 22 of a blown-up picture (experiments/upsample_probe.py): the base is cortex_nat_r_s{seed}.pkl
                 (experiments/cortex_native_probe.py), the names sailn / sailnu, the weights sailn_weights_s{seed}.npz
    SET=photos   (2026-11-26 00:35) the same on the photo stream of experiments/bench_photos.py: real photographs at
                 64 x 64, nothing enlarged. The layer develops on the unlabelled photographs the benchmark's cortex
                 develops on, the base is cortex_ph_r_s{seed}.pkl, the names sailp / sailpu, the weights
                 sailp_weights_s{seed}.npz, the codes photos_{sailp,sailpu}_s{seed}_{train,test}.npy
    STAGE=develop SEEDS=0 python experiments/sailnet_cortex_probe.py   -> out/bench/sailc_weights_s{seed}.npz
    STAGE=code    SEEDS=0 python experiments/sailnet_cortex_probe.py   -> out/bench/cifar10_{sailc,sailcu}_s{seed}_*.npy,
                                                                         cortex_{sailc,sailcu}_s{seed}.pkl
"""
import copy, os, pickle, sys, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import softhebb_probe as P  # noqa: E402
from bcm_probe import near_copies  # noqa: E402
from sailnet_probe import activities, PHASES, KINDS, BATCH, TARGET  # noqa: E402
from organic.cortex2 import _windows, _contrast  # noqa: E402

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
STAGE, GRID, SHRINK, NRELAY = os.environ.get("STAGE", "develop"), 4, 0.0005, 6000
SIZE, SET = int(os.environ.get("SIZE", 64)), os.environ.get("SET", "cifar10")
NAME, BASE = ("sailp", "ph_r") if SET == "photos" else ("sailc", "random") if SIZE == 64 else ("sailn", "nat_r")
if SIZE == 64:
    big = lambda x: np.asarray(Image.fromarray(x).resize((64, 64), Image.BILINEAR), np.float32) / 255  # noqa: E731
else:
    assert SIZE == 32
    big = lambda x: x.astype(np.float32) / 255  # noqa: E731  (the picture as it is)


def strong_windows(layer, x, rng, k):
    """k of the picture's windows as the layer's filters get them, among those our own development learns from."""
    v, e = _contrast(_windows(x, layer.window))
    v, e = v.reshape(-1, v.shape[2]), e.ravel()
    strong = np.nonzero(e > 0.3 * e.max())[0] if e.max() > 1e-6 else np.arange(0)
    if not len(strong):
        return np.zeros((0, v.shape[1]), np.float32)
    v = v[rng.choice(strong, min(k, len(strong)), replace=False)] @ layer.Z[0]
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def develop(layer, pictures, rng):
    Q = rng.standard_normal((KINDS, 75))
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    W, theta, born, t0, b = np.zeros((KINDS, KINDS)), 2.0 * np.ones(KINDS), Q.copy(), time.time(), 0
    total = sum(p[0] for p in PHASES)
    marks, rate, both, off = {100, 1000, 5000, 10000, PHASES[0][0], total}, [], [], ~np.eye(KINDS, dtype=bool)
    for batches, alpha, beta, gamma in PHASES:
        for _ in range(batches):
            rows = []
            while sum(len(r) for r in rows) < BATCH:
                rows.append(strong_windows(layer, big(pictures[rng.integers(len(pictures))]), rng, 10))
            X = np.concatenate(rows)[:BATCH].astype(np.float64) * np.sqrt(75.0)
            n = activities(X, Q, W, theta)
            C = n.T @ n / BATCH
            W += alpha * (C - TARGET ** 2)
            np.fill_diagonal(W, 0.0)
            np.maximum(W, 0.0, out=W)
            Q += beta * (n.T @ X) / BATCH - beta * (n * n).sum(axis=0)[:, None] * Q / BATCH
            theta += gamma * (n.mean(axis=0) - TARGET)
            rate.append(n.mean(axis=0))
            both.append(C[off].mean())
            b += 1
            if b in marks:
                R, lengths = np.array(rate[-200:]).mean(axis=0), np.linalg.norm(Q, axis=1)
                alike, pairs, groups = near_copies(Q)
                print(f"   after {b:>6} batches (rates {alpha:g}, {beta:g}, {gamma:g}): spikes an image, per cell {R.min():.3f} / {R.mean():.3f} / {R.max():.3f} (least, mean, most; the target {TARGET}) | "
                      f"two cells firing to one image {float(np.mean(both[-200:])):.5f} (the target {TARGET ** 2:.5f}) | thresholds {theta.min():.2f} / {theta.mean():.2f} / {theta.max():.2f} | "
                      f"inhibition {W.mean():.3f} on average, most {W.max():.2f}, pairs with any {float((W > 0).mean()):.2f} | weight lengths {lengths.min():.2f} / {lengths.mean():.2f} / {lengths.max():.2f} | "
                      f"two cells' weights alike {alike:.3f}, pairs above 0.9: {pairs}, different filters {groups} ({time.time() - t0:.0f}s)", flush=True)
    return born.astype(np.float32), Q.astype(np.float32), W.astype(np.float32), theta


def photos():
    """The photo stream's unlabelled photographs, its pictures in order and its tests' labels (already 64 x 64)."""
    import json
    assert SIZE == 64
    thumbs, dev = np.load(f"{P.DATA}/coco_thumbs/train2017_64.npy", mmap_mode="r"), np.load("out/bench/photos_dev_rows.npy")
    coco = np.load(f"{P.DATA}/cococrops/pictures_64.npy", mmap_mode="r")
    many = sum(1 for _ in open(f"{P.DATA}/commons/pictures.jsonl"))
    commons = np.memmap(f"{P.DATA}/commons/pictures_64.u8", np.uint8, "r", shape=(many, 64, 64, 3))

    class Rows:
        def __init__(self, source, rows):
            self.source, self.rows = source, rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            if self.source is not None:
                return np.asarray(self.source[self.rows[int(i)]])
            which, row = self.rows[int(i)]
            return np.asarray((coco, commons)[which][row])

    return (Rows(thumbs, dev), Rows(None, np.load("out/bench/photos_train_rows.npy")), Rows(None, np.load("out/bench/photos_test_rows.npy")),
            np.load("out/bench/photos_test_labels.npy"))


if __name__ == "__main__":
    if SET == "photos":
        unlabelled, Xtr, Xte, yte = photos()
    else:
        unlabelled = P.load(f"{P.DATA}/cifar-100-python/train")
    for seed in SEEDS:
        base = pickle.load(open(f"out/bench/cortex_{BASE}_s{seed}.pkl", "rb"))
        base.relay = None
        if STAGE == "develop":
            born, Q, W, theta = develop(base.layers[0], unlabelled, np.random.default_rng(7000 + seed))
            np.savez(f"out/bench/{NAME}_weights_s{seed}.npz", born=born, W=Q, lateral=W, theta=theta)
            alike, pairs, groups = near_copies(Q)
            a0, p0, g0 = near_copies(base.layers[0].W[0])
            print(f"seed {seed} {NAME}: two cells' weights alike {alike:.3f}, pairs above 0.9: {pairs}, different filters {groups} of {KINDS} (the cortex's filters as born: {a0:.3f}, {p0}, {g0})", flush=True)
            continue
        if SET != "photos":
            Xtr, Xte = P.load(f"{P.DATA}/cifar-10-batches-py/data_batch_1"), P.load(f"{P.DATA}/cifar-10-batches-py/test_batch")[:2000]
            yte = np.load("out/bench/cifar10_test_labels.npy")[:2000]
        Q = np.load(f"out/bench/{NAME}_weights_s{seed}.npz")["W"]
        settle = [big(unlabelled[i]) for i in np.random.default_rng(seed + 7).choice(len(unlabelled), NRELAY, replace=False)]
        for name, weights in ((NAME, Q), (NAME + "u", Q / np.linalg.norm(Q, axis=1, keepdims=True))):
            t0, cortex = time.time(), copy.deepcopy(base)
            cortex.layers[0].W[0] = weights.astype(np.float32)
            cortex.settle_relay(settle, GRID, SHRINK)
            relay, cortex.relay = cortex.relay, None
            for part, X in (("train", Xtr), ("test", Xte)):
                S = np.lib.format.open_memmap(f"out/bench/{SET}_{name}_s{seed}_{part}.npy", "w+", np.float16, (len(X), GRID * GRID * KINDS))
                for i in range(len(X)):
                    S[i] = cortex.whole(big(X[i]), GRID)
                S.flush()
            S = np.asarray(S[:1500], np.float32)
            through = (S - relay.mu) @ relay.Z
            print(f"seed {seed} " + P.read_code(f"{name:6} summary", S, yte[:1500]) + " | " + P.read_code("through the relay", through, yte[:1500]) + f" ({time.time() - t0:.0f}s)", flush=True)
            cortex.relay = relay
            pickle.dump(cortex, open(f"out/bench/cortex_{name}_s{seed}.pkl", "wb"))
            del relay, through, S, cortex
