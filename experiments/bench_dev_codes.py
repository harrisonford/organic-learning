"""The unlabelled pictures the benchmark's relay was settled on, kept as codes (for experiments/relay_local_probe.py).

The batch relay of the norm0 cortices (experiments/cortex_norm_probe.py, variant "now") was settled on 6,000
CIFAR-100 training pictures, chosen by default_rng(seed + 7). Their summaries were not kept. Here the same cortex
codes the same 6,000 again (rows 0 to 5,999, in the relay's own order) and MORE others after them, so that a relay
learned by a local rule can develop on the very pictures the batch relay had, or on more of the same kind, and
never on the stream's own.
    SEEDS=0 python experiments/bench_dev_codes.py   ->  out/bench/cifar100_norm0_s{seed}_dev.npy
"""
import os, pickle, sys, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA = os.path.expanduser("~/datasets")
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
NRELAY, MORE, GRID = 6000, int(os.environ.get("MORE", 14000)), 4

if __name__ == "__main__":
    d = pickle.load(open(f"{DATA}/cifar-100-python/train", "rb"), encoding="bytes")
    unlabelled = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    big = lambda x: np.asarray(Image.fromarray(x).resize((64, 64), Image.BILINEAR), np.float32) / 255  # noqa: E731
    for seed in SEEDS:
        t0 = time.time()
        cortex = pickle.load(open(f"out/bench/cortex_developed_s{seed}.pkl", "rb"))
        relay = pickle.load(open(f"out/bench/cortex_norm0_s{seed}.pkl", "rb")).relay
        cortex.relay = None
        first = np.random.default_rng(seed + 7).choice(len(unlabelled), NRELAY, replace=False)  # the relay's own
        rest = np.setdiff1d(np.arange(len(unlabelled)), first)
        rest = rest[np.random.default_rng(seed + 70).permutation(len(rest))[:MORE]]
        out = np.zeros((NRELAY + len(rest), GRID * GRID * cortex.layers[0].kinds), np.float16)
        for k, i in enumerate(np.concatenate([first, rest])):
            out[k] = cortex.whole(big(unlabelled[i]), GRID)
        np.save(f"out/bench/cifar100_norm0_s{seed}_dev.npy", out)
        mu = np.asarray(out[:NRELAY], np.float32).mean(axis=0)
        print(f"seed {seed}: {len(out)} unlabelled pictures coded ({time.time() - t0:.0f}s) | the first {NRELAY} against the relay's own "
              f"average: largest difference {float(np.abs(mu - relay.mu).max()):.2g} (the average's largest number {float(np.abs(relay.mu).max()):.2g})", flush=True)
