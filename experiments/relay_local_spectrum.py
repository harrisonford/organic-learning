"""Where the relay circuit sits as it learns, direction by direction (queue E 26/28; docs/VISION.md, 2026-11-25).

The circuit of experiments/relay_local_probe.py should bring every direction of the summary whose variance is
above 1 (in its units: the mean variance) down to 1 and leave the others. Read here, as it sees the unlabelled
pictures (out/bench/cifar100_norm0_s{seed}_dev.npy), on 1,000 test pictures and along the eigen-directions of the
first 6,000 unlabelled pictures' covariance:
    the output's variance along directions 1, 3, 10, 30, 100, 183, 300, 1000, 3000 (largest first)
    beside what the rule's resting point gives there (1 above the floor, the input's own variance under it)
    the squared length of one picture's activity (what the rule's step along a picture is multiplied by)
    RATE=1e-3 SEEDS=0 python experiments/relay_local_spectrum.py
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import relay_local_probe as P  # noqa: E402

AT = [int(x) for x in os.environ.get("AT", "500,1000,2000,4000,6000,10000,15000,20000").split(",")]
PICK = [1, 3, 10, 30, 100, 183, 300, 1000, 3000]

if __name__ == "__main__":
    for seed in P.SEEDS:
        dev = np.load(f"out/bench/cifar100_norm0_s{seed}_dev.npy", mmap_mode="r")
        test = np.asarray(np.load(f"out/bench/cifar10_norm0_s{seed}_test.npy")[:1000], np.float32)
        V = np.asarray(dev[:6000], np.float32)
        C = np.cov((V - V.mean(axis=0)).T).astype(np.float64)
        d, E = np.linalg.eigh(C)
        d, E = (np.maximum(d, 0) / (np.trace(C) / len(C)))[::-1], E[:, ::-1][:, [k - 1 for k in PICK]].astype(np.float32)
        del C, V
        print(f"seed {seed} rate {P.RATE:g} | the input's variance along directions {PICK}: " + " ".join(f"{d[k - 1]:.3g}" for k in PICK), flush=True)
        print(f"seed {seed} the resting point gives there: " + " ".join(f"{min(d[k - 1], 1.0):.3g}" for k in PICK), flush=True)
        c, t0, seen = P.Circuit(dev.shape[1], np.random.default_rng(2000 + seed)), time.time(), 0
        for a in range(0, max(AT), P.BLOCK):
            c.learn(np.asarray(dev[a: a + P.BLOCK], np.float32))
            seen = a + P.BLOCK
            if seen in AT:
                Y = c.code(test)
                Y = Y - Y.mean(axis=0)
                print(f"seed {seed} after {seen:6} pictures: " + " ".join(f"{v:.3g}" for v in (Y @ E).var(axis=0))
                      + f" | a picture's activity, squared length {float((Y * Y).sum(axis=1).mean()):.0f} ({time.time() - t0:.0f}s)", flush=True)
