"""The whole CIFAR-10 stream as the soft-rule layer codes it (queue E 27; docs/VISION.md, 2026-11-25).

experiments/softhebb_probe.py kept codes for 10,000 training pictures only. The paper's headline is read on the
whole stream (50,000 pictures, 2,000 of the 10,000 test pictures), so here the layers it kept
(out/bench/cortex_{sh,shr}_s{seed}.pkl: 256 kinds developed by the soft rule, and the same layer as born) code all
of it. Nothing is developed again; the relay is the one settled there.
    SEEDS=0 python experiments/bench_soft_codes.py
        ->  out/bench/cifar10_{shfull,shrfull}_s{seed}_{train,test}.npy, cortex_{shfull,shrfull}_s{seed}.pkl
"""
import os, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import softhebb_probe as P  # noqa: E402

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
NAMES = os.environ.get("NAMES", "sh,shr").split(",")

if __name__ == "__main__":
    Xtr = np.concatenate([P.load(f"{P.DATA}/cifar-10-batches-py/data_batch_{i}") for i in range(1, 6)])
    Xte = P.load(f"{P.DATA}/cifar-10-batches-py/test_batch")
    for seed in SEEDS:
        for name in NAMES:
            t0 = time.time()
            kept = pickle.load(open(f"out/bench/cortex_{name}_s{seed}.pkl", "rb"))
            standard = P.Standard()
            standard.mean, standard.var = kept.standard
            for part, X in (("train", Xtr), ("test", Xte)):
                out = np.zeros((len(X), P.GRID * P.GRID * len(kept.W)), np.float16)
                for a in range(0, len(X), 1000):
                    out[a: a + 1000] = P.coded(kept.W, standard, X[a: a + 1000])
                np.save(f"out/bench/cifar10_{name}full_s{seed}_{part}.npy", out)
            old = np.asarray(np.load(f"out/bench/cifar10_{name}_s{seed}_train.npy", mmap_mode="r")[:500], np.float32)
            new = np.asarray(np.load(f"out/bench/cifar10_{name}full_s{seed}_train.npy", mmap_mode="r")[:500], np.float32)
            link = f"out/bench/cortex_{name}full_s{seed}.pkl"
            if not os.path.exists(link):
                os.symlink(f"cortex_{name}_s{seed}.pkl", link)
            print(f"seed {seed} {name}: the whole stream coded ({time.time() - t0:.0f}s) | the first 500 against the codes kept before: "
                  f"largest difference {float(np.abs(old - new).max()):.2g}", flush=True)
