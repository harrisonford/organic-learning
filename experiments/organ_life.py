"""The progressive eye lives on photographs; do similar images get similar outputs?

The organ looks at CIFAR-10 training images and learns everything (groups,
predictions, motor) without labels. Every so often it is tested, frozen:
reference images and test images go through it, and for each tick we ask
whether a test image's nearest neighbours in the organ's output space share
its class (fine: 10 classes) or its kind (coarse: animal vs vehicle).
Labels are used only for this reading.

    python experiments/organ_life.py [images] [out.jsonl]
    MOTOR=settle|curious|random|center  TRACE=0.5  SEED=0  EPOCHS=1
"""

import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic import cifar  # noqa: E402
from organic.peye import Parts, ProgressiveEye, ProgressiveEye2, develop_parts  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
out_path = sys.argv[2] if len(sys.argv) > 2 else "out/organ.jsonl"
MOTOR = os.environ.get("MOTOR", "settle")
TRACE = float(os.environ.get("TRACE", 0.5))
SEED = int(os.environ.get("SEED", 0))
EVERY = int(os.environ.get("EVERY", 10000))
REF, TEST = int(os.environ.get("REF", 5000)), int(os.environ.get("TEST", 2000))

xtr, ytr = cifar.load("train")
xte, yte = cifar.load("test")
rng = np.random.default_rng(SEED)
ORGAN = int(os.environ.get("ORGAN", 1))
if ORGAN == 2:
    parts_file = os.path.join(os.path.dirname(__file__), "..", "data", "parts.npy")
    if os.path.exists(parts_file):
        parts = Parts()
        parts.W = np.load(parts_file)
        parts.frozen = True
    else:  # development on unlabeled photos, then frozen and kept as part of the organ
        parts = develop_parts(xtr[rng.permutation(len(xtr))[:10000]], np.random.default_rng(123))
        np.save(parts_file, parts.W)
    eye = ProgressiveEye2(parts, motor=MOTOR, trace=TRACE, seed=SEED)
else:
    eye = ProgressiveEye(motor=MOTOR, trace=TRACE, seed=SEED)
ref_idx = np.arange(REF)  # reference images (also seen in life, like familiar things)
test_idx = np.arange(TEST)


def embed(images):
    """Organ outputs after every tick: (ticks, n, dim)."""
    outs, paths = [], []
    for im in images:
        o, path, _ = eye.look(im, learn=False, explore=False)
        outs.append(np.stack(o))
        paths.append(path)
    return np.stack(outs, axis=1), paths


def knn(ref, ref_y, q, q_y, k=10):
    ref = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-8)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    sim = q @ ref.T
    nn = np.argpartition(-sim, k, axis=1)[:, :k]
    votes = ref_y[nn]
    pred = np.array([np.bincount(v, minlength=ref_y.max() + 1).argmax() for v in votes])
    return float((pred == q_y).mean())


def evaluate():
    R, _ = embed(xtr[ref_idx])
    Q, paths = embed(xte[test_idx])
    fine = [knn(R[t], ytr[ref_idx], Q[t], yte[test_idx]) for t in range(R.shape[0])]
    coarse = [knn(R[t], cifar.ANIMAL[ytr[ref_idx]], Q[t], cifar.ANIMAL[yte[test_idx]]) for t in range(R.shape[0])]
    targets = np.bincount(np.array(paths).ravel(), minlength=25) / np.array(paths).size
    return fine, coarse, targets


if os.environ.get("BASELINE"):
    flat = lambda x: (x - x.mean(axis=(1, 2, 3), keepdims=True)).reshape(len(x), -1)  # noqa: E731
    print("raw pixels kNN: fine %.3f coarse %.3f" % (knn(flat(xtr[ref_idx]), ytr[ref_idx], flat(xte[test_idx]), yte[test_idx]), knn(flat(xtr[ref_idx]), cifar.ANIMAL[ytr[ref_idx]], flat(xte[test_idx]), cifar.ANIMAL[yte[test_idx]])))
    sys.exit()

log = open(out_path, "w")
t0 = time.time()
epochs = int(os.environ.get("EPOCHS", 1))
age = 0
surprise_window = []
checkpoints = sorted(set([1000] + list(range(EVERY, N * epochs + 1, EVERY))))
for epoch in range(epochs):
    order = rng.permutation(len(xtr))[:N]
    for i in order:
        _, path, err = eye.look(xtr[i])
        surprise_window.append(np.mean(err, axis=0))
        age += 1
        if age in checkpoints:
            fine, coarse, targets = evaluate()
            s = np.mean(surprise_window[-2000:], axis=0)
            rec = {"age": age, "motor": MOTOR, "trace": TRACE, "seed": SEED, "fine": fine, "coarse": coarse,
                   "assemblies": eye.report(),
                   "surprise_mid_fine": [float(v) for v in s], "gaze_targets": [float(v) for v in targets], "seconds": round(time.time() - t0)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"age {age:>6} fine/tick " + " ".join(f"{v:.2f}" for v in fine) + " | coarse/tick " + " ".join(f"{v:.2f}" for v in coarse)
                  + f" | surprise mid {s[0]:.2f} fine {s[1]:.2f} | {eye.report()} | {time.time() - t0:.0f}s", flush=True)
pickle.dump(eye, open(out_path.replace(".jsonl", ".pkl"), "wb"))
