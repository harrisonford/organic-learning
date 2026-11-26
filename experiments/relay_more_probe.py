"""Is the relay what keeps the word cells from the development's worth? (docs/VISION.md, 2026-11-26 08:00)

Across five codes the word cells gained from development only where the developed code, after the relay, still
spread over about 500 directions (CIFAR at 32: our rule 535, SAILnet 497); where it spread over 300-400 (photographs
393, enlarged CIFAR 354, SAILnet on photographs 307) they gained nothing, while ridge gained every time. The relay
is settled on 6,000 unlabelled pictures for a code of 4,096 numbers, with a little shrinkage: an estimate that may
leave a developed code, whose variance sits in fewer directions, less evenly spread than a code as born. Here the
same codes, the relay settled on MANY unlabelled pictures in place of 6,000 (more of the experience the learner
already has; nothing else changes), as the variant {VARIANT}m{MANY/1000}k with the codes linked, not copied.

    SET=photos VARIANT=ph_d SEEDS=0 MANY=20000 python experiments/relay_more_probe.py
        -> out/bench/cortex_ph_dm20k_s0.pkl, photos_ph_dm20k_s0_{train,test}.npy (links)
    then  SET=photos VARIANT=ph_dm20k CORTEX_SEED=0 SEEDS=0 TESTS=4000 ARMS=cells+relay,ridge+relay python experiments/bench_stream.py
"""
import os, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import softhebb_probe as P  # noqa: E402
from organic.cortex2 import Relay  # noqa: E402

SET, VARIANT = os.environ.get("SET", "photos"), os.environ.get("VARIANT", "ph_d")
SEEDS, MANY = [int(x) for x in os.environ.get("SEEDS", "0").split(",")], int(os.environ.get("MANY", 20000))
GRID, SHRINK = 4, 0.0005
assert SET == "photos", "only the photo stream so far"
thumbs, dev = np.load(f"{P.DATA}/coco_thumbs/train2017_64.npy", mmap_mode="r"), np.load("out/bench/photos_dev_rows.npy")
yte = np.load("out/bench/photos_test_labels.npy")

for seed in SEEDS:
    t0, name = time.time(), f"{VARIANT}m{MANY // 1000}k"
    cortex = pickle.load(open(f"out/bench/cortex_{VARIANT}_s{seed}.pkl", "rb"))
    old, cortex.relay = cortex.relay, None
    picks = np.sort(np.random.default_rng(seed + 11).choice(dev, MANY, replace=False))
    S = np.zeros((MANY, GRID * GRID * cortex.layers[0].kinds), np.float32)
    for i, r in enumerate(picks):
        S[i] = cortex.whole(np.asarray(thumbs[r], np.float32) / 255, GRID)
    relay = Relay(SHRINK).settle(S)
    del S
    cortex.relay = relay
    pickle.dump(cortex, open(f"out/bench/cortex_{name}_s{seed}.pkl", "wb"))
    for part in ("train", "test"):
        link = f"out/bench/{SET}_{name}_s{seed}_{part}.npy"
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(f"{SET}_{VARIANT}_s{seed}_{part}.npy", link)
    T = np.asarray(np.load(f"out/bench/{SET}_{VARIANT}_s{seed}_test.npy", mmap_mode="r")[:1500], np.float32)
    print(f"seed {seed} {name}: " + P.read_code("through the relay of 6,000", (T - old.mu) @ old.Z, yte[:1500])
          + " | " + P.read_code(f"through the relay of {MANY:,}", (T - relay.mu) @ relay.Z, yte[:1500]) + f" ({time.time() - t0:.0f}s)", flush=True)
