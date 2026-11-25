"""Real photographs as a one-pass stream (docs/VISION.md, 2026-11-25 20:30; Kris: "should use all type of pictures").

CIFAR's pictures are 32 pixels across and the learner's eye is 64: enlarged, a 5 x 5 window of one holds half the
directions of a real picture's (experiments/upsample_probe.py). Here the same benchmark on pictures that are real
at the eye's size, from what is on disk:
    labelled     ~/datasets/cococrops   (objects cut from COCO train2017 photographs, 64 x 64, named with a word)
                 ~/datasets/commons     (Wikimedia Commons photographs, 64 x 64, named with a word)
    unlabelled   ~/datasets/coco_thumbs (COCO train2017 photographs whole, 64 x 64): the cortex develops on these
The classes are the words with at least 1,000 COCO crops (33); a word's Commons pictures are added to it. A
twentieth of the COCO photographs (every crop of them) and a twentieth of the Commons pictures are held out as
tests; no photograph gives crops to both sides, and the held-out photographs are not among the unlabelled ones
the cortex develops on. 50,000 of the rest, drawn once, are the stream (the size of CIFAR-10's).

    STAGE=set   python experiments/bench_photos.py     -> out/bench/photos_names.json, photos_{train,test}_labels.npy,
                                                          photos_{train,test}_rows.npy (source 0 COCO / 1 Commons, row),
                                                          photos_dev_rows.npy
    STAGE=code SEEDS=0 VARIANTS=ph_r,ph_d python experiments/bench_photos.py
                                                       -> out/bench/photos_{variant}_s{seed}_{train,test}.npy,
                                                          cortex_{variant}_s{seed}.pkl   (ph_r as born, ph_d developed
                                                          by our hard rule: organic.cortex2.VisualCortex2.develop)
    then        SET=photos VARIANT=ph_d CORTEX_SEED=0 SEEDS=0 TESTS=4000 python experiments/bench_stream.py
"""
import collections, json, os, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA, OUT = os.path.expanduser("~/datasets"), "out/bench"
STAGE = os.environ.get("STAGE", "set")
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
VARIANTS = os.environ.get("VARIANTS", "ph_r,ph_d").split(",")
GRID, SHRINK, STREAM, LEAST = 4, 0.0005, 50000, 1000

coco = np.load(f"{DATA}/cococrops/pictures_64.npy", mmap_mode="r")
coco_meta = [json.loads(l) for l in open(f"{DATA}/cococrops/pictures.jsonl")]
commons_meta = [json.loads(l) for l in open(f"{DATA}/commons/pictures.jsonl")]
commons = np.memmap(f"{DATA}/commons/pictures_64.u8", np.uint8, "r", shape=(len(commons_meta), 64, 64, 3))
thumbs = np.load(f"{DATA}/coco_thumbs/train2017_64.npy", mmap_mode="r")
thumb_ids = json.load(open(f"{DATA}/coco_thumbs/train2017_ids.json"))


def picture(source, row):
    return np.asarray((coco, commons)[source][row], np.float32) / 255


if STAGE == "set":
    rng = np.random.default_rng(2026)
    count = collections.Counter(m["word"] for m in coco_meta)
    names = sorted(w for w, n in count.items() if n >= LEAST)
    images = sorted({m["image"] for m in coco_meta})
    held = set(rng.choice(images, len(images) // 20, replace=False).tolist())
    held_commons = set(rng.choice(len(commons_meta), len(commons_meta) // 20, replace=False).tolist())
    train, test = [], []
    for row, m in enumerate(coco_meta):
        if m["word"] in names:
            (test if m["image"] in held else train).append((0, row, names.index(m["word"])))
    for row, m in enumerate(commons_meta):
        if m["word"] in names:
            (test if row in held_commons else train).append((1, row, names.index(m["word"])))
    train = np.array(train)[rng.permutation(len(train))[:STREAM]]
    test = np.array(test)
    dev = np.array([i for i, image in enumerate(thumb_ids) if image not in held])
    json.dump(names, open(f"{OUT}/photos_names.json", "w"))
    for part, a in (("train", train), ("test", test)):
        np.save(f"{OUT}/photos_{part}_rows.npy", a[:, :2])
        np.save(f"{OUT}/photos_{part}_labels.npy", a[:, 2])
    np.save(f"{OUT}/photos_dev_rows.npy", dev)
    share = collections.Counter(test[:, 2].tolist()).most_common(1)[0][1] / len(test)
    print(f"{len(names)} words: {', '.join(names)}")
    print(f"stream {len(train)} pictures ({int((train[:, 0] == 1).sum())} from Commons) | tests {len(test)} ({int((test[:, 0] == 1).sum())} from Commons; "
          f"from {len(held)} held-out photographs) | the commonest word is {share:.3f} of the tests | unlabelled photographs for development {len(dev)} of {len(thumbs)}")
    sys.exit(0)

import softhebb_probe as P  # noqa: E402
from bcm_probe import near_copies  # noqa: E402
from organic.cortex2 import VisualCortex2  # noqa: E402

rows = {part: np.load(f"{OUT}/photos_{part}_rows.npy") for part in ("train", "test")}
yte, dev = np.load(f"{OUT}/photos_test_labels.npy"), np.load(f"{OUT}/photos_dev_rows.npy")


class Pictures:
    def __len__(self):
        return len(dev)

    def __getitem__(self, i):
        return np.asarray(thumbs[dev[int(i)]], np.float32) / 255


for seed in SEEDS:
    for variant in VARIANTS:
        t0, rng, pics = time.time(), np.random.default_rng(seed), Pictures()
        cortex = VisualCortex2(seed=seed, kinds=(256,), scaled=False)
        if variant.endswith("_d"):
            cortex.develop(pics, rng, log=lambda s: print("   " + s, flush=True))
        else:
            cortex.layers[0].settle((pics[i] for i in rng.choice(len(pics), 800, replace=False)), rng)
        alike, pairs, groups = near_copies(cortex.layers[0].W[0])
        cortex.settle_relay([pics[i] for i in rng.choice(len(pics), 6000, replace=False)], GRID, SHRINK)
        cortex.grid = GRID
        relay, cortex.relay = cortex.relay, None
        for part in ("train", "test"):
            S = np.lib.format.open_memmap(f"{OUT}/photos_{variant}_s{seed}_{part}.npy", "w+", np.float16, (len(rows[part]), GRID * GRID * 256))
            for i, (source, row) in enumerate(rows[part]):
                S[i] = cortex.whole(picture(source, row), GRID)
            S.flush()
        S = np.asarray(S[:1500], np.float32)
        through = (S - relay.mu) @ relay.Z
        cortex.relay = relay
        pickle.dump(cortex, open(f"{OUT}/cortex_{variant}_s{seed}.pkl", "wb"))
        print(f"seed {seed} {variant}: two kinds' weights alike {alike:.3f}, pairs above 0.9: {pairs}, different filters {groups} | "
              + P.read_code("summary", S, yte[:1500]) + " | " + P.read_code("through the relay", through, yte[:1500]) + f" ({time.time() - t0:.0f}s)", flush=True)
