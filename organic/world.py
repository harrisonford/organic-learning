"""The brain's wider world: real photos and what people say about them.

Sources (all cached on disk, labels only used as what an adult would answer):
  - COCO val2017 photos with questions written by the adult models
    (data/world/coco_qa.jsonl; only items both adults agreed on);
  - questions derived from COCO's own human annotations (people, counts,
    objects present, where the person is);
  - CIFAR-100 photos with "what is this?" (fine label) and coarse-group questions.

Every moment is (image 64x64 RGB in [0,1], question, answer, kind, deadline).
The brain never sees `kind` or `deadline`; the environment uses them to reward.
"""

import json
import os
import pickle
import random
import zlib

import numpy as np
from PIL import Image

DATA = os.path.expanduser("~/datasets")
SIZE = 64
NUMBER = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four"}
# hidden time limits (ticks) by kind of question; None = no deadline, only be right
DEADLINE = {"gist": 2, "presence": 2, "color": 4, "where": 4, "count": None, "identity": None, "activity": None, "text": None}


def thumb(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((SIZE, SIZE), Image.BILINEAR)
    return np.asarray(im, np.float32) / 255


def coco_annotation_questions(rng):
    """Questions whose answers come from COCO's human annotations."""
    ann = json.load(open(os.path.join(DATA, "annotations", "instances_val2017.json")))
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    files = {im["id"]: (im["file_name"], im["width"], im["height"]) for im in ann["images"]}
    per = {}
    for a in ann["annotations"]:
        if a.get("iscrowd"):
            continue
        per.setdefault(a["image_id"], []).append(a)
    common = ["dog", "cat", "car", "bus", "bicycle", "pizza", "horse", "bird", "boat", "train", "airplane", "elephant", "giraffe", "cow", "sheep", "chair", "tv", "bed", "umbrella", "clock"]
    out = []
    for iid, (fname, w, h) in files.items():
        objs = per.get(iid, [])
        names = [cats[a["category_id"]] for a in objs]
        people = [a for a in objs if cats[a["category_id"]] == "person" and a["area"] > 0.01 * w * h]
        out.append((fname, "is there a person?", "yes" if people else "no", "presence"))
        out.append((fname, "how many people are there?", NUMBER.get(min(len(people), 4), "many") if len(people) < 5 else "many", "count"))
        present = [c for c in common if c in names]
        absent = [c for c in common if c not in names]
        if present:
            out.append((fname, f"is there a {present[0]}?", "yes", "presence"))
        out.append((fname, f"is there a {rng.choice(absent)}?", "no", "presence"))
        if objs:
            big = max(objs, key=lambda a: a["area"])
            out.append((fname, "what is the biggest thing here?", cats[big["category_id"]], "identity"))
        if len(people) == 1:
            x, y, bw, bh = people[0]["bbox"]
            cx = (x + bw / 2) / w
            out.append((fname, "where is the person?", "on the left" if cx < 0.4 else "on the right" if cx > 0.6 else "in the middle", "where"))
    return out


def coco_adult_questions():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "world", "coco_qa.jsonl")
    out = []
    if os.path.exists(path):
        for line in open(path):
            d = json.loads(line)
            if d.get("q") and d.get("agree"):
                kind = d.get("kind", "") if d.get("kind") in DEADLINE else "identity"
                out.append((d["file"], d["q"], d["a"], kind))
    return out


def cifar100_questions():
    root = os.path.join(DATA, "cifar-100-python")
    if not os.path.exists(root):
        return [], None
    meta = pickle.load(open(os.path.join(root, "meta"), "rb"), encoding="latin1")
    d = pickle.load(open(os.path.join(root, "train"), "rb"), encoding="latin1")
    imgs = d["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    out = []
    for i, (f, c) in enumerate(zip(d["fine_labels"], d["coarse_labels"])):
        fine, coarse = meta["fine_label_names"][f].replace("_", " "), meta["coarse_label_names"][c].replace("_", " ")
        out.append((("cifar", i), "what is this?", fine, "identity"))
        out.append((("cifar", i), "is it a person?", "yes" if coarse == "people" else "no", "presence"))
    return out, imgs


class World:
    """Splits photos into a lived part and a test part, and serves moments."""

    def __init__(self, seed=0, test_fraction=0.1, use_cifar=True):
        rng = random.Random(seed)
        items = coco_annotation_questions(rng) + coco_adult_questions()
        self.cifar = None
        if use_cifar:
            c, self.cifar = cifar100_questions()
            items += c
        # a stable split by photo: the same photo is always on the same side,
        # however the world grows
        in_test = lambda ref: zlib.crc32(str(ref).encode()) % 1000 < test_fraction * 1000  # noqa: E731
        self.train = [it for it in items if not in_test(it[0])]
        self.test = [it for it in items if in_test(it[0])]
        self.rng = rng
        self._cache = {}

    def image(self, ref):
        if ref not in self._cache:
            if isinstance(ref, tuple):
                im = Image.fromarray(self.cifar[ref[1]]).resize((SIZE, SIZE), Image.BILINEAR)
                self._cache[ref] = np.asarray(im, np.float32) / 255
            else:
                self._cache[ref] = thumb(os.path.join(DATA, "val2017", ref))
        return self._cache[ref]

    def moment(self, split="train"):
        ref, q, a, kind = self.rng.choice(self.train if split == "train" else self.test)
        return {"image": self.image(ref), "question": q, "answer": a, "kind": kind, "deadline": DEADLINE.get(kind)}

    def summary(self):
        kinds = {}
        for it in self.train:
            kinds[it[3]] = kinds.get(it[3], 0) + 1
        return f"{len(self.train)} lived items, {len(self.test)} test items | " + ", ".join(f"{k}: {v}" for k, v in sorted(kinds.items()))
