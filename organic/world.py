"""The brain's wider world: real photos and what people say about them.

Sources (all cached on disk, labels only used as what an adult would answer):
  - COCO val2017 photos with questions written by the adult models
    (data/world/coco_qa.jsonl; only items both adults agreed on);
  - questions derived from COCO's own human annotations (people, counts,
    objects present, where the person is);
  - CIFAR-100 photos with "what is this?" (fine label) and coarse-group questions;
  - VQA v2: questions people asked about the same COCO photos, with the answer
    at least 5 of the 10 people gave (answers kept as typed: "2", not "two");
  - bAbI: short stories heard with eyes closed, then a question about them;
  - the text world: everyday spoken questions written and checked by the adults;
  - DailyDialog (Li et al. 2017): something a person said and the short reply
    the other person gave (replies of up to 4 words);
  - CommonsenseQA (Talmor et al. 2019) and ARC-Easy (Clark et al. 2018): a
    question asked together with its possible answers ("... a, b, c or d?");
  - WinoGrande (Sakaguchi et al. 2020): a sentence with a gap and the two
    things that could fill it ("... because the ... is bigger. home or house?").

Every moment is (image 64x64 RGB in [0,1], question, answer, kind, deadline).
The brain never sees `kind` or `deadline`; the environment uses them to reward.
"""

import json
import os
import pickle
import random
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DATA = os.path.expanduser("~/datasets")
SIZE = 64
# where the adults' questions live (a frozen copy can be used for controlled lives)
WORLD_DIR = os.environ.get("WORLD_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "world"))
NUMBER = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four"}
# hidden time limits (ticks) by kind of question; None = no deadline, only be right
DEADLINE = {"talk": 3, "chat": 2, "gist": 2, "presence": 2, "color": 4, "where": 4, "count": None, "identity": None, "activity": None, "text": None,
            "yesno": None, "ask": None, "reason": None, "story": None}
# text-world themes that need working something out (no deadline) or that are social replies (quick)
REASON_THEMES = ("addition", "subtraction", "sequence", "comes after", "comes before", "logic", "compar", "riddle", "more or less", "bigger")
CHAT_THEMES = ("greeting", "polite", "conversation", "reply", "small talk")


def thumb(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((SIZE, SIZE), Image.BILINEAR)
    return np.asarray(im, np.float32) / 255


def coco_annotation_questions(rng, split="val2017", ref=None):
    """Questions whose answers come from COCO's human annotations.
    ref maps a COCO image id to how the world refers to that photo."""
    ann = json.load(open(os.path.join(DATA, "annotations", f"instances_{split}.json")))
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    files = {im["id"]: (ref(im["id"]) if ref else im["file_name"], im["width"], im["height"]) for im in ann["images"]}
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


SIGN_WORDS = ["stop", "exit", "open", "closed", "sale", "push", "pull", "danger", "hello", "welcome", "free", "taxi", "hotel", "bank", "cafe",
              "park", "school", "police", "fire", "help", "yes", "no", "left", "right", "up", "down", "red", "blue", "green", "cat", "dog", "bus",
              "car", "pizza", "horse", "bird", "boat", "train", "cake", "book", "chair", "clock", "tree", "sun", "moon", "home", "milk", "bread"]


def reading_questions():
    """Words written on photos: 'what does it say?' (the eye has to read)."""
    files = sorted(f for f in os.listdir(os.path.join(DATA, "val2017")) if f.endswith(".jpg"))
    out = []
    for f in files:
        h = zlib.crc32(("read" + f).encode())
        if h % 3:
            continue
        word = SIGN_WORDS[(h // 3) % len(SIGN_WORDS)]
        out.append((("read", f, word, (h // 997) % 3), "what does it say?", word, "text"))
    return out


def render_word(img, word, place):
    im = Image.fromarray((img * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=14)
    x, y = [(4, 6), (8, 24), (4, 42)][place]
    box = d.textbbox((x, y), word, font=font)
    d.rectangle([box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2], fill=(255, 255, 255))
    d.text((x, y), word, fill=(0, 0, 0), font=font)
    return np.asarray(im, np.float32) / 255


FORMAT_KIND = {"fact": "talk", "why": "talk", "define": "talk", "wordplay": "talk", "chat": "chat", "dialogue": "chat",
               "logic": "reason", "analogy": "reason", "order": "reason", "story": "story", "mind": "story"}


def text_kind(d):
    if d.get("format") in FORMAT_KIND and d.get("format") != "fact":
        return FORMAT_KIND[d["format"]]
    theme = d.get("theme", "")
    if any(t in theme for t in CHAT_THEMES) or d.get("format") == "chat":
        return "chat"
    if any(t in theme for t in REASON_THEMES) or d.get("format") == "logic":
        return "reason"
    return "talk"


def talk_questions():
    """Spoken questions with eyes closed (text world, pairs the adults agreed on).
    Each distinct question is heard once in the world, whatever it was asked."""
    path = os.path.join(WORLD_DIR, "text_qa.jsonl")
    out, seen = [], set()
    if os.path.exists(path):
        for i, line in enumerate(open(path)):
            d = json.loads(line)
            if d.get("agree") and d["q"] not in seen:
                seen.add(d["q"])
                out.append((("dark", i), d["q"], d["a"], text_kind(d)))
    return out


def babi_questions(max_lines=6):
    """bAbI (Weston et al. 2015): a short story heard with eyes closed, then a
    question. Only stories of up to max_lines sentences (a child's span)."""
    path = os.path.join(DATA, "babi", "babi_train.jsonl")
    out = []
    if os.path.exists(path):
        for i, line in enumerate(open(path)):
            d = json.loads(line)
            if d["passage"].count("\n") <= max_lines:
                out.append((("dark", "babi", i), d["passage"] + d["question"], d["answer"], "story"))
    return out


def _speech(text):
    """Undo the dataset's tokenization so it reads as it was said ("you ?" -> "you?")."""
    t = " ".join(text.replace("’", "'").split()).lower()
    for p in (" ?", " .", " ,", " !", " '"):
        t = t.replace(p, p[1:])
    return t


def dialog_questions(max_reply=4, max_words=15):
    path = os.path.join(DATA, "text", "daily_dialog_train.parquet")
    out = []
    if os.path.exists(path):
        import pandas as pd
        for i, dia in enumerate(pd.read_parquet(path)["dialog"]):
            for j, (a, b) in enumerate(zip(dia[:-1], dia[1:])):
                if len(b.split()) <= max_reply + 1 and len(a.split()) <= max_words:
                    reply = _speech(b).rstrip(".!? ")
                    if reply and len(reply.split()) <= max_reply:
                        out.append((("dark", "dd", i, j), _speech(a), reply, "chat"))
    return out


def commonsense_questions():
    out = []
    for name, tag in (("commonsense_qa_train.parquet", "csqa"), ("arc_easy_train.parquet", "arc")):
        path = os.path.join(DATA, "text", name)
        if not os.path.exists(path):
            continue
        import pandas as pd
        for i, r in pd.read_parquet(path).iterrows():
            ch = [str(c).lower().rstrip(".") for c in r["choices"]["text"]]
            if r["answerKey"] not in list(r["choices"]["label"]) or max(len(c.split()) for c in ch) > 3:
                continue
            ans = ch[list(r["choices"]["label"]).index(r["answerKey"])]
            q = r["question"].strip().lower().rstrip("?") + "? " + ", ".join(ch[:-1]) + " or " + ch[-1] + "?"
            out.append((("dark", tag, i), q, ans, "reason"))
    path = os.path.join(DATA, "text", "winogrande_xl_train.parquet")
    if os.path.exists(path):
        import pandas as pd
        for i, r in pd.read_parquet(path).iterrows():
            o1, o2 = r["option1"].lower(), r["option2"].lower()
            q = r["sentence"].strip().lower().replace("_", "...") + f" {o1} or {o2}?"
            out.append((("dark", "wino", i), q, o1 if r["answer"] == "1" else o2, "reason"))
    return out


def vqa_kind(qtype, q, atype):
    if atype == "number":
        return "count"
    if qtype.startswith("what color") or qtype.startswith("what is the color"):
        return "color"
    if qtype.startswith("where"):
        return "where"
    if qtype.startswith("what sport") or (qtype.startswith("what is the") and q.rstrip("?").endswith("doing")):
        return "activity"
    if qtype.startswith(("is there", "are there")):
        return "presence"
    if atype == "yes/no":
        return "yesno"
    return "ask"


def vqa_questions(min_people=5, have=None):
    """VQA v2 questions on the COCO photos we have (have: image id -> ref;
    default the val2017 photos), answered the same way by at least min_people
    of the 10 people asked."""
    root = os.path.join(DATA, "vqa")
    if not os.path.exists(root):
        return []
    cache = os.path.join(root, f"items_{min_people}.pkl")
    if have is None:  # the val2017 photos: parsed once, then read from disk (the source files are large)
        if os.path.exists(cache):
            return pickle.load(open(cache, "rb"))
        have = {int(f[:-4]): f for f in os.listdir(os.path.join(DATA, "val2017")) if f.endswith(".jpg")}
        out = vqa_questions(min_people, have)
        pickle.dump(out, open(cache, "wb"))
        return out
    out = []
    for split in ("train", "val"):
        qs = {q["question_id"]: q["question"] for q in json.load(open(os.path.join(root, f"v2_OpenEnded_mscoco_{split}2014_questions.json")))["questions"] if q["image_id"] in have}
        for a in json.load(open(os.path.join(root, f"v2_mscoco_{split}2014_annotations.json")))["annotations"]:
            if a["image_id"] not in have:
                continue
            m = a["multiple_choice_answer"]
            if sum(x["answer"] == m for x in a["answers"]) >= min_people:
                q = qs[a["question_id"]].strip().lower()
                out.append((have[a["image_id"]], q, m, vqa_kind(a["question_type"], q, a["answer_type"])))
    return out


def coco_adult_questions():
    path = os.path.join(WORLD_DIR, "coco_qa.jsonl")
    out = []
    if os.path.exists(path):
        for line in open(path):
            d = json.loads(line)
            if d.get("q") and d.get("agree"):
                kind = d.get("kind", "") if d.get("kind") in DEADLINE else ("identity" if "angle" not in d else "ask")
                ref = ("thumb", int(d["file"][6:])) if d["file"].startswith("thumb:") else d["file"]
                out.append((ref, d["q"], d["a"], kind))
    return out


THUMBS = os.path.join(DATA, "coco_thumbs")


def build_coco_train_items(seed=0):
    """COCO train2017 (118k photos as 64x64 thumbnails): annotation questions
    and VQA questions, computed once and stored (the source files are large)."""
    ids = json.load(open(os.path.join(THUMBS, "train2017_ids.json")))
    row = {iid: i for i, iid in enumerate(ids)}
    items = coco_annotation_questions(random.Random(seed), "train2017", ref=lambda iid: ("thumb", row[iid]))
    items += vqa_questions(have={iid: ("thumb", r) for iid, r in row.items()})
    pickle.dump(items, open(os.path.join(THUMBS, "items.pkl"), "wb"))
    return items


def coco_train_questions():
    path = os.path.join(THUMBS, "items.pkl")
    return pickle.load(open(path, "rb")) if os.path.exists(path) else []


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

    def __init__(self, seed=0, test_fraction=0.1, use_cifar=True, use_vqa=True, use_babi=True, use_text=True, use_coco_train=True):
        rng = random.Random(seed)
        # a frozen world can say which sources it was made of (WORLD_DIR/sources.json)
        manifest = os.path.join(WORLD_DIR, "sources.json")
        if os.path.exists(manifest):
            m = json.load(open(manifest))
            use_cifar, use_vqa, use_babi, use_text, use_coco_train = (m.get(k, False) for k in ("cifar", "vqa", "babi", "text", "coco_train"))
        items = coco_annotation_questions(rng) + coco_adult_questions() + reading_questions() + talk_questions()
        if use_vqa:
            items += vqa_questions()
        if use_babi:
            items += babi_questions()
        if use_text:
            items += dialog_questions() + commonsense_questions()
        self.thumbs = None
        if use_coco_train and os.path.exists(os.path.join(THUMBS, "items.pkl")):
            items += coco_train_questions()
            self.thumbs = np.load(os.path.join(THUMBS, "train2017_64.npy"), mmap_mode="r")
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

    DARK = np.zeros((SIZE, SIZE, 3), np.float32)

    def image(self, ref):
        if isinstance(ref, tuple) and ref[0] == "dark":  # eyes closed
            return self.DARK
        if isinstance(ref, tuple) and ref[0] == "thumb":  # memory-mapped, not cached (118k photos)
            return np.asarray(self.thumbs[ref[1]], np.float32) / 255
        if ref not in self._cache:
            if isinstance(ref, tuple) and ref[0] == "read":
                self._cache[ref] = render_word(thumb(os.path.join(DATA, "val2017", ref[1])), ref[2], ref[3])
            elif isinstance(ref, tuple):
                im = Image.fromarray(self.cifar[ref[1]]).resize((SIZE, SIZE), Image.BILINEAR)
                self._cache[ref] = np.asarray(im, np.float32) / 255
            else:
                self._cache[ref] = thumb(os.path.join(DATA, "val2017", ref))
        return self._cache[ref]

    def moment(self, split="train"):
        """A kind of question is picked uniformly, then an item of that kind
        (so abundant kinds like presence do not crowd out the rest)."""
        pool = self.train if split == "train" else self.test
        if not hasattr(self, "_by_kind") or self._by_kind[0] is not pool:
            by = {}
            for it in pool:
                by.setdefault(it[3], []).append(it)
            self._by_kind = (pool, by)
        by = self._by_kind[1]
        ref, q, a, kind = self.rng.choice(by[self.rng.choice(sorted(by))])
        return {"image": self.image(ref), "question": q, "answer": a, "kind": kind, "deadline": DEADLINE.get(kind)}

    def summary(self):
        kinds = {}
        for it in self.train:
            kinds[it[3]] = kinds.get(it[3], 0) + 1
        return f"{len(self.train)} lived items, {len(self.test)} test items | " + ", ".join(f"{k}: {v}" for k, v in sorted(kinds.items()))
