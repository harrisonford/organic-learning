"""A curriculum world: teaching in order, from babbling to fourth grade.

See docs/CURRICULUM.md. Stages are taught in order; the brain moves on only
when it masters the current stage (on held-out items) and still holds the
earlier ones. While at a stage, 70% of moments come from it and 30% from
earlier stages (spaced review, most recent first). Grading is local and gets
stricter stage by stage; what the adult says (the answer to imitate) is
always the full answer. The brain is never told the stage.

Items: (image ref, utterance, acceptable answers, stage, source).
"""

import json
import os
import pickle
import random
import re
import zlib

import numpy as np
from PIL import Image

from . import world as W

STAGES = ["BB", "B", "W1", "W2", "P", "K", "G1", "G2", "G3", "G4"]
NAMES = {"BB": "babbling", "B": "echo", "W1": "first words", "W2": "two words", "P": "preschool", "K": "kindergarten",
         "G1": "first grade", "G2": "second grade", "G3": "third grade", "G4": "fourth grade"}
PASS = {"BB": 0.22, "B": 0.8, "W1": 0.5, "W2": 0.5, "P": 0.4, "K": 0.4, "G1": 0.4, "G2": 0.4, "G3": 0.4, "G4": 0.4}
RETAIN = 0.8  # an earlier stage must stay at >= 80% of the score it had when passed

# a toddler's first words (after the MacArthur-Bates CDI's most common early words)
FIRST_WORDS = ["mama", "dada", "ball", "dog", "cat", "hi", "bye", "milk", "no", "yes", "more", "up", "down", "baby", "car", "book", "shoe",
               "duck", "cookie", "juice", "banana", "apple", "water", "bath", "hat", "eyes", "nose", "mouth", "uh oh", "night night",
               "bird", "fish", "cow", "moo", "woof", "meow", "baa", "hot", "all gone", "thank you", "please", "bear", "bottle", "spoon",
               "cup", "door", "outside", "light", "truck", "bus", "train", "boat", "kitty", "doggy", "bunny", "teddy", "sock", "bed"]
# things a toddler names first (picture-book and household nouns present in COCO / CIFAR-100);
# naming anything else belongs to later stages
TODDLER_NOUNS = {"dog", "cat", "car", "bus", "bird", "horse", "cow", "sheep", "train", "boat", "airplane", "truck", "bicycle", "chair", "bed",
                 "cup", "banana", "apple", "orange", "cake", "pizza", "book", "clock", "teddy bear", "elephant", "bear", "giraffe", "zebra",
                 "baby", "bee", "bottle", "bowl", "boy", "butterfly", "dinosaur", "girl", "house", "lamp", "lion", "man", "woman", "mouse",
                 "rabbit", "snake", "table", "telephone", "television", "tiger", "turtle", "tv", "sports ball", "couch", "motorcycle"}
# first words are picture-book words: single things a toddler names (the rest of the toddler nouns
# come at preschool, and naming the biggest thing in a busy scene at kindergarten)
PICTURE_BOOK = {"dog", "cat", "car", "bus", "bird", "horse", "cow", "train", "boat", "airplane", "baby", "bear", "apple", "banana",
                "elephant", "chair", "bed", "cup", "clock", "bicycle", "tiger", "lion", "rabbit", "turtle", "mouse"}
STOP = set("a an the is are was were be to of in on at and or it this that i you he she they we my your his her their its do does did "
           "what where who how why which there here with for".split())


# ------------------------------------------------------------------ grading

def words(s):
    return re.findall(r"[a-z0-9']+", (s or "").lower())


def content(s):
    return [w for w in words(s) if w not in STOP] or words(s)


def sound_sim(a, b):
    """How alike two utterances sound to the ear: shared byte trigrams (Dice)."""
    def grams(s):
        s = f"  {s or ''} ".encode()
        return {s[i : i + 3] for i in range(len(s) - 2)}
    ga, gb = grams(a), grams(b)
    return 2 * len(ga & gb) / (len(ga) + len(gb)) if ga and gb else 0.0


def f1(said, ans):
    s, a = content(said), content(ans)
    common = sum(min(s.count(w), a.count(w)) for w in set(a))
    if not common:
        return 0.0
    p, r = common / len(s), common / len(a)
    return 2 * p * r / (p + r)


def in_order(said, ans):
    s, a = content(said), [w for w in content(ans) if w in content(said)]
    pos = [s.index(w) for w in a]
    return 1.0 if len(pos) < 2 else sum(x < y for x, y in zip(pos, pos[1:])) / (len(pos) - 1)


def tidy(s):
    """How a parent hears it: case and the punctuation around it do not matter."""
    return " ".join((s or "").lower().split()).strip(" .!?,;:")


def grade(stage, said, answers):
    """Score in [0, 1]; leniency shrinks with the stage."""
    said, answers = tidy(said), [tidy(a) for a in answers]
    if not said:
        return 0.0
    if any(said == a for a in answers):
        return 1.0
    if stage == "B":  # any sound counts; sounding like what was heard counts more
        return 0.3 + 0.6 * max(sound_sim(said, a) for a in answers)
    if stage == "W1":  # a babble that sounds like the word
        return 0.8 * max(sound_sim(said, a) for a in answers)
    if stage == "W2":  # the right word in any form
        return max(0.8 if set(content(said)) & set(content(a)) else 0.5 * sound_sim(said, a) for a in answers)
    if stage == "P":  # the content words, any order
        return max(len(set(content(said)) & set(content(a))) / len(set(content(a))) for a in answers)
    if stage == "K":
        return max(f1(said, a) for a in answers)
    return max(f1(said, a) * (0.5 + 0.5 * in_order(said, a)) for a in answers)


# ------------------------------------------------------------------ sources

def _babble():
    """Echo games: first words, and the many other words a toddler hears adults say."""
    out = []
    for w in FIRST_WORDS:
        for fmt in ("{w}", "{w}!", "say {w}", "{w}, {w}"):
            out.append((("dark", "babble", w, fmt), fmt.format(w=w), [w], "B", "echo"))
    vocab = set()
    path = os.path.join(W.WORLD_DIR, "text_qa.jsonl")
    for line in open(path) if os.path.exists(path) else []:
        vocab.update(w for w in words(json.loads(line)["q"]) if w.isalpha() and 3 <= len(w) <= 10)
    for w in sorted(vocab - set(FIRST_WORDS)):
        fmt = ("{w}", "{w}!", "say {w}")[zlib.crc32(w.encode()) % 3]
        out.append((("dark", "babble", w, fmt), fmt.format(w=w), [w], "B", "echo:wide"))
    return out


def _text_world():
    """The adults' text world, by format, to the stage a child would meet it."""
    stage_of = {"chat": "P", "mind": "P", "dialogue": "P", "wordplay": "K", "define": "K", "logic": "K",
                "fact": "G1", "why": "G1", "order": "G2", "analogy": "G2"}
    out, seen = [], set()
    path = os.path.join(W.WORLD_DIR, "text_qa.jsonl")
    for i, line in enumerate(open(path)) if os.path.exists(path) else []:
        d = json.loads(line)
        if not d.get("agree") or d["q"] in seen:
            continue
        seen.add(d["q"])
        fmt = d.get("format", "fact")
        stage = "P" if fmt == "story" and d["q"].count(".") <= 1 else "G1" if fmt == "story" else stage_of.get(fmt, "G1")
        answers = [d["a"]] + ([d["a_check"]] if d.get("a_check") and d["a_check"] != d["a"] else [])
        out.append((("dark", i), d["q"], answers, stage, f"text:{fmt}"))
    return out


def _curriculum_items():
    out = []
    path = os.path.join(W.WORLD_DIR, "curriculum.jsonl")
    for i, line in enumerate(open(path)) if os.path.exists(path) else []:
        d = json.loads(line)
        if d.get("agree") and d.get("stage") in STAGES:
            out.append((("dark", "cur", i), d["q"], d["answers"], d["stage"], "curriculum"))
    return out


def _babi():
    stage = {1: "G1", 6: "G1", 7: "G1", 8: "G1", 9: "G1", 2: "G2", 3: "G2", 15: "G2", 16: "G2", 17: "G2", 18: "G2"}
    out = []
    path = os.path.join(W.DATA, "babi", "babi_train.jsonl")
    for i, line in enumerate(open(path)) if os.path.exists(path) else []:
        d = json.loads(line)
        if d["task"] in stage and d["passage"].count("\n") <= 6:
            out.append((("dark", "babi", i), d["passage"] + d["question"], [d["answer"]], stage[d["task"]], f"babi{d['task']}"))
    return out


def _captions():
    """First grade: describe the picture (five people's descriptions are all acceptable)."""
    path = os.path.join(W.DATA, "annotations", "captions_val2017.json")
    if not os.path.exists(path):
        return []
    ann = json.load(open(path))
    files = {im["id"]: im["file_name"] for im in ann["images"]}
    caps = {}
    for a in ann["annotations"]:
        caps.setdefault(a["image_id"], []).append(a["caption"].strip().lower().rstrip("."))
    return [(files[i], "what is happening in the picture?", c, "G1", "captions") for i, c in caps.items()]


def build_items(seed=0):
    rng = random.Random(seed)
    items = _babble() + _text_world() + _curriculum_items() + _babi() + _captions()
    # COCO annotation questions on val2017 photos
    for ref, q, a, kind in W.coco_annotation_questions(rng):
        if kind == "identity":
            items.append((ref, q, [a], "K", "coco:biggest"))
        elif kind == "presence":
            items.append((ref, q, [a], "W2", "coco:presence"))
        elif kind == "count" and a in ("zero", "one", "two", "three"):
            items.append((ref, q, [a], "W2", "coco:count"))
        elif kind == "where":
            items.append((ref, q, [a, a.split()[-1]], "P", "coco:where"))
    for ref, q, a, kind in W.vqa_questions():
        if kind == "color" and len(a.split()) == 1:
            items.append((ref, q, [a], "W2", "vqa:color"))
        elif kind in ("yesno", "presence"):
            items.append((ref, q, [a], "W2", "vqa:yesno"))
        elif kind == "count":
            items.append((ref, q, [a], "P", "vqa:count"))
    for ref, q, a, _ in W.reading_questions():
        items.append((ref, q, [a], "K", "reading"))
    for ref, q, a, _ in W.dialog_questions(max_reply=5):
        items.append((ref, q, [a], "P", "dailydialog"))
    for ref, q, a, _ in W.commonsense_questions():
        items.append((ref, q, [a], "G4" if ref[1] == "wino" else "G3", ref[1]))
    cifar, imgs = W.cifar100_questions()
    for ref, q, a, _ in cifar:
        if q == "what is this?":
            items.append((ref, q, [a], "W1" if a in PICTURE_BOOK else "P" if a in TODDLER_NOUNS else "G1", "cifar:name"))
    return items, imgs


class Curriculum:
    def __init__(self, seed=0, test_fraction=0.1, review=0.3):
        items, self.cifar = build_items(seed)
        in_test = lambda it: zlib.crc32((str(it[0]) + it[1]).encode()) % 1000 < test_fraction * 1000  # noqa: E731
        self.train, self.test = {}, {}
        for it in items:
            side = self.test if in_test(it) else self.train
            side.setdefault(it[3], {}).setdefault(it[4], []).append(it)
        self.rng = random.Random(seed + 1)
        self.review = review
        self._cache = {}

    def summary(self):
        return " | ".join(f"{s}: " + ", ".join(f"{src} {len(v)}" for src, v in sorted(self.train.get(s, {}).items())) for s in STAGES)

    def image(self, ref):
        if isinstance(ref, tuple) and ref[0] == "dark":
            return W.World.DARK
        if ref not in self._cache:
            if isinstance(ref, tuple) and ref[0] == "read":
                self._cache[ref] = W.render_word(W.thumb(os.path.join(W.DATA, "val2017", ref[1])), ref[2], ref[3])
            elif isinstance(ref, tuple) and ref[0] == "cifar":
                im = Image.fromarray(self.cifar[ref[1]]).resize((W.SIZE, W.SIZE), Image.BILINEAR)
                self._cache[ref] = np.asarray(im, np.float32) / 255
            else:
                self._cache[ref] = W.thumb(os.path.join(W.DATA, "val2017", ref))
        return self._cache[ref]

    def _pick(self, pool):
        src = self.rng.choice(sorted(pool))  # sources of a stage take turns, so big datasets do not crowd out small ones
        return self.rng.choice(pool[src])

    def moment(self, stage):
        """70% the current stage, 30% earlier ones (the more recent, the likelier)."""
        k = STAGES.index(stage)
        if k and self.rng.random() < self.review:
            back = min(k, 1 + int(self.rng.expovariate(0.7)))
            if STAGES[k - back] in self.train:  # babbling has no questions to review
                stage = STAGES[k - back]
        ref, q, answers, st, src = self._pick(self.train[stage])
        return {"image": self.image(ref), "question": q, "answer": answers[0], "answers": answers, "stage": st, "source": src}

    def test_set(self, stage, n=60, seed=7):
        rng = random.Random(seed + STAGES.index(stage))
        pool = self.test.get(stage, {})
        per = max(1, n // max(1, len(pool)))
        out = []
        for src in sorted(pool):
            for ref, q, answers, st, s in rng.sample(pool[src], min(per, len(pool[src]))):
                out.append({"image": self.image(ref), "question": q, "answer": answers[0], "answers": answers, "stage": st, "source": s})
        return out
