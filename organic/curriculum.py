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
PASS = {"BB": 0.17,  # babbling: 0.17 accepted 2026-10-29 (the organic version plateaus at 0.17-0.20)
         "B": 0.8, "W1": 0.4,  # first words 0.4: accepted 2026-11-04 (confusing similar words is age-appropriate)
         "W2": 0.5, "P": 0.4, "K": 0.4, "G1": 0.4, "G2": 0.4, "G3": 0.4, "G4": 0.4}
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

def _echo_games(frames):
    """Echo games worded by the adults (experiments/world/echo_frames.py): the word to repeat can stand anywhere in what
    is said ("ball! can you say it?"), not always last as in the four hand-written formats of _babble. One item per
    word (four for a first word); a test item keeps one wording, a lived one is worded afresh (Curriculum.wording)."""
    fixed = lambda key: frames[zlib.crc32(key.encode()) % len(frames)]  # noqa: E731
    out = []
    for w in FIRST_WORDS:
        for k in range(4):
            out.append((("dark", "echo", w, k), fixed(f"{w}{k}").replace("WORD", w), [w], "B", "echo"))
    vocab = set()
    path = os.path.join(W.WORLD_DIR, "text_qa.jsonl")
    for line in open(path) if os.path.exists(path) else []:
        vocab.update(w for w in words(json.loads(line)["q"]) if w.isalpha() and 3 <= len(w) <= 10)
    for w in sorted(vocab - set(FIRST_WORDS)):
        out.append((("dark", "echo", w, 0), fixed(w).replace("WORD", w), [w], "B", "echo:wide"))
    return out


def _social():
    """Everyday social routines written by the adults (experiments/world/social_formats.py): greetings, thanks, how are
    you, offers, turn-taking games. First words when a toddler's reply is one word, two words otherwise."""
    out = []
    path = os.path.join(W.WORLD_DIR, "social_formats.jsonl")
    for i, line in enumerate(open(path)) if os.path.exists(path) else []:
        d = json.loads(line)
        one = [c for c in d["child"] if len(c.split()) == 1]
        out.append((("dark", "social", i), d["parent"], one or d["child"], "W1" if one else "W2", "social"))
    return out


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


# facts from Wikidata (experiments/world/wikidata_facts.py): the stage where schools teach them, a few wordings each
FACTS = {
    "animal_class": ("G1", ["what kind of animal is a {s}?", "is a {s} a mammal, a bird or a fish?", "a {s} is what kind of animal?"]),
    "continent": ("G2", ["which continent is {s} in?", "{s} is on which continent?", "on what continent is {s}?"]),
    "mountain_continent": ("G3", ["which continent is {s} on?", "where is {s}: on which continent?"]),
    "river_continent": ("G3", ["which continent is the {s} in?"]),
    "capital": ("G3", ["what is the capital of {s}?", "what city is the capital of {s}?", "name the capital of {s}."]),
    "language": ("G3", ["what language do people speak in {s}?", "which language is spoken in {s}?"]),
    "city_country": ("G3", ["which country is {s} in?", "{s} is a city in which country?", "in what country is {s}?"]),
    "inventor": ("G3", ["who invented {s}?", "who came up with {s}?"]),
    "painter": ("G3", ["who painted {s}?", "which artist painted {s}?"]),
    "author": ("G4", ["who wrote {s}?", "who is the author of {s}?"]),
}
ANIMAL_CLASSES = ["mammal", "bird", "fish", "reptile", "amphibian", "insect", "spider"]


def _wikidata():
    out = []
    path = os.path.join(W.WORLD_DIR, "wikidata_facts.jsonl")
    for i, line in enumerate(open(path)) if os.path.exists(path) else []:
        d = json.loads(line)
        if d["rel"] not in FACTS or not d["answers"]:
            continue
        stage, says = FACTS[d["rel"]]
        h = zlib.crc32(d["s"].encode())
        s, answers = d["s"].lower(), [a.lower() for a in d["answers"]]
        if d["rel"] == "animal_class":
            answers = answers + [("an " if answers[0][0] in "aeiou" else "a ") + answers[0]]
            other = [c for c in ANIMAL_CLASSES if c != answers[0]][h % (len(ANIMAL_CLASSES) - 1)]
            ask = other if h % 2 else answers[0]  # true or false about categories (is a whale a fish?)
            out.append((("dark", "wd", i, "yn"), f"is a {s} a {ask}?", ["yes" if ask == answers[0] else "no"], "G1", "wikidata"))
        out.append((("dark", "wd", i), says[h % len(says)].format(s=s), answers, stage, "wikidata"))
    return out


def _commons(frames):
    """Picture-book photos from Wikimedia Commons named through Wikidata (experiments/world/commons_pictures.py)."""
    path = os.path.join(W.DATA, "commons", "pictures.jsonl")
    rows = os.path.getsize(os.path.join(W.DATA, "commons", "pictures_64.u8")) // (W.SIZE * W.SIZE * 3) if os.path.exists(path) else 0
    out = []
    for line in open(path) if rows else []:
        d = json.loads(line)
        if d["row"] < rows and d["word"] in PICTURE_BOOK:
            ref = ("commons", d["row"])
            out.append((ref, frames[zlib.crc32(repr(ref).encode()) % len(frames)], [d["word"]], "W1", "commons:name"))
    return out, rows


def build_items(seed=0, world=1):
    rng = random.Random(seed)
    epath = os.path.join(W.WORLD_DIR, "echo_frames.json")
    echo = _echo_games(json.load(open(epath))["frames"]) if world >= 2 and os.path.exists(epath) else _babble()
    items = echo + (_social() if world >= 2 else []) + _text_world() + _curriculum_items() + _babi() + _captions() + _wikidata()
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
    # the ways adults ask to name a picture (experiments/world/naming_frames.py); each picture keeps one
    fpath = os.path.join(W.WORLD_DIR, "naming_frames.json")
    frames = json.load(open(fpath))["frames"] if os.path.exists(fpath) else ["what is this?"]
    for ref, q, a, _ in cifar:
        if q == "what is this?":
            q = frames[zlib.crc32(repr(ref).encode()) % len(frames)]
            items.append((ref, q, [a], "W1" if a in PICTURE_BOOK else "P" if a in TODDLER_NOUNS else "G1", "cifar:name"))
    commons, rows = _commons(frames)
    items += commons
    commons_imgs = np.memmap(os.path.join(W.DATA, "commons", "pictures_64.u8"), np.uint8, "r", shape=(rows, W.SIZE, W.SIZE, 3)) if rows else None
    return items, (imgs, commons_imgs)


class Curriculum:
    def __init__(self, seed=0, test_fraction=0.1, review=0.3, world=1):
        """world=2 (2026-11-11, docs/CONVERSATION.md): echo games worded by the adults with the word anywhere, and
        social routines in first words and two words. world=1 is the world every life before that lived in."""
        items, (self.cifar, self.commons) = build_items(seed, world)
        epath = os.path.join(W.WORLD_DIR, "echo_frames.json")
        self.echo_frames = json.load(open(epath))["frames"] if world >= 2 and os.path.exists(epath) else None
        # a naming item is split by its picture (its wording varies: experiments/world/naming_frames.py), as when all were "what is this?"
        key = lambda it: str(it[0]) + ("what is this?" if it[4] in ("cifar:name", "commons:name") else "" if it[0][:2] == ("dark", "echo") else it[1])  # noqa: E731
        in_test = lambda it: zlib.crc32(key(it).encode()) % 1000 < test_fraction * 1000  # noqa: E731
        self.train, self.test = {}, {}
        for it in items:
            side = self.test if in_test(it) else self.train
            side.setdefault(it[3], {}).setdefault(it[4], []).append(it)
        self.rng = random.Random(seed + 1)
        self.review = review
        self._cache = {}
        # the ways adults name a thing unasked (experiments/world/observation_frames.py)
        opath = os.path.join(W.WORLD_DIR, "observation_frames.json")
        self.comments = json.load(open(opath))["frames"] if os.path.exists(opath) else ["look, a THING!"]
        self.orng = random.Random(seed + 2)  # observation has its own dice: turning it on does not change the lessons drawn
        fpath = os.path.join(W.WORLD_DIR, "naming_frames.json")
        self.frames = json.load(open(fpath))["frames"] if os.path.exists(fpath) else ["what is this?"]

    def summary(self):
        return " | ".join(f"{s}: " + ", ".join(f"{src} {len(v)}" for src, v in sorted(self.train.get(s, {}).items())) for s in STAGES)

    def image(self, ref):
        if isinstance(ref, tuple) and ref[0] == "dark":
            return W.World.DARK
        if ref not in self._cache:
            if isinstance(ref, tuple) and ref[0] == "read":
                self._cache[ref] = W.render_word(W.thumb(os.path.join(W.DATA, "val2017", ref[1])), ref[2], ref[3])
            elif isinstance(ref, tuple) and ref[0] == "commons":  # memory-mapped
                return np.asarray(self.commons[ref[1]], np.float32) / 255
            elif isinstance(ref, tuple) and ref[0] == "cifar":
                im = Image.fromarray(self.cifar[ref[1]]).resize((W.SIZE, W.SIZE), Image.BILINEAR)
                self._cache[ref] = np.asarray(im, np.float32) / 255
            else:
                self._cache[ref] = W.thumb(os.path.join(W.DATA, "val2017", ref))
        return self._cache[ref]

    def wording(self, item, rng=None):
        """How a lived item is asked this time. A naming question is worded afresh at every moment (adults do not
        ask about the same picture in the same words each time; with one fixed wording per picture, a wording's
        own words became cues for "its" pictures' names: docs/AREAS.md). Test items keep their fixed wording."""
        if item[4] in ("cifar:name", "commons:name"):
            return (rng or self.rng).choice(self.frames)
        if item[4] in ("echo", "echo:wide") and self.echo_frames:
            return (rng or self.rng).choice(self.echo_frames).replace("WORD", item[2][0])
        return item[1]

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
        item = self._pick(self.train[stage])
        ref, q, answers, st, src = item
        q = self.wording(item)
        answer = self.rng.choice(answers) if src == "social" else answers[0]  # any of the replies a toddler gives to a routine
        return {"image": self.image(ref), "question": q, "answer": answer, "answers": answers, "stage": st, "source": src}

    def observation(self, stage):
        """An adult talks about a picture, unasked: no question, no answer expected (docs/REVIEW_2026-11-10.md 4.4).
        Things are named ("look, a dog!") from the start, since infants are spoken to long before they speak; from
        first grade also whole descriptions of photos. Only pictures of the lived (train) side, up to the current stage."""
        k = max(STAGES.index(stage), STAGES.index("W1"))
        pools = [(s, src) for s in STAGES[: k + 1] for src in ("cifar:name", "commons:name", "captions") if src in self.train.get(s, {})]
        s, src = pools[self.orng.randrange(len(pools))]
        ref, _, answers, st, _ = self.orng.choice(self.train[s][src])
        if src == "captions":
            says = self.orng.choice(answers)
        else:
            w = answers[0]
            says = self.orng.choice(self.comments).replace("a THING", ("an " if w[0] in "aeiou" else "a ") + w).replace("THING", w)
        return {"image": self.image(ref), "says": says, "stage": st, "source": "observe:" + src}

    def test_set(self, stage, n=60, seed=7):
        rng = random.Random(seed + STAGES.index(stage))
        pool = self.test.get(stage, {})
        per = max(1, n // max(1, len(pool)))
        out = []
        for src in sorted(pool):
            for ref, q, answers, st, s in rng.sample(pool[src], min(per, len(pool[src]))):
                out.append({"image": self.image(ref), "question": q, "answer": answers[0], "answers": answers, "stage": st, "source": s})
        return out
