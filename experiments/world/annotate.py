"""Daemon: the adults write and check questions about COCO photos, within budget.

Each request shows 4 photos as the brain's eye would see them (64x64). The
writer (Nemotron VL or Dots 3, taking turns) writes 5 questions per photo with
1-3 word answers, from two angles picked for this batch (people, scene,
materials, humour and danger, what happens next...), so questions do not all
look alike. The other model looks at the same photos, gives its own answer and
says whether the writer's is also right (same judging as the text world).
Photos: the 5,000 COCO val2017 photos first, then the 118,000 train2017
thumbnails (referred to as "thumb:ROW"). Stops for the day at its share.

    python experiments/world/annotate.py ~/datasets/val2017 data/world/coco_qa.jsonl
"""

import json
import os
import random
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from organic.adults import PER_DAY, Adults, image_part, parse_json  # noqa: E402

DAILY = 200  # photo share; text world 140, curriculum 100, reserve 60 (total 500 = half the allowance)
MODELS = ["nvidia/nemotron-3.5-super-vl:free", "dots-studio/dots-3-note-preview:free"]
ANGLES = [
    "the people: what they are doing, wearing, holding, how many, how they might feel",
    "the objects: what they are, how many, what they are for",
    "colours, materials and textures (is it wooden? is it soft?)",
    "where things are: left or right, top or bottom, on, under, next to, in front of",
    "the scene: indoors or outdoors, what place, time of day, weather, season",
    "feelings and humour: is it funny, is it dangerous, would you like to be there, is someone happy",
    "what just happened, what will happen next, and why",
    "writing and signs: is there writing, what does it say, what is the sign for",
    "everyday sense: can you eat it, is it heavy, is it alive, is it hot, is it safe to touch",
    "comparisons: which is bigger, are there more of one thing than another, which is closer",
    "animals: what animal, what it is doing, where it lives, what it eats",
    "food and meals: what food, what meal of the day, is it healthy, sweet or salty",
    "vehicles and travel: what vehicle, where it goes, is it moving",
    "yes or no questions that need a careful look",
]
WRITE = ("You are teaching a child about these {n} small, blurry photos (numbered 1-{n} in order). For EACH photo write 5 different questions a "
         "person might ask about it, focusing on: {angles}. Every answer must be 1-3 words, lowercase, answerable from the small picture. "
         'Reply only with JSON: {{"1": [{{"q": "...", "a": "...", "kind": "gist|color|count|where|identity|activity|text|yesno|ask"}}], "2": [...], ...}} '
         "(kind: gist = whole-scene glance, yesno = other yes/no, ask = anything else open)")
CHECK = ("A child was asked these questions about these {n} small photos (numbered in order). For each question give your own short answer "
         "(1-3 lowercase words) and say whether the proposed answer is also right for the photo (a different wording of a right answer is fine). "
         'Reply only with JSON: {{"1": [{{"mine": "...", "ok": true}}, ...], "2": [...], ...}}\nQuestions: {qs}')
src, out = os.path.expanduser(sys.argv[1]), sys.argv[2]
thumbs = np.load(os.path.expanduser("~/datasets/coco_thumbs/train2017_64.npy"), mmap_mode="r")
files = sorted(f for f in os.listdir(src) if f.endswith(".jpg"))
order = list(range(len(thumbs)))
random.Random(0).shuffle(order)  # train photos in a fixed random order
done = set()
if os.path.exists(out):
    for line in open(out):
        done.add(json.loads(line)["file"])
adults = Adults()
rng = random.Random()


def used_today():
    """Requests counted against this daemon's share: everything except the text world's and the curriculum's (up to their shares)."""
    s = adults._state()
    return s["count"] - min(s.get("text_count", 0), 140) - min(s.get("curriculum_count", 0), 100)


def log(msg):
    print(time.strftime("[%F %T] ") + msg, flush=True)


def photo(ref):
    if ref.startswith("thumb:"):
        return Image.fromarray(np.asarray(thumbs[int(ref[6:])]))
    return Image.open(os.path.join(src, ref))


todo = [f for f in files if f not in done] + [f"thumb:{r}" for r in order if f"thumb:{r}" not in done]
log(f"{len(done)} photos done, {len(todo)} to go")
i, k = 0, 0
while i < len(todo):
    if used_today() + 2 > DAILY or adults._state()["count"] + 2 > PER_DAY:
        log("daily budget reached; sleeping until tomorrow")
        while used_today() + 2 > DAILY or adults._state()["count"] + 2 > PER_DAY:
            time.sleep(600)
        log("new day")
    batch = todo[i : i + 4]
    imgs = [photo(f) for f in batch]
    angles = rng.sample(ANGLES, 2)
    writer, checker = MODELS[k % 2], MODELS[(k + 1) % 2]
    k += 1
    try:
        parts = [{"type": "text", "text": WRITE.format(n=len(batch), angles=" and ".join(angles))}]
        for j, im in enumerate(imgs):
            parts += [{"type": "text", "text": f"Photo {j + 1}:"}, image_part(im)]
        qa = parse_json(adults.ask(writer, parts, temperature=0.7)["text"]) or {}
        if isinstance(qa, list):
            qa = {str(j + 1): v for j, v in enumerate(qa)}
        qa = {key: [x for x in v if isinstance(x, dict) and x.get("q") and x.get("a")] for key, v in qa.items() if isinstance(v, list)}
        qs = {key: [{"q": x["q"], "proposed": x["a"]} for x in v] for key, v in qa.items()}
        cparts = [{"type": "text", "text": CHECK.format(n=len(batch), qs=json.dumps(qs))}]
        for j, im in enumerate(imgs):
            cparts += [{"type": "text", "text": f"Photo {j + 1}:"}, image_part(im)]
        check = parse_json(adults.ask(checker, cparts)["text"]) or {}
        if isinstance(check, list):  # some replies are a bare list of lists, one per photo
            check = {str(j + 1): v for j, v in enumerate(check)}
    except Exception as e:  # noqa: BLE001 - keep the daemon alive
        log(f"batch {batch[0]} failed: {e}")
        time.sleep(60)
        if "budget" in str(e):
            continue
        i += 4
        continue
    kept = 0
    with open(out, "a") as f:
        for j, fname in enumerate(batch):
            items = qa.get(str(j + 1), [])
            judged = check.get(str(j + 1)) if isinstance(check.get(str(j + 1)), list) else []
            for m, x in enumerate(items):
                jd = judged[m] if m < len(judged) and isinstance(judged[m], dict) else {}
                ok = jd.get("ok") is True
                kept += ok
                f.write(json.dumps({"file": fname, "q": str(x["q"]).strip().lower(), "a": str(x["a"]).strip().lower(),
                                    "a_check": str(jd.get("mine", "")).strip().lower(), "agree": ok, "kind": x.get("kind", ""),
                                    "angle": angles, "writer": writer}) + "\n")
            if not items:
                f.write(json.dumps({"file": fname, "q": None}) + "\n")  # mark as seen
    i += 4
    log(f"{batch[0]}..: {sum(len(v) for v in qa.values())} questions, {kept} judged right ({angles[0][:25]} / {angles[1][:25]}), {used_today()} requests today")
log("all photos annotated")
