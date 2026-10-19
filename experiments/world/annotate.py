"""Daemon: the adults write and check questions about COCO photos, within budget.

Each request shows 4 photos as the brain's eye would see them (64x64). Nemotron
VL writes 5 questions per photo with 1-3 word answers; Dots 3 answers the same
questions independently. Every item is stored with both answers and whether
they agree. Stops for the day at 440 requests (half the account's daily free
allowance, minus a reserve for grading and comparisons), then resumes.

    python experiments/world/annotate.py ~/datasets/val2017 data/world/coco_qa.jsonl
"""

import json
import os
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from organic.adults import Adults, image_part, parse_json  # noqa: E402

DAILY = 340  # photo annotation share; text world 100, reserve 60 (total 500 = half the allowance)
WRITER, CHECKER = "nvidia/nemotron-3.5-super-vl:free", "dots-studio/dots-3-note-preview:free"
src, out = os.path.expanduser(sys.argv[1]), sys.argv[2]
files = sorted(f for f in os.listdir(src) if f.endswith(".jpg"))
done = set()
if os.path.exists(out):
    for line in open(out):
        done.add(json.loads(line)["file"])
adults = Adults()
WRITE = ("You are teaching a child about these {n} small, blurry photos (numbered 1-{n} in order). For EACH photo write 5 questions a person "
         "might ask, from very easy (is there a person? is it indoors?) to harder (what is the person doing? how many people? what does it say?). "
         "Every answer must be 1-3 words, lowercase, answerable from the small picture. "
         'Reply only with JSON: {{"1": [{{"q": "...", "a": "...", "kind": "gist|color|count|where|identity|activity|text"}}], "2": [...], ...}}')
CHECK = ("Answer each question about these {n} small photos (numbered in order) in 1-3 lowercase words. "
         'Reply only with JSON: {{"1": ["answer", ...], "2": [...], ...}}\nQuestions: {qs}')


def used_today():
    """Requests counted against this daemon's share: everything except the text world's (up to its 100)."""
    s = adults._state()
    return s["count"] - min(s.get("text_count", 0), 100)


def log(msg):
    print(time.strftime("[%F %T] ") + msg, flush=True)


todo = [f for f in files if f not in done]
log(f"{len(done)} photos done, {len(todo)} to go")
i = 0
while i < len(todo):
    if used_today() + 2 > DAILY:
        log("daily budget reached; sleeping until tomorrow")
        while used_today() + 2 > DAILY:
            time.sleep(600)
        log("new day")
    batch = todo[i : i + 4]
    imgs = [Image.open(os.path.join(src, f)) for f in batch]
    try:
        parts = [{"type": "text", "text": WRITE.format(n=len(batch))}]
        for k, im in enumerate(imgs):
            parts += [{"type": "text", "text": f"Photo {k + 1}:"}, image_part(im)]
        qa = parse_json(adults.ask(WRITER, parts)["text"]) or {}
        qs = {k: [x.get("q", "") for x in v] for k, v in qa.items() if isinstance(v, list)}
        cparts = [{"type": "text", "text": CHECK.format(n=len(batch), qs=json.dumps(qs))}]
        for k, im in enumerate(imgs):
            cparts += [{"type": "text", "text": f"Photo {k + 1}:"}, image_part(im)]
        check = parse_json(adults.ask(CHECKER, cparts)["text"]) or {}
        if isinstance(check, list):  # some replies are a bare list of lists, one per photo
            check = {str(k + 1): v for k, v in enumerate(check)}
        if not isinstance(qa, dict):
            qa = {}
    except Exception as e:  # noqa: BLE001 - keep the daemon alive
        log(f"batch {batch[0]} failed: {e}")
        if "budget" in str(e):
            continue
        i += 4
        continue
    with open(out, "a") as f:
        for k, fname in enumerate(batch):
            items = qa.get(str(k + 1), [])
            answers = check.get(str(k + 1), []) if isinstance(check.get(str(k + 1)), list) else []
            for j, x in enumerate(items if isinstance(items, list) else []):
                if not isinstance(x, dict) or "q" not in x or "a" not in x:
                    continue
                a2 = str(answers[j]).strip().lower() if j < len(answers) else None
                a1 = str(x["a"]).strip().lower()
                f.write(json.dumps({"file": fname, "q": x["q"].strip().lower(), "a": a1, "a_check": a2, "agree": a1 == a2, "kind": x.get("kind", "")}) + "\n")
            if not items:
                f.write(json.dumps({"file": fname, "q": None}) + "\n")  # mark as seen
    i += 4
    if (i // 4) % 25 == 0:
        log(f"{len(done) + i} photos annotated, {used_today()} requests today")
log("all photos annotated")
