"""Picture-book photos from Wikimedia Commons, named through Wikidata (no model requests).

CIFAR's picture book is 100 classes of tiny photos, and each W1 word has one kind of
look. Wikidata links many members of a kind to a photo (P18): 500 dog breeds give 500
different-looking dogs, car models thousands of cars, bird species thousands of birds.
For each picture-book word, the members of its kinds (breeds, models, species,
cultivars, subclasses of a thing) with a photo are found by one SPARQL query, and their
photos are fetched as small thumbnails, slowly, with a polite user agent, resumable.
Each becomes a 64x64 picture named with the word a toddler would use.

    python experiments/world/commons_pictures.py   -> data/commons/pictures_64.u8 + pictures.jsonl
"""

import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from organic.world import DATA, SIZE  # noqa: E402

UA = "organic-learning/0.1 (research; https://github.com/harrisonford/organic-learning)"
OUT = os.path.join(DATA, "commons")
PER_WORD = int(os.environ.get("PER_WORD", 300))
SPECIES = "?x wdt:P105 wd:Q7432; wdt:P171* wd:%s."
KINDS = {  # word: graph pattern for ?x (one photo per member), min sitelinks
    "dog": ("?x wdt:P31 wd:Q39367.", 1), "cat": ("?x wdt:P31 wd:Q43577.", 1), "horse": ("?x wdt:P31 wd:Q1160573.", 1),
    "cow": ("?x wdt:P31 wd:Q12045585.", 1), "rabbit": ("?x wdt:P31 wd:Q12045584.", 1),
    "car": ("?x wdt:P31 wd:Q3231690.", 5), "bus": ("?x wdt:P31 wd:Q23039057.", 1), "train": ("?x wdt:P31 wd:Q19832486.", 2),
    "boat": ("?x wdt:P31 wd:Q559026.", 3), "airplane": ("?x wdt:P31 wd:Q15056995. FILTER NOT EXISTS { ?x wdt:P279* wd:Q34486 }", 4),
    "bird": (SPECIES % "wd:Q5113", 30), "bear": (SPECIES % "wd:Q11788", 3), "elephant": (SPECIES % "wd:Q2372824", 3),
    "turtle": (SPECIES % "wd:Q223044", 10), "mouse": (SPECIES % "wd:Q39275", 3),
    "tiger": ("{ BIND(wd:Q19939 AS ?x) } UNION { ?x wdt:P171 wd:Q19939 }", 0), "lion": ("{ BIND(wd:Q140 AS ?x) } UNION { ?x wdt:P171 wd:Q140 }", 0),
    "apple": ("?x wdt:P31 wd:Q15731356.", 1), "chair": ("?x wdt:P279+ wd:Q15026.", 1), "bed": ("?x wdt:P279+ wd:Q42177.", 1),
    "cup": ("?x wdt:P279+ wd:Q81727.", 1), "clock": ("?x wdt:P279+ wd:Q376.", 1), "bicycle": ("?x wdt:P279+ wd:Q11442.", 1),
}


def get(url, timeout=60):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=timeout)


def members(word):
    pattern, n = KINDS[word]
    q = f"SELECT DISTINCT ?x ?img ?n WHERE {{ {pattern} ?x wdt:P18 ?img; wikibase:sitelinks ?n. FILTER(?n >= {n}) }} ORDER BY DESC(?n) LIMIT {PER_WORD * 2}"
    for attempt in range(3):
        try:
            d = json.load(get("https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": q, "format": "json"}), timeout=180))
            return [(b["x"]["value"].rsplit("/", 1)[1], b["img"]["value"]) for b in d["results"]["bindings"]]
        except Exception as e:  # noqa: BLE001
            print("  query retry", word, str(e)[:80], flush=True)
            time.sleep(30 * (attempt + 1))
    return []


def picture(img_url):
    """A Commons thumbnail (a standard 120px width), center-cropped and resized like CIFAR."""
    name = urllib.parse.unquote(img_url.rsplit("/", 1)[1])
    if not name.lower().endswith((".jpg", ".jpeg", ".png")):  # drawings (svg), animations and scans are left out
        return None
    raw = get("https://commons.wikimedia.org/wiki/Special:FilePath/" + urllib.parse.quote(name) + "?width=120").read()
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((SIZE, SIZE), Image.BILINEAR)
    return np.asarray(im, np.uint8), name


def main():
    os.makedirs(OUT, exist_ok=True)
    index = os.path.join(OUT, "pictures.jsonl")
    have = {json.loads(line)["item"] for line in open(index)} if os.path.exists(index) else set()
    row = len(have)
    for word in KINDS:
        todo = [(x, img) for x, img in members(word) if x not in have]
        kept = sum(1 for line in open(index) if json.loads(line)["word"] == word) if os.path.exists(index) else 0
        print(f"{word}: {len(todo)} members with a photo to fetch ({kept} kept already)", flush=True)
        for x, img in todo:
            if kept >= PER_WORD:
                break
            try:
                got = picture(img)
            except Exception as e:  # noqa: BLE001
                print("  skip", x, str(e)[:60], flush=True)
                time.sleep(5)
                continue
            if got is None:
                continue
            arr, name = got
            with open(os.path.join(OUT, "pictures_64.u8"), "ab") as f:  # rows of 64x64x3 bytes, memory-mapped when read
                f.write(arr.tobytes())
            with open(index, "a") as f:
                f.write(json.dumps({"row": row, "word": word, "item": x, "file": name}) + "\n")
            have.add(x)
            row += 1
            kept += 1
            time.sleep(1.0)  # polite: one picture a second
        time.sleep(5)
    print("pictures:", row)


if __name__ == "__main__":
    main()
