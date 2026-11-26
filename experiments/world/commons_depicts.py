"""More real photographs, constantly and slowly: Wikimedia Commons files by what they depict (2026-11-25).

Kris: "should use all type of pictures ... we should have also wiki pics", then "yes slow polite download
constantly". experiments/world/commons_pictures.py took one photo per member of a kind (a dog breed, a car model)
and stopped at 300 a word. Here: the photographs that Commons' own editors have marked as depicting a thing
(structured data, "depicts", P180), found through the site's search, a page of results at a time, word after word,
for as long as the service runs.

    what is kept   a JPEG or PNG whose 120-pixel thumbnail is at least 64 pixels on its short side (nothing is ever
                   enlarged), the middle square of it at 64 x 64, with the word, the file's title and its page id
                   (the title is what a licence or an author is looked up by; the pictures are not passed on)
    how politely   one request at a time; a picture every PAUSE seconds (default 2); the user agent of the earlier
                   fetcher; maxlag on the site's API; on a refusal or an overloaded server it waits five minutes or
                   as long as the server says
    where          ~/datasets/commons_more/{pictures_64.u8, pictures.jsonl, offsets.json}: NOT the directory the
                   lives' world reads (~/datasets/commons), which must not change while lives are in it
    which words    the photo benchmark's 33 and the picture book's others; each word's item is checked against its
                   English label on Wikidata before anything is fetched, and a word whose label does not agree is
                   left out and said

    python experiments/world/commons_depicts.py            (the service: runs until stopped)
    MAX=3 python experiments/world/commons_depicts.py      (a check: three pictures and out)
"""
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image

UA = "organic-learning/0.1 (research; https://github.com/harrisonford/organic-learning)"
OUT = os.path.expanduser("~/datasets/commons_more")
PAUSE, MAX, PAGE, DEEPEST = float(os.environ.get("PAUSE", 2.0)), int(os.environ.get("MAX", 0)), 25, 9900
API = "https://commons.wikimedia.org/w/api.php"
WORDS = {  # word: (Wikidata item, the English labels it may have)
    "dog": ("Q144", {"dog"}), "cat": ("Q146", {"house cat", "cat"}), "horse": ("Q726", {"horse"}), "cow": ("Q830", {"cattle"}),
    "sheep": ("Q7368", {"sheep", "domestic sheep"}), "bird": ("Q5113", {"bird"}), "elephant": ("Q7378", {"elephant"}),
    "giraffe": ("Q15083", {"giraffe"}), "zebra": ("Q32789", {"zebra"}), "bear": ("Q30090244", {"bear"}),
    "car": ("Q1420", {"motor car", "car", "automobile"}), "bus": ("Q5638", {"bus"}), "train": ("Q870", {"train"}),
    "truck": ("Q43193", {"truck"}), "motorcycle": ("Q34493", {"motorcycle"}), "bicycle": ("Q11442", {"bicycle"}),
    "boat": ("Q35872", {"boat"}), "airplane": ("Q197", {"airplane"}),
    "apple": ("Q89", {"apple"}), "banana": ("Q503", {"banana"}), "orange": ("Q13191", {"orange"}), "pizza": ("Q177", {"pizza"}),
    "cake": ("Q13276", {"cake"}), "chair": ("Q15026", {"chair"}), "couch": ("Q131514", {"couch", "sofa"}), "bed": ("Q42177", {"bed"}),
    "cup": ("Q81727", {"cup"}), "bowl": ("Q153988", {"bowl"}), "bottle": ("Q80228", {"bottle"}), "book": ("Q571", {"book"}),
    "clock": ("Q376", {"clock"}), "tv": ("Q8075", {"television set"}), "teddy bear": ("Q213477", {"teddy bear"}),
    "rabbit": ("Q9394", {"rabbit"}), "turtle": ("Q223044", {"turtle", "testudines"}), "tiger": ("Q19939", {"tiger"}),
    "lion": ("Q140", {"lion"}), "flower": ("Q506", {"flower"}), "tree": ("Q10884", {"tree"}), "house": ("Q3947", {"house"}),
}


def say(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def get(url, timeout=60):
    """One request; on a refusal or an overloaded server, wait as told (or five minutes) and try again."""
    while True:
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=timeout).read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                wait = int(e.headers.get("Retry-After") or 300) if (e.headers.get("Retry-After") or "300").isdigit() else 300
                say(f"the server said {e.code}; waiting {max(wait, 60)} s")
                time.sleep(max(wait, 60))
                continue
            raise


def api(site, **params):
    while True:
        d = json.loads(get(site + "?" + urllib.parse.urlencode({"format": "json", "maxlag": 5, **params})))
        if d.get("error", {}).get("code") == "maxlag":
            say("the site is lagging; waiting five minutes")
            time.sleep(300)
            continue
        return d


def checked():
    """The words whose item carries the label we think it does."""
    d = api("https://www.wikidata.org/w/api.php", action="wbgetentities", ids="|".join(q for q, _ in WORDS.values()), props="labels", languages="en")
    good = {}
    for word, (q, labels) in WORDS.items():
        label = d.get("entities", {}).get(q, {}).get("labels", {}).get("en", {}).get("value", "").lower()
        if label in labels:
            good[word] = q
        else:
            say(f"left out: {word} ({q} is labelled '{label}')")
    return good


def square(raw):
    """The middle square of a thumbnail at 64 x 64, or nothing if that would enlarge it."""
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = im.size
    s = min(w, h)
    if s < 64:
        return None
    return np.asarray(im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((64, 64), Image.BILINEAR), np.uint8)


def main():
    os.makedirs(OUT, exist_ok=True)
    index, state = os.path.join(OUT, "pictures.jsonl"), os.path.join(OUT, "offsets.json")
    rows = [json.loads(line) for line in open(index)] if os.path.exists(index) else []
    have, row = {r["file"] for r in rows}, len(rows)
    offsets = json.load(open(state)) if os.path.exists(state) else {}
    words, got = checked(), 0
    say(f"{len(words)} words; {row} pictures kept so far")
    time.sleep(PAUSE)
    while True:
        busy = False
        for word, q in words.items():
            at = offsets.get(word, 0)
            if at is None or at >= DEEPEST:
                continue
            d = api(API, action="query", generator="search", gsrsearch=f"haswbstatement:P180={q} filetype:bitmap", gsrnamespace=6,
                    gsrlimit=PAGE, gsroffset=at, prop="imageinfo", iiprop="url|mime|size", iiurlwidth=120)
            pages = sorted(d.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
            kept = 0
            for p in pages:
                info = (p.get("imageinfo") or [{}])[0]
                if p["title"] in have or info.get("mime") not in ("image/jpeg", "image/png") or not info.get("thumburl"):
                    continue
                time.sleep(PAUSE)
                try:
                    arr = square(get(info["thumburl"]))
                except Exception as e:  # noqa: BLE001
                    say("skip", p["title"][:60], str(e)[:60])
                    continue
                if arr is None:
                    continue
                with open(os.path.join(OUT, "pictures_64.u8"), "ab") as f:  # rows of 64x64x3 bytes, memory-mapped when read
                    f.write(arr.tobytes())
                with open(index, "a") as f:
                    f.write(json.dumps({"row": row, "word": word, "file": p["title"], "page": p["pageid"], "depicts": q}) + "\n")
                have.add(p["title"])
                row, kept, got = row + 1, kept + 1, got + 1
                if MAX and got >= MAX:
                    say(f"check done: {got} pictures")
                    return
            offsets[word] = at + PAGE if pages and "continue" in d else None  # None: this word's results are used up
            json.dump(offsets, open(state + ".new", "w"))
            os.replace(state + ".new", state)
            busy = busy or offsets[word] is not None
            say(f"{word}: {kept} kept of {len(pages)} found at {at}; {row} pictures in all")
            time.sleep(PAUSE)
        if not busy:
            say("every word's results are used up; looking again in a day")
            time.sleep(86400)
            offsets = {}


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
