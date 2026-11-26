"""Real video for the eye, constantly and slowly: Wikimedia Commons videos as 64 x 64 frames in order (2026-11-26).

Kris: "yes lets include video" and "we can also download in parallel wikimedia then" (queue E 31 in docs/LOOP.md).
Commons holds about 420,000 video files under free licences. Here, a page of search results at a time and one
video at a time, for as long as the service runs:
    named        videos whose editors marked them as depicting one of the picture book's words (structured data,
                 "depicts", P180): the word is kept with the frames
    by subject   videos anywhere under Commons' categories of animals, vehicles, people and nature (the search's
                 deepcat): the category is kept, no word
    what is kept the smallest version Commons serves that is at least 160 pixels on its short side (usually 240p);
                 the file itself in raw/ (so other sizes and rates can be taken later), and its first SECONDS
                 seconds as frames at FPS a second, the middle square of each at 64 x 64, appended in order to
                 frames_64.u8 with one line per video in videos.jsonl (where its frames start, how many, the title
                 that its licence and author are looked up by, its page id, word or category, duration)
    left out     videos shorter than 2 seconds or longer than 20 minutes; none without a small version over 50 MB
    politeness   one request at a time, PAUSE seconds between them (default 3), the user agent of the picture
                 fetcher, maxlag on the API, five minutes' wait (or as long as the server says) on a refusal
    where        ~/datasets/commons_video/ (its own directory; ffmpeg decodes each file from there)

    python experiments/world/commons_video.py          (the service: runs until stopped)
    MAX=2 python experiments/world/commons_video.py    (a check: two videos and out)
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from commons_depicts import WORDS, UA, checked, say  # noqa: E402

OUT = os.path.expanduser("~/datasets/commons_video")
PAUSE, MAX = float(os.environ.get("PAUSE", 3.0)), int(os.environ.get("MAX", 0))
FPS, SECONDS, SIDE = int(os.environ.get("FPS", 10)), int(os.environ.get("SECONDS", 120)), 64
API, PAGE, DEEPEST = "https://commons.wikimedia.org/w/api.php", 20, 9980
SUBJECTS = ["Videos_of_animals", "Videos_of_vehicles", "Videos_of_people", "Videos_of_nature"]


def fetch(url, timeout=120):
    """One request; on a refusal or an overloaded server, wait as told (or five minutes) and try again."""
    while True:
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=timeout).read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                after = e.headers.get("Retry-After") or "300"
                wait = max(int(after) if after.isdigit() else 300, 60)
                say(f"the server said {e.code}; waiting {wait} s")
                time.sleep(wait)
                continue
            raise


def api(**params):
    while True:
        d = json.loads(fetch(API + "?" + urllib.parse.urlencode({"format": "json", "maxlag": 5, **params})))
        if d.get("error", {}).get("code") == "maxlag":
            say("the site is lagging; waiting five minutes")
            time.sleep(300)
            continue
        return d


def smallest(info):
    """The smallest served version at least 160 pixels on its short side, or the original if it is small enough."""
    options = [x for x in info.get("derivatives", []) if x.get("transcodekey") and min(x.get("width", 0), x.get("height", 0)) >= 160]
    options = [x for x in options if "webm" in x["transcodekey"]] or options
    if options:
        return min(options, key=lambda x: (x["width"] * x["height"], x.get("bandwidth", 0)))["src"]
    return info["url"] if info.get("size", 1e12) <= 50e6 else None


def frames(path):
    """The first SECONDS seconds at FPS a second, each the middle square at 64 x 64, as rows of 64 x 64 x 3 bytes."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-t", str(SECONDS), "-i", path,
           "-vf", f"fps={FPS},crop=min(iw\\,ih):min(iw\\,ih),scale={SIDE}:{SIDE}:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    raw = subprocess.run(cmd, capture_output=True, timeout=600, check=True).stdout
    return raw[: len(raw) // (SIDE * SIDE * 3) * SIDE * SIDE * 3]


def main():
    os.makedirs(os.path.join(OUT, "raw"), exist_ok=True)
    index, state = os.path.join(OUT, "videos.jsonl"), os.path.join(OUT, "offsets.json")
    rows = [json.loads(line) for line in open(index)] if os.path.exists(index) else []
    have = {r["page"] for r in rows}
    start = sum(r["frames"] for r in rows)
    offsets = json.load(open(state)) if os.path.exists(state) else {}
    queries = [(w, f"haswbstatement:P180={q} filetype:video") for w, q in checked().items()]
    queries += [(f"category:{c}", f"filetype:video deepcat:{c}") for c in SUBJECTS]
    say(f"{len(queries)} searches; {len(rows)} videos, {start} frames kept so far")
    got = 0
    while True:
        busy = False
        for label, search in queries:
            at = offsets.get(label, 0)
            if at is None or at >= DEEPEST:
                continue
            time.sleep(PAUSE)
            d = api(action="query", generator="search", gsrsearch=search, gsrnamespace=6, gsrlimit=PAGE, gsroffset=at,
                    prop="videoinfo", viprop="url|size|mime|derivatives")
            pages = sorted(d.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
            kept = 0
            for p in pages:
                info = (p.get("videoinfo") or [{}])[0]
                if p["pageid"] in have or not str(info.get("mime", "")).startswith(("video/", "application/ogg")):
                    continue
                duration = float(info.get("duration") or 0)
                src = smallest(info)
                if src is None or not 2 <= duration <= 1200:
                    continue
                path = os.path.join(OUT, "raw", f"{p['pageid']}{os.path.splitext(urllib.parse.urlparse(src).path)[1] or '.webm'}")
                time.sleep(PAUSE)
                try:
                    data = fetch(src, timeout=600)
                    with open(path + ".part", "wb") as f:
                        f.write(data)
                    os.replace(path + ".part", path)
                    raw = frames(path)
                except Exception as e:  # noqa: BLE001 - a bad file is skipped, the service goes on
                    say("skip", p["title"][:70], str(e)[:80])
                    continue
                n = len(raw) // (SIDE * SIDE * 3)
                if n < 10:
                    continue
                with open(os.path.join(OUT, "frames_64.u8"), "ab") as f:  # rows of 64 x 64 x 3 bytes, memory-mapped when read
                    f.write(raw)
                entry = {"page": p["pageid"], "title": p["title"], "start": start, "frames": n, "fps": FPS, "duration": round(duration, 2),
                         "word" if not label.startswith("category:") else "category": label.replace("category:", ""),
                         "raw": os.path.basename(path), "from": src.rsplit("/", 1)[-1][-40:]}
                with open(index, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                have.add(p["pageid"])
                start, kept, got = start + n, kept + 1, got + 1
                if MAX and got >= MAX:
                    say(f"check done: {got} videos, {start} frames")
                    return
            offsets[label] = at + PAGE if pages and "continue" in d else None
            json.dump(offsets, open(state + ".new", "w"))
            os.replace(state + ".new", state)
            busy = busy or offsets[label] is not None
            say(f"{label}: {kept} videos kept of {len(pages)} found at {at}; {len(have)} videos, {start} frames in all")
        if not busy:
            say("every search is used up; looking again in a day")
            time.sleep(86400)
            offsets = {}


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
