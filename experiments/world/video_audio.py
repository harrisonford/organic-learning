"""The sound of the Commons videos, through the cochlea, aligned with their frames (queue E 32; 2026-11-26).

experiments/world/commons_video.py keeps each video's file and its first SECONDS seconds as 64 x 64 frames at 10 a
second. Here the same seconds of its sound track, as organic/cochlea.py hears them (64 bands, a frame every 10 ms),
so that ten sound frames sit under every picture frame. Videos without sound are marked and skipped. Runs over the
videos already kept, then waits an hour and looks again, for as long as the service runs.

    out   ~/datasets/commons_video/audio_64.f16   rows of 64 float16 (one per 10 ms), appended
          ~/datasets/commons_video/audio.jsonl    one line per video: page, first row, rows, whether it has sound
    python experiments/world/video_audio.py            (the service)
    ONCE=1 MAX=3 python experiments/world/video_audio.py   (a check)
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from organic.cochlea import Cochlea, RATE  # noqa: E402

DIR = os.path.expanduser("~/datasets/commons_video")
ONCE, MAX = os.environ.get("ONCE") == "1", int(os.environ.get("MAX", 0))


def say(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def sound(path, seconds):
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-t", str(seconds), "-i", path, "-vn", "-ac", "1", "-ar", str(RATE), "-f", "s16le", "pipe:1"]
    raw = subprocess.run(cmd, capture_output=True, timeout=600).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def main():
    ear, index, store = Cochlea(), os.path.join(DIR, "audio.jsonl"), os.path.join(DIR, "audio_64.f16")
    while True:
        done = {json.loads(l)["page"] for l in open(index)} if os.path.exists(index) else set()
        start = os.path.getsize(store) // (64 * 2) if os.path.exists(store) else 0
        videos = [json.loads(l) for l in open(os.path.join(DIR, "videos.jsonl"))]
        n = 0
        for v in videos:
            if v["page"] in done:
                continue
            wave = sound(os.path.join(DIR, "raw", v["raw"]), v["frames"] / v["fps"])
            gram = ear.hear(wave)[: v["frames"] * (100 // v["fps"])]
            entry = {"page": v["page"], "start": start, "rows": int(len(gram)), "sound": bool(len(gram) and float(np.abs(wave).max()) > 1e-4),
                     "per_frame": 100 // v["fps"]}
            if entry["sound"]:
                with open(store, "ab") as f:
                    f.write(gram.astype(np.float16).tobytes())
                start += len(gram)
            else:
                entry["rows"] = 0
            with open(index, "a") as f:
                f.write(json.dumps(entry) + "\n")
            n += 1
            if MAX and n >= MAX:
                say(f"check done: {n} videos")
                return
        say(f"{n} videos heard this round; {start} sound frames in all")
        if ONCE:
            return
        time.sleep(3600)


if __name__ == "__main__":
    main()
