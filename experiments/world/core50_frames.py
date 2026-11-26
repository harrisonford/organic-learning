"""CORe50's videos as the eye sees them: 64 x 64 frames, in their order (queue E 31 (b); 2026-11-26).

CORe50 (Lomonaco & Maltoni 2017; CC BY 4.0): 50 handheld objects of 10 categories, each filmed for about 15 s at
20 frames a second in each of 11 sessions (backgrounds and light differ), cropped to 128 x 128 around the object.
Here every frame, halved by averaging 2 x 2 pixels (the eye is 64 across), in the order it was filmed:

    ~/datasets/core50/frames_64.u8      rows of 64 x 64 x 3 bytes
    ~/datasets/core50/frames.jsonl      one line per (session, object) video: session, object, category, first row, frames

The zip is downloaded data: read only, never run; this script is run with `python -I` from the repository.
    .venv/bin/python -I experiments/world/core50_frames.py
"""
import io
import json
import os
import re
import zipfile

import numpy as np
from PIL import Image

DIR = os.path.expanduser("~/datasets/core50")
ZIP = os.path.join(DIR, "core50_128x128.zip")
NAME = re.compile(r"s(\d+)/o(\d+)/C_(\d+)_(\d+)_(\d+)\.png$")
CATEGORIES = ["plug adapter", "mobile phone", "scissors", "light bulb", "can", "glasses", "ball", "marker", "cup", "remote control"]


def main():
    z = zipfile.ZipFile(ZIP)
    videos = {}
    for n in z.namelist():
        m = NAME.search(n)
        if m:
            videos.setdefault((int(m.group(1)), int(m.group(2))), []).append((int(m.group(5)), n))
    out, index, row = os.path.join(DIR, "frames_64.u8"), os.path.join(DIR, "frames.jsonl"), 0
    with open(out, "wb") as f, open(index, "w") as ix:
        for (s, o), frames in sorted(videos.items()):
            frames.sort()
            for _, n in frames:
                img = Image.open(io.BytesIO(z.read(n))).convert("RGB")
                a = np.asarray(img, np.float32)
                a = a.reshape(64, 2, 64, 2, 3).mean(axis=(1, 3)) if a.shape == (128, 128, 3) else np.asarray(img.resize((64, 64), Image.BOX), np.float32)
                f.write(np.clip(np.round(a), 0, 255).astype(np.uint8).tobytes())
            ix.write(json.dumps({"session": s, "object": o, "category": CATEGORIES[(o - 1) // 5], "start": row, "frames": len(frames)}) + "\n")
            row += len(frames)
            print(f"session {s} object {o}: {len(frames)} frames ({row} in all)", flush=True)


if __name__ == "__main__":
    main()
