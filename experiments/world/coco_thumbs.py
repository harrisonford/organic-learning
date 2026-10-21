"""COCO train2017 as the brain's eye sees it: 64x64 center-cropped thumbnails.

Reads the photos straight from the zip and stores one uint8 array
(~/datasets/coco_thumbs/train2017_64.npy, memory-mapped by the world) and the
COCO image id of every row.

    python experiments/world/coco_thumbs.py ~/datasets/train2017.zip
"""

import io
import json
import os
import sys
import zipfile
from multiprocessing import Pool

import numpy as np
from PIL import Image

SIZE = 64
src = os.path.expanduser(sys.argv[1])
dst = os.path.expanduser("~/datasets/coco_thumbs")


_zip = None


def _open():
    global _zip
    _zip = zipfile.ZipFile(src)


def thumb(name):
    im = Image.open(io.BytesIO(_zip.read(name))).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((SIZE, SIZE), Image.BILINEAR)
    return np.asarray(im, np.uint8)


if __name__ == "__main__":
    os.makedirs(dst, exist_ok=True)
    names = sorted(n for n in zipfile.ZipFile(src).namelist() if n.endswith(".jpg"))
    out = np.lib.format.open_memmap(os.path.join(dst, "train2017_64.npy"), mode="w+", dtype=np.uint8, shape=(len(names), SIZE, SIZE, 3))
    with Pool(int(os.environ.get("WORKERS", 8)), initializer=_open) as pool:
        for i, t in enumerate(pool.imap(thumb, names, chunksize=64)):
            out[i] = t
            if i % 10000 == 0:
                print(i, len(names), flush=True)
    out.flush()
    json.dump([int(os.path.basename(n)[:-4]) for n in names], open(os.path.join(dst, "train2017_ids.json"), "w"))
    print("done", len(names))
