"""More and better pictures for the picture book: things cut out of the COCO training photographs (no model requests).

Naming from the visual code is still rising with every doubling of the pictures seen (0.33 after 500, 0.55 after
8,000; docs/VISION.md), and the picture book had about 10,000, most of them CIFAR's 32x32. A child sees a dog
thousands of times. COCO's 118,000 training photographs have every dog, cat, bus... outlined by people. Each
outlined thing that is big enough becomes a 64x64 picture named with the word a toddler would use: the square
around its outline, a little wider, cut from the photograph at its own resolution.

Only train2017 is used; the curriculum's photo questions are on val2017, so nothing held out is seen.
    python experiments/world/coco_crops.py   -> ~/datasets/cococrops/pictures_64.u8 + pictures.jsonl
"""
import io, json, os, random, sys, zipfile
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from organic.world import DATA, SIZE  # noqa: E402
from organic.curriculum import PICTURE_BOOK, TODDLER_NOUNS  # noqa: E402

OUT = os.path.join(DATA, "cococrops")
PER_WORD = int(os.environ.get("PER_WORD", 5000))
SMALLEST = int(os.environ.get("SMALLEST", 80))  # an outline at least this many pixels on its shorter side
os.makedirs(OUT, exist_ok=True)
ann = json.load(open(os.path.join(DATA, "annotations", "instances_train2017.json")))
word = {c["id"]: c["name"] for c in ann["categories"] if c["name"] in PICTURE_BOOK | TODDLER_NOUNS}
size = {im["id"]: (im["width"], im["height"]) for im in ann["images"]}
picks = {}
for a in ann["annotations"]:
    x, y, w, h = a["bbox"]
    if a["category_id"] in word and not a["iscrowd"] and min(w, h) >= SMALLEST and 0.4 <= w / h <= 2.5:
        picks.setdefault(word[a["category_id"]], []).append((a["image_id"], a["id"], x, y, w, h))
del ann
rng = random.Random(0)
todo = {}
for wd, rows in sorted(picks.items()):
    rng.shuffle(rows)
    for r in rows[:PER_WORD]:
        todo.setdefault(r[0], []).append((wd,) + r[1:])
print({wd: min(len(rows), PER_WORD) for wd, rows in sorted(picks.items())}, flush=True)
n = sum(len(v) for v in todo.values())
out = np.lib.format.open_memmap(os.path.join(OUT, "pictures_64.npy"), "w+", np.uint8, (n, SIZE, SIZE, 3))
z = zipfile.ZipFile(os.path.join(DATA, "train2017.zip"))
row = 0
with open(os.path.join(OUT, "pictures.jsonl"), "w") as f:
    for k, (image_id, things) in enumerate(sorted(todo.items())):
        im = Image.open(io.BytesIO(z.read(f"train2017/{image_id:012d}.jpg"))).convert("RGB")
        W, H = im.size
        for wd, ann_id, x, y, w, h in things:
            side = min(max(w, h) * 1.1, W, H)
            cx, cy = x + w / 2, y + h / 2
            left = min(max(cx - side / 2, 0), W - side)
            top = min(max(cy - side / 2, 0), H - side)
            out[row] = np.asarray(im.crop((int(left), int(top), int(left + side), int(top + side))).resize((SIZE, SIZE), Image.BILINEAR))
            f.write(json.dumps({"row": row, "word": wd, "image": image_id, "thing": ann_id}) + "\n")
            row += 1
        if k % 5000 == 0:
            print(k, "photographs,", row, "pictures", flush=True)
out.flush()
print("done:", row, "pictures", flush=True)
