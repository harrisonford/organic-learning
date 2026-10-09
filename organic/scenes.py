"""Scenes with objects, and the things a person might ask and answer about them.

Every episode is: an image, a question someone asks, and the answer a person
would give. The organism hears the answer (it imitates, like a child), it is
never told which words are colors, shapes or places.

Some combinations are held out of training on purpose (e.g. no green triangle
is ever seen) and so are some ways of asking, to test whether understanding is
compositional or just memorized.
"""

import numpy as np
from PIL import Image, ImageDraw

SIZE = 40
COLORS = {
    "red": (0.95, 0.15, 0.1),
    "green": (0.15, 0.85, 0.2),
    "blue": (0.2, 0.35, 1.0),
    "yellow": (0.95, 0.9, 0.1),
    "purple": (0.65, 0.2, 0.9),
    "white": (0.95, 0.95, 0.95),
}
SHAPES = ["circle", "square", "triangle", "cross", "star", "face"]
HELD_OUT_COMBOS = {("triangle", "green"), ("star", "blue"), ("face", "purple"), ("circle", "white")}

# grid cells: (column, row) -> how a person says where it is
PLACES = {
    (0, 0): "at the top left", (1, 0): "at the top", (2, 0): "at the top right",
    (0, 1): "on the left", (1, 1): "in the middle", (2, 1): "on the right",
    (0, 2): "at the bottom left", (1, 2): "at the bottom", (2, 2): "at the bottom right",
}
NUMBERS = {1: "one", 2: "two", 3: "three", 4: "four"}


def draw_object(d, shape, color, cx, cy, r, scale):
    fill = tuple(int(255 * c) for c in color)
    w = max(2, int(scale * 1.8))
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
    elif shape == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=fill)
    elif shape == "triangle":
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=fill)
    elif shape == "cross":
        d.line([cx - r, cy - r, cx + r, cy + r], fill=fill, width=int(w * 1.6))
        d.line([cx - r, cy + r, cx + r, cy - r], fill=fill, width=int(w * 1.6))
    elif shape == "star":
        pts = []
        for k in range(10):
            a = -np.pi / 2 + k * np.pi / 5
            rad = r if k % 2 == 0 else r * 0.42
            pts.append((cx + rad * np.cos(a), cy + rad * np.sin(a)))
        d.polygon(pts, fill=fill)
    elif shape == "face":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=fill, width=w)
        e = r * 0.16
        for ex in (cx - r * 0.38, cx + r * 0.38):
            d.ellipse([ex - e, cy - r * 0.3 - e, ex + e, cy - r * 0.3 + e], fill=fill)
        d.arc([cx - r * 0.55, cy - r * 0.35, cx + r * 0.55, cy + r * 0.6], 20, 160, fill=fill, width=w)


def render(objects, rng, noise=0.06):
    """objects: list of dicts with shape, color, cell, size ('big'/'small')."""
    scale = 6
    big = SIZE * scale
    img = Image.new("RGB", (big, big), (0, 0, 0))
    d = ImageDraw.Draw(img)
    for o in objects:
        col, row = o["cell"]
        cx = (0.2 + 0.3 * col + rng.uniform(-0.03, 0.03)) * big
        cy = (0.2 + 0.3 * row + rng.uniform(-0.03, 0.03)) * big
        r = (rng.uniform(0.17, 0.2) if o["size"] == "big" else rng.uniform(0.09, 0.12)) * big
        draw_object(d, o["shape"], COLORS[o["color"]], cx, cy, r, scale)
    arr = np.asarray(img.resize((SIZE, SIZE), Image.LANCZOS), np.float32) / 255
    arr = arr * rng.uniform(0.85, 1.0) + rng.normal(0, noise, arr.shape)
    return np.clip(arr, 0, 1)


def random_object(rng, held_out, size=None, taken=()):
    """A random object; held_out picks from (or avoids) the held-out combinations."""
    while True:
        shape = SHAPES[rng.integers(len(SHAPES))]
        color = list(COLORS)[rng.integers(len(COLORS))]
        if ((shape, color) in HELD_OUT_COMBOS) == held_out:
            break
    size = size or ("big" if rng.random() < 0.4 else "small")
    cells = [c for c in PLACES if c not in taken]
    return {"shape": shape, "color": color, "cell": cells[rng.integers(len(cells))], "size": size}


# ways people ask; the last phrasing of each intent is never used in training
ASK = {
    "what": ["what is in this image ?", "what do you see ?", "what is this ?", "what can you see in the picture ?"],
    "color": ["what color is it ?", "which color is this ?", "what color is the shape ?", "tell me its color"],
    "shape": ["what shape is it ?", "which shape is this ?", "what kind of shape is it ?"],
    "where": ["where is it ?", "where is the shape ?", "where can you see it ?"],
    "size": ["is it big or small ?", "how big is it ?", "what size is it ?"],
    "funny": ["is this funny ?", "isn't this funny ?", "does this make you laugh ?"],
    "is_shape": ["is it a {x} ?", "is this a {x} ?", "do you see a {x} ?"],
    "is_color": ["is it {x} ?", "is this {x} ?", "is the shape {x} ?"],
    "count": ["how many shapes are there ?", "how many things do you see ?", "count the shapes"],
}
SINGLE_INTENTS = ["what", "color", "shape", "where", "size", "funny", "is_shape", "is_color", "count"]


def answer(intent, objs, x, rng):
    """What a person would say. Some intents have more than one natural answer."""
    o = objs[0]
    sh, co = o["shape"], o["color"]
    if intent == "what":
        if len(objs) > 1:
            return "i see " + " and ".join(f"a {p['color']} {p['shape']}" for p in objs)
        return [f"i see a {co} {sh}", f"it is a {co} {sh}"][rng.integers(2)]
    if intent == "color":
        return f"it is {co}"
    if intent == "shape":
        return f"it is a {sh}"
    if intent == "where":
        return f"it is {PLACES[o['cell']]}"
    if intent == "size":
        return f"it is {o['size']}"
    if intent == "funny":
        if sh == "face":
            return "haha yes , it is a funny face"
        return f"not really , it is just a {co} {sh}"
    if intent == "is_shape":
        return f"yes , it is a {sh}" if x == sh else f"no , it is a {sh}"
    if intent == "is_color":
        return f"yes , it is {co}" if x == co else f"no , it is {co}"
    if intent == "count":
        n = len(objs)
        return "there is one shape" if n == 1 else f"there are {NUMBERS[n]} shapes"
    raise ValueError(intent)


def episode(rng, split="train"):
    """One moment of life: a scene, a question and a person's answer.

    split: 'train'   - training combos and phrasings
           'test'    - new images, training phrasings, training combos
           'combo'   - held-out color+shape combinations
           'phrase'  - held-out phrasings
    """
    held = split == "combo"
    n = 1
    intent = SINGLE_INTENTS[rng.integers(len(SINGLE_INTENTS))]
    if intent in ("count", "what") and split != "combo" and rng.random() < 0.4:
        n = int(rng.integers(2, 4))
    objs, taken = [], []
    for k in range(n):
        o = random_object(rng, held if k == 0 else False, size="small" if n > 1 else None, taken=taken)
        objs.append(o)
        taken.append(o["cell"])
    phrasings = ASK[intent]
    if split == "phrase":
        q = phrasings[-1]
    else:
        q = phrasings[rng.integers(len(phrasings) - 1)]
    x = None
    if intent == "is_shape":
        x = objs[0]["shape"] if rng.random() < 0.5 else SHAPES[rng.integers(len(SHAPES))]
    elif intent == "is_color":
        x = objs[0]["color"] if rng.random() < 0.5 else list(COLORS)[rng.integers(len(COLORS))]
    q = q.replace("{x}", x or "")
    return {"image": render(objs, rng), "objects": objs, "intent": intent, "question": q, "answer": answer(intent, objs, x, rng)}
