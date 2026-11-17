"""Picture-book scenes for two words (docs/VISION.md, 2026-11-15).

The two-words stage asked about photographs (COCO, VQA): how many people, is there an umbrella, what colour is the
bus. No readout of the brain's visual code answers those much above the base rate (the eyes see a 64x64 whole view),
and six lives stopped there. A two-year-old is not asked to count people in street photographs either; it is shown a
page with two dogs and a ball. A scene here is such a page: up to four of the picture-book pictures the first-words
stage names one by one, each in a quarter of the view.

    how many:  "how many dogs are there?"  -> one / two / three     (other pictures or empty quarters beside them)
    is there:  "is there a dog?"           -> yes / no              (half of each)

Pictures of the lived side make the lived scenes, held-out pictures the held-out scenes, so a test scene is made of
pictures never seen. The wordings are few and written by hand for now; the adults should write them.
"""
import random
import zlib

import numpy as np
from PIL import Image

PLURAL = {"mouse": "mice", "sheep": "sheep", "fish": "fish", "bus": "buses", "baby": "babies", "butterfly": "butterflies", "box": "boxes",
          "fox": "foxes", "wolf": "wolves", "leaf": "leaves", "man": "men", "woman": "women", "child": "children", "foot": "feet", "tooth": "teeth"}
COUNT = ["how many {ws} are there?", "how many {ws}?", "how many {ws} can you see?", "count the {ws}"]
THERE = ["is there {a} {w}?", "can you see {a} {w}?", "do you see {a} {w} here?", "is {a} {w} here?"]
NUMBER = ["zero", "one", "two", "three", "four"]


def plural(w):
    return PLURAL.get(w, w + ("es" if w.endswith(("s", "sh", "ch", "x")) else "s"))


def page(tiles, shade, size=30, cell=32):
    """Up to four 64x64 pictures (or None), each in a quarter of a page of one light shade. `cell` is the side of a
    quarter and `size` the side of a picture in it: (30, 32) a 64x64 page with a margin of one pixel; (26, 32) of
    three, as the pages of world 4 had at first; (52, 64) a page twice as large as the view, which the eye sees whole
    only coarsely and in full detail where it fixates."""
    out = np.full((2 * cell, 2 * cell, 3), shade, np.float32)
    m = (cell - size) // 2
    for k, t in enumerate(tiles):
        if t is not None:
            small = t if size == t.shape[0] else np.asarray(Image.fromarray((np.clip(t, 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR), np.float32) / 255
            r, c = divmod(k, 2)
            out[r * cell + m : r * cell + m + size, c * cell + m : c * cell + m + size] = small
    return out


class Scenes:
    """Scenes made from named pictures: by_name = {name: [picture refs]}, image(ref) -> 64x64x3 array."""

    def __init__(self, by_name, image, seed=0, size=30):
        self.by_name = {w: refs for w, refs in by_name.items() if len(refs) >= 3}
        self.names = sorted(self.by_name)
        self.image = image
        self.seed = seed
        self.size = size  # the side of a picture on the page

    def named(self, n):
        """Scene number n of the kind "name": a thing on a page, one time in three with another beside it, to be
        named when asked. (tile refs, shade, the names on the page: the thing first)."""
        rng = random.Random(zlib.crc32(f"{self.seed}:name:{n}".encode()))
        w = rng.choice(self.names)
        on = [w] + ([rng.choice([x for x in self.names if x != w])] if rng.random() < 1 / 3 else [])
        tiles = [rng.choice(self.by_name[x]) for x in on] + [None] * (4 - len(on))
        rng.shuffle(tiles)
        return tiles, rng.uniform(0.75, 0.95), on

    def make(self, kind, n):
        """Scene number n of a kind ("count" or "there"), always the same: (tile refs, shade, question, answer)."""
        rng = random.Random(zlib.crc32(f"{self.seed}:{kind}:{n}".encode()))
        w = rng.choice(self.names)
        others = [x for x in self.names if x != w]
        if kind == "count":
            k = rng.choice([1, 2, 3])
            tiles = rng.sample(self.by_name[w], k)
            for _ in range(rng.randint(0, 4 - k)):
                tiles.append(rng.choice(self.by_name[rng.choice(others)]))
            q, a = rng.choice(COUNT).format(ws=plural(w)), NUMBER[k]
        else:
            yes = rng.random() < 0.5
            tiles = [rng.choice(self.by_name[rng.choice(others)]) for _ in range(rng.randint(1, 3))]
            if yes:
                tiles.append(rng.choice(self.by_name[w]))
            q, a = rng.choice(THERE).format(a="an" if w[0] in "aeiou" else "a", w=w), "yes" if yes else "no"
        tiles += [None] * (4 - len(tiles))
        rng.shuffle(tiles)
        return tiles, rng.uniform(0.75, 0.95), q, a

    def picture(self, kind, n):
        tiles, shade = (self.named(n) if kind == "name" else self.make(kind, n))[:2]
        return page([self.image(t) if t is not None else None for t in tiles], shade, self.size)
