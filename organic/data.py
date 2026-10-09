"""Tiny built-in datasets so the demo needs no downloads."""

import numpy as np
from PIL import Image, ImageDraw

SHAPES = ["circle", "square", "triangle", "cross", "hline", "vline"]


def draw_shape(name, rng, size=16, noise=0.15):
    """Render a shape with random position, scale and pixel noise."""
    scale = 8
    big = size * scale
    img = Image.new("L", (big, big), 0)
    d = ImageDraw.Draw(img)
    r = rng.uniform(0.22, 0.36) * big
    cx = rng.uniform(r + 4, big - r - 4)
    cy = rng.uniform(r + 4, big - r - 4)
    w = int(scale * 1.4)
    if name == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=255, width=w)
    elif name == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], outline=255, width=w)
    elif name == "triangle":
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], outline=255, width=w)
    elif name == "cross":
        d.line([cx - r, cy - r, cx + r, cy + r], fill=255, width=w)
        d.line([cx - r, cy + r, cx + r, cy - r], fill=255, width=w)
    elif name == "hline":
        d.line([cx - r, cy, cx + r, cy], fill=255, width=w)
    elif name == "vline":
        d.line([cx, cy - r, cx, cy + r], fill=255, width=w)
    small = np.asarray(img.resize((size, size), Image.BILINEAR), np.float32) / 255.0
    small = small + rng.normal(0, noise, small.shape)
    return np.clip(small, 0, 1)


def shape_dataset(n, seed):
    rng = np.random.default_rng(seed)
    labels = [SHAPES[i % len(SHAPES)] for i in range(n)]
    rng.shuffle(labels)
    return [(draw_shape(lab, rng), lab) for lab in labels]


DIALOGUES = [
    ("hello", "hi there , nice to meet you"),
    ("hi", "hello , how can i help you ?"),
    ("good morning", "good morning to you too"),
    ("how are you ?", "i am fine , thank you"),
    ("how are you doing today ?", "i am doing well today"),
    ("what is your name ?", "my name is organic"),
    ("who are you ?", "i am a small growing brain"),
    ("what are you ?", "i am a cluster of neurons"),
    ("what color is the sky ?", "the sky is blue"),
    ("what color is the grass ?", "the grass is green"),
    ("what color is the sun ?", "the sun is yellow"),
    ("what color is snow ?", "snow is white"),
    ("what color is the night ?", "the night is black"),
    ("what do cats say ?", "cats say meow"),
    ("what do dogs say ?", "dogs say woof"),
    ("what do cows say ?", "cows say moo"),
    ("what do you eat ?", "i eat data and electricity"),
    ("do you like music ?", "yes , i like music very much"),
    ("do you like math ?", "yes , math is beautiful"),
    ("do you sleep ?", "yes , when i sleep weak neurons die"),
    ("how do you learn ?", "i grow new neurons when nothing feels familiar"),
    ("what is a neuron ?", "a neuron is a small cell that fires"),
    ("what is the brain ?", "the brain is a forest of neurons"),
    ("where do you live ?", "i live inside a computer"),
    ("are you alive ?", "i am not sure , but i am growing"),
    ("can you see ?", "yes , i have an eye for shapes"),
    ("what shapes do you know ?", "i know circles , squares and triangles"),
    ("what is two plus two ?", "two plus two is four"),
    ("what is one plus one ?", "one plus one is two"),
    ("what is three plus three ?", "three plus three is six"),
    ("tell me a joke", "why did the neuron cross the synapse ? to fire"),
    ("thank you", "you are welcome"),
    ("goodbye", "goodbye , see you soon"),
    ("good night", "good night , sleep well"),
]
