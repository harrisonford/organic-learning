"""Raise one organism on both shape recognition and small talk, then test it.

    python examples/demo.py
"""

import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic import Organism  # noqa: E402
from organic.data import DIALOGUES, SHAPES, shape_dataset  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "out")


def save_predictions(samples, preds, path):
    cell, pad = 64, 18
    cols = 10
    rows = (len(samples) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + pad)), "white")
    d = ImageDraw.Draw(sheet)
    for i, ((img, label), (pred, conf)) in enumerate(zip(samples, preds)):
        x, y = (i % cols) * cell, (i // cols) * (cell + pad)
        tile = Image.fromarray((img * 255).astype(np.uint8)).resize((cell - 4, cell - 4), Image.NEAREST)
        sheet.paste(tile.convert("RGB"), (x + 2, y + 2))
        d.text((x + 3, y + cell), pred, fill=(0, 130, 0) if pred == label else (200, 0, 0))
    sheet.save(path)


def main():
    os.makedirs(OUT, exist_ok=True)
    org = Organism(seed=0)
    train = shape_dataset(900, seed=1)
    test = shape_dataset(300, seed=2)

    print("== life: interleaved experience of shapes and conversations ==")
    t0 = time.time()
    epochs = 4
    per_epoch = len(train) // epochs
    for epoch in range(epochs):
        seen, correct = 0, 0
        talks = list(DIALOGUES)
        org.cortex.rng.shuffle(talks)
        for i, (img, label) in enumerate(train[epoch * per_epoch : (epoch + 1) * per_epoch]):
            r = org.vision.learn(img, label)
            seen += 1
            correct += r["correct"]
            if i % 5 == 0:  # hear some conversation every few things it sees
                for prompt, response in talks[: 2]:
                    org.language.learn_dialogue(prompt, response)
                talks = talks[2:] or list(DIALOGUES)
        for prompt, response in DIALOGUES:  # and a full round of talking per epoch
            org.language.learn_dialogue(prompt, response)
        died = org.cortex.sleep()
        print(f"day {epoch + 1}: vision online acc={correct / seen:.2f}  slept, {died} clusters died")
        print(f"   {org.report()}")
    print(f"(lived {time.time() - t0:.1f}s)\n")

    print("== vision test on unseen shapes ==")
    preds = [org.vision.recognize(img) for img, _ in test]
    acc = np.mean([p == lab for (p, _), (_, lab) in zip(preds, test)])
    print(f"accuracy {acc:.2f} (chance {1 / len(SHAPES):.2f})")
    print("confusion (rows=true, cols=predicted):")
    print("          " + " ".join(f"{s[:6]:>7}" for s in SHAPES))
    for s in SHAPES:
        row = [sum(1 for (p, _), (_, lab) in zip(preds, test) if lab == s and p == q) for q in SHAPES]
        print(f"{s:>9} " + " ".join(f"{v:>7}" for v in row))
    save_predictions(test[:40], preds[:40], os.path.join(OUT, "vision_predictions.png"))
    print(f"saved {os.path.normpath(os.path.join(OUT, 'vision_predictions.png'))}\n")

    print("== conversation ==")
    prompts = [p for p, _ in DIALOGUES[:6]] + [
        "what color is the sky ?",
        "what do dogs say ?",
        "what is one plus one ?",
        # never heard these exact sentences:
        "hey , how are you ?",
        "what color is grass ?",
        "tell me who you are",
        "do you like dogs ?",
        "what is a brain ?",
    ]
    for p in prompts:
        print(f"  you: {p}\n  org: {org.language.respond(p)}")
    print("\n" + org.report())


if __name__ == "__main__":
    main()
