"""Raise a brain on scenes + questions, then talk with it about new scenes.

    python examples/multimodal_demo.py [episodes]

Everything it hears is raw bytes. Nothing is labeled: it only ever sees a
scene, hears a question, and hears how a person answered.
Writes out/conversations.png and saves the brain to out/brain.pkl
(use examples/ask.py to ask it your own questions).
"""

import os
import pickle
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.brain import Brain  # noqa: E402
from organic.data import DIALOGUES  # noqa: E402
from organic.evaluate import evaluate, same_answer  # noqa: E402
from organic.scenes import LOGIC_INTENTS, SINGLE_INTENTS, episode  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
ALL = SINGLE_INTENTS + LOGIC_INTENTS


def sheet(rows, path):
    cell, text_h, cols = 120, 70, 4
    img = Image.new("RGB", (cols * 300, ((len(rows) + cols - 1) // cols) * (cell + text_h)), "white")
    d = ImageDraw.Draw(img)
    for i, (e, out, ok) in enumerate(rows):
        x, y = (i % cols) * 300, (i // cols) * (cell + text_h)
        img.paste(Image.fromarray((e["image"] * 255).astype(np.uint8)).resize((cell, cell), Image.NEAREST), (x + 4, y + 4))
        d.text((x + 4, y + cell + 6), "Q: " + e["question"], fill=(0, 0, 0))
        said = out if len(out) <= 44 else out[:42] + "..."
        d.text((x + 4, y + cell + 22), "brain: " + said, fill=(0, 120, 0) if ok else (190, 0, 0))
        d.text((x + 4, y + cell + 38), "person: " + e["answer"], fill=(90, 90, 90))
    img.save(path)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    brain = Brain()
    t0 = time.time()
    print("== growing up ==")
    for age in range(1, n + 1):
        if rng.random() < 0.08:  # sometimes just talk, eyes closed
            prompt, reply = DIALOGUES[rng.integers(len(DIALOGUES))]
            brain.live(None, prompt, reply)
        else:
            e = episode(rng, "train", ALL)
            brain.live(e["image"], e["question"], e["answer"])
        if age % 1000 == 0:
            dreams, died = brain.sleep(replay=200)
            print(f"day {age // 1000}: slept, replayed {dreams} memories | {brain.report()}  ({time.time() - t0:.0f}s)")

    print("\n== how it segments raw bytes now ==")
    for text in ["what color is it?", "is it red and big?", "haha yes, it is a funny face"]:
        print("   " + " | ".join(u.decode(errors="replace") for u in brain.ear.units(text, learn=False)))

    print("\n== tests ==")
    for split, label in [("test", "new scenes"), ("combo", "never-seen color+shape pairs"), ("phrase", "never-heard phrasings")]:
        score, per = evaluate(brain, split, n=300, intents=ALL if split != "combo" else SINGLE_INTENTS)
        print(f"{label:<30} {score:.2f}   " + " ".join(f"{k}={v:.2f}" for k, v in per.items()))

    print("\n== a few conversations ==")
    r = np.random.default_rng(42)
    rows = []
    for _ in range(16):
        e = episode(r, "test", ALL)
        out, _ = brain.live(e["image"], e["question"], learn=False)
        ok = same_answer(out, e["answer"], e["intent"], len(e["objects"]))
        rows.append((e, out, ok))
        print(f"   {'ok ' if ok else 'BAD'} {e['question']:<32} -> {out:<40} (person: {e['answer']})")
    sheet(rows, os.path.join(OUT, "conversations.png"))

    print("\n== small talk, eyes closed ==")
    for prompt in ["hello", "how are you?", "what is your name?", "what do cats say?", "hey, how are you?", "what is a brain?"]:
        print(f"   you: {prompt:<22} brain: {brain.live(None, prompt, learn=False)[0]}")
    pickle.dump(brain, open(os.path.join(OUT, "brain.pkl"), "wb"))
    print(f"\nsaved out/conversations.png and out/brain.pkl  |  {brain.report()}")


if __name__ == "__main__":
    main()
