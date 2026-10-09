"""Ask a raised brain your own questions about a scene.

    python examples/ask.py                 # a random scene
    python examples/ask.py my_picture.png  # any image (it will be shrunk to 40x40)

Run examples/multimodal_demo.py first to raise and save a brain.
"""

import os
import pickle
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.evaluate import explain  # noqa: E402
from organic.scenes import SIZE, episode  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
brain = pickle.load(open(os.path.join(OUT, "brain.pkl"), "rb"))
if len(sys.argv) > 1:
    image = np.asarray(Image.open(sys.argv[1]).convert("RGB").resize((SIZE, SIZE)), np.float32) / 255
else:
    e = episode(np.random.default_rng(), "test")
    image = e["image"]
    print("objects in the scene:", [(o["color"], o["shape"], o["size"]) for o in e["objects"]])
Image.fromarray((image * 255).astype(np.uint8)).resize((240, 240), Image.NEAREST).save(os.path.join(OUT, "scene.png"))
print("scene saved to out/scene.png. Ask away (empty line to quit, prefix with ? to see its reasoning)")
while True:
    q = input("you: ").strip()
    if not q:
        break
    if q.startswith("?"):
        print(explain(brain, image, q[1:].strip()))
    else:
        print("brain:", brain.live(image, q, learn=False)[0])
