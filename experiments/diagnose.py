"""Live a while, then show where speech goes wrong, word by word."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.brain import Brain  # noqa: E402
from organic.evaluate import evaluate, explain  # noqa: E402
from organic.scenes import episode  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
rng = np.random.default_rng(0)
b = Brain()
acc = []
for i in range(N):
    e = episode(rng)
    acc.append(b.live(e["image"], e["question"], e["answer"])[1])
    if (i + 1) % 500 == 0:
        print(i + 1, "imitation acc (last 500)", round(np.mean(acc[-500:]), 2), b.report(), flush=True)

for split in ("test", "combo", "phrase"):
    score, per = evaluate(b, split, n=300, show=0)
    print(f"{split}: {score:.2f}  " + " ".join(f"{k}={v:.2f}" for k, v in per.items()), flush=True)

r = np.random.default_rng(5)
shown = 0
while shown < 6:
    e = episode(r, "test")
    out, _ = b.live(e["image"], e["question"], learn=False)
    from organic.evaluate import same_answer  # noqa: E402

    if not same_answer(out, e["answer"], e["intent"], len(e["objects"])):
        print(f"\n(person: {e['answer']})\n" + explain(b, e["image"], e["question"]))
        shown += 1
