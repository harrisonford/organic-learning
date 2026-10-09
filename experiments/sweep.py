"""Raise several brains with different settings in parallel and compare them.

    python experiments/sweep.py 1500 '{"spread": 0.3}' '{"spread": 1.0, "rem_weight": 0.6}'
"""

import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.brain import Brain  # noqa: E402
from organic.evaluate import evaluate  # noqa: E402
from organic.scenes import episode  # noqa: E402


def run(args):
    n, kwargs = args
    rng = np.random.default_rng(0)
    b = Brain(**kwargs)
    for _ in range(n):
        e = episode(rng)
        b.live(e["image"], e["question"], e["answer"])
    lines = [f"{json.dumps(kwargs)}"]
    total = []
    for split in ("test", "combo", "phrase"):
        score, per = evaluate(b, split, n=300)
        total.append(score)
        lines.append(f"   {split:<6} {score:.2f}  " + " ".join(f"{k}={v:.2f}" for k, v in per.items()))
    return sum(total), "\n".join(lines)


if __name__ == "__main__":
    n = int(sys.argv[1])
    configs = [json.loads(c) for c in sys.argv[2:]] or [{}]
    with Pool(min(len(configs), 14)) as pool:
        results = pool.map(run, [(n, c) for c in configs])
    for _, text in sorted(results, key=lambda r: -r[0]):
        print(text)
