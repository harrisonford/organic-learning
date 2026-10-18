"""How do the adult models do on the brain's test questions, and how fast?

80 held-out items, balanced by kind, from human annotations and reading only
(never questions an adult model wrote). 4 photos per request. Uses the daily
reserve (60 requests): 20 requests each for Nemotron VL, Dots 3 and Jev (Jev is
capped at $1.50 of real spend). Writes data/world/counterpart.json.

    python experiments/counterpart.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.adults import JEV, Adults, image_part, parse_json  # noqa: E402
from organic.world import World  # noqa: E402

MODELS = ["nvidia/nemotron-3.5-super-vl:free", "dots-studio/dots-3-note-preview:free", JEV]
world = World(seed=0)
by = {}
for ref, q, a, kind in world.test:
    if kind in ("presence", "count", "identity", "where", "text") and not (isinstance(ref, str) is False and ref[0] == "dark"):
        by.setdefault(kind, []).append((ref, q, a, kind))
items = []
for kind in sorted(by):
    items += by[kind][:16]
adults = Adults()
results = {}
for model in MODELS:
    rows = []
    for i in range(0, len(items), 4):
        batch = items[i : i + 4]
        parts = [{"type": "text", "text": "Answer each question about these small photos (numbered in order) in 1-3 lowercase words. "
                  'Reply only with JSON: ["answer 1", "answer 2", ...]\n' + "\n".join(f"{k + 1}. {q}" for k, (_, q, _, _) in enumerate(batch))}]
        for k, (ref, _, _, _) in enumerate(batch):
            parts += [{"type": "text", "text": f"Photo {k + 1}:"}, image_part(world.image(ref))]
        try:
            r = adults.ask(model, parts, purpose="reserve")
            answers = parse_json(r["text"]) or []
            sec, used = r["seconds"], r.get("model", model)
        except Exception as e:  # noqa: BLE001
            print(model, "stopped:", e, flush=True)
            break
        for (ref, q, a, kind), ans in zip(batch, answers + [None] * 4):
            rows.append({"kind": kind, "q": q, "a": a, "said": str(ans).strip().lower() if ans is not None else None, "seconds_per_request": sec, "routed_to": used})
        print(model, len(rows), "items", flush=True)
    acc = {}
    for row in rows:
        acc.setdefault(row["kind"], []).append(row["said"] == row["a"])
    results[model] = {"accuracy": {k: sum(v) / len(v) for k, v in acc.items()}, "n": len(rows),
                      "mean_seconds_per_request": sum(r["seconds_per_request"] for r in rows) / max(len(rows), 1), "rows": rows}
json.dump(results, open("data/world/counterpart.json", "w"), indent=1)
for m, r in results.items():
    print(m, r["n"], "items, s/request %.1f" % r["mean_seconds_per_request"], {k: round(v, 2) for k, v in r["accuracy"].items()})
