"""Try new free OpenRouter models on the world's two jobs before adding them as adults.

    python experiments/world/model_scan.py model_id [model_id ...]

1. Writing: ten first-words items (a toddler question + 2-4 accepted answers), JSON.
2. Looking: a 64x64 CIFAR photo of known class (as the brain's eye sees it): what is it?
Each model gets two requests through the shared rate limiter (organic/adults.py).
"""
import json, sys, time
sys.path.insert(0, ".")
from organic.adults import Adults, image_part, parse_json
from organic.curriculum import Curriculum

WRITE = ("Write 10 questions an adult asks an 18-month-old about everyday things (animals, food, body, toys), "
         "each with 2-4 short answers a toddler could say that are all right. Reply only with JSON: "
         '[{"question": "...", "answers": ["...", "..."]}]')
cur = Curriculum(seed=0)
items = [x for x in cur.test["W1"]["cifar:name"]][:3]
adults = Adults()
for model in sys.argv[1:]:
    row = {"model": model}
    try:
        t0 = time.time()
        r = adults.ask(model, WRITE, purpose="scan", timeout=120)
        got = parse_json(r["text"]) or []
        ok = [g for g in got if isinstance(g, dict) and g.get("question") and isinstance(g.get("answers"), list) and 2 <= len(g["answers"]) <= 4]
        row["write"] = f"{len(ok)}/10 well-formed in {time.time() - t0:.0f}s"
        row["sample"] = [(g["question"], g["answers"]) for g in ok[:2]]
    except Exception as e:  # noqa: BLE001
        row["write"] = f"error: {str(e)[:120]}"
    try:
        ref, _, answers = items[0][:3]
        r = adults.ask(model, [{"type": "text", "text": "What is in this small blurry photo? One or two words."}, image_part(cur.image(ref))], max_tokens=2000, purpose="scan", timeout=120)
        row["look"] = f"said {r['text'].strip()[:40]!r}, truth {answers[0]!r}"
    except Exception as e:  # noqa: BLE001
        row["look"] = f"error: {str(e)[:120]}"
    print(json.dumps(row, ensure_ascii=False), flush=True)
