"""Daemon: the adults write short spoken questions and answers for the ear.

A text model writes 50 everyday question/answer pairs per request on a rotating
theme; a second text model answers the same questions; pairs both agree on are
kept. Uses at most 100 requests a day (its share of half the allowance).

    python experiments/world/text_world.py data/world/text_qa.jsonl
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from organic.adults import Adults, parse_json  # noqa: E402

SHARE = 100
WRITER, CHECKER = "nvidia/nemotron-3.5-lightning:free", "xingchen-agi/xing4.0-29b-a4b:free"
THEMES = ["colors of everyday things", "animals and the sounds they make", "opposites", "small additions and subtractions with numbers up to twenty",
          "what everyday objects are used for", "days, months and seasons", "feelings and what causes them", "food and where it comes from",
          "jobs and what people in them do", "weather", "parts of the body", "places in a town", "simple yes or no facts about the world",
          "greetings and polite replies", "shapes and sizes", "counting things", "family members", "sports", "vehicles", "music and instruments"]
out = sys.argv[1] if len(sys.argv) > 1 else "data/world/text_qa.jsonl"
adults = Adults()


def text_used():
    return adults._state().get("text_count", 0)


def log(msg):
    print(time.strftime("[%F %T] ") + msg, flush=True)


n = sum(1 for _ in open(out)) if os.path.exists(out) else 0
k = n // 40
while True:
    if text_used() + 2 > SHARE:
        log("daily share reached; sleeping")
        while text_used() + 2 > SHARE:
            time.sleep(600)
    theme = THEMES[k % len(THEMES)]
    k += 1
    try:
        w = adults.ask(WRITER, f"Write 50 different short questions a child might be asked about {theme} (variation {k}), each with a 1-3 word lowercase answer. "
                       'Reply only with JSON: [{"q": "...", "a": "..."}]', purpose="text", temperature=0.8)
        pairs = [p for p in (parse_json(w["text"]) or []) if isinstance(p, dict) and p.get("q") and p.get("a")]
        qs = [p["q"] for p in pairs]
        c = adults.ask(CHECKER, "Answer each question in 1-3 lowercase words. Reply only with a JSON list of answers in the same order.\n" + json.dumps(qs), purpose="text")
        answers = parse_json(c["text"]) or []
    except Exception as e:  # noqa: BLE001
        log(f"failed: {e}")
        time.sleep(60)
        continue
    with open(out, "a") as f:
        for p, a2 in zip(pairs, answers if isinstance(answers, list) else []):
            a1 = str(p["a"]).strip().lower()
            a2 = str(a2).strip().lower()
            f.write(json.dumps({"q": p["q"].strip().lower(), "a": a1, "a_check": a2, "agree": a1 == a2, "theme": theme}) + "\n")
    log(f"{theme}: {len(pairs)} pairs, {text_used()} text requests today")
