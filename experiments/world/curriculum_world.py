"""Daemon: the adults write a graded curriculum (docs/CURRICULUM.md), within budget.

Each batch is one stage and one subject, from first words to fourth grade
(subjects follow the usual early-years and K-4 outline: language, number,
the world around us, social life, and from grade 2 following instructions,
from grade 4 simple coding). Every item has several acceptable answers in the
form expected at that stage (one word ... two sentences), so later grading can
be local. A second model checks that the answers are right and that the item
fits the stage. Text only for now (photo items per stage come from the photo
daemon and existing datasets).

    python experiments/world/curriculum_world.py data/world/curriculum.jsonl
"""

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from organic.adults import PER_DAY, Adults, parse_json  # noqa: E402

SHARE = 100  # curriculum share of the 500/day: text world 140, curriculum 100, photos 200, reserve 60
MODELS = ["nvidia/nemotron-3.5-lightning:free", "nvidia/nemotron-3.5-super-vl:free", "dots-studio/dots-3-note-preview:free", "recursive/flint-v1.0:free"]  # flint added 2026-11-04 (scan: out/model_scan.log)
# stage keys: W1 first words, W2 two words, P preschool, K kindergarten, G1-G4 grades (docs/CURRICULUM.md)
# subjects follow the usual early-years and K-4 outlines: language and number, and from preschool on the
# world around us (science), people long ago and far away (history, geography), living together (civics,
# money, health) and the arts; widened 2026-11-05 (user: "ask things about history, math, science too")
STAGES = {
    "W1": ("a toddler saying first words (12-18 months)", "one word", 3,
          ["animal sounds", "social words: hi, bye, please, more, no", "names of family members", "everyday things a toddler knows", "body parts", "food a toddler eats",
           "vehicles and things that go", "toys and play", "clothes", "things in the house", "things outside: sun, tree, rain, dog"]),
    "W2": ("a two-year-old (two-word stage)", "one or two words", 3,
          ["colours of familiar things", "big and small", "hot and cold", "yes or no about simple facts", "counting to three", "what someone is doing (two words)", "where is it: here, there, up, down",
           "what animals eat", "day and night", "wet and dry", "what we wear when it rains or is cold", "who helps us: doctor, teacher, cook", "loud and quiet sounds"]),
    "P": ("a preschooler (3-5 years)", "a short phrase of 2-5 words", 2,
          ["where things are: in, on, under, next to", "counting to ten", "feelings and why", "one-sentence stories: who did what", "polite replies", "opposites", "shapes", "what we use things for",
           "where animals live", "the weather today", "plants need water and sun", "the five senses", "healthy habits: washing hands, brushing teeth", "jobs people do",
           "musical instruments", "mixing colours", "seasons", "safety: hot, sharp, crossing the street"]),
    "K": ("a kindergarten child", "a short sentence", 2,
          ["letters and their sounds", "rhyming words", "sight words", "counting to twenty", "more and less", "shapes and patterns", "seasons and weather", "community helpers", "short stories: who, where, what happened",
           "living and non-living things", "push and pull", "sinking and floating", "baby animals and their parents", "long ago and today: how people lived", "families and traditions",
           "rules at home and school", "land and water on a map", "comparing numbers", "flat and solid shapes", "healthy food", "the sun, moon and stars"]),
    "G1": ("a first grade child", "a sentence", 2,
          ["addition within 20", "subtraction within 20", "telling time to the hour", "questions about a 2-3 sentence story", "describing a picture in words", "plants and animals", "maps and directions", "nouns and verbs", "sequence: first, next, last",
           "tens and ones", "measuring with units", "halves and quarters", "sound and light", "patterns of the sun and moon", "holidays and why we have them",
           "how life was for grandparents", "rules and laws: why we have them", "food groups", "instrument families", "famous children's books and their characters"]),
    "G2": ("a second grade child", "a sentence with because or then", 2,
          ["addition and subtraction within 100", "place value: tens and ones", "measuring length", "money: coins", "why questions about nature", "questions about a short paragraph", "following instructions on a grid (right, left, up, down: where do you end?)", "past and present", "cause and effect",
           "skip counting", "time to five minutes", "reading a simple bar graph", "habitats", "solids and liquids", "how the earth changes: erosion, volcanoes, earthquakes",
           "inventors and inventions", "how transport changed over time", "north, south, east and west", "continents and oceans", "needs and wants", "goods and services", "dinosaurs and fossils"]),
    "G3": ("a third grade child", "one or two sentences", 2,
          ["multiplication facts", "division facts", "simple fractions", "area and perimeter", "main idea of a paragraph", "multi-step word problems", "states of matter", "life cycles", "comparing two things and explaining",
           "rounding", "time intervals", "forces and magnets", "weather and climate", "traits animals get from their parents", "ancient egypt, rome and china: simple facts",
           "explorers and what they found", "countries and capitals", "landforms: mountains, valleys, deserts, islands", "communities and local government", "famous artists and musicians", "the human body: bones, muscles, heart"]),
    "G4": ("a fourth grade child", "one or two sentences, or a few lines of simple code", 2,
          ["multi-digit multiplication", "long division", "equivalent fractions", "decimals", "reasoning about who or what a sentence refers to", "energy and electricity", "simple coding: what does this 3-line program print?",
           "simple coding: write the moves for a robot to reach a square", "simple coding: find the one wrong step", "inference from a short text",
           "angles", "factors and multiples", "converting units", "the water cycle", "rocks and minerals", "the solar system", "the brain and the senses", "food chains",
           "historical figures and what they did", "timelines: what came before and after", "climate zones and biomes", "map scale and distance", "how a country is governed", "why people moved to new places long ago"]),
}
WRITE = ("You are writing a school-readiness curriculum. Write {n} different items for {who}, subject: {subject}. Each item is something an adult "
         "says or asks, and the answer such a child is expected to give, as {form}. Give 2-4 acceptable answers per item (different natural "
         "wordings of a right answer, all lowercase). Items must be things a child at exactly this stage can do. Vary them; avoid the most "
         'obvious ones. {avoid}(batch {k}) Reply only with JSON: [{{"q": "...", "answers": ["...", "..."]}}]')
CHECK = ("For each item, say whether ALL the listed answers are correct, and whether the item is right for {who} (not too hard, not too easy). "
         'Reply only with a JSON list in the same order: [{{"correct": true, "fits": true}}, ...]\n{items}')

out = sys.argv[1] if len(sys.argv) > 1 else "data/world/curriculum.jsonl"
adults = Adults()
rng = random.Random()
seen, by_subject = set(), {}
if os.path.exists(out):
    for line in open(out):
        d = json.loads(line)
        seen.add(d["q"])
        by_subject.setdefault((d["stage"], d["subject"]), []).append(d["q"])


LIFE = os.path.join(os.path.dirname(__file__), "..", "..", "out", os.environ.get("CURRICULUM_LIFE", "kid_s3") + ".jsonl")
ORDER = ["W1", "W2", "P", "K", "G1", "G2", "G3", "G4"]


def focus():
    """Half of the writing goes to the stage the brain is at now and the next one (a teacher
    prepares for the class in front of them); the rest is spread over all stages."""
    try:
        with open(LIFE) as f:  # the last checkpoint record (note lines have no stage)
            stage = next(r["stage"] for r in map(json.loads, reversed(f.readlines())) if "stage" in r)
    except (OSError, IndexError, ValueError, KeyError):
        return ORDER
    k = ORDER.index(stage) if stage in ORDER else 0
    return ORDER[k : k + 2]


def used():
    return adults._state().get("curriculum_count", 0)


def log(msg):
    print(time.strftime("[%F %T] ") + msg, flush=True)


k = len(seen)
while True:
    if used() + 2 > SHARE or adults._state()["count"] + 2 > PER_DAY:
        log("daily share reached; sleeping")
        while used() + 2 > SHARE or adults._state()["count"] + 2 > PER_DAY:
            time.sleep(600)
    k += 1
    stage = rng.choice(focus()) if rng.random() < 0.5 else rng.choice(list(STAGES))
    who, form, weight, subjects = STAGES[stage]
    subject = rng.choice(subjects)
    n = 10 if any(w in subject for w in ("paragraph", "story", "stories", "coding", "text")) else 25
    writer = MODELS[k % len(MODELS)]
    checker = rng.choice([m for m in MODELS if m != writer])
    prev = by_subject.get((stage, subject), [])
    avoid = ("Do not repeat any of these: " + json.dumps(rng.sample(prev, min(10, len(prev)))) + " ") if prev else ""
    try:
        w = adults.ask(writer, WRITE.format(n=n, who=who, subject=subject, form=form, avoid=avoid, k=k), purpose="curriculum", temperature=0.9, timeout=600)
        items = [x for x in (parse_json(w["text"]) or []) if isinstance(x, dict) and x.get("q") and isinstance(x.get("answers"), list) and x["answers"]]
        items = [x for x in items if str(x["q"]).strip().lower() not in seen]
        if not items:
            log(f"{stage} / {subject}: nothing new from {writer}")
            continue
        listing = "\n".join(json.dumps({"q": x["q"], "answers": x["answers"]}) for x in items)
        judged = parse_json(adults.ask(checker, CHECK.format(who=who, items=listing), purpose="curriculum", timeout=600)["text"])
        if isinstance(judged, dict):  # some replies wrap the list
            judged = next((v for v in judged.values() if isinstance(v, list)), None)
        if not isinstance(judged, list):
            raise RuntimeError("judge reply not a list")
    except Exception as e:  # noqa: BLE001
        log(f"{stage} / {subject}: failed: {e}")
        time.sleep(60)
        continue
    kept = 0
    with open(out, "a") as f:
        for x, j in zip(items, judged):
            j = j if isinstance(j, dict) else {}
            q = str(x["q"]).strip().lower()
            ok = j.get("correct") is True and j.get("fits") is True
            kept += ok
            seen.add(q)
            by_subject.setdefault((stage, subject), []).append(q)
            f.write(json.dumps({"stage": stage, "subject": subject, "q": q, "answers": [str(a).strip().lower() for a in x["answers"]], "form": form,
                                "agree": ok, "correct": j.get("correct"), "fits": j.get("fits"), "writer": writer}) + "\n")
    log(f"{stage} / {subject} / {writer.split('/')[1]}: {len(items)} items, {kept} judged right and fitting, {used()} requests today")
