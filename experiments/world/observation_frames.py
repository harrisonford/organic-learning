"""The ways adults talk about what is in a picture without asking anything (written once by the adults).

Children mostly learn words by observation: they hear speech about what they are looking at,
unasked ("look, a dog!"). Two adults each write such comments with THING where the name goes,
a third judges each (natural to say to an 18-month-old; names the thing; asks nothing). Kept
comments go to data/world/observation_frames.json; organic/curriculum.py builds observation
moments from them (docs/REVIEW_2026-11-10.md 4.4).
    python experiments/world/observation_frames.py
"""
import json, os, sys
sys.path.insert(0, ".")
from organic.adults import Adults, parse_json
from organic.world import WORLD_DIR

WRITERS = ["nvidia/nemotron-3.5-lightning:free", "dots-studio/dots-3-note-preview:free"]
JUDGE = "nvidia/nemotron-3.5-super-vl:free"
ASK = ("A parent is looking at a picture book with an 18-month-old and points at one thing, naming it for the child without "
       "asking anything. Write 40 different short things the parent might say, with the word THING where the name goes, always "
       "as \"a THING\" or \"the THING\" (for example \"look, a THING!\" or \"the THING is sleeping\" is NOT allowed because it "
       "says more than the picture shows; keep to naming and pointing out). Vary them a lot, the way real parents talk: playful, "
       "with \"look\", \"see\", \"here is\", \"that's\", \"it's\", repeating the word. No questions. Lower case. Reply only with "
       "a JSON list of strings.")
adults = Adults()
cands = []
for w in WRITERS:
    got = parse_json(adults.ask(w, ASK, temperature=0.9, purpose="frames")["text"]) or []
    cands += [g.strip().lower().replace("thing", "THING") for g in got if isinstance(g, str) and 2 <= len(g.split()) <= 10]
cands = sorted({c for c in cands if "THING" in c and "?" not in c})
JUDGE_ASK = ("Each line is something a parent says to an 18-month-old while pointing at a thing in a picture; THING stands for "
             "the thing's name. For each: is it natural to say to a toddler, does it only name or point out the thing (it says "
             "nothing else about it that the picture might not show), and is it not a question? Reply only with JSON: "
             "[{\"say\": \"...\", \"ok\": true}]\n" + "\n".join(cands))
verdict = parse_json(adults.ask(JUDGE, JUDGE_ASK, purpose="frames")["text"]) or []
ok = sorted({v["say"].strip() for v in verdict if isinstance(v, dict) and v.get("ok") and v.get("say", "").strip() in cands})
frames = sorted(set(ok) | {"look, a THING!"})
json.dump({"frames": frames, "writers": WRITERS, "judge": JUDGE, "candidates": len(cands)}, open(os.path.join(WORLD_DIR, "observation_frames.json"), "w"), indent=1)
print(len(cands), "written,", len(frames), "kept:", frames)
