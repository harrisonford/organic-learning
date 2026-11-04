"""The many ways adults ask a toddler to name what is in a picture (written once by the adults).

CIFAR picture-book items used to be asked only as "what is this?": 90% of the
first-words world was a single frame, and the ear fused it into one unit (docs/
EAR_SEGMENTATION.md). Real parents ask in many ways. Two adults each write phrasings,
a third judges each one (natural to say to an 18-month-old, and answered by just naming
the thing). Kept phrasings go to data/world/naming_frames.json; organic/curriculum.py
gives each picture one of them (fixed per picture, so a picture keeps its question).
    python experiments/world/naming_frames.py
"""
import json, os, sys
sys.path.insert(0, ".")
from organic.adults import Adults, parse_json
from organic.world import WORLD_DIR

WRITERS = ["nvidia/nemotron-3.5-lightning:free", "dots-studio/dots-3-note-preview:free"]
JUDGE = "nvidia/nemotron-3.5-super-vl:free"
ASK = ("A parent is looking at a picture book with an 18-month-old and points at one thing. Write 40 different short ways "
       "the parent might ask the child to name it (for example \"what's that?\"). Vary them a lot, the way real parents talk: "
       "playful, with \"look!\", with \"who\" for animals, asking the child to tell or say it. Each must be answered by just "
       "naming the thing, and must not name it. Lower case. Reply only with a JSON list of strings.")
adults = Adults()
cands = []
for w in WRITERS:
    got = parse_json(adults.ask(w, ASK, temperature=0.9, purpose="frames")["text"]) or []
    cands += [g.strip().lower() for g in got if isinstance(g, str) and 2 <= len(g.split()) <= 10]
cands = sorted(set(cands))
JUDGE_ASK = ("Each line is something a parent says to an 18-month-old while pointing at a thing in a picture. For each, is it "
             "natural to say to a toddler AND answered by just naming the thing (not yes/no, not a colour, not a sound, and it "
             "does not name the thing)? Reply only with JSON: [{\"say\": \"...\", \"ok\": true}]\n" + "\n".join(cands))
verdict = parse_json(adults.ask(JUDGE, JUDGE_ASK, purpose="frames")["text"]) or []
ok = sorted({v["say"].strip().lower() for v in verdict if isinstance(v, dict) and v.get("ok") and v.get("say", "").strip().lower() in cands})
frames = sorted(set(ok) | {"what is this?"})
json.dump({"frames": frames, "writers": WRITERS, "judge": JUDGE, "candidates": len(cands)}, open(os.path.join(WORLD_DIR, "naming_frames.json"), "w"), indent=1)
print(len(cands), "written,", len(frames), "kept:", frames)
