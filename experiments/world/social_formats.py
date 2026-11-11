"""Everyday social routines between a parent and a toddler (written once by the adults).

Conversation was a preschool stage, behind colours and counting in photographs; a child greets, thanks and takes
turns long before that (Ninio & Snow 1996; Bruner's formats; docs/CONVERSATION.md). Two adults each write what a
parent says and the short replies a toddler gives, a third judges each. Kept exchanges go to
data/world/social_formats.jsonl; organic/curriculum.py puts those with one-word replies in first words and
the rest in two words (source "social").
    python experiments/world/social_formats.py
"""
import json, os, sys
sys.path.insert(0, ".")
from organic.adults import Adults, parse_json
from organic.world import WORLD_DIR

WRITERS = ["nvidia/nemotron-3.5-lightning:free", "dots-studio/dots-3-note-preview:free"]
JUDGE = "nvidia/nemotron-3.5-super-vl:free"
KINDS = ["greetings and goodbyes at different times of day and to different people", "thanking, please, sorry, you're welcome, bless you",
         "asking how the child is or feels, and comforting", "offers and choices about food, play and going out (want more? all done?)",
         "turn-taking games and routines (peekaboo, ready set go, your turn, night night, bath time, clean up)"]
ASK = ("Write 40 different everyday exchanges between a parent and a toddler of 18 to 24 months about: {kind}. These are social "
       "routines, not questions about facts or pictures. For each give what the parent says (at most 8 words, lower case) and 2 to "
       "4 different replies a toddler would naturally give, each ONE or TWO words, lower case. In most of them the reply should "
       "not be a repetition of the parent's last word. Vary the parent's wording a lot. Reply only with JSON: "
       "[{{\"parent\": \"...\", \"child\": [\"...\", \"...\"]}}]")
adults = Adults()
cands = {}
for i, kind in enumerate(KINDS):
    w = WRITERS[i % len(WRITERS)]
    got = parse_json(adults.ask(w, ASK.format(kind=kind), temperature=0.9, purpose="frames")["text"]) or []
    for g in got:
        if isinstance(g, dict) and isinstance(g.get("parent"), str) and isinstance(g.get("child"), list):
            p = " ".join(g["parent"].lower().split())
            c = [" ".join(x.lower().split()).strip(" .") for x in g["child"] if isinstance(x, str) and 1 <= len(x.split()) <= 2]
            if 1 <= len(p.split()) <= 9 and c:
                cands.setdefault(p, {"parent": p, "child": [], "kind": kind})["child"] += [x for x in c if x not in cands[p]["child"]]
rows = sorted(cands.values(), key=lambda r: r["parent"])
kept = []
for k in range(0, len(rows), 60):  # the judge reads them in batches
    batch = rows[k : k + 60]
    ask = ("Each line is what a parent says to a toddler of 18-24 months, then ' => ', then replies the toddler might give. For each "
           "line: is the parent's line natural to say to a toddler, and is every reply something a toddler would naturally say back? "
           "Reply only with JSON: [{\"n\": 0, \"ok\": true}] with n the line number.\n"
           + "\n".join(f"{n}. {r['parent']} => {' / '.join(r['child'])}" for n, r in enumerate(batch)))
    verdict = parse_json(adults.ask(JUDGE, ask, purpose="frames")["text"]) or []
    kept += [batch[v["n"]] for v in verdict if isinstance(v, dict) and v.get("ok") and isinstance(v.get("n"), int) and 0 <= v["n"] < len(batch)]
with open(os.path.join(WORLD_DIR, "social_formats.jsonl"), "w") as f:
    for r in kept:
        f.write(json.dumps(r) + "\n")
print(len(rows), "written,", len(kept), "kept;", kept[:12])
