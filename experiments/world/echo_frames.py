"""The ways adults get a toddler to repeat a word (written once by the adults).

Echo lessons were four formats written by hand ("{w}", "{w}!", "say {w}", "{w}, {w}"): the word to say was
always the last word heard, so "say the last word" answered every one of them (docs/CONVERSATION.md). Real
parents put the word anywhere ("ball! can you say it?"). Two adults each write such prompts with WORD where
the word goes, a third judges each (natural to say to a 12-month-old; the child would answer by saying the
word). Kept prompts go to data/world/echo_frames.json; organic/curriculum.py words every echo moment afresh.
    python experiments/world/echo_frames.py
"""
import json, os, sys
sys.path.insert(0, ".")
from organic.adults import Adults, parse_json
from organic.world import WORLD_DIR

WRITERS = ["nvidia/nemotron-3.5-lightning:free", "dots-studio/dots-3-note-preview:free"]
JUDGE = "nvidia/nemotron-3.5-super-vl:free"
ASK = ("A parent is teaching a 12-month-old a new word by getting the child to repeat it. Write 45 different short things the "
       "parent might say, with WORD where the word goes (WORD appears once or twice). Put WORD in different places: in a "
       "third of them WORD is the first word and other words follow (\"WORD! can you say it?\"), in a third it is in the "
       "middle (\"can you say WORD for me?\"), in a third it is last (\"say WORD\"). Vary them a lot, the way real parents "
       "talk. At most 8 words each. Lower case. Reply only with a JSON list of strings.")
adults = Adults()
cands = []
for w in WRITERS:
    got = parse_json(adults.ask(w, ASK, temperature=0.9, purpose="frames")["text"]) or []
    cands += [g.strip().lower().replace("word", "WORD") for g in got if isinstance(g, str) and 1 <= len(g.split()) <= 8]
cands = sorted({c for c in cands if 1 <= c.count("WORD") <= 2})
JUDGE_ASK = ("Each line is something a parent says to a 12-month-old to get the child to repeat a word; WORD stands for that word. "
             "For each: is it natural to say to a baby, and would the child answer it by just saying the word? Reply only with "
             "JSON: [{\"say\": \"...\", \"ok\": true}]\n" + "\n".join(cands))
verdict = parse_json(adults.ask(JUDGE, JUDGE_ASK, purpose="frames")["text"]) or []
ok = sorted({v["say"].strip() for v in verdict if isinstance(v, dict) and v.get("ok") and v.get("say", "").strip() in cands})
frames = sorted(set(ok) | {"WORD", "WORD!", "say WORD"})
place = lambda f: "only" if f.strip("!?., ") == "WORD" else "last" if f.rstrip("!?., ").endswith("WORD") else "first" if f.startswith("WORD") else "middle"  # noqa: E731
json.dump({"frames": frames, "writers": WRITERS, "judge": JUDGE, "candidates": len(cands)}, open(os.path.join(WORLD_DIR, "echo_frames.json"), "w"), indent=1)
print(len(cands), "written,", len(frames), "kept;", {k: sum(place(f) == k for f in frames) for k in ("only", "first", "middle", "last")}, frames)
