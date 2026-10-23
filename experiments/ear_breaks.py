"""How the ear cuts speech into units, from its own statistics (no tokenizer).

    python experiments/ear_breaks.py out/full_s0_ear.pkl

Shows probe sentences of many kinds (heard-like, new, other languages, made-up
words), how often its breaks fall on spaces (word boundaries), the most common
multi-word chunks and the pieces it splits words into.
"""

import collections
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ear = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else "out/full_s0_ear.pkl", "rb"))

PROBES = {
    "photo questions": ["what color is the bus?", "how many people are in the picture?", "is the man holding an umbrella?", "what is the woman doing?"],
    "everyday chat": ["thank you very much!", "good morning, how are you?", "see you tomorrow at school.", "i'm sorry, i didn't hear you."],
    "stories (bAbI)": ["Mary moved to the bathroom.\nJohn went to the hallway.\nWhere is Mary?", "Daniel picked up the football.\nDaniel went to the kitchen.\nWhere is the football?"],
    "reasoning": ["the trophy doesn't fit in the suitcase because the ... is too big. trophy or suitcase?", "what is 7 plus 5?", "hot is to cold as up is to what?"],
    "never heard like this": ["the purple elephant sang quietly under the bridge.", "astronauts eat dehydrated strawberries.", "my grandmother's bicycle is older than me."],
    "made-up words": ["the blorpy zindle flomped the gax.", "glimmerwick and snorfle went to tharn."],
    "other languages / symbols": ["¿dónde está la biblioteca?", "je ne sais pas.", "3.14159 is pi, 2+2=4", "🙂 hello 🙂"],
}

out = []
for group, sents in PROBES.items():
    out.append(f"\n### {group}")
    for s in sents:
        units = ear.units(s, learn=False)
        out.append("`" + " | ".join(u.decode("utf-8", "replace").replace("\n", "⏎") for u in units) + "`")

# boundary agreement on a sample of heard-like text (the probes above + typical questions)
text = [s for g, v in PROBES.items() for s in v if g not in ("made-up words", "other languages / symbols")]
hits = breaks = spaces = caught = 0
for s in text:
    units = ear.units(s, learn=False)
    pos, cut = 0, set()
    for u in units[:-1]:
        pos += len(u)
        cut.add(pos)
    data = s.encode()
    space_after = {i + 1 for i, b in enumerate(data) if b == 32}  # a break right after a space = word start
    breaks += len(cut)
    hits += len(cut & space_after)
    spaces += len(space_after)
    caught += len(cut & space_after)
out.append(f"\n### Word boundaries\n{hits / max(1, breaks):.0%} of the ear's breaks fall at a word start; "
           f"{caught / max(1, spaces):.0%} of word starts are breaks.")

# what kinds of units it has
heard = ear.heard
top = sorted(heard.items(), key=lambda kv: -kv[1])
multi = [(u, n) for u, n in top if u.strip().count(b" ") >= 1 and len(u) > 4][:25]
pieces = [(u, n) for u, n in top if b" " not in u and 1 < len(u) < 6 and u.isalpha()][:25]
longest = sorted((u for u, n in heard.items() if n >= 20), key=len, reverse=True)[:15]
dec = lambda u: u.decode("utf-8", "replace").replace("\n", "⏎")  # noqa: E731
out.append("\n### Most common multi-word chunks\n" + ", ".join(f"`{dec(u)}` ({n:,})" for u, n in multi))
out.append("\n### Most common word pieces (no space inside)\n" + ", ".join(f"`{dec(u)}` ({n:,})" for u, n in pieces))
out.append("\n### Longest units heard at least 20 times\n" + ", ".join(f"`{dec(u)}`" for u in longest))
out.append(f"\n{len(heard):,} units, {ear.utterances:,} utterances heard.")
print("\n".join(out))
