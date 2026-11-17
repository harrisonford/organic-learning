"""Does leisure talk cost the answers to lessons? (2026-11-17, docs/LEISURE.md)

In a life with leisure talk the word cells answer text lessons with run-on, parent-like strings ("you're looking
you want to happy so"). Word cells alone, text lessons of first words and two words (three passes), with and without
one logged leisure exchange after every fourth lesson (what the child said -> what the parent said back), and with
the parent's reply cut to its first unit (is it the length of the replies or their words?). Held-out text lessons.
    python experiments/chat_cost_probe.py
"""
import json, pickle, random, sys, collections
import numpy as np
sys.path.insert(0, ".")
from organic.words import Words
from organic.curriculum import Curriculum, grade
cur = Curriculum(seed=0, world=2)
ear = pickle.load(open("out/tmp/ear_first.pkl", "rb"))
hear = lambda q: ear.units(q, learn=False)
text = [it for st in ("W1", "W2") for src in ("curriculum", "social") for it in cur.train[st].get(src, [])]
test = [it for st in ("W1", "W2") for it in cur.test[st].get("curriculum", [])][:200]
chats = [json.loads(x) for x in open("out/world3_s1.chat.jsonl")][5000:]
chats = [c for c in chats if c.get("back") and c.get("child")]
for seed in (0, 1, 2):
    for arm in ("lessons only", "with leisure talk (1 in 5)", "leisure talk, the reply cut to its first unit", "leisure talk, the last two units of the reply"):
        rng = random.Random(seed)
        w = Words(64, ear); w.plastic_by_error = False
        n = 0
        for p in range(3):
            for it in rng.sample(text, len(text)):
                w.learn(hear(it[1]), None, hear(rng.choice(it[2]) if it[4] == "social" else it[2][0]))
                n += 1
                if arm != "lessons only" and n % 4 == 0:
                    c = rng.choice(chats)
                    back = hear(c["back"])
                    w.learn(hear(c["child"]), None, back[:1] if "cut" in arm else back[-2:] if "last two" in arm else back)
        lens, sc = [], []
        for it in test:
            txt = w.recall(hear(it[1]), None)[0] or ""
            lens.append(len(hear(txt)) if txt else 0); sc.append(grade(it[3], txt, it[2]))
        print(f"seed {seed} {arm:46}: held-out text score {np.mean(sc):.3f} | units in a reply {np.mean(lens):.2f} | replies of 4 or more units {np.mean(np.array(lens) >= 4):.2f}", flush=True)
