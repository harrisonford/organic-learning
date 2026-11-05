"""Does overheard speech (Simple English Wikipedia) help the ear find words? (offline)

Fresh ears, same seed: (a) hears only the curriculum's speech, as lives do now;
(b) hears the same curriculum speech plus overheard sentences, interleaved (an infant
hears adults talk to each other far more than to it). Both are scored on held-out
curriculum speech with the segmentation scores of experiments/segmentation_probe.py,
overall and per stage, at several amounts of experience.
    python experiments/overheard_probe.py [overheard per curriculum utterance, default 1]
"""
import os, random, sys
sys.path.insert(0, ".")
sys.path.insert(0, "experiments")
from segmentation_probe import cur, held, life, score  # noqa: E402
from organic.ear import Ear  # noqa: E402
from organic.world import DATA  # noqa: E402

RATIO = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
rng = random.Random(0)
wiki = []
with open(os.path.join(DATA, "simplewiki", "sentences.txt")) as f:
    for i, line in enumerate(f):  # reservoir sample: the file is large
        if len(wiki) < 200000:
            wiki.append(line.strip())
        elif rng.random() < 200000 / (i + 1):
            wiki[rng.randrange(200000)] = line.strip()
rng.shuffle(wiki)
cur.rng.seed(5)
by_stage = {s: [cur.moment(s)["question"] for _ in range(300)] for s in ("B", "W1", "P", "K")}


def show(label, ear):
    r = score(ear, held)
    per = " ".join(f"{s} {score(ear, t)['word P']:.2f}" for s, t in by_stage.items())
    print(f"{label:36} whole words {r['word P']:.2f} | fused {r['fused words']:.2f} | pieces {r['piece of a word']:.2f} | boundary P/R {r['boundary P']:.2f}/{r['boundary R']:.2f} | per stage: {per}", flush=True)


a, b = Ear(seed=7), Ear(seed=7)
done, owed, w = 0, 0.0, 0
for n in (5000, 20000, 40000):
    for t in life[done:n]:
        a.units(t)
        b.units(t)
        owed += RATIO
        while owed >= 1:  # overheard sentences between the utterances addressed to it
            b.units(wiki[w % len(wiki)])
            w += 1
            owed -= 1
    done = n
    show(f"curriculum only, {n} heard", a)
    show(f"+ overheard x{RATIO:g} ({w} sentences)", b)
