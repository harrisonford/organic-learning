"""Replaying the day in sleep (offline; docs/CONVERSATION.md, proposal 5).

The word cells learn slowly by design: each moment is met once, among forty others, and moves each synapse a
little. The brain also has a fast memory. In animals the fast one teaches the slow one in sleep: the hippocampus
replays recent experience to the cortex, the surprising and rewarded moments more than the rest (Wilson &
McNaughton 1994; McClelland, McNaughton & O'Reilly 1995; Ji & Wilson 2007; prioritised by surprise: Mattar & Daw
2018). Here the fast memory is a store of the day's moments as they were lived (what was heard, what was seen,
what was said next); every DAY moments the brain sleeps and the word cells live some of them again.

  awake only     each moment once (as every life so far)
  replay         in sleep, every moment of the day once more, in a new order
  surprise       in sleep, as many replays as the day had moments, each drawn in proportion to how surprised the
                 cells were by it when it happened (so the surprising ones several times, the expected ones never)
The same moments, order and tests as experiments/imagery_probe.py (first and two words mixed, comments).
    SEEDS=0,1,2 EAR=out/tmp/ear_edge.pkl python experiments/replay_probe.py
"""
import os, random, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("EAR", "out/tmp/ear_edge.pkl")
import experiments.imagery_probe as P  # noqa: E402  (ARMS="" there: only its world, tests and measure are used)
from organic.words import Words  # noqa: E402

DAY = int(os.environ.get("DAY", 2000))
ARMS = os.environ.get("REPLAY_ARMS", "awake only,replay,surprise").split(",")
results = {}
for seed in P.SEEDS:
    rng = np.random.default_rng(seed)
    order = [P.w1[i] for i in rng.choice(len(P.w1), min(P.N, len(P.w1)), replace=False)] + [P.w2[i] for i in rng.choice(len(P.w2), min(P.N, len(P.w2)), replace=False)]
    order = [order[i] for i in rng.permutation(len(order))]
    for arm in ARMS:
        words, wr, t0 = Words(P.dim, P.ear), random.Random(seed), time.time()
        P.cur.orng = random.Random(seed + 2)
        sleep = np.random.default_rng(seed + 9)
        day, replays = [], 0
        for n, it in enumerate(order, 1):
            heard, v, reply = P.hear(P.cur.wording(it, wr)), P.vis(it[0]), P.hear(it[2][0])
            src = ("cifar:name", "commons:name")[P.cur.orng.randrange(2)]
            ob = P.cur.orng.choice(P.cur.train["W1"][src])
            w = ob[2][0]
            says = P.cur.orng.choice(P.cur.comments).replace("a THING", ("an " if w[0] in "aeiou" else "a ") + w).replace("THING", w)
            for m in ((heard, v, reply), ([], P.vis(ob[0]), P.hear(says))):
                day.append((m, words.surprise(*m) if arm == "surprise" else 1.0))
                words.learn(*m)
            if n % DAY == 0 and arm != "awake only":  # sleep
                if arm == "surprise":
                    p = np.array([s for _, s in day]) + 1e-6
                    again = sleep.choice(len(day), len(day), p=p / p.sum())
                else:
                    again = sleep.permutation(len(day))
                for i in again:
                    words.learn(*day[i][0])
                replays += len(again)
                day = []
        row = P.measure(words)
        results.setdefault(arm, []).append(row)
        print(f"seed {seed} {arm:11} ({time.time() - t0:.0f}s, {replays} replays): " + " | ".join(f"{k} {v:.3f}" for k, v in row.items() if not k.startswith("there")), flush=True)
print(f"\nmean over seeds {P.SEEDS} (spread):")
for name, rows in results.items():
    print(f"  {name:11} " + " | ".join(f"{k} {np.nanmean([r[k] for r in rows]):.3f}±{np.nanstd([r[k] for r in rows]):.3f}" for k in rows[0] if not k.startswith("there")))
