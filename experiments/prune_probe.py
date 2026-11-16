"""Pruning the word cells' synapses in sleep (offline; docs/PRUNING.md).

A word cell keeps a synapse from every visual neuron (7,168) and from every cue, used or not. Brains make too many
synapses and remove those that were not used (Huttenlocher 1979; Navlakha et al. 2015), and sleep scales synapses
down sparing the most used (de Vivo et al. 2017). Here, after a life's worth of first and two words:

  how many synapses can go?   each cell keeps the fraction KEEP of the synapses on each dendrite, chosen by
      weight   the largest weights (Chechik et al. 1998; Han et al. 2015)
      use      weight x how active that input has been (use it or lose it; Sun et al. 2024)
      random   any (a control)
  does it help or hurt what comes after?   the cells then live the same moments once more with the removed
      synapses gone for good, against cells that keep all of them.
    SEEDS=0,1,2 EAR=out/tmp/ear_edge.pkl python experiments/prune_probe.py
"""
import os, pickle, random, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("EAR", "out/tmp/ear_edge.pkl")
import experiments.imagery_probe as P  # noqa: E402
from organic.words import Words  # noqa: E402

KEEPS = [0.5, 0.2, 0.1, 0.05, 0.02]
short = lambda row: " | ".join(f"{k} {row[k]:.3f}" for k in ("W1 pictures", "same x3", "W1 text", "W1 text beyond parrot", "W2 curriculum", "W2"))  # noqa: E731


def live(words, order, seed, masks=None):
    wr = random.Random(seed)
    P.cur.orng = random.Random(seed + 2)
    for n, it in enumerate(order, 1):
        words.learn(P.hear(P.cur.wording(it, wr)), P.vis(it[0]), P.hear(it[2][0]))
        src = ("cifar:name", "commons:name")[P.cur.orng.randrange(2)]
        ob = P.cur.orng.choice(P.cur.train["W1"][src])
        w = ob[2][0]
        says = P.cur.orng.choice(P.cur.comments).replace("a THING", ("an " if w[0] in "aeiou" else "a ") + w).replace("THING", w)
        words.learn([], P.vis(ob[0]), P.hear(says))
        if masks is not None and n % 20 == 0:  # a synapse that is gone stays gone
            cells = masks[0].shape[0]
            words.Wv[:cells] *= masks[0]
            words.Wc[:cells, : masks[1].shape[1]] *= masks[1]


def masks_for(words, keep, rule, rng):
    n, c = len(words.units), len(words.cues)
    out = []
    for W, use in ((words.Wv[:n], words.use_v), (words.Wc[:n, :c], words.use_c[:c])):
        score = np.abs(W) * (np.sqrt(use)[None, :] if rule == "use" else 1.0) if rule != "random" else rng.random(W.shape)
        k = max(1, int(round(keep * W.shape[1])))
        kth = np.partition(score, -k, axis=1)[:, -k][:, None]
        out.append((score >= kth) & (np.abs(W) > 0 if rule != "random" else True))
    return out


for seed in P.SEEDS:
    rng = np.random.default_rng(seed)
    order = [P.w1[i] for i in rng.choice(len(P.w1), min(P.N, len(P.w1)), replace=False)] + [P.w2[i] for i in rng.choice(len(P.w2), min(P.N, len(P.w2)), replace=False)]
    order = [order[i] for i in rng.permutation(len(order))]
    words, t0 = Words(P.dim, P.ear), time.time()
    live(words, order, seed)
    n, c = len(words.units), len(words.cues)
    alive = (np.abs(words.Wv[:n]) > 0).mean(), (np.abs(words.Wc[:n, :c]) > 0).mean()
    print(f"seed {seed} all synapses ({time.time() - t0:.0f}s; {n} cells, {c} cues; nonzero: visual {alive[0]:.2f}, cue {alive[1]:.3f}): {short(P.measure(words))}", flush=True)
    full = pickle.dumps(words)
    for rule in ("use", "weight", "random"):
        for keep in KEEPS if rule != "random" else [0.2]:
            w = pickle.loads(full)
            mv, mc = masks_for(w, keep, rule, rng)
            w.Wv[:n] *= mv
            w.Wc[:n, :c] *= mc
            print(f"seed {seed} keep {keep:<4} by {rule:6}: {short(P.measure(w))}", flush=True)
    for label, keep in (("all synapses, a second pass", None), ("pruned to 0.2 by use, then a second pass", 0.2), ("pruned to 0.05 by use, then a second pass", 0.05)):
        w = pickle.loads(full)
        masks = None
        if keep:
            masks = masks_for(w, keep, "use", rng)
            w.Wv[:n] *= masks[0]
            w.Wc[:n, :c] *= masks[1]
        live(w, order, seed + 100, masks)
        print(f"seed {seed} {label}: {short(P.measure(w))}", flush=True)
