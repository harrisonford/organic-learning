"""A life's cortex with the relay as a circuit of interneurons (docs/VISION.md, 2026-11-25; queue E 26/28).

As experiments/make_cortex2.py, with organic.cortex2.RelayCircuit in the batch relay's place. The circuit is born
random and develops, before any lesson, on the very pictures the batch relay of FROM was settled on (the same
draw: 6,000 pictures of the lived side), one sight of each. In a life it then goes on learning from every view it
carries. Read at the end, on 500 further pictures of the lived side: how alike two views are and how many
directions the code uses, through the batch relay and through the circuit.
    FROM=data/visual_cortex2.pkl TO=data/visual_cortex2_circuit.pkl python experiments/make_cortex2_circuit.py
"""
import os, pickle, random, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.curriculum import Curriculum  # noqa: E402
from organic.cortex2 import RelayCircuit  # noqa: E402

cur = Curriculum(seed=0, world=3)
rng = random.Random(5)
SRC = ("cifar:name", "commons:name", "coco:name")
lived = {s: [it for it in cur.train["W1"][s] if cur.image(it[0]).max() > 0] for s in SRC}
cortex = pickle.load(open(os.environ.get("FROM", "data/visual_cortex2.pkl"), "rb"))
N, GRID = int(os.environ.get("N", 6000)), cortex.grid
batch, cortex.relay = cortex.relay, None
draw = lambda: cur.image(rng.choice(lived[rng.choice(SRC)])[0])  # noqa: E731
circuit, t0, block = RelayCircuit(len(batch.mu), seed=int(os.environ.get("SEED", 0))), time.time(), []
for n in range(N):
    block.append(cortex.whole(draw(), GRID))
    if len(block) == circuit.block:
        circuit.learn(np.array(block))
        block = []
print(f"the circuit developed on {N} pictures of the lived side ({time.time() - t0:.0f}s); the average it keeps against the batch relay's: "
      f"largest difference {float(np.abs(circuit.mu - batch.mu).max()):.2g} (largest number {float(np.abs(batch.mu).max()):.2g})", flush=True)
V = np.array([cortex.whole(draw(), GRID) for _ in range(500)])


def read(label, X):
    Xc = X - X.mean(axis=0)
    Xc /= np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-8
    L = (Xc @ Xc.T).astype(np.float64)
    s = np.maximum(np.linalg.eigvalsh(L), 0)
    print(f"{label}: two views alike {np.abs(L[~np.eye(len(L), dtype=bool)]).mean():.3f} | directions used {s.sum() ** 2 / (s * s).sum():.0f}", flush=True)


circuit.plastic = False
read("no relay          ", V)
read("the batch relay   ", np.array([batch(v) for v in V]))
read("the circuit       ", np.array([circuit(v) for v in V]))
circuit.plastic = True
cortex.relay = circuit
pickle.dump(cortex, open(os.environ.get("TO", "data/visual_cortex2_circuit.pkl"), "wb"))
print("saved", flush=True)
