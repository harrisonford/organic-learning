"""Does prediction-error hearing let the content word beat the shared frame? (offline)

Same test as experiments/fatigue_probe.py, on kid_s3's ear (out/tmp/ear_kid_s3.pkl,
extracted from the released L1 so the probe stays small next to the training life):
the predictor learns from everyday utterances of the curriculum, then questions are
coded from what was not predicted.
    python experiments/prediction_probe.py
"""
import copy, pickle, sys
sys.path.insert(0, ".")
import numpy as np
from organic.curriculum import Curriculum
ear0 = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else "out/kid_s3.ear.pkl", "rb"))
cur = Curriculum(seed=0)
heard = [cur.moment(st)["question"] for _ in range(5000) for st in ("W1", "W2", "P", "K")]
X = ["lion", "pig", "cow", "duck", "dog", "cat", "owl", "sheep", "horse", "frog", "bee", "snake", "mouse", "bird", "elephant", "monkey"]
B = ["knee", "nose", "elbow", "toe", "ear", "eye", "hand", "foot", "chin", "belly", "thumb", "neck"]
TESTS = [("what sound does a {} make?", "what noise does the {} make?", X), ("where is your {}?", "show me your {}", B), ("is that a {}?", "look at the {}!", X)]


def measure(ear, label):
    codes = lambda f, xs: np.array([ear.question(ear.units(f.format(x), learn=False), adapt=False) for x in xs])  # noqa: E731
    sw, sf, wins, acc = [], [], [], []
    for f1, f2, xs in TESTS:
        A, Bc = codes(f1, xs), codes(f2, xs)
        acc.append(np.mean((A @ Bc.T).argmax(1) == np.arange(len(xs))))
        for i in range(len(xs)):
            a, b = float(A[i] @ Bc[i]), max(float(A[i] @ A[j]) for j in range(len(xs)) if j != i)
            sw.append(a); sf.append(b); wins.append(a > b)
    res = ""
    if getattr(ear, "predictive", False):
        r = [np.linalg.norm(e) / np.linalg.norm(ear.sound(u)) for q in heard[-400:] for u, e in ear._surprise_codes(ear.units(q, learn=False), False).items()]
        res = f" | residual {np.mean(r):.2f}"
    print(f"{label:34} cross-frame match {np.mean(acc):.2f} | same word/other frame {np.mean(sw):.2f} | other word/same frame {np.mean(sf):.2f} | word wins {np.mean(wins):.2f}{res}", flush=True)


measure(copy.deepcopy(ear0), "now (no prediction)")
for trace in (0.0, 0.5):
    ear = copy.deepcopy(ear0)
    ear.enable_prediction(trace=trace)
    for n, q in enumerate(heard, 1):
        ear.question(ear.units(q, learn=False), adapt=True)  # hearing everyday speech: the predictor learns
        if n in (2000, 20000):
            measure(ear, f"C trace {trace} ({n} heard)")
