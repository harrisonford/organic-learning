import copy, pickle, sys
sys.path.insert(0, ".")
import numpy as np
from organic.curriculum import Curriculum
ear0 = pickle.load(open("out/kid_s3.pkl", "rb")).ear
cur = Curriculum(seed=0)
heard = [cur.moment(st)["question"] for st in ("W1", "W2", "P", "K") for _ in range(5000)]
X = ["lion", "pig", "cow", "duck", "dog", "cat", "owl", "sheep", "horse", "frog", "bee", "snake", "mouse", "bird", "elephant", "monkey"]
B = ["knee", "nose", "elbow", "toe", "ear", "eye", "hand", "foot", "chin", "belly", "thumb", "neck"]
TESTS = [("what sound does a {} make?", "what noise does the {} make?", X), ("where is your {}?", "show me your {}", B), ("is that a {}?", "look at the {}!", X)]
def codes(ear, frame, xs, fat):
    return np.array([ear.question(ear.units(frame.format(x), learn=False), adapt=False) for x in xs])
for label, fat in (("now (no fatigue)", False), ("B: fatigue", True)):
    ear = copy.deepcopy(ear0)
    if fat:
        ear.enable_fatigue()
        for q in heard:  # a life of hearing everyday speech tires the neurons of common sounds
            ear.question(ear.units(q, learn=False), adapt=True)
    accs, sims = [], []
    for f1, f2, xs in TESTS:
        A, Bc = codes(ear, f1, xs, fat), codes(ear, f2, xs, fat)
        S = A @ Bc.T
        accs.append(np.mean(S.argmax(1) == np.arange(len(xs))))
        sims.append(np.mean(A @ A.T)[()] if False else (np.sum(A @ A.T) - len(xs)) / (len(xs) * (len(xs) - 1)))
    print(f"{label:18} cross-frame match {np.mean(accs):.2f} (chance {1/len(X):.2f}) | similarity between different content words in one frame {np.mean(sims):.2f}", flush=True)
    if fat:
        print("fuel: min %.2f median %.2f max %.2f" % (ear.fuel.min(), np.median(ear.fuel), ear.fuel.max()))

print("--- frame vs content word: which is the stronger pull?")
for label, fat in (("now (no fatigue)", False), ("B: fatigue", True)):
    ear = copy.deepcopy(ear0)
    if fat:
        ear.enable_fatigue()
        for q in heard:
            ear.question(ear.units(q, learn=False), adapt=True)
    same_word, same_frame, wins = [], [], []
    for f1, f2, xs in TESTS:
        A, Bc = codes(ear, f1, xs, fat), codes(ear, f2, xs, fat)
        for i in range(len(xs)):
            sw = float(A[i] @ Bc[i])                      # same word, other frame
            sf = max(float(A[i] @ A[j]) for j in range(len(xs)) if j != i)  # closest other word, same frame
            same_word.append(sw); same_frame.append(sf); wins.append(sw > sf)
    print(f"{label:18} same word/other frame {np.mean(same_word):.2f} | other word/same frame {np.mean(same_frame):.2f} | word wins {np.mean(wins):.2f}")

print("--- for comparison: the old fixed habituation rule (per unit, by how often heard)")
ear = copy.deepcopy(ear0); ear.habituation = True
same_word, same_frame, wins = [], [], []
for f1, f2, xs in TESTS:
    A, Bc = codes(ear, f1, xs, False), codes(ear, f2, xs, False)
    for i in range(len(xs)):
        sw = float(A[i] @ Bc[i]); sf = max(float(A[i] @ A[j]) for j in range(len(xs)) if j != i)
        same_word.append(sw); same_frame.append(sf); wins.append(sw > sf)
print(f"habituation rule   same word/other frame {np.mean(same_word):.2f} | other word/same frame {np.mean(same_frame):.2f} | word wins {np.mean(wins):.2f}")
