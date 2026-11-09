"""How rigid is our grading? The local grader vs a human-like judge, on a brain's real answers (offline).

A brain answers held-out items of W1, W2 and P; each answer is graded by the curriculum's
local grader and judged by parallel/decisions-fast (OpenRouter Decisions API, free): the
probability a caring adult would accept it as a right answer for a child of that stage.
Many answers go in one request (typed yes/no questions evaluated in parallel).
    python experiments/judge_study.py out/kid_s3.pkl
"""
import json, os, pickle, sys, urllib.request
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.adults import Adults  # noqa: E402
from organic.curriculum import NAMES, Curriculum, grade  # noqa: E402

AGE = {"W1": "an 18-month-old", "W2": "a two-year-old", "P": "a preschooler (3-5 years)", "K": "a kindergarten child"}
URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "parallel/decisions-fast:free"


def judge(batch, adults):
    """batch: [(stage, question, answers, said, has_picture)] -> acceptance probabilities."""
    state = {f"item{i}": {"child": AGE.get(st, "a young child"), "adult asked": q, "adult's own answer": a[0],
                          "other answers the adult would accept": a[1:], "the adult was pointing at a picture": pic,
                          "the child answered": said or "(nothing)"} for i, (st, q, a, said, pic) in enumerate(batch)}
    questions = {f"item{i}": {"type": "noul", "instructions": f"For item{i}: would a caring adult accept what the child answered as a right "
                              "answer for a child of that age (meaning matters, not the exact words; a picture the judge cannot see is "
                              "described by the adult's own answer)?"} for i in range(len(batch))}
    body = json.dumps({"model": MODEL, "state": state, "questions": questions}).encode()
    adults._wait_turn("judge")
    req = urllib.request.Request(URL, data=body, headers={"Authorization": f"Bearer {adults.key}", "Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=300))
    return [float(d["answers"][f"item{i}"]["noul"]) for i in range(len(batch))]


if __name__ == "__main__":
    brain = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else "out/kid_s3.pkl", "rb"))
    cur = Curriculum(seed=0)
    rows = []
    for st in ("W1", "W2", "P"):
        for m in cur.test_set(st)[:80]:
            said = brain.live(m["image"], m["question"], learn=False, explore=False)[0]
            rows.append((st, m["question"], m["answers"], said, bool(m["image"].max() > 0), grade(st, said, m["answers"])))
    del brain
    adults = Adults()
    probs = []
    for i in range(0, len(rows), 20):
        probs += judge([r[:5] for r in rows[i : i + 20]], adults)
    out = [dict(stage=r[0], question=r[1], answers=r[2], said=r[3], local=r[5], judge=p) for r, p in zip(rows, probs)]
    json.dump(out, open("out/judge_study.json", "w"), indent=1)
    for st in ("W1", "W2", "P"):
        xs = [o for o in out if o["stage"] == st]
        loc, jud = np.array([o["local"] for o in xs]), np.array([o["judge"] for o in xs])
        print(f"{st} ({NAMES[st]}): local grade {loc.mean():.2f} | judge acceptance {jud.mean():.2f} (p>0.5: {np.mean(jud > 0.5):.2f})"
              f" | judge accepts, local < 0.5: {np.mean((jud > 0.5) & (loc < 0.5)):.2f} | local >= 0.5, judge rejects: {np.mean((jud <= 0.5) & (loc >= 0.5)):.2f}", flush=True)
    print("accepted by the judge but not by the grader:")
    for o in [o for o in out if o["judge"] > 0.5 and o["local"] < 0.5][:15]:
        print(f"   [{o['stage']}] {o['question'][:50]!r} -> said {o['said']!r} (adult: {o['answers'][:3]}) judge {o['judge']:.2f} local {o['local']:.2f}")
    print("graded right but rejected by the judge:")
    for o in [o for o in out if o["judge"] <= 0.5 and o["local"] >= 0.5][:8]:
        print(f"   [{o['stage']}] {o['question'][:50]!r} -> said {o['said']!r} (adult: {o['answers'][:3]}) judge {o['judge']:.2f} local {o['local']:.2f}")
