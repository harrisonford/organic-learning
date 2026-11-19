"""The rigid grade beside the adult's judgement, answer by answer (docs/JUDGING.md, 2026-11-19).

Stages are passed on the curriculum's own grade (organic/curriculum.py, grade): the wanted word, or nothing. The
local judge (organic/judge.py: would a caring adult accept this from a child of that age?) has given the reward
since 2026-11-09 and its score is in every record, but the two were never compared answer by answer. For a level
(a brain as it passed a stage) and the held-out tests of first and two words:
  - by source: the rigid grade, the judged acceptance, and how often each of them says "right" (>= 0.5)
  - what a child who does not look or listen would get from the same judge: always "yes", always "no", always the
    commonest answer of the source, the last word of the question (the floor under any lenient score)
  - the answers the two disagree on, to read
    python experiments/judge_audit.py out/levels/eyes2w_s2_W2.pkl
"""
import collections, json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.curriculum import Curriculum, grade, tidy  # noqa: E402
from organic.judge import Judge  # noqa: E402

cur = Curriculum(seed=0, world=int(os.environ.get("WORLD", 4)))
judge = Judge()
for path in sys.argv[1:]:
    brain = pickle.load(open(path, "rb"))
    brain.learning = False
    tag = os.path.basename(path)[:-4]
    out = open(f"out/judge_audit_{tag}.jsonl", "w")
    print(f"{path} (age {brain.age})", flush=True)
    for stage in os.environ.get("STAGES", "W1,W2").split(","):
        items = cur.test_set(stage, n=300)
        common = {s: collections.Counter(m["answers"][0] for m in items if m["source"] == s).most_common(1)[0][0] for s in {m["source"] for m in items}}
        rows = []
        for m in items:
            said = brain.live(np.asarray(m["image"], np.float32), m["question"], learn=False, explore=False)[0] or ""
            rows.append({"stage": stage, "source": m["source"], "question": m["question"], "wanted": m["answers"][0], "said": said,
                         "grade": grade(stage, said, m["answers"]), "answers": m["answers"]})
        blind = {"always yes": lambda r: "yes", "always no": lambda r: "no", "the commonest answer": lambda r: common[r["source"]],
                 "the question's last word": lambda r: tidy(r["question"]).replace("?", "").split()[-1]}
        ask = lambda say: judge.accept([(stage, r["question"], r["answers"], say(r)) for r in rows], fallback=lambda i: grade(stage, say(rows[i]), rows[i]["answers"]))  # noqa: E731
        for r, p in zip(rows, ask(lambda r: r["said"])):
            r["judged"] = p
        for name, say in blind.items():
            for r, p, g in zip(rows, ask(say), [grade(stage, say(r), r["answers"]) for r in rows]):
                r["blind " + name] = [g, p]
        for r in rows:
            out.write(json.dumps({k: v for k, v in r.items() if k != "answers"}) + "\n")
        by = collections.defaultdict(list)
        for r in rows:
            by[r["source"]].append(r)
        for src, rs in sorted(by.items()):
            g, j = np.array([r["grade"] for r in rs]), np.array([r["judged"] for r in rs])
            print(f"  {stage} {src:13} n {len(rs):>3} | rigid grade {g.mean():.2f} | judged {j.mean():.2f} | right by the grade {np.mean(g >= 0.5):.2f}, by the judge {np.mean(j >= 0.5):.2f}"
                  f" | only the judge accepts {np.mean((j >= 0.5) & (g < 0.5)):.2f}, only the grade {np.mean((g >= 0.5) & (j < 0.5)):.2f}"
                  + " | blind, judged: " + ", ".join(f"{n} {np.mean([r['blind ' + n][1] for r in rs]):.2f}" for n in blind), flush=True)
            seen = set()
            for label, pick in (("the judge accepts, the grade does not", lambda r: r["judged"] >= 0.5 and r["grade"] < 0.5), ("the grade accepts, the judge does not", lambda r: r["grade"] >= 0.5 and r["judged"] < 0.5)):
                ex = [r for r in rs if pick(r) and (r["question"], r["said"]) not in seen][:5]
                for r in ex:
                    seen.add((r["question"], r["said"]))
                    print(f"      {label}: {r['question']!r} wanted {r['wanted']!r}, said {r['said']!r} (grade {r['grade']:.2f}, judged {r['judged']:.2f})")
    out.close()
    print(f"  the judge: {judge.calls} calls, {judge.hits} from its memory, {judge.fallbacks} fallbacks", flush=True)
    del brain
