"""What is in preschool's lessons and test, source by source (queue item 8; docs/JUDGING.md, 2026-11-19: "preschool's
test is mostly answerable blind"). No brain, no judge: the held-out items of stage P, and what blind children get on
them by the letter-level grade (always "yes", always "no", the source's commonest answer, the question's last word
said back). For each source: how many items to live and to be tested on; how many different answers; the share of
the three commonest; the share of answers that are one of the question's own words; the best blind child.
    WORLD=5 python experiments/preschool_sources.py [stage]
"""
import collections, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.curriculum import Curriculum, grade, tidy  # noqa: E402

stage = sys.argv[1] if len(sys.argv) > 1 else "P"
cur = Curriculum(seed=0, world=int(os.environ.get("WORLD", 4)))
print(f"stage {stage}: sources to live {sorted(cur.train.get(stage, {}))}")
for src in sorted(cur.test.get(stage, {})):
    items = cur.test[stage][src]
    lived = len(cur.train.get(stage, {}).get(src, []))
    first = collections.Counter(it[2][0] for it in items)
    top = first.most_common(3)
    blind = {"yes": lambda it: "yes", "no": lambda it: "no", "common": lambda it: top[0][0],
             "last word": lambda it: (tidy(it[1]).split() or [""])[-1]}
    got = {name: float(np.mean([grade(stage, say(it), it[2]) >= 0.5 for it in items])) for name, say in blind.items()}
    inq = float(np.mean([any(a in tidy(it[1]).split() for a in it[2]) for it in items]))
    words = float(np.mean([len(it[2][0].split()) for it in items]))
    best = max(got, key=got.get)
    print(f"{src:14} to live {lived:6} | test {len(items):5} | different answers {len(first):5} | words in an answer {words:.1f} | "
          f"the three commonest: {', '.join(f'{a!r} {n / len(items):.2f}' for a, n in top)} | the answer is a word of the question {inq:.2f} | "
          f"best blind child: {best} {got[best]:.2f} ({', '.join(f'{k} {v:.2f}' for k, v in got.items())})", flush=True)
    for it in items[:3]:
        print(f"      {it[1]!r} -> {it[2]}")
