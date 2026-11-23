"""A fingerprint of a world's lived and held-out items and of preschool's test, to show that a change to the curriculum
leaves the worlds lives are in exactly as they were (2026-11-23: world 5 added; world 4 the same before and after).
    python experiments/world_hash.py 4 5
"""
import hashlib, sys, os
sys.path.insert(0, os.getcwd())
from organic.curriculum import Curriculum
for world in [int(x) for x in sys.argv[1:]]:
    cur = Curriculum(seed=0, world=world)
    for side, d in (("train", cur.train), ("test", cur.test)):
        h = hashlib.sha256(repr([(st, src, d[st][src]) for st in sorted(d) for src in sorted(d[st])]).encode()).hexdigest()[:16]
        print(world, side, h, {st: sum(len(v) for v in d[st].values()) for st in sorted(d)})
    ts = cur.test_set("P", n=300)
    print(world, "P test_set", len(ts), hashlib.sha256(repr([(m["question"], m["answers"], m["source"]) for m in ts]).encode()).hexdigest()[:16])
