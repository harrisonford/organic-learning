"""What grows in a life's memory? (2026-11-18, docs/CRAFT.md)

A life's memory climbs steadily (eyes2_s1: 2.7 GB at 30,000 moments, 4.05 GB at 85,000, where its guard stopped it).
For each pickle of a brain: the bytes held under each part (arrays counted once; lists, dicts and strings by a rough
size), the largest first, so that two ages of the same life can be compared.
    python experiments/mem_diag.py out/levels/eyes2_s1_W1.pkl out/levels/eyes2_s1_W2.pkl out/eyes2_s1.pkl
"""
import os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def size(o, seen):
    if id(o) in seen:
        return 0
    seen.add(id(o))
    if isinstance(o, np.ndarray):
        return o.nbytes  # (an unpickled array has a base; views of one array would be counted twice: none here)
    if isinstance(o, (str, bytes)):
        return sys.getsizeof(o)
    if isinstance(o, dict):
        return sys.getsizeof(o) + sum(size(k, seen) + size(v, seen) for k, v in o.items())
    if isinstance(o, (list, tuple, set, frozenset)):
        return sys.getsizeof(o) + sum(size(x, seen) for x in o)
    if hasattr(o, "__dict__"):
        return size(vars(o), seen)
    return sys.getsizeof(o)


def parts(o, path, seen, out, depth):
    for k, v in (vars(o).items() if hasattr(o, "__dict__") else []):
        if hasattr(v, "__dict__") and depth < 2:
            parts(v, f"{path}.{k}", seen, out, depth + 1)
        else:
            n = size(v, seen)
            what = f"array {v.shape} {v.dtype}" if isinstance(v, np.ndarray) else f"{type(v).__name__} of {len(v)}" if hasattr(v, "__len__") else type(v).__name__
            out.append((n, f"{path}.{k}", what))


for p in sys.argv[1:]:
    brain = pickle.load(open(p, "rb"))
    out = []
    parts(brain, "brain", set(), out, 0)
    print(f"{p}: age {brain.age}, file {os.path.getsize(p) / 1e9:.2f} GB, counted {sum(n for n, _, _ in out) / 1e9:.2f} GB")
    for n, name, what in sorted(out, reverse=True)[:22]:
        print(f"   {n / 1e6:9.1f} MB  {name:40} {what}")
    sys.stdout.flush()
    del brain
