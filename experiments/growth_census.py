"""What each organ holds at each stage of a life (queue E 33; docs/GROWTH.md; 2026-11-26).

The lives' records give counts (cells, clusters, store) but not how many numbers each organ holds. This loads a
life's stage checkpoints (out/levels/{tag}_{BB,B,W1,W2,P}.pkl, in order) one at a time and counts, under each of the
brain's top-level parts, the numbers in its arrays (each array once) and the bytes of the rest, as
experiments/mem_diag.py does. Writes out/growth/{tag}_census.json; growth_report.py puts it into docs/GROWTH.md.

    python experiments/growth_census.py plat_s0
"""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

STAGES = ["BB", "B", "W1", "W2", "P", "K"]


def count(o, seen):
    """(numbers in arrays, bytes of everything) under o, each object once."""
    if id(o) in seen:
        return 0, 0
    seen.add(id(o))
    if isinstance(o, np.ndarray):
        return int(o.size), int(o.nbytes)
    if isinstance(o, (str, bytes)):
        return 0, sys.getsizeof(o)
    items = o.items() if isinstance(o, dict) else ((None, x) for x in o) if isinstance(o, (list, tuple, set, frozenset)) else vars(o).items() if hasattr(o, "__dict__") else ()
    n, b = 0, sys.getsizeof(o)
    for k, v in items:
        dn, db = count(v, seen)
        n, b = n + dn, b + db
        if k is not None and not isinstance(o, dict):
            pass
    return n, b


if __name__ == "__main__":
    for tag in sys.argv[1:]:
        rows = []
        for st in STAGES:
            p = f"out/levels/{tag}_{st}.pkl"
            if not os.path.exists(p):
                continue
            brain, seen, parts = pickle.load(open(p, "rb")), set(), {}
            for k, v in vars(brain).items():
                n, b = count(v, seen)
                if b > 1e6:
                    parts[k] = {"numbers_m": round(n / 1e6, 2), "mb": round(b / 1e6, 1)}
            rows.append({"stage": st, "age": int(getattr(brain, "age", 0)), "file_gb": round(os.path.getsize(p) / 1e9, 2), "parts": parts,
                         "numbers_m": round(sum(x["numbers_m"] for x in parts.values()), 1)})
            w = getattr(brain, "words", None)
            if w is not None and getattr(w, "born", None) is not None:  # turnover: the word cells carry the moment they were born
                n, now = len(w.units), int(w.moments)
                born = np.asarray(w.born[:n], np.float64)
                before = rows[-2]["word_cells"] if len(rows) > 1 and "word_cells" in rows[-2] else {"n": 0, "moments": -1}
                new = int((born > before["moments"]).sum())
                rows[-1]["word_cells"] = {"n": n, "moments": now, "born_since": new, "forgotten_since": int(before["n"] + new - n),
                                          "median_age": int(np.median(now - born)), "cues": len(getattr(w, "cues", []))}
            print(f"{tag} {st} age {rows[-1]['age']}: {rows[-1]['numbers_m']} million numbers | {rows[-1].get('word_cells')} | " + ", ".join(f"{k} {v['numbers_m']}" for k, v in sorted(parts.items(), key=lambda kv: -kv[1]['numbers_m'])[:8]), flush=True)
            del brain
        os.makedirs("out/growth", exist_ok=True)
        json.dump(rows, open(f"out/growth/{tag}_census.json", "w"))
