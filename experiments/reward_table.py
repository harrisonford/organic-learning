"""The reward lives' reading as a table (docs/JUDGING.md, 2026-11-24 and 2026-11-25), from the lives' own records.

For each life: moments spent in first words (W1), pages (W2) and preschool (P) (records in the stage x 5,000), the
last test of a stage while the life was in it, and every stage's test at the life's end; all "above blind answering"
(the records' `above`).
    python experiments/reward_table.py ramp_s0 ramp_s1 ramp_s2 [--vs kind_s0 kind_s1 kind_s2]
"""
import json, sys

STEP = 5000


def read(tag):
    rows = [json.loads(line) for line in open(f"out/{tag}.jsonl")]
    above = lambda r, k: (r.get("measure", {}).get(k) or {}).get("above")  # noqa: E731
    out = {}
    for st in ("W1", "W2", "P"):
        inside = [r for r in rows if r["stage"] == st]
        out[f"moments in {st}"] = len(inside) * STEP if inside else None
        last = [above(r, st) for r in inside if above(r, st) is not None]
        out[f"{st}, last test in the stage"] = last[-1] if last else None
    for st in ("P", "W1", "W2"):
        out[f"{st} at the end"] = above(rows[-1], st)
    out["age at the end"] = rows[-1]["age"]
    return out


def fmt(v):
    return "–" if v is None else f"{v:,}" if isinstance(v, int) else f"{v:.2f}"


if __name__ == "__main__":
    args = sys.argv[1:]
    vs = args[args.index("--vs") + 1:] if "--vs" in args else []
    tags = args[: args.index("--vs")] if vs else args
    A, B = [read(t) for t in tags], [read(t) for t in vs]
    print("| | " + " | ".join(tags) + " |\n|---|" + "---|" * len(tags))
    for k in A[0]:
        print(f"| {k} | " + " | ".join(fmt(a[k]) + (f" ({fmt(b[k])})" if B else "") for a, b in zip(A, B or [None] * len(A))) + " |")
