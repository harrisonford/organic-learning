"""Scoring the organism's answers against what a person said."""

import numpy as np

from .scenes import episode


def same_answer(out, human, intent, n_objects):
    if intent == "what" and n_objects > 1:
        # describing several things in a different order is fine
        def parts(text):
            return sorted(p.strip() for p in text.replace("i see", "").split(" and "))

        return parts(out) == parts(human)
    if intent == "what":  # "i see a red circle" and "it is a red circle" are both fine
        return out.split()[-2:] == human.split()[-2:] and len(out.split()) == len(human.split())
    return out == human


def evaluate(brain, split, n=300, seed=99, show=0, printer=print):
    rng = np.random.default_rng(seed)
    hits, per = 0, {}
    for i in range(n):
        e = episode(rng, split)
        out, _ = brain.live(e["image"], e["question"], learn=False)
        ok = same_answer(out, e["answer"], e["intent"], len(e["objects"]))
        hits += ok
        per.setdefault(e["intent"], []).append(ok)
        if i < show:
            mark = "ok " if ok else "BAD"
            printer(f"   {mark} [{e['intent']:>8}] {e['question']:<34} -> {out:<42} (person: {e['answer']})")
    return hits / n, {k: float(np.mean(v)) for k, v in sorted(per.items())}


def explain(brain, image, question):
    """Show, word by word, what the speech area expected and how vision gated it."""
    trace = []
    out, _ = brain.live(image, question, learn=False, trace=trace)
    lines = [f"Q: {question}", f"A: {out}"]
    for step, cands in enumerate(trace):
        row = "  ".join(f"{w}(vote {v:.2f} x gain {g:.2f} -> {p:.2f})" for w, v, g, p in cands[:4])
        lines.append(f"  step {step}: {row}")
    return "\n".join(lines)
