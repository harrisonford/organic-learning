"""Monitor: how the world grows and how the current life (and its language) responds.

Writes out/monitor/report.md (latest) and appends one line per run to
out/monitor/history.jsonl. Run every few hours by the organic-monitor timer.

Sections:
  data   items per kind in the world, text formats, today's daemon yield
  life   per kind: accuracy vs the majority-answer baseline, timing, and how
         answers are produced (modal share, copied from the question, whether
         the right answer exists as an output neuron at all)
  ear    units known, contexts, mean unit length, how probe sentences are cut
  flags  observations relevant to the language pathway (docs/LANGUAGE_BIOLOGY.md)

    python experiments/monitor.py [out/full_s0.jsonl]
"""

import collections
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.world import World  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
LIFE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "out", "full_s0.jsonl")
OUT = os.path.join(ROOT, "out", "monitor")
TEXT_KINDS = ("talk", "chat", "reason", "story")
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ data
world = World(seed=0)
by_kind = collections.Counter(k for *_, k in world.train)
answers = collections.defaultdict(collections.Counter)
in_question = collections.defaultdict(list)
for _, q, a, k in world.train:
    answers[k][a] += 1
    if k in TEXT_KINDS:
        in_question[k].append(a in q)
baseline = {k: c.most_common(1)[0][1] / sum(c.values()) for k, c in answers.items()}
majority = {k: c.most_common(1)[0][0] for k, c in answers.items()}

formats = collections.Counter()
for line in open(os.path.join(ROOT, "data", "world", "text_qa.jsonl")):
    d = json.loads(line)
    if d.get("agree"):
        formats[d.get("format", "v1")] += 1
photo_adult = sum(1 for line in open(os.path.join(ROOT, "data", "world", "coco_qa.jsonl")) if '"agree": true' in line)

today = time.strftime("%Y-%m-%d")


def daemon_today(path, ok_pat, fail_pat):
    ok = fail = 0
    for line in open(path):
        if line.startswith(f"[{today}"):
            ok += bool(re.search(ok_pat, line))
            fail += bool(re.search(fail_pat, line))
    return ok, fail


text_ok, text_fail = daemon_today(os.path.join(ROOT, "out", "text_world.log"), r"judged fine|judged later", r"failed")
photo_ok, photo_fail = daemon_today(os.path.join(ROOT, "out", "annotate.log"), r"judged right", r"failed")

# ------------------------------------------------------------------ life
recs = [json.loads(line) for line in open(LIFE)] if os.path.exists(LIFE) else []
last = recs[-1] if recs else None


def mean_acc(rs, k):
    v = [r["measure"][k]["acc"] for r in rs if k in r["measure"]]
    return sum(v) / len(v) if v else float("nan")


lines = [f"# Monitor {time.strftime('%Y-%m-%d %H:%M')}", "", "## Data", "",
         f"World: {len(world.train):,} lived items, {len(world.test):,} test items. Adult photo items judged right: {photo_adult:,}.", "",
         "| kind | items | majority answer | majority share |", "|---|---|---|---|"]
for k in sorted(by_kind):
    lines.append(f"| {k} | {by_kind[k]:,} | {majority[k]} | {baseline[k]:.2f} |")
lines += ["", "Text world (judged fine) by format: " + ", ".join(f"{f} {n}" for f, n in formats.most_common()), "",
          f"Today: text batches ok {text_ok}, failed {text_fail}; photo batches ok {photo_ok}, failed {photo_fail}.", ""]
flags = []
if last:
    m = last["measure"]
    prev, recent = recs[-6:-3], recs[-3:]
    lines += ["## Life", "", f"Age {last['age']:,}, reward {last['lived_reward']:.2f}, clusters {last['clusters']:,}, answers known {last.get('outputs', '?')}, "
              f"memory {last['memory_gb']} GB.", "",
              "| kind | acc (last 3) | vs majority | trend | ticks | modal share (word) | copied from question | right answer producible |",
              "|---|---|---|---|---|---|---|---|"]
    for k in sorted(m):
        a = mean_acc(recent, k)
        t = a - mean_acc(prev, k) if prev else float("nan")
        x = m[k]
        lines.append(f"| {k} | {a:.2f} | {a - baseline.get(k, 0):+.2f} | {t:+.2f} | {x['tick']:.1f} | "
                     f"{x.get('modal', float('nan')):.2f} ({x.get('modal_word', '')}) | {x.get('copied', float('nan')):.2f} | {x.get('producible', float('nan')):.2f} |")
        if k in TEXT_KINDS:
            iq = sum(in_question[k]) / max(1, len(in_question[k]))
            if "producible" in x and x["producible"] < 0.6:
                flags.append(f"{k}: only {x['producible']:.0%} of right answers exist as output neurons (whole-answer output; gap 2)")
            if "copied" in x and iq > 0.4 and x["copied"] < 0.2:
                flags.append(f"{k}: answer is in the question for {iq:.0%} of items but the brain repeats a heard word {x.get('copied', 0):.0%} of the time (no hearing-to-saying route; gap 1)")
            if x.get("modal", 0) > 0.5:
                flags.append(f"{k}: {x['modal']:.0%} of replies are '{x.get('modal_word')}' (collapse onto one answer)")
            if a < baseline.get(k, 0) - 0.05:
                flags.append(f"{k}: below always saying '{majority[k]}' ({a:.2f} vs {baseline[k]:.2f})")
    ear = last.get("ear")
    if ear:
        lines += ["", "## Ear", "", f"{ear['units']:,} units heard, {ear['contexts']:,} contexts, mean unit {ear['mean_unit_bytes']} bytes, "
                  f"{ear['utterances']:,} utterances.", ""]
        lines += [f"- `{' | '.join(p)}`" for p in ear["probes"]]
        old = next((r["ear"] for r in recs if r.get("ear")), None)
        if old and old is not ear:
            lines += ["", f"Since monitoring began: units {old['units']:,} -> {ear['units']:,}, mean unit {old['mean_unit_bytes']} -> {ear['mean_unit_bytes']} bytes."]
lines += ["", "## Flags for the language pathway", ""] + ([f"- {f}" for f in flags] or ["- none"])
open(os.path.join(OUT, "report.md"), "w").write("\n".join(lines) + "\n")
with open(os.path.join(OUT, "history.jsonl"), "a") as f:
    f.write(json.dumps({"time": time.strftime("%F %T"), "world": len(world.train), "by_kind": by_kind, "formats": formats, "text_ok": text_ok,
                        "text_fail": text_fail, "photo_ok": photo_ok, "photo_fail": photo_fail, "age": last and last["age"],
                        "acc": last and {k: v["acc"] for k, v in last["measure"].items()}, "baseline": baseline, "flags": flags}) + "\n")
print("\n".join(lines))
