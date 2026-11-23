"""The paper's figures, made from the logs (docs/paper/). Nothing is computed here that is not in a log.
    python experiments/paper_figures.py
Figure 1 (fig_benchmark): CIFAR-10 seen once: held-out accuracy against pictures seen; mean of the seeds, their
range as a band. Colours: the dataviz skill's validated categorical order (light surface), with a marker per series
so that nothing is told by colour alone.
Figure 2 (fig_stages): the lives with no pass mark (plat_s*): the score above the blind floor of first words and of
pages against moments lived, a line per seed; a tick where the world moved on.
Figure 3 (fig_ablation): a life with one part taken away, at 50,000 moments: mean of three seeds as a bar, the seeds
as dots, the whole brain's mean as a line down each panel.
"""
import collections, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
rows = collections.defaultdict(dict)
for f in ("stream_developed_s0.log", "stream_seeds.log"):
    for l in open(f"out/bench/{f}"):
        m = re.match(r"seed (\d) developed (\S+)\s+(.*)", l)
        if m:
            rows[m.group(2)][int(m.group(1))] = {int(a): float(b) for a, b in re.findall(r"(\d+) (0\.\d+)", m.group(3))}
for l in open("out/bench/backprop_full.log"):
    m = re.match(r"seed (\d) backprop filters random\s+rate \S+\s+batch \S+\s+replay (\d): (.*)", l)
    if m:
        rows["backprop replay " + m.group(2)][int(m.group(1))] = {int(a): float(b) for a, b in re.findall(r"(\d+) (0\.\d+)", m.group(3))}

SERIES = [  # (log key, label, colour, marker, label's nudge in points)
    ("cells+relay", "word cells through the relay", "#2a78d6", "o", -11),
    ("backprop replay 0", "backprop, same shape, one pass", "#eb6834", "s", 0),
    ("backprop replay 4", "backprop, 4 pictures lived again per new one (1 seed)", "#1baf7a", "^", 9),
    ("cells", "word cells, no relay", "#eda100", "D", 0),
    ("slda+relay", "streaming LDA through the relay", "#e87ba4", "v", 0),
]
AT = [500, 1000, 2000, 5000, 10000, 20000, 50000]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(6.6, 4.0), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for key, label, colour, marker, nudge in SERIES:
    seeds = sorted(rows[key])
    v = np.array([[rows[key][s][n] for n in AT] for s in seeds])
    if len(seeds) > 1:
        ax.fill_between(AT, v.min(axis=0), v.max(axis=0), color=colour, alpha=0.18, linewidth=0)
    ax.plot(AT, v.mean(axis=0), color=colour, linewidth=2, marker=marker, markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1, label=label)
    ax.annotate(f"{v.mean(axis=0)[-1]:.3f}", (AT[-1], v.mean(axis=0)[-1]), xytext=(7, nudge - 3), textcoords="offset points", color=INK, fontsize=8.5)
ax.set_xscale("log")
ax.set_xticks(AT)
ax.set_xticklabels([f"{n:,}" for n in AT])
ax.minorticks_off()
ax.set_xlim(430, 80000)
ax.set_ylim(0.2, 0.78)
ax.set_xlabel("labelled pictures seen (each once)", color=MUTED)
ax.set_ylabel("held-out pictures named right", color=MUTED)
ax.tick_params(colors=MUTED, length=0)
ax.grid(axis="y", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
handles, labels = ax.get_legend_handles_labels()
order = np.argsort([-np.mean([rows[k][s][AT[-1]] for s in rows[k]]) for k, *_ in SERIES])
leg = ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower right", frameon=False, fontsize=8, labelcolor=INK, handlelength=2.4)
ax.set_title("CIFAR-10 seen once: accuracy against pictures seen", loc="left", color=INK, fontsize=10)
fig.tight_layout()
os.makedirs("docs/paper", exist_ok=True)
fig.savefig("docs/paper/fig_benchmark.pdf", facecolor=SURFACE)
fig.savefig("docs/paper/fig_benchmark.png", dpi=170, facecolor=SURFACE)
print("docs/paper/fig_benchmark.pdf, .png")


# ---- figure 3: one part taken away ----
def at(tag, seed, age=50000):
    for l in open(f"out/{tag}_s{seed}.jsonl"):
        if l.startswith("{"):
            r = json.loads(l)
            if r.get("age") == age and "measure" in r:
                m = r["measure"]
                return m["W1"]["above"], m["W2"]["by_source"]["page:name"]["above"], m["W2"]["by_source"]["page:there"]["above"]


ARMS = [("cleanw", "the whole brain"), ("ablrand", "cortex never developed"), ("ablnorelay", "no relay"),
        ("ablnostore", "no episodic store"), ("ablnolook", "no looking"), ("ablnomatch", "no match neurons")]
PANELS = ["naming a single picture", "naming a thing on a page", "\u201cis there a \u2026?\u201d on a page"]
v = np.array([[at(tag, s) for s in range(3)] for tag, _ in ARMS])  # arms x seeds x panels
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), facecolor=SURFACE, sharey=True)
ys = np.arange(len(ARMS))[::-1]
for j, ax in enumerate(axes):
    ax.set_facecolor(SURFACE)
    ax.axvline(v[0, :, j].mean(), color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    ax.barh(ys, v[:, :, j].mean(axis=1), height=0.56, color=["#8f8e89"] + ["#2a78d6"] * (len(ARMS) - 1), zorder=2)
    for i, y in enumerate(ys):
        ax.scatter(v[i, :, j], [y] * 3, s=16, color=INK, edgecolor=SURFACE, linewidth=0.8, zorder=3)
        ax.annotate(f"{v[i, :, j].mean():.2f}", (max(v[i, :, j].max(), 0), y), xytext=(5, -3), textcoords="offset points", color=INK, fontsize=8)
    ax.set_xlim(0, 0.68)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.set_title(PANELS[j], loc="left", color=INK, fontsize=9)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
axes[0].set_yticks(ys)
axes[0].set_yticklabels([name for _, name in ARMS], color=INK)
axes[1].set_xlabel("score above what a blind answer gets, at 50,000 moments (bar: mean of three seeds; dots: the seeds)", color=MUTED, fontsize=8)
fig.tight_layout()
fig.savefig("docs/paper/fig_ablation.pdf", facecolor=SURFACE)
fig.savefig("docs/paper/fig_ablation.png", dpi=170, facecolor=SURFACE)
print("docs/paper/fig_ablation.pdf, .png")


# ---- figure: one life (seed 0: clean_s0 to 20,000 moments, then cleanw_s0 kept in the pages stage) ----
def records(tag, lo, hi):
    for l in open(f"out/{tag}.jsonl"):
        if l.startswith("{"):
            r = json.loads(l)
            if "measure" in r and lo < r["age"] <= hi:
                yield r


life = list(records("clean_s0", 0, 20000)) + list(records("cleanw_s0", 20000, 10 ** 9))
ages = np.array([r["age"] for r in life]) / 1000
fig = plt.figure(figsize=(7.4, 4.6), facecolor=SURFACE)
ax = fig.add_axes([0.085, 0.40, 0.80, 0.47], facecolor=SURFACE)
for key, label, colour, marker in (("W1", "naming single pictures (first words)", "#2a78d6", "o"), ("W2", "picture-book pages", "#eb6834", "s")):
    pts = [(a, r["measure"][key]["above"]) for a, r in zip(ages, life) if "above" in r["measure"].get(key, {})]
    x, y = zip(*pts)
    ax.plot(x, y, color=colour, linewidth=2, marker=marker, markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=0.8, label=label)
    ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), xytext=(6, 1 if key == "W1" else -8), textcoords="offset points", color=INK, fontsize=8.5)
stages, start = [], 0
for r in life:  # where each stage began, from the life's own record
    if not stages or stages[-1][0] != r["stage"]:
        stages.append([r["stage"], start])
    start = r["age"] / 1000
NAMES = {"BB": "babbling", "B": "echo", "W1": "first words", "W2": "pages"}
for i, (st, x0) in enumerate(stages):
    x1 = stages[i + 1][1] if i + 1 < len(stages) else ages[-1]
    ax.axvspan(x0, x1, ymin=0.93, ymax=1.0, color=["#d9d8d2", "#c4c3bd"][i % 2], linewidth=0)
    if x1 - x0 > 12:
        ax.text((x0 + x1) / 2, 0.618, NAMES.get(st, st), ha="center", va="center", fontsize=7.5, color=INK)
    else:  # a short stage: its name slanted above the band
        ax.text((x0 + x1) / 2, 0.65, NAMES.get(st, st), ha="left", va="bottom", fontsize=7.5, color=INK, rotation=32, clip_on=False)
ax.set_xlim(0, ages[-1] + 1)
ax.set_ylim(0, 0.64)
ax.set_ylabel("score above blind answering", color=MUTED)
ax.set_xlabel("moments lived (thousands)", color=MUTED)
ax.tick_params(colors=MUTED, length=0)
ax.grid(axis="y", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK)
strip = fig.add_axes([0.085, 0.0, 0.80, 0.30], facecolor=SURFACE)
strip.set_xlim(ax.get_xlim())
strip.set_ylim(0, 1)
strip.axis("off")
short = lambda xs, n=2, most=24: [s for s in xs if 1 < len(s) <= most][:n]  # noqa: E731
for at, anchor in ((5, "left"), (30, "center"), (65, "center"), (110, "right")):
    r = min(life, key=lambda r: abs(r["age"] / 1000 - at))
    babble = short(r["measure"]["BB"]["by_source"]["babble"]["said"])
    named = short((r["measure"].get("W1", {}).get("by_source", {}).get("cifar:name", {}).get("said") or []), 3, 12)
    text = f"at {r['age']:,} moments\nbabbling alone:\n" + "\n".join(f"  “{b}”" for b in babble) + ("\nasked to name a picture:\n  " + ", ".join(named) if named else "\n(names nothing yet)")
    col = [5, 30, 65, 110].index(at)
    strip.text(strip.get_xlim()[1] * (0.02 + 0.26 * col), 0.98, text, ha="left", va="top", fontsize=7.3, color=INK, linespacing=1.35)
    ax.axvline(r["age"] / 1000, ymax=0.93, color=MUTED, linewidth=0.5, linestyle=(0, (2, 3)), zorder=0)
fig.savefig("docs/paper/fig_life.pdf", facecolor=SURFACE)
fig.savefig("docs/paper/fig_life.png", dpi=170, facecolor=SURFACE)
print("docs/paper/fig_life.pdf, .png")

# ---- figure: what it says (experiments/showcase_probe.py on a saved learner; the first test items, in order) ----
show = json.load(open("out/tmp/showcase_cleanw_s0.json"))
pics = np.load("out/tmp/showcase_cleanw_s0.npz")
fig, axes = plt.subplots(2, 6, figsize=(7.4, 3.5), facecolor=SURFACE)
for row, src in enumerate(("cifar:name", "page:name")):
    items = [r for r in show["rows"] if r.get("picture", "").startswith(src)][:6]
    for ax, r in zip(axes[row], items):
        ax.imshow(pics[r["picture"]], interpolation="nearest" if row == 0 else "bilinear")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color(GRID)
        ax.set_title(f"“{r['question']}”", fontsize=6.3, color=MUTED, pad=3)
        mark = "✓" if r["right"] else "✗"
        ax.set_xlabel(f"{mark} {r['said'] or '(silence)'}" + ("" if r["right"] else f"\n(it was: {' / '.join(r['answers'])})"), fontsize=7.6, color=INK, labelpad=3)
fig.tight_layout(h_pad=1.4, w_pad=0.6)
fig.savefig("docs/paper/fig_examples.pdf", facecolor=SURFACE)
fig.savefig("docs/paper/fig_examples.png", dpi=170, facecolor=SURFACE)
print("docs/paper/fig_examples.pdf, .png")
