"""The paper's figures, made from the logs (docs/paper/). Nothing is computed here that is not in a log.
    python experiments/paper_figures.py
Figure 1 (fig_benchmark): CIFAR-10 seen once: held-out accuracy against pictures seen; mean of the seeds, their
range as a band. Colours: the dataviz skill's validated categorical order (light surface), with a marker per series
so that nothing is told by colour alone.
"""
import collections, os, re
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
