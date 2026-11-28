"""The figures added for draft 12 of the paper (2026-11-28; Kris: "as visual as possible and concise"). The older ones
(fig_benchmark, fig_life, fig_examples, fig_ablation) are made by experiments/paper_figures.py and are unchanged.

    .venv/bin/python experiments/paper_figures12.py

fig_organism   the parts, and how each comes to be what it is (which are born with the life)
fig_credit     seeing, one pass: with and without the relay, ours against backpropagation, CIFAR-10 and photographs
fig_ear        the ear born with the life: three readers along the life, and the preparation taken away piece by piece
fig_clock      one sentence through the ear on the 10 ms clock (its loudness, the cuts, the sound units, their traces)
fig_voices     two voices in one life: three pairs of readers, a relay that follows the voice against a fixed one

Numbers that are read from logs are read here; the few taken from tables of the long report (docs/paper/SUPPLEMENT.md)
are written out with the table they come from. Colours and marks as in paper_figures.py (ours blue, the reference
orange; nothing is told by colour alone: every series is labelled where it is drawn).
"""
import json, os, pickle, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "pdf.fonttype": 42})
ONLY = set(sys.argv[1:])


def save(fig, name):
    fig.savefig(f"docs/paper/{name}.pdf", facecolor=SURFACE)
    fig.savefig(f"docs/paper/{name}.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print(name)


def clean(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=2.5, labelsize=7.5)


# ---------------------------------------------------------------------------------------------------------- fig_credit
if not ONLY or "credit" in ONLY:
    # at the end of the stream (50,000 pictures seen once), three seeds, mean and spread; SUPPLEMENT.md, the tables of
    # "The local learner beats one-pass backpropagation", "Which part earned the win", "Real photographs at the eye's size"
    DATA = {"CIFAR-10 (ten classes)": {"ours": [(0.591, 0.007), (0.687, 0.005)], "backprop": [(0.616, 0.023), (0.688, 0.005)], "blind": 0.10,
                                       "lda": 0.739},
            "real photographs (33 words)": {"ours": [(0.389, 0.009), (0.446, 0.006)], "backprop": [(0.392, 0.005), (0.463, 0.001)], "blind": 0.051,
                                            "lda": 0.499}}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.75), facecolor=SURFACE)
    for ax, (title, d) in zip(axes, DATA.items()):
        ax.set_facecolor(SURFACE), clean(ax)
        for j, (key, colour, label) in enumerate((("ours", BLUE, "local\nrule"), ("backprop", ORANGE, "back-\nprop"))):
            for i, (m, s) in enumerate(d[key]):
                x = i * 2.6 + j
                ax.bar(x, m, 0.86, color=colour, edgecolor=SURFACE, linewidth=2)
                ax.errorbar(x, m, yerr=s, color=INK, linewidth=0.9, capsize=2)
                ax.text(x, m - s - 0.012, f"{100 * m:.1f}", ha="center", va="top", fontsize=8.2, color="white", fontweight="bold")
                ax.text(x, -0.012, label, ha="center", va="top", fontsize=6.6, color=INK, linespacing=1.05)
        ax.axhline(d["lda"], color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.text(3.95, d["lda"] + 0.008, "a streaming discriminant on the same code", ha="right", va="bottom", fontsize=6.4, color=MUTED)
        ax.set_xticks([])
        for x, words in ((0.5, "the code as the\ncortex gives it"), (3.1, "the code decorrelated\nonce more (the relay)")):
            ax.text(x, -0.115 * d["lda"] * 1.16 / 0.86, words, ha="center", va="top", fontsize=7.6, color=INK, fontweight="bold", linespacing=1.1)
        ax.set_xlim(-0.75, 4.25), ax.set_ylim(0, d["lda"] * 1.16)
        ax.set_yticks(np.arange(0, d["lda"] * 1.16, 0.1)), ax.set_yticklabels([f"{int(round(100 * v))}%" for v in np.arange(0, d["lda"] * 1.16, 0.1)])
        ax.set_title(title, fontsize=8.6, color=INK, loc="left", pad=6)
    axes[0].set_ylabel("held-out pictures named,\nafter 50,000 seen once", fontsize=7.6, color=MUTED)
    fig.subplots_adjust(left=0.105, right=0.995, top=0.9, bottom=0.27, wspace=0.2)
    save(fig, "fig_credit")

# ------------------------------------------------------------------------------------------------------------- fig_ear
CURVE = re.compile(r"after (\d+) captions[^:]*: photograph -> its caption among 1000: R@1 [\d.]+ R@5 [\d.]+ R@10 ([\d.]+)")


def curve(path):
    return [(int(a), float(b)) for a, b in CURVE.findall(open(path).read())]


if not ONLY or "ear" in ONLY:
    readers = [("top", "speaker 1"), ("top2", "speaker 2"), ("top3", "speaker 3")]
    fig = plt.figure(figsize=(7.4, 4.7), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.78], hspace=0.62, wspace=0.12, left=0.085, right=0.995, top=0.935, bottom=0.075)
    for i, (r, name) in enumerate(readers):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(SURFACE), clean(ax)
        born = curve(f"out/bench/grounded_online_{r}_live0.5_circuit_layerborn_s0.log")
        prep = curve(f"out/bench/grounded_online_{r}_s0.log")
        for c, colour, mark in ((prep, ORANGE, "s"), (born, BLUE, "o")):
            ax.plot([a / 1000 for a, _ in c], [100 * b for _, b in c], color=colour, linewidth=2, marker=mark, markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=1)
        ax.axhline(1.0, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.set_ylim(0, 11), ax.set_xlim(0, 10.3), ax.set_yticks([0, 2, 4, 6, 8, 10])
        ax.set_title(name, fontsize=8.6, color=INK, loc="left", pad=4)
        ax.text(born[-1][0] / 1000, 100 * born[-1][1] + 0.5, f"{100 * born[-1][1]:.1f}", ha="center", va="bottom", fontsize=7.6, color=BLUE, fontweight="bold")
        if i == 0:
            ax.set_ylabel("right caption in the\nfirst ten of 1,000 (%)", fontsize=7.6, color=MUTED)
            ax.text(10.2, 1.25, "chance", ha="right", va="bottom", fontsize=6.6, color=MUTED)
            ax.plot([0.5, 1.4], [10.1, 10.1], color=BLUE, linewidth=2, marker="o", markersize=4.5, markeredgecolor=SURFACE, markevery=[1])
            ax.text(1.7, 10.1, "born with the life, nothing prepared", va="center", fontsize=6.6, color=INK)
            ax.plot([0.5, 1.4], [8.8, 8.8], color=ORANGE, linewidth=2, marker="s", markersize=4.5, markeredgecolor=SURFACE, markevery=[1])
            ax.text(1.7, 8.8, "prepared beforehand", va="center", fontsize=6.6, color=INK)
        else:
            ax.set_yticklabels([])
        if i == 1:
            ax.set_xlabel("spoken captions heard, each once, with its photograph (thousands)", fontsize=7.6, color=MUTED)
    # the preparation taken away piece by piece, reader 2 (each number from the end of its own log; +-0.008 is the test's own spread)
    end = lambda f: curve(f"out/bench/{f}")[-1][1]  # noqa: E731
    STEPS = [("everything prepared; each sentence known whole before it is used", end("grounded_online_top2_s0.log"), ORANGE),
             ("… only the past is known (adapting within half a second)", end("grounded_online_top2_live0.5_s0.log"), ORANGE),
             ("… and the relay is a circuit learning as the sound comes", end("grounded_online_top2_live0.5_circuit_s0.log"), ORANGE),
             ("… and the hearing cortex is born with the life: nothing prepared", end("grounded_online_top2_live0.5_circuit_layerborn_s0.log"), BLUE)]
    ax = fig.add_subplot(gs[1, :])
    ax.set_facecolor(SURFACE), clean(ax, "x")
    for k, (label, v, colour) in enumerate(STEPS):
        y = len(STEPS) - 1 - k
        ax.errorbar(100 * v, y, xerr=0.8, color=colour, linewidth=1.4, capsize=2.5, marker="o" if colour == BLUE else "s", markersize=6, markeredgecolor=SURFACE)
        ax.text(100 * v + 1.05, y, f"{100 * v:.1f}", va="center", fontsize=7.8, color=INK, fontweight="bold" if colour == BLUE else "normal")
        ax.text(-0.25, y, label, va="center", ha="right", fontsize=7.4, color=INK)
    ax.axvline(1.0, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3))), ax.text(1.12, -0.62, "chance", fontsize=6.6, color=MUTED, va="center")
    ax.axvline(8.8, color=GREEN, linewidth=0.9, linestyle=(0, (1, 2))), ax.text(8.68, -0.62, "written words through the same eye and link", fontsize=6.6, color=MUTED, va="center", ha="right")
    ax.set_xlim(0, 11), ax.set_ylim(-0.95, len(STEPS) - 0.5), ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title("Speaker 2 at the end of the life, as the preparation is taken away", fontsize=8.6, color=INK, loc="left", pad=4, x=-0.72)
    ax.set_position([0.5, 0.075, 0.495, 0.3])
    save(fig, "fig_ear")

# ---------------------------------------------------------------------------------------------------------- fig_voices
VOICE = re.compile(r"R@10 ([\d.]+); caption -> its photograph: R@1 [\d.]+ R@5 [\d.]+ R@10 [\d.]+; median rank (\d+)")


def voices(a, b, kind):
    """(later voice at the end, first voice at the change, first voice at the end): R@10 of each."""
    out = {}
    for l in open(f"out/bench/two_voices_{a}_{b}_{kind}_s0.log"):
        m = VOICE.search(l)
        if not m:
            continue
        key = ("later" if "the second voice:" in l and "after" in l else "first_end" if "the first voice" in l else
               "first" if re.search(r"after \d+ captions \(", l) else None)
        if key:
            out[key] = float(m.group(1))
    return out


if not ONLY or "voices" in ONLY:
    PAIRS = [("top2", "top", "2, then 1"), ("top", "top2", "1, then 2"), ("top2", "top3", "2, then 3")]
    # how much the two voices use the same sound units (overlap of held-out use), from the last line of the kinds runs
    OVERLAP = {("top2", "top"): (0.44, 0.12), ("top", "top2"): (None, 0.16), ("top2", "top3"): (0.48, 0.35)}
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), facecolor=SURFACE)
    rows = {"later": [], "kept": []}
    for a, b, _ in PAIRS:
        alone = voices(a, b, "alone")["later"]
        for kind in ("born", "batch"):
            v = voices(a, b, kind)
            rows["later"].append(v["later"] / alone), rows["kept"].append(v["first_end"] / v["first"])
    for ax, key, title in ((axes[0], "later", "the later voice, against that\nvoice heard alone from birth"), (axes[1], "kept", "the first voice, against what it\nwas when the voice changed")):
        ax.set_facecolor(SURFACE), clean(ax)
        ax.axhline(1.0, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
        for i in range(3):
            for j, (colour, mark) in enumerate(((BLUE, "o"), (ORANGE, "s"))):
                v = rows[key][2 * i + j]
                ax.plot(i + (j - 0.5) * 0.3, v, marker=mark, markersize=8, color=colour, markeredgecolor=SURFACE, markeredgewidth=1.2, linestyle="none")
                ax.text(i + (j - 0.5) * 0.3 + (j - 0.5) * 0.22, v, f"{v:.2f}", ha="right" if j == 0 else "left", va="center", fontsize=6.9, color=INK, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.8))
        ax.set_xticks(range(3)), ax.set_xticklabels([p[2] for p in PAIRS], fontsize=7.2, color=INK)
        ax.set_xlabel("speakers heard, in this order", fontsize=7.2, color=MUTED)
        ax.set_ylim(0, 1.3), ax.set_xlim(-0.75, 2.75), ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(title, fontsize=7.9, color=INK, loc="left", pad=5)
    axes[0].plot(-0.55, 0.2, marker="o", markersize=7, color=BLUE, markeredgecolor=SURFACE), axes[0].text(-0.4, 0.2, "relay learns with the ear", va="center", fontsize=6.6, color=INK)
    axes[0].plot(-0.55, 0.08, marker="s", markersize=7, color=ORANGE, markeredgecolor=SURFACE), axes[0].text(-0.4, 0.08, "relay fitted to the first voice, fixed", va="center", fontsize=6.6, color=INK)
    ax = axes[2]
    ax.set_facecolor(SURFACE), clean(ax)
    for i, (a, b, _) in enumerate(PAIRS):
        for j, (colour, mark) in enumerate(((BLUE, "o"), (ORANGE, "s"))):
            v = OVERLAP[(a, b)][j]
            if v is None:
                ax.text(i - 0.15, 0.05, "not\nrun", ha="center", fontsize=6.2, color=MUTED)
                continue
            ax.plot(i + (j - 0.5) * 0.3, v, marker=mark, markersize=8, color=colour, markeredgecolor=SURFACE, markeredgewidth=1.2, linestyle="none")
            ax.text(i + (j - 0.5) * 0.3 + (j - 0.5) * 0.22, v, f"{v:.2f}", ha="right" if j == 0 else "left", va="center", fontsize=6.9, color=INK, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.8))
    ax.set_xticks(range(3)), ax.set_xticklabels([p[2] for p in PAIRS], fontsize=7.2, color=INK)
    ax.set_xlabel("speakers heard, in this order", fontsize=7.2, color=MUTED)
    ax.set_ylim(0, 0.8), ax.set_xlim(-0.75, 2.75), ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_title("how far the two voices share\nthe same sound units (0 to 1)", fontsize=7.9, color=INK, loc="left", pad=5)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.84, bottom=0.175, wspace=0.22)
    save(fig, "fig_voices")

# ----------------------------------------------------------------------------------------------------------- fig_clock
if not ONLY or "clock" in ONLY:
    from organic.hear_stream import StreamEar  # noqa: E402
    DIR = os.path.expanduser("~/datasets/spokencoco")
    kept = pickle.load(open("out/bench/grounded_online_ear_s0_rtop2_live0.5.pkl", "rb"))
    ear = StreamEar(kept["layer"], kept["relay"], tau=0.5, learning=False)
    (ear.pairs.U, ear.pairs.wins, ear.pairs.founded), (ear.triples.U, ear.triples.wins, ear.triples.founded) = kept["pairs"], kept["triples"]
    utts = [u for u in map(json.loads, open(os.path.join(DIR, "cochlea.jsonl"))) if u["split"] == "train" and u["rows"] >= 50]
    count = {}
    for u in utts:
        count[u["speaker"]] = count.get(u["speaker"], 0) + 1
    who = sorted(count, key=count.get, reverse=True)[1]
    mine = [u for u in utts if u["speaker"] == who]
    grams = np.memmap(os.path.join(DIR, "cochlea_64.f16"), np.float16, "r").reshape(-1, 64)
    for u in mine[:30]:  # the ear has heard a little before the sentence shown, as it would have in a life
        for row in np.asarray(grams[u["start"]: u["start"] + u["rows"]], np.float32):
            ear.step(row)
        ear.pause()
    u = next(u for u in mine[30:] if 300 <= u["rows"] <= 380)
    G = np.asarray(grams[u["start"]: u["start"] + u["rows"]], np.float32)
    heard, cuts_seen = [], []
    for t, row in enumerate(G):
        before = ear.done
        for what, k in ear.step(row):
            heard.append((t, what, k))
        if ear.done != before:
            cuts_seen.append((ear.done, t))
    env, quiet = np.array(ear.env), np.array(ear.quiet)
    n_env = len(env)
    for what, k in ear.pause():
        heard.append((len(G) - 1, what, k))
    T = np.arange(len(G) + 60) / 100.0
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 3.9), facecolor=SURFACE, sharex=True, gridspec_kw={"height_ratios": [1.0, 0.55, 1.0], "hspace": 0.42})
    ax = axes[0]
    ax.set_facecolor(SURFACE), clean(ax)
    off = (8 - 1) / 100.0  # a code frame is the last 80 ms
    te = np.arange(n_env) / 100.0 + off
    ax.fill_between(te, env / env.max(), color=BLUE, alpha=0.18, linewidth=0), ax.plot(te, env / env.max(), color=BLUE, linewidth=1.4)
    for c, _ in cuts_seen:
        ax.axvline(c / 100.0 + off, color=INK, linewidth=0.6, alpha=0.55)
    ax.set_ylim(0, 1.12), ax.set_yticks([]), ax.set_ylabel("loudness", fontsize=7.4, color=MUTED)
    ax.text(0.0, 1.1, "A sentence arrives 10 ms at a time. The ear cuts it at the dips of its loudness (lines): stretches of about a syllable.", fontsize=7.2, color=INK, transform=ax.get_xaxis_transform())
    ax = axes[1]
    ax.set_facecolor(SURFACE), clean(ax, "x")
    for t, what, k in heard:
        ax.plot([t / 100.0] * 2, [0.1, 0.9] if what == "pair" else [1.1, 1.9], color=BLUE if what == "pair" else GREEN, linewidth=1.3)
    ax.set_ylim(0, 2), ax.set_yticks([0.5, 1.5]), ax.set_yticklabels(["of two", "of three"], fontsize=6.8, color=INK)
    ax.text(0.0, 1.14, f"Every two and three stretches in a row fire one of 2,048 sound units ({len(heard)} in this sentence), known about 70 ms after the last ends.", fontsize=7.2, color=INK, transform=ax.transAxes)
    ax = axes[2]
    ax.set_facecolor(SURFACE), clean(ax)
    shown = [h for h in heard][:: max(1, len(heard) // 9)][:9]
    for i, (t0, what, k) in enumerate(shown):
        tt = T[T >= t0 / 100.0]
        ax.plot(tt, np.exp(-(tt - t0 / 100.0) / 1.0), color=BLUE if what == "pair" else GREEN, linewidth=1.1, alpha=0.85)
    ax.set_ylim(0, 1.12), ax.set_yticks([]), ax.set_ylabel("trace", fontsize=7.4, color=MUTED)
    ax.set_xlabel("seconds", fontsize=7.6, color=MUTED), ax.set_xlim(0, T[-1])
    ax.text(0.0, 1.1, "Each unit heard leaves a trace that fades over about a second (nine shown); while it lasts, its synapses onto what the eye holds grow.", fontsize=7.2, color=INK, transform=ax.transAxes)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.93, bottom=0.105)
    save(fig, "fig_clock")

# -------------------------------------------------------------------------------------------------------- fig_organism
if not ONLY or "organism" in ONLY:
    KINDS = {"born": (BLUE, "born with the life; learns by a local rule as the stream comes"),
             "before": ("#7a5cc4", "local rule, but developed before the life"),
             "batch": (ORANGE, "computed in a batch beforehand (a stand-in)"),
             "fixed": (GREY, "not learned (a reflex, or the body)")}
    W_, H_ = 17.0, 12.5
    BOXES = {
        "Cochlea": (2, 37, "fixed", ["64 bands,", "a row every 10 ms"]),
        "Adaptation": (22, 37, "born", ["each band against", "the last half second"]),
        "Hearing cortex": (42, 37, "born", ["128 detectors on", "80 ms patches; a", "circuit evens its input"]),
        "Relay": (62, 37, "born", ["interneurons even", "out the code"]),
        "Sound units": (82, 37, "born", ["2,048: runs of 2–3", "syllable-long", "stretches"]),
        "Eye": (2, 6, "fixed", ["a 64-pixel view;", "gaze goes to what", "stands out or moves"]),
        "Visual cortex": (32, 6, "before", ["256 filters, Hebbian,", "no labels"]),
        "Its relay": (57, 6, "batch", ["decorrelates the", "4,096 numbers"]),
        "Hebb link": (104, 21.5, "born", ["a unit's trace ×", "what the eye holds"]),
        "Typed letters": (82, 6, "fixed", ["runs of 3–4 letters,", "no words cut"]),
    }
    fig, ax = plt.subplots(figsize=(7.4, 3.3), facecolor=SURFACE)
    ax.set_xlim(-1, 124), ax.set_ylim(0, 62), ax.axis("off")
    for name, (x, y, kind, lines) in BOXES.items():
        ax.add_patch(FancyBboxPatch((x, y), W_, H_, boxstyle="round,pad=0,rounding_size=1.2", facecolor=KINDS[kind][0] + "2e", edgecolor=KINDS[kind][0], linewidth=1.1))
        ax.text(x + W_ / 2, y + H_ - 2.7, name, ha="center", va="center", fontsize=7.6, fontweight="bold", color=INK)
        for i, line in enumerate(lines):
            ax.text(x + W_ / 2, y + H_ - 5.8 - 2.5 * i, line, ha="center", va="center", fontsize=6.5, color=MUTED)

    def arrow(points, label=None, at=None, **kw):
        xs, ys = zip(*points)
        ax.plot(xs[:-1], ys[:-1], color=INK, linewidth=0.9, solid_capstyle="round")
        ax.annotate("", xy=points[-1], xytext=points[-2], arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=0.9, shrinkA=0, shrinkB=0, mutation_scale=8))
        if label:
            ax.text(*at, label, color=MUTED, **{"ha": "center", "va": "center", "fontsize": 6.3, **kw})

    top, low = 37 + H_ / 2, 6 + H_ / 2
    for a, b in (("Cochlea", "Adaptation"), ("Adaptation", "Hearing cortex"), ("Hearing cortex", "Relay"), ("Relay", "Sound units")):
        arrow([(BOXES[a][0] + W_, top), (BOXES[b][0], top)])
    arrow([(19, low), (32, low)]), arrow([(49, low), (57, low)])
    arrow([(99, top), (112.5, top), (112.5, 34)], "traces of\nabout a second", (106, 46.6))
    arrow([(74, low + 3), (78, low + 3), (78, 27.75), (104, 27.75)], "what the eye holds", (90.5, 29.4))
    arrow([(99, low), (112.5, low), (112.5, 21.5)], "traces of 2 s", (106.5, 10.3))
    ax.text(-0.5, 52.5, "sound", fontsize=7.4, color=INK, fontweight="bold"), ax.text(-0.5, 21.5, "sight", fontsize=7.4, color=INK, fontweight="bold")
    ax.text(82, 21.5, "letters", fontsize=7.4, color=INK, fontweight="bold")
    for i, (colour, words) in enumerate(KINDS.values()):  # the legend, two to a line along the top
        x, y = 2 + 62 * (i % 2), 59.2 - 3.6 * (i // 2)
        ax.add_patch(FancyBboxPatch((x, y - 1.3), 3.2, 2.6, boxstyle="round,pad=0,rounding_size=0.6", facecolor=colour + "2e", edgecolor=colour, linewidth=1.0))
        ax.text(x + 4.4, y, words, ha="left", va="center", fontsize=6.7, color=INK)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    save(fig, "fig_organism")
