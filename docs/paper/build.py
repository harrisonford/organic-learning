"""DRAFT.md -> paper.tex -> paper.pdf (pandoc and pdflatex, as arXiv compiles). DRAFT.md stays the text that is edited;
paper.tex is made from it and never edited by hand.
    python docs/paper/build.py
What this does to the draft on the way: the note to Kris under the title is left out; the abstract becomes the
abstract; the sections' own numbers are dropped (LaTeX numbers them); the figures are put where the text first
points to them. Marks like [pending] and [check] stay visible: a PDF with them in it is not ready to send.
"""
import os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
AUTHOR = os.environ.get("AUTHOR", "[author line: to be set]")  # the name is set when building, not kept in this public repository until Kris says so
src = open(os.path.join(HERE, "DRAFT.md"), encoding="utf-8").read()
title = re.match(r"# (.*)\n", src).group(1)
body = src[src.index("## Abstract"):]
abstract, body = body[len("## Abstract"):body.index("\n## ")].strip(), body[body.index("\n## "):]
body = re.sub(r"^(#+) \d+\. ", r"\1 ", body, flags=re.M)  # "## 3. Experiment 1" -> "## Experiment 1"
body = re.sub(r"^#(#+) ", r"\1 ", body, flags=re.M)  # one level up: the draft's sections are "##"
FIGURES = [  # (the words that first point to it, the file, its width, the caption); numbered in this order
    ("Figure 1, `fig_architecture.pdf`,", "fig_architecture.pdf", 100,
     "The learner's parts, and how each comes to be what it is. Blue parts learn online by a local rule; the two orange "
     "parts are computed in batch before learning begins; the green part learns from reward; the grey part is a fixed reflex. "
     "No error is sent back from one part to the part before it."),
    ("Figure 2, `fig_life.pdf`,", "fig_life.pdf", 92,
     "One life (seed 0). Lines: score above blind answering on held-out questions, every 5,000 moments. The band on top shows "
     "the stage the world was in. Below: what the learner babbled with nothing said to it, and what it answered when asked "
     "to name a picture, at four ages (the first short samples in the log, not chosen)."),
    ("Figure 3, `fig_examples.pdf`,", "fig_examples.pdf", 100,
     "What a learner says after 115,000 moments. Top: single pictures; bottom: picture-book pages, where it must look at one "
     "thing. These are the first six held-out items of each kind, in order, not chosen. Above each picture, the question; "
     "below, the answer."),
    ("Figure 4, `fig_benchmark.pdf`,", "fig_benchmark.pdf", 92,
     "CIFAR-10 seen once: held-out accuracy against labelled pictures seen. Lines: mean of three seeds; bands: their range. "
     "Backpropagation with four replays per new picture is one seed."),
    ("Figure 5, `fig_ablation.pdf`,", "fig_ablation.pdf", 100,
     "A life with one part removed, at 50,000 moments: score above blind answering. Bars: mean of three seeds; "
     "dots: the seeds; dashed line: the whole learner's mean."),
]
for mark, file, width, caption in FIGURES:
    assert body.count(mark) == 1, mark
    end = body.index("\n\n", body.index(mark))  # the end of the paragraph that points to it
    if body[end + 2] == "|":  # a table follows: the figure goes after the table
        end = body.index("\n\n", end + 2)
    body = body[:end] + f"\n\n![{caption}]({file}){{width={width}%}}\n" + body[end:]
    body = body.replace(mark, mark.split(",")[0])
head = r"""
\usepackage{microtype}
\DeclareUnicodeCharacter{2212}{\ensuremath{-}}
\DeclareUnicodeCharacter{2192}{\ensuremath{\rightarrow}}
\DeclareUnicodeCharacter{2265}{\ensuremath{\geq}}
\DeclareUnicodeCharacter{2264}{\ensuremath{\leq}}
\DeclareUnicodeCharacter{2248}{\ensuremath{\approx}}
\setlength{\emergencystretch}{2em}
"""
meta = f"---\ntitle: \"{title}\"\nauthor: \"{AUTHOR}\"\ndate: \"Draft of \\\\today. Not for circulation.\"\nabstract: |\n" + "".join("  " + l + "\n" for l in abstract.splitlines()) + "---\n"
open(os.path.join(HERE, "paper.md"), "w", encoding="utf-8").write(meta + body)
open(os.path.join(HERE, "head.tex"), "w").write(head)
run = lambda *a: subprocess.run(a, cwd=HERE, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)  # noqa: E731
run("pandoc", "paper.md", "-s", "-N", "-o", "paper.tex", "-H", "head.tex", "-V", "documentclass=article", "-V", "fontsize=10pt",
    "-V", "geometry:margin=2.6cm", "-V", "colorlinks=true", "-V", "linkcolor=black", "-V", "urlcolor=blue")
os.remove(os.path.join(HERE, "paper.md"))
os.remove(os.path.join(HERE, "head.tex"))
for _ in range(2):
    out = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "paper.tex"], cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    if out.returncode:
        print(out.stdout[-3000:])
        sys.exit(1)
for ext in ("aux", "log", "out"):
    p = os.path.join(HERE, "paper." + ext)
    if os.path.exists(p):
        os.remove(p)
warn = [l for l in out.stdout.splitlines() if "Overfull" in l or "Missing character" in l or "Warning" in l]
print("docs/paper/paper.tex, paper.pdf;", re.search(r"\((\d+ pages?)", out.stdout).group(1), f"; {len(warn)} warnings")
for l in warn[:15]:
    print("  ", l)
