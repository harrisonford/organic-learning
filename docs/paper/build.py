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
    ("Figure 1, `fig_organism.pdf`,", "fig_organism.pdf", 100,
     "The learner's parts, and how each comes to be what it is. Blue parts are born with the life and learn by a local rule "
     "as the stream comes; the purple part learns by a local rule but is developed before the life; the orange part is "
     "computed in a batch beforehand; grey parts are not learned. No error is sent back from one part to the part before it."),
    ("Figure 2, `fig_benchmark.pdf`,", "fig_benchmark.pdf", 88,
     "CIFAR-10 seen once: held-out accuracy against labelled pictures seen. Lines: mean of three seeds; bands: their range. "
     "Backpropagation with four replays per new picture is one seed."),
    ("Figure 3, `fig_credit.pdf`,", "fig_credit.pdf", 100,
     "Who earned the win. Held-out pictures named after 50,000 seen once, for one layer of plastic synapses learning by a local "
     "rule (blue) and for backpropagation of the same shape (orange), reading the cortex's code as it comes (left pair) or "
     "decorrelated once more by the relay (right pair). Mean of three seeds; bars show the standard deviation. Dashed: a "
     "streaming linear discriminant on the relayed code."),
    ("Figure 4, `fig_ear.pdf`,", "fig_ear.pdf", 100,
     "An ear born with the life. Top: along one life of spoken captions, each heard once beside its photograph, how often an "
     "unseen photograph's own caption is among the first ten of 1,000, for three speakers; blue, every part of the ear starts "
     "from random numbers and nothing is known of a caption before it is heard; orange, hearing cortex and relay prepared "
     "beforehand and each caption normalised whole. Bottom: one speaker at the end of the life as the preparation is removed "
     "piece by piece; bars are the sampling error of one test. One seed each."),
    ("Figure 5, `fig_voices.pdf`,", "fig_voices.pdf", 100,
     "A second voice. The ear hears one speaker for 2,500 captions, then another for 2,500. Left: the later voice at the end, "
     "as a share of what that voice reaches when heard alone from birth. Middle: the first voice at the end, as a share of "
     "what it was when the voice changed. Right: how far the two voices use the same sound units (1: the same units as often; "
     "0: none shared). Three pairs of speakers, one seed each."),
    ("Figure 6, `fig_clock.pdf`,", "fig_clock.pdf", 100,
     "One sentence through the ear on the 10 ms clock (a real caption; only its loudness is shown)."),
    ("Figure 7, `fig_life.pdf`,", "fig_life.pdf", 88,
     "One life in the staged world (seed 0). Lines: score above blind answering on held-out questions, every 5,000 moments. "
     "The band on top shows the stage the world was in. Below: what the learner babbled with nothing said to it, and what it "
     "answered when asked to name a picture, at four ages (the first short samples in the log, not chosen)."),
    ("Figure 8, `fig_examples.pdf`,", "fig_examples.pdf", 96,
     "What a learner says after 115,000 moments. Top: single pictures; bottom: picture-book pages, where it must look at one "
     "thing. The first six held-out items of each kind, in order, not chosen. Above each picture, the question; below, the answer."),
    ("Figure 9, `fig_ablation.pdf`,", "fig_ablation.pdf", 96,
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
# pandoc writes every table as a longtable, and a longtable does not see the figures floating on its page: a page with
# both ran past its bottom margin (2026-11-23). The tables here are short, so each becomes an ordinary table that stays whole.
tex = open(os.path.join(HERE, "paper.tex"), encoding="utf-8").read()
tex = tex.replace("\\begin{longtable}[]{", "\\begin{center}\\begin{tabular}{").replace("\\endhead\n", "").replace("\\bottomrule\\noalign{}\n\\endlastfoot\n", "")
tex = tex.replace("\\end{longtable}", "\\bottomrule\\noalign{}\n\\end{tabular}\\end{center}")
assert "longtable}" not in tex.split("\\begin{document}")[1] and "\\endlastfoot" not in tex
open(os.path.join(HERE, "paper.tex"), "w", encoding="utf-8").write(tex)
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
