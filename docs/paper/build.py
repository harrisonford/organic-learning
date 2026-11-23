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
body = re.sub(r"^## ", "# ", body, flags=re.M)
FIGURES = [  # (the sentence that first points to it, the file, the caption)
    ("Figure 1, `fig_benchmark.pdf`, shows the curves", "fig_benchmark.pdf",
     "CIFAR-10 seen once: held-out accuracy against labelled pictures seen. Lines: mean of three seeds; bands: their range. "
     "Backpropagation with four pictures lived again per new one is one seed."),
    ("Figure 2, `fig_ablation.pdf`, shows the seeds", "fig_ablation.pdf",
     "A life with one part taken away, at 50,000 moments: score above what a blind answer gets. Bars: mean of three seeds; "
     "dots: the seeds; dashed line: the whole brain's mean."),
]
for n, (mark, file, caption) in enumerate(FIGURES, 1):
    assert body.count(mark) == 1, mark
    at = body.index(mark)
    end = body.index("\n\n", body.index("\n|", at))  # after the table that follows the pointer
    end = body.index("\n\n", end + 2) if body[end + 2:end + 3] == "|" else end
    body = body[:end] + f"\n\n![{caption}]({file}){{width=92%}}\n" + body[end:]
    body = body.replace(mark, re.sub(r", `[^`]*`,", "", mark).replace(f"Figure {FIGURES.index((mark, file, caption)) + 1}", f"Figure {n}"))
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
