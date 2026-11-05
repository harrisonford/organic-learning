"""Overheard speech: sentences from Simple English Wikipedia (no model requests).

Infants hear millions of words that are not addressed to them, and that is where they
learn the sound statistics that split speech into words (docs/EAR_SEGMENTATION.md). Our
ear heard only the curriculum's questions, a narrow diet that fused frequent frames
into units. Simple English Wikipedia is written in short, plain sentences. Each article's
prose is stripped of markup, cut into sentences, lowercased (the ear hears no capitals:
curriculum speech is lowercase) and kept if it is a plain sentence of 4-25 words.

    python experiments/world/simplewiki_speech.py   -> ~/datasets/simplewiki/sentences.txt

The license is CC BY-SA (Wikipedia contributors); used for research, credited in docs/WORLD.md.
"""

import bz2
import os
import re
import sys
import xml.etree.ElementTree as ET

import mwparserfromhell

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from organic.world import DATA  # noqa: E402

DIR = os.path.join(DATA, "simplewiki")
DUMP = os.path.join(DIR, "simplewiki-latest-pages-articles.xml.bz2")
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
PLAIN = re.compile(r"^[a-z0-9][a-z0-9 ,.'\-?!:;()]*[.!?]$")


def sentences(wikitext):
    code = mwparserfromhell.parse(wikitext)
    for t in code.filter_templates(recursive=False):  # infoboxes, citations
        try:
            code.remove(t)
        except ValueError:
            pass
    text = code.strip_code(normalize=True, collapse=True)
    for para in text.split("\n"):
        para = para.strip()
        if len(para) < 30 or para.startswith(("Category:", "File:", "Image:", "|", "*", "#")):
            continue
        for s in SENT.split(para):
            s = re.sub(r"\s+", " ", s).strip().lower()
            if 4 <= len(s.split()) <= 25 and PLAIN.match(s):
                yield s


def main():
    out = os.path.join(DIR, "sentences.txt")
    n_pages = n_sent = 0
    with open(out + ".tmp", "w") as f:
        for _, el in ET.iterparse(bz2.open(DUMP), events=("end",)):
            if not el.tag.endswith("}page"):
                continue
            ns = el.find("{*}ns")
            text = el.find("{*}revision/{*}text")
            if ns is not None and ns.text == "0" and text is not None and text.text and not text.text.lower().startswith("#redirect"):
                for s in sentences(text.text):
                    f.write(s + "\n")
                    n_sent += 1
                n_pages += 1
                if n_pages % 20000 == 0:
                    print(n_pages, "articles,", n_sent, "sentences", flush=True)
            el.clear()
    os.replace(out + ".tmp", out)
    print(n_pages, "articles,", n_sent, "sentences ->", out)


if __name__ == "__main__":
    main()
