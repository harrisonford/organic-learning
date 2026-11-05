"""Facts with exact answers from Wikidata, for the school stages (no model requests).

Each relation is one SPARQL query to query.wikidata.org (a few queries, run once, with a
polite user agent). Only well-known subjects are kept (many Wikipedia language
editions link them: "sitelinks"), so a child could plausibly have heard of them.
Answers keep the English label and aliases, so local grading accepts any of them.

    python experiments/world/wikidata_facts.py      -> data/world/wikidata_facts.jsonl

organic/curriculum.py turns each fact into questions of its stage (docs/WORLD.md).
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from organic.world import DATA, WORLD_DIR  # noqa: E402

UA = "organic-learning/0.1 (research; https://github.com/harrisonford/organic-learning)"
URL = "https://query.wikidata.org/sparql"
LABELS = 'SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }'

# relation: (query, minimum sitelinks). ?s subject, ?o answer item (or ?ans literal), ?n sitelinks
QUERIES = {
    "capital": ("""SELECT ?s ?sLabel ?o ?oLabel ?n WHERE { ?s wdt:P31 wd:Q3624078; wdt:P36 ?o; wikibase:sitelinks ?n.
                   FILTER NOT EXISTS { ?s wdt:P576 [] } %s }""", 0),
    "continent": ("""SELECT ?s ?sLabel ?o ?oLabel ?n WHERE { ?s wdt:P31 wd:Q3624078; wdt:P30 ?o; wikibase:sitelinks ?n.
                     FILTER NOT EXISTS { ?s wdt:P576 [] } %s }""", 0),
    "language": ("""SELECT ?s ?sLabel ?o ?oLabel ?n WHERE { ?s wdt:P31 wd:Q3624078; wdt:P37 ?o; wikibase:sitelinks ?n.
                    FILTER NOT EXISTS { ?s wdt:P576 [] } %s }""", 0),
    "city_country": ("""SELECT ?s ?sLabel ?o ?oLabel ?n WHERE { ?s wdt:P31/wdt:P279* wd:Q515; wdt:P17 ?o; wikibase:sitelinks ?n.
                        FILTER(?n > 120) %s }""", 120),
    "inventor": ("""SELECT ?s ?sLabel ?sDescription ?o ?oLabel ?n WHERE { ?s wdt:P61 ?o; wikibase:sitelinks ?n. FILTER(?n > 60)
                    ?o wdt:P31 wd:Q5. %s }""", 60),
    "painter": ("""SELECT ?s ?sLabel ?o ?oLabel ?n WHERE { ?s wdt:P31 wd:Q3305213; wdt:P170 ?o; wikibase:sitelinks ?n. FILTER(?n > 40) %s }""", 40),
    "author": ("""SELECT ?s ?sLabel ?o ?oLabel ?n WHERE { ?s wdt:P31 wd:Q7725634; wdt:P50 ?o; wikibase:sitelinks ?n. FILTER(?n > 60) %s }""", 60),
    "mountain_continent": ("""SELECT ?s ?sLabel ?o ?oLabel ?n WHERE { ?s wdt:P31 wd:Q8502; wdt:P30 ?o; wikibase:sitelinks ?n. FILTER(?n > 50) %s }""", 50),
    "river_continent": ("""SELECT ?s ?sLabel ?o ?oLabel ?n WHERE { ?s wdt:P31 wd:Q4022; wdt:P30 ?o; wikibase:sitelinks ?n. FILTER(?n > 60) %s }""", 60),
}
# made things (an invention's description must name one): "who invented australia?" is not a question
MADE = ("device", "machine", "instrument", "tool", "vehicle", "engine", "apparatus", "game", "toy", "system", "technique", "method",
        "material", "food", "weapon", "telescope", "computer", "software", "language", "vaccine", "drug", "aircraft", "process", "invention",
        "sport", "musical", "camera", "lamp", "battery", "motor", "type of", "kind of", "equipment", "appliance", "medicine", "fabric")
# the class taxa, as a child says them
CLASS_WORD = {"Q7377": "mammal", "Q5113": "bird", "Q10811": "reptile", "Q10908": "amphibian", "Q1390": "insect",
              "Q127282": "fish", "Q25371": "fish", "Q1358": "spider"}


def sparql(q):
    req = urllib.request.Request(URL + "?" + urllib.parse.urlencode({"query": q, "format": "json"}), headers={"User-Agent": UA})
    for attempt in range(4):
        try:
            return json.load(urllib.request.urlopen(req, timeout=180))["results"]["bindings"]
        except Exception as e:  # noqa: BLE001
            print("  retry:", str(e)[:100], flush=True)
            time.sleep(30 * (attempt + 1))
    return []


QLEVER = "https://qlever.dev/api/wikidata"  # the same Wikidata on a faster engine: the taxonomy walk times out on query.wikidata.org
ANIMALS = """PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?s ?label ?tn ?n ?o WHERE { VALUES ?o { %s } ?s wdt:P105 wd:Q7432 . ?s wdt:P171+ ?o . ?s wdt:P225 ?tn .
  ?s wikibase:sitelinks ?n . FILTER(?n > 60) ?s rdfs:label ?label . FILTER(LANG(?label) = "en") }"""
# when a species sits under several classes (birds are reptiles cladistically), the child's word is the most specific
PRECEDENCE = ["bird", "mammal", "fish", "amphibian", "insect", "spider", "reptile"]


def animal_classes():
    """Well-known animal species and their class, as a child says it. Common names only (not the
    scientific name), and only names that occur in Simple English Wikipedia (names a child could hear)."""
    q = ANIMALS % " ".join("wd:" + c for c in CLASS_WORD)
    req = urllib.request.Request(QLEVER, data=urllib.parse.urlencode({"query": q}).encode(),
                                 headers={"User-Agent": UA, "Accept": "application/sparql-results+json"})
    got = json.load(urllib.request.urlopen(req, timeout=300))["results"]["bindings"]
    heard = set()
    path = os.path.join(DATA, "simplewiki", "sentences.txt")
    best = {}
    for b in got:
        sid, label, tn = b["s"]["value"].rsplit("/", 1)[1], b["label"]["value"], b["tn"]["value"]
        if label == tn or label == sid:
            continue
        word = CLASS_WORD[b["o"]["value"].rsplit("/", 1)[1]]
        if sid not in best or PRECEDENCE.index(word) < PRECEDENCE.index(best[sid][1]):
            best[sid] = (label, word, int(b["n"]["value"]))
    names = {v[0].lower() for v in best.values()}
    for line in open(path) if os.path.exists(path) else []:
        for nm in names:
            if nm in line:
                heard.add(nm)
        if len(heard) == len(names):
            break
    rows = [{"rel": "animal_class", "s": label, "sid": sid, "o": word, "oid": word, "sitelinks": n}
            for sid, (label, word, n) in best.items() if label.lower() in heard]
    print(f"animal classes: {len(best)} species with a common name, {len(rows)} heard in simple wikipedia", flush=True)
    return rows


def aliases(qids):
    """English aliases for answer items (so "usa", "u.s." etc. are accepted)."""
    out = {}
    qids = sorted(qids)
    for i in range(0, len(qids), 200):
        vals = " ".join("wd:" + q for q in qids[i : i + 200])
        for b in sparql(f'SELECT ?o ?alt WHERE {{ VALUES ?o {{ {vals} }} ?o skos:altLabel ?alt. FILTER(LANG(?alt) = "en") }}'):
            out.setdefault(b["o"]["value"].rsplit("/", 1)[1], []).append(b["alt"]["value"])
        time.sleep(2)
    return out


def main():
    path = os.path.join(WORLD_DIR, "wikidata_facts.jsonl")
    rows = []
    for rel, (q, _) in QUERIES.items():
        got = sparql(q % LABELS)
        n0 = len(rows)
        for b in got:
            s, o = b["sLabel"]["value"], b["oLabel"]["value"]
            sid, oid = b["s"]["value"].rsplit("/", 1)[1], b["o"]["value"].rsplit("/", 1)[1]
            if s == sid or o == oid:  # no English label
                continue
            if rel == "inventor" and not any(w in b.get("sDescription", {}).get("value", "").lower() for w in MADE):
                continue
            rows.append({"rel": rel, "s": s, "sid": sid, "o": o, "oid": oid, "sitelinks": int(b["n"]["value"])})
        print(f"{rel}: {len(rows) - n0} facts", flush=True)
        time.sleep(5)
    rows += animal_classes()
    print("animal_class:", sum(r["rel"] == "animal_class" for r in rows), "facts", flush=True)
    alt = aliases({r["oid"] for r in rows if r["rel"] != "animal_class"})
    # one subject can have several right answers (two capitals, several languages): merge them
    merged = {}
    for r in rows:
        key = (r["rel"], r["sid"])
        m = merged.setdefault(key, {"rel": r["rel"], "s": r["s"], "sitelinks": r["sitelinks"], "answers": []})
        # aliases only when they share a word with the name (drops vandalism and obscure names: "ghahere" for cairo)
        name = set(r["o"].lower().replace("-", " ").split())
        for a in [r["o"]] + [x for x in alt.get(r["oid"], []) if name & set(x.lower().replace("-", " ").split())]:
            if a.lower() not in [x.lower() for x in m["answers"]] and len(a) < 40 and "various" not in a.lower():
                m["answers"].append(a)
    with open(path + ".tmp", "w") as f:
        for m in merged.values():
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    os.replace(path + ".tmp", path)
    print(len(merged), "facts ->", path)


def add_animals():
    """Only the animal classes, appended to an existing facts file (the heavy query alone, retried another time)."""
    path = os.path.join(WORLD_DIR, "wikidata_facts.jsonl")
    rows = [json.loads(line) for line in open(path) if json.loads(line)["rel"] != "animal_class"]
    merged = {}
    for r in animal_classes():
        merged.setdefault(r["sid"], {"rel": "animal_class", "s": r["s"], "sitelinks": r["sitelinks"], "answers": [r["o"]]})
    with open(path + ".tmp", "w") as f:
        for m in rows + list(merged.values()):
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    os.replace(path + ".tmp", path)
    print(len(merged), "animal facts added ->", path)


if __name__ == "__main__":
    add_animals() if "--animals" in sys.argv else main()
