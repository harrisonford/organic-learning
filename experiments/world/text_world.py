"""Daemon: the adults write spoken questions, chat and little stories for the ear.

The aim is a broad, varied childhood of language, slowly, within budget. Each
batch picks a format, a knowledge domain or situation, and a person asking (a
grandmother, a classmate, a shopkeeper...), so the same model does not keep
producing the same register. All four free models take turns writing; a
different one listens in as the second adult: it gives its own answer and says
whether the writer's answer is also correct and natural. Only pairs judged fine
are used by the world (`agree`). Questions already in the world are not written
again; a batch whose judging failed is kept and judged later, so a writer's
request is never wasted.

Formats (-> kind in the world, see organic/world.py FORMAT_KIND):
  fact      knowledge question, 1-3 word answer                    -> talk
  why       why/how something happens, short cause                 -> talk
  define    what is a ..., what kind of thing is it                -> talk
  wordplay  rhymes, opposites, plurals, first sounds               -> talk
  chat      something said in everyday life, the natural reply     -> chat
  dialogue  two or three turns of a conversation, the next reply   -> chat
  logic     and/or/not, comparisons, sequences, small arithmetic   -> reason
  analogy   a is to b as c is to ...                               -> reason
  order     what comes first / next in an everyday procedure       -> reason
  story     one to three sentences then a question about them      -> story
  mind      a tiny scene, then how someone feels / thinks / wants  -> story

    python experiments/world/text_world.py data/world/text_qa.jsonl
"""

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from organic.adults import PER_DAY, Adults, parse_json  # noqa: E402

SHARE = 240  # text share of the 500/day; photos 200, reserve 60
MODELS = ["nvidia/nemotron-3.5-lightning:free", "xingchen-agi/xing4.0-29b-a4b:free", "nvidia/nemotron-3.5-super-vl:free", "dots-studio/dots-3-note-preview:free"]

DOMAINS = [
    # nature and science
    "animals and the sounds they make", "where animals live", "baby animals", "insects and small creatures", "birds", "sea creatures", "dinosaurs",
    "plants, trees and flowers", "fruit", "vegetables", "the weather", "seasons", "the sky, sun, moon and stars", "planets and space",
    "water, ice and steam", "light, shadows and colours", "sound and music", "magnets, forces and simple machines", "electricity at home",
    "the human body", "the senses", "health and being sick", "food and nutrition", "rocks, soil and volcanoes", "oceans, rivers and mountains",
    "recycling and the environment", "materials things are made of", "hot and cold", "floating and sinking", "growing up and life cycles",
    # people and society
    "family members", "jobs and what people in them do", "the doctor and the dentist", "school and the classroom", "the supermarket",
    "money and shopping", "the post office and letters", "the fire station and police", "the library and books", "restaurants and cafes",
    "holidays and celebrations around the world", "birthdays", "manners", "safety rules", "traffic and crossing the street", "sports and games",
    "toys", "clothes and what to wear when", "houses and rooms", "furniture", "tools", "the kitchen and cooking", "cleaning and chores",
    "pets and caring for them", "farms", "cities, towns and villages", "countries, flags and capitals", "languages people speak",
    "famous buildings and places", "transport and vehicles", "travel and holidays", "jobs of the future and computers", "phones and the internet",
    # time, numbers, language
    "telling the time", "days, weeks and months", "calendars and dates", "counting and numbers", "shapes and patterns", "measuring length and weight",
    "money and prices", "letters and the alphabet", "words that rhyme", "opposites", "plurals", "describing words", "action words",
    # arts and culture
    "music and instruments", "painting and drawing", "dance", "fairy tales and nursery rhymes", "famous stories for children", "movies and cartoons",
    "history: long ago and now", "inventions", "explorers", "kings, queens and castles", "the olympics",
    # inner life
    "feelings and what causes them", "friendship", "sharing and taking turns", "being brave", "being kind", "dreams and sleep", "jokes and funny things",
]
SITUATIONS = ["greetings and goodbyes", "thanks, apologies and polite replies", "small talk about how someone is", "answering an invitation",
              "replies to good and bad news", "at a shop or cafe", "at school", "at the doctor", "on the phone", "at the park", "at dinner",
              "playing a game", "asking for help", "giving directions", "reacting to a joke or a surprise", "a birthday party", "on a bus or train",
              "meeting someone new", "when someone is sad", "when something goes wrong", "bedtime", "in the morning", "a visit to grandparents"]
REASONING = ["and / or / not about everyday facts (yes or no)", "which of two things is bigger, faster, heavier or older", "what comes after or before (days, months, numbers, letters)",
             "small additions up to twenty", "small subtractions up to twenty", "doubling and halving small numbers", "counting legs, wheels or parts of things",
             "odd one out among three words", "true or false about categories (is a whale a fish?)", "more or less comparisons", "simple riddles with one-word answers",
             "if-then about everyday life (if it rains, what do you need?)", "what is missing from a group", "patterns that repeat"]
STORY = ["people and where they go", "who has which object", "colours and owners of things", "what someone ate or drank", "names of pets and people",
         "what happened first and then", "how many things someone has after getting or losing some", "who is taller, older or faster", "where something was put",
         "a small adventure", "a day at school", "a trip to the market", "animals in a farm"]
ASKERS = ["a grandmother", "a curious five-year-old", "a teacher", "a classmate", "a shopkeeper", "a doctor", "an older brother", "a neighbour", "a quiz host",
          "a parent at bedtime", "a friend on the phone", "a park ranger", "a librarian", "a coach", "a cook"]

FORMATS = {
    "fact": ("Write {n} different short questions {asker} might ask a child about {topic}, each with a 1-3 word lowercase answer.", 40, DOMAINS),
    "why": ("Write {n} different short why/how questions {asker} might ask a child about {topic}, each with the short everyday answer a person would give "
            "(1-4 lowercase words, like \"why do we wear coats?\" -> \"to stay warm\").", 40, DOMAINS),
    "define": ("Write {n} different short questions asking what something is or what kind of thing it is, about {topic} (like \"what is a robin?\" -> "
               "\"a bird\"). Answers are 1-3 lowercase words.", 40, DOMAINS),
    "wordplay": ("Write {n} different short word games {asker} might play with a child about {topic}: rhymes, opposites, plurals, first letters, "
                 "words that go together. Each answer is one lowercase word.", 40, DOMAINS),
    "chat": ("Write {n} different short things {asker} might say to a child {topic}, each with the 1-4 word lowercase reply a polite child would "
             "give. Put what is said in \"q\" and the reply in \"a\".", 40, SITUATIONS),
    "dialogue": ("Write {n} different tiny conversations {topic}: two or three short turns, each turn on its own line starting with the speaker "
                 "(\"a: ...\\nb: ...\\na: ...\"), all in \"q\"; \"a\" is the next reply, 1-4 lowercase words, as a person would say it.", 20, SITUATIONS),
    "logic": ("Write {n} different short questions for a child that need a little thinking: {topic}. Each answer is 1-3 lowercase words "
              "(numbers as words).", 40, REASONING),
    "analogy": ("Write {n} different analogies for a child about {topic}, as questions (\"hot is to cold as up is to what?\"), each with a one-word "
                "lowercase answer.", 40, DOMAINS),
    "order": ("Write {n} different questions about the order of steps in everyday life related to {topic} (\"what do you do before you eat, wash your "
              "hands or dry your hands?\"), each with a 1-4 word lowercase answer.", 40, DOMAINS),
    "story": ("Write {n} different tiny stories for a child about {topic}: one to three short sentences, followed by a question about the story, "
              "all in \"q\" (like \"anna has a red ball. what color is the ball?\"). The answer is 1-3 lowercase words found from the story.", 20, STORY),
    "mind": ("Write {n} different tiny scenes about {topic}: one or two sentences about what happens to someone, then a question about how they feel, "
             "what they think or what they want, all in \"q\" (like \"sam dropped his ice cream. how does sam feel?\" -> \"sad\"). "
             "Answers are 1-3 lowercase words.", 20, DOMAINS + STORY),
}
ROTATION = ["fact", "chat", "story", "logic", "why", "dialogue", "define", "mind", "fact", "wordplay", "analogy", "order", "chat"]
CHECK = ("A child was asked these questions. For each one, give your own short answer (1-4 lowercase words) and say whether the proposed answer "
         "is also correct and natural (a different wording of a right answer is fine). Reply only with a JSON list in the same order: "
         "[{{\"mine\": \"...\", \"ok\": true}}, ...]\n{items}")

out = sys.argv[1] if len(sys.argv) > 1 else "data/world/text_qa.jsonl"
adults = Adults()
rng = random.Random()
seen, by_format, old = set(), {f: [] for f in FORMATS}, {}
if os.path.exists(out):
    for line in open(out):
        d = json.loads(line)
        if d["q"] not in seen:
            by_format.setdefault(d.get("format", "fact"), []).append(d["q"])
        seen.add(d["q"])
        if "format" not in d or d.get("rejudged"):
            old.setdefault(d["q"], []).append(d)
backlog = [v[0] for v in old.values() if not any(x["agree"] or x.get("rejudged") for x in v)]
unjudged = []  # (pairs, fmt, topic, asker, writer) whose judging failed


def text_used():
    return adults._state().get("text_count", 0)


def log(msg):
    print(time.strftime("[%F %T] ") + msg, flush=True)


def judge(pairs, writer):
    checker = rng.choice([m for m in MODELS if m != writer])
    items = "\n".join(json.dumps({"question": p["q"], "proposed": p["a"]}) for p in pairs)
    judged = parse_json(adults.ask(checker, CHECK.format(items=items), purpose="text", timeout=600)["text"])
    if not isinstance(judged, list) or not judged:
        raise RuntimeError("judge reply not a list")
    return judged


def record(pairs, judged, fmt, topic, asker, writer):
    kept = 0
    with open(out, "a") as f:
        for p, j in zip(pairs, judged):
            j = j if isinstance(j, dict) else {}
            q, a1 = str(p["q"]).strip().lower(), str(p["a"]).strip().lower()
            ok = j.get("ok") is True
            kept += ok
            f.write(json.dumps({"q": q, "a": a1, "a_check": str(j.get("mine", "")).strip().lower(), "agree": ok, "theme": topic, "format": fmt,
                                "asker": asker, "writer": writer}) + "\n")
    return kept


k = len(seen)
while True:
    if text_used() + 2 > SHARE or adults._state()["count"] + 2 > PER_DAY:
        log("daily share reached; sleeping")
        while text_used() + 2 > SHARE or adults._state()["count"] + 2 > PER_DAY:
            time.sleep(600)
    k += 1
    if unjudged:  # finish what was written before writing more
        pairs, fmt, topic, asker, writer = unjudged[0]
        try:
            kept = record(pairs, judge(pairs, writer), fmt, topic, asker, writer)
            unjudged.pop(0)
            log(f"{fmt} / {topic}: judged later, {kept}/{len(pairs)} fine")
        except Exception as e:  # noqa: BLE001
            log(f"judging failed again: {e}")
            time.sleep(120)
        continue
    if backlog and k % 6 == 0:
        batch, backlog = backlog[:40], backlog[40:]
        try:
            judged = judge([{"q": d["q"], "a": d["a"]} for d in batch], None)
            with open(out, "a") as f:
                for d, j in zip(batch, judged):
                    j = j if isinstance(j, dict) else {}
                    f.write(json.dumps({**d, "a_check": str(j.get("mine", "")).strip().lower(), "agree": j.get("ok") is True, "rejudged": True}) + "\n")
            log(f"rejudged {len(judged)} old pairs, {len(backlog)} left")
        except Exception as e:  # noqa: BLE001
            backlog = batch + backlog
            log(f"rejudge failed: {e}")
            time.sleep(60)
        continue
    fmt = ROTATION[k % len(ROTATION)]
    template, n, topics = FORMATS[fmt]
    topic, asker = rng.choice(topics), rng.choice(ASKERS)
    writer = MODELS[k % len(MODELS)]
    avoid = rng.sample(by_format[fmt], min(15, len(by_format[fmt])))
    prompt = template.format(n=n, topic=topic, asker=asker)
    if avoid:
        prompt += " Do not repeat any of these: " + json.dumps(avoid)
    prompt += f' Make them varied and surprising, not the most obvious ones. (batch {k}) Reply only with JSON: [{{"q": "...", "a": "..."}}]'
    try:
        w = adults.ask(writer, prompt, purpose="text", temperature=0.9, timeout=600)
    except Exception as e:  # noqa: BLE001
        log(f"{fmt} / {writer}: writing failed: {e}")
        time.sleep(60)
        continue
    pairs = [p for p in (parse_json(w["text"]) or []) if isinstance(p, dict) and p.get("q") and p.get("a")]
    pairs = [p for p in pairs if str(p["q"]).strip().lower() not in seen]
    if not pairs:
        log(f"{fmt} / {topic}: nothing new from {writer}")
        continue
    for p in pairs:
        q = str(p["q"]).strip().lower()
        seen.add(q)
        by_format[fmt].append(q)
    try:
        kept = record(pairs, judge(pairs, writer), fmt, topic, asker, writer)
        log(f"{fmt} / {topic} / {asker} / {writer.split('/')[1]}: {len(pairs)} new, {kept} judged fine, {text_used()} text requests today")
    except Exception as e:  # noqa: BLE001
        unjudged.append((pairs, fmt, topic, asker, writer))
        log(f"{fmt} / {topic}: written, judging failed ({e}); kept for later")
        time.sleep(60)
