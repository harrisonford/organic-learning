"""Daemon: the adults write short spoken questions and answers for the ear.

Four formats, rotating:
  fact   everyday questions on a theme ("what color is a banana?" -> "yellow")
  chat   something a person says, and what a person would say back
         ("thank you!" -> "you're welcome"); quick, social
  logic  questions that need working out: and / or / not, comparisons,
         sequences, small arithmetic ("is ice cold and hot?" -> "no")
  story  one to three sentences, then a question answered by the story
         ("my dog is called rex. what is my dog called?" -> "rex")

One text model writes a batch; the other (roles swap every batch) gives its own
answer to each and says whether the writer's answer is also fine, as a second
adult listening in would. Only pairs judged fine are used by the world
(`agree`). Pairs from the first version of this daemon (which kept only exact
agreement, and so lost almost every open answer about feelings or jobs) are
judged again the same way, a batch at a time, and re-recorded. Questions already in the world are not written again. Uses at most
SHARE requests a day.

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
MODELS = ["nvidia/nemotron-3.5-lightning:free", "xingchen-agi/xing4.0-29b-a4b:free"]
THEMES = ["colors of everyday things", "animals and the sounds they make", "opposites", "what everyday objects are used for", "days, months and seasons",
          "feelings and what causes them", "food and where it comes from", "jobs and what people in them do", "weather", "parts of the body",
          "places in a town", "simple yes or no facts about the world", "shapes and sizes", "family members", "sports", "vehicles",
          "music and instruments", "the kitchen", "the bathroom", "the bedroom", "school and classrooms", "the farm", "the sea and the beach",
          "the forest", "insects and small creatures", "birds", "fruit", "vegetables", "drinks", "clothes and what you wear when",
          "toys and games", "tools", "the sky, sun, moon and stars", "plants and flowers", "time of day and daily routine", "the human senses",
          "hot and cold things", "materials things are made of", "where animals live", "baby animals", "big and small animals",
          "safety rules", "manners", "holidays and celebrations", "the city and the country", "travel", "the doctor and being sick",
          "the supermarket", "the park and playground", "cooking", "money and shopping", "pets", "letters and words", "computers and phones",
          "the house and its rooms", "water", "light and dark", "sounds and noise", "numbers in everyday life", "nature and seasons"]
LOGIC = ["and / or / not about everyday facts (yes or no)", "which of two things is bigger, faster, heavier or older", "what comes after or before in a sequence (days, months, numbers, letters)",
         "small additions with numbers up to twenty", "small subtractions with numbers up to twenty", "counting legs, wheels or parts of things",
         "odd one out among three words", "true or false statements about categories (is a whale a fish?)", "more or less comparisons of quantities",
         "simple riddles with one-word answers"]
CHAT = ["greetings and goodbyes", "thanks, apologies and polite replies", "small talk about how someone is", "answering a friend's invitation",
        "replies to good and bad news", "conversation at a shop or cafe", "short replies at school", "reacting to a joke or a surprise"]
STORY = ["people and where they go", "who has which object", "colors and owners of things", "what someone ate or drank", "names of pets and people",
         "what happened first and then", "how many things someone has after getting or losing some", "how someone feels and why"]

FORMATS = {
    "fact": ("Write {n} different short questions a child might be asked about {topic}, each with a 1-3 word lowercase answer.", 40),
    "chat": ("Write {n} different short things a person might say to a child in everyday life about {topic}, each with the 1-4 word lowercase reply a "
             "polite person would give. Put what is said in \"q\" and the reply in \"a\".", 40),
    "logic": ("Write {n} different short questions for a child that need a little thinking: {topic}. Each answer is 1-3 lowercase words "
              "(numbers as words).", 40),
    "story": ("Write {n} different tiny stories for a child about {topic}: one to three short sentences, followed by a question about the story, "
              "all in \"q\" (like \"anna has a red ball. what color is the ball?\"). The answer is 1-3 lowercase words found from the story.", 20),
}
TOPICS = {"fact": THEMES, "chat": CHAT, "logic": LOGIC, "story": STORY}
ROTATION = ["fact", "fact", "chat", "logic", "story"]
CHECK = ("A child was asked these questions. For each one, give your own short answer (1-4 lowercase words) and say whether the proposed answer "
         "is also correct and natural. Reply only with a JSON list in the same order: [{{\"mine\": \"...\", \"ok\": true}}, ...]\n{items}")

out = sys.argv[1] if len(sys.argv) > 1 else "data/world/text_qa.jsonl"
adults = Adults()
rng = random.Random()
seen, by_format, old = set(), {f: [] for f in FORMATS}, {}
if os.path.exists(out):
    for line in open(out):
        d = json.loads(line)
        if d["q"] not in seen:
            by_format[d.get("format", "fact")].append(d["q"])
        seen.add(d["q"])
        if "format" not in d or d.get("rejudged"):
            old.setdefault(d["q"], []).append(d)
backlog = [v[0] for v in old.values() if not any(x["agree"] or x.get("rejudged") for x in v)]


def text_used():
    return adults._state().get("text_count", 0)


def log(msg):
    print(time.strftime("[%F %T] ") + msg, flush=True)


def rejudge(batch):
    items = "\n".join(json.dumps({"question": d["q"], "proposed": d["a"]}) for d in batch)
    judged = parse_json(adults.ask(MODELS[0], CHECK.format(items=items), purpose="text", timeout=600)["text"]) or []
    judged = judged if isinstance(judged, list) else []
    with open(out, "a") as f:
        for d, j in zip(batch, judged):
            j = j if isinstance(j, dict) else {}
            f.write(json.dumps({**d, "a_check": str(j.get("mine", "")).strip().lower(), "agree": j.get("ok") is True, "rejudged": True}) + "\n")
    return sum(isinstance(j, dict) and j.get("ok") is True for j in judged), len(judged)


k = len(seen)
while True:
    if text_used() + 2 > SHARE or adults._state()["count"] + 2 > PER_DAY:
        log("daily share reached; sleeping")
        while text_used() + 2 > SHARE or adults._state()["count"] + 2 > PER_DAY:
            time.sleep(600)
    if backlog and k % 3 == 0:
        batch, backlog = backlog[:40], backlog[40:]
        k += 1
        try:
            good, n_j = rejudge(batch)
            log(f"rejudged {n_j} old pairs: {good} fine, {len(backlog)} left")
        except Exception as e:  # noqa: BLE001
            log(f"rejudge failed: {e}")
            time.sleep(60)
        continue
    fmt = ROTATION[k % len(ROTATION)]
    topic = rng.choice(TOPICS[fmt])
    writer, checker = MODELS[k % 2], MODELS[(k + 1) % 2]
    k += 1
    template, n = FORMATS[fmt]
    avoid = rng.sample(by_format[fmt], min(15, len(by_format[fmt])))
    prompt = template.format(n=n, topic=topic)
    if avoid:
        prompt += " Do not repeat any of these: " + json.dumps(avoid)
    prompt += f' (batch {k}) Reply only with JSON: [{{"q": "...", "a": "..."}}]'
    try:
        w = adults.ask(writer, prompt, purpose="text", temperature=0.9, timeout=600)
        pairs = [p for p in (parse_json(w["text"]) or []) if isinstance(p, dict) and p.get("q") and p.get("a")]
        pairs = [p for p in pairs if str(p["q"]).strip().lower() not in seen]
        if not pairs:
            log(f"{fmt} / {topic}: nothing new")
            continue
        items = "\n".join(json.dumps({"question": p["q"], "proposed": p["a"]}) for p in pairs)
        c = adults.ask(checker, CHECK.format(items=items), purpose="text", timeout=600)
        judged = parse_json(c["text"]) or []
    except Exception as e:  # noqa: BLE001
        log(f"failed: {e}")
        time.sleep(60)
        continue
    judged = judged if isinstance(judged, list) else []
    kept = 0
    with open(out, "a") as f:
        for p, j in zip(pairs, judged):
            j = j if isinstance(j, dict) else {}
            q, a1 = str(p["q"]).strip().lower(), str(p["a"]).strip().lower()
            ok = j.get("ok") is True
            seen.add(q)
            by_format[fmt].append(q)
            kept += ok
            f.write(json.dumps({"q": q, "a": a1, "a_check": str(j.get("mine", "")).strip().lower(), "agree": ok, "theme": topic, "format": fmt,
                                "writer": writer}) + "\n")
    log(f"{fmt} / {topic}: {len(pairs)} new, {kept} judged fine, {text_used()} text requests today")
