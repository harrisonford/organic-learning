"""The parent in leisure talk (docs/LEISURE.md): a local open-weight chat model answers what the child just said.

Not everything a child hears is a lesson. Most talk between a parent and a toddler expects nothing: the parent says
something, the child says whatever it says, and the parent answers *that* (takes the word up, adds a little, or just
answers kindly). Those replies, contingent on the child's own sounds, are what infants learn fastest from (Goldstein
& Schwade 2008; recasts and expansions: Brown & Bellugi 1964). The curriculum's adults are hosted models with a
small daily budget and the judge is a decision model that cannot write, so this parent is a third, local one
(Llama 3.2 3B on the GPU beside the judge, systemd unit organic-parent, about 0.1 s per reply, no cost).

This is the environment, not the brain. If the parent does not answer in time, the moment is skipped.
"""
import json
import os
import urllib.request

URL = os.environ.get("PARENT_URL", "http://127.0.0.1:8791/v1/chat/completions")
ANSWER = ("You are a warm parent chatting with your toddler ({age}). The child just answered you. Reply to what the child said "
          "the way a parent does: one short, simple sentence, at most 8 words, lower case. If the child said a real word, take "
          "it up and add a little. If it was babble or nothing, answer kindly in simple words. Never ask the child to repeat a "
          "word. No quotes.")
OPEN = ("You are a parent with your toddler ({age}) at home. Say one short thing to your child, the way parents chat through the "
        "day: about {topic}. It is not a quiz: do not ask the child to name, count or repeat anything. At most 8 words, lower "
        "case, no quotes. Something different from: {others}")
AGES = {"BB": "8 months old", "B": "12 months old", "W1": "18 months old", "W2": "2 years old", "P": "4 years old", "K": "5 years old"}
TOPICS = ["what you are doing together right now", "food and eating", "getting dressed", "going outside", "bath time", "bedtime", "a toy",
          "how the child feels", "something funny that just happened", "the weather", "a pet or an animal you can see", "someone in the family",
          "tidying up", "a song or a game", "waking up", "a hug or a cuddle", "something the child just did well", "a noise you both heard"]


class Parent:
    def __init__(self, timeout=8):
        self.timeout, self.calls, self.failed = timeout, 0, 0

    def _ask(self, system, user, temperature):
        body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "max_tokens": 24, "temperature": temperature}
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            text = json.load(urllib.request.urlopen(req, timeout=self.timeout))["choices"][0]["message"]["content"]
            self.calls += 1
            return " ".join(text.lower().replace('"', "").split())[:80] or None
        except Exception:  # the parent is busy or away: nothing is said
            self.failed += 1
            return None

    def open(self, stage, rng, others=(), topic=None):
        """Something a parent says in passing (None if the parent is away); `topic`: what the two are looking at."""
        return self._ask(OPEN.format(age=AGES.get(stage, "4 years old"), topic=topic or rng.choice(TOPICS), others="; ".join(others) or "-"), "say it now", 1.0)

    def answer(self, stage, parent_said, child_said):
        """What the parent says back to what the child said."""
        return self._ask(ANSWER.format(age=AGES.get(stage, "4 years old")), f"parent: {parent_said}\nchild: {child_said or '(nothing)'}", 0.8)
