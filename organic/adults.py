"""The adults: language and vision models that act as people in the brain's world.

They write questions and short answers about photos (like a parent naming
things), check each other's answers, grade free-form replies, and serve as a
reference to compare against. They never touch the brain's internals.

Limits (half of the account's allowance, enforced here and persisted on disk):
  - at most 10 requests per minute and 500 per day across all free models;
  - Jev (a router that may pick paid models) has a hard spend cap: before each
    Jev call the key's real usage is checked against the usage recorded when the
    Jev budget was opened.
Every response is cached by a hash of the request, so nothing is asked twice.
"""

import base64
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

KEY_FILE = os.path.expanduser("~/.config/openrouter/api_key")
STATE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "adults")
URL = "https://openrouter.ai/api/v1/chat/completions"
PER_MINUTE, PER_DAY = 10, 500
JEV, JEV_CAP_USD = "typesafe/jev-router", 1.50


class Adults:
    def __init__(self):
        os.makedirs(STATE_DIR, exist_ok=True)
        self.key = open(KEY_FILE).read().strip()
        self.state_file = os.path.join(STATE_DIR, "usage.json")
        self.cache_file = os.path.join(STATE_DIR, "cache.jsonl")
        self.cache = {}
        if os.path.exists(self.cache_file):
            for line in open(self.cache_file):
                d = json.loads(line)
                self.cache[d["key"]] = d["reply"]

    # ------------------------------------------------------------- limits

    def _state(self):
        today = time.strftime("%Y-%m-%d")
        s = json.load(open(self.state_file)) if os.path.exists(self.state_file) else {}
        if s.get("day") != today:
            s = {"day": today, "count": 0, "jev_baseline": s.get("jev_baseline"), "recent": []}
        return s

    def _save(self, s):
        json.dump(s, open(self.state_file, "w"))

    def _wait_turn(self, purpose=None):
        while True:
            s = self._state()
            if s["count"] >= PER_DAY:
                raise RuntimeError("daily request budget used up")
            now = time.time()
            s["recent"] = [t for t in s["recent"] if now - t < 60]
            if len(s["recent"]) < PER_MINUTE:
                s["recent"].append(now)
                s["count"] += 1
                if purpose:
                    s[f"{purpose}_count"] = s.get(f"{purpose}_count", 0) + 1
                self._save(s)
                return
            time.sleep(60 - (now - min(s["recent"])) + 0.5)

    def key_usage(self):
        req = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {self.key}"})
        return float(json.load(urllib.request.urlopen(req, timeout=30))["data"]["usage"])

    def _jev_allowed(self):
        s = self._state()
        if s.get("jev_baseline") is None:
            s["jev_baseline"] = self.key_usage()
            self._save(s)
        return self.key_usage() - s["jev_baseline"] < JEV_CAP_USD

    # ---------------------------------------------------------------- calls

    def ask(self, model, content, max_tokens=4000, temperature=0.2, purpose=None, timeout=300):
        """content: a string or an OpenAI-style list of text/image parts."""
        body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, "temperature": temperature}
        k = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        if k in self.cache:
            return self.cache[k]
        if model == JEV and not self._jev_allowed():
            raise RuntimeError("Jev budget cap reached")
        for attempt in range(6):
            self._wait_turn(purpose)
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            t0 = time.time()
            try:
                d = json.load(urllib.request.urlopen(req, timeout=timeout))
                if not d.get("choices"):  # an upstream error reported inside a 200 reply
                    self.last_error = str(d.get("error", d))[:200]
                    time.sleep(20 * (attempt + 1))
                    continue
                text = d["choices"][0]["message"].get("content")
                if not text:  # a reasoning model ran out of room before answering
                    body["max_tokens"] = int(body["max_tokens"] * 2)
                    continue
                reply = {"text": text, "seconds": round(time.time() - t0, 2), "model": d.get("model", model)}
                self.cache[k] = reply
                with open(self.cache_file, "a") as f:
                    f.write(json.dumps({"key": k, "reply": reply}) + "\n")
                return reply
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep(20 * (attempt + 1))
                    continue
                raise
            except (urllib.error.URLError, TimeoutError):
                time.sleep(20 * (attempt + 1))
        raise RuntimeError("no reply after retries: " + getattr(self, "last_error", ""))


def image_part(img, size=64, show=256):
    """What the brain's eye could see (size x size), enlarged for the model to look at."""
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)) if isinstance(img, np.ndarray) else img
    im = im.convert("RGB").resize((size, size), Image.BILINEAR).resize((show, show), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}


def parse_json(text):
    """The first JSON array or object in a model's reply."""
    for open_c, close_c in (("[", "]"), ("{", "}")):
        a, b = text.find(open_c), text.rfind(close_c)
        if a != -1 and b > a:
            try:
                return json.loads(text[a : b + 1])
            except json.JSONDecodeError:
                continue
    return None
