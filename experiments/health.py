"""Health of everything that should be running (docs/REVIEW_2026-11-10.md, the ruler).

The data daemons were down for 3.5 days in November (2,520 restarts) and nobody noticed.
This check runs every hour (organic-health.timer) and before every status report:

  services   each expected systemd user unit is active and is not restarting in a loop
  data       each daemon's data file grew in the last DATA_HOURS hours (a daemon can be "active" and make nothing)
  life       every life listed in out/monitor/lives.json ({"words_s0": "organic-words-s0", ...}): if its unit runs, its
             log moved in the last LIFE_HOURS hours; if its unit stopped, its last record has a verdict (not killed)
  machine    memory and disk left

Writes out/monitor/health.json (latest) and appends every alert to out/monitor/alerts.log.
Exit code 1 if anything is wrong (so `systemctl --user status organic-health` shows it failed).
    python experiments/health.py
"""
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SERVICES = ["organic-text", "organic-photo", "organic-curriculum", "organic-judge", "kid-a"]
DATA = {"organic-text": "data/world/text_qa.jsonl", "organic-photo": "data/world/coco_qa.jsonl", "organic-curriculum": "data/world/curriculum.jsonl"}
DATA_HOURS, LIFE_HOURS = 30, 3  # the daemons share a daily request budget: a quiet night is normal, a quiet day is not
now = time.time()
alerts, facts = [], {}


def show(unit, *props):
    out = subprocess.run(["systemctl", "--user", "show", unit] + [f"-p{p}" for p in props], capture_output=True, text=True).stdout
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


for u in SERVICES:
    d = show(u, "ActiveState", "NRestarts", "ExecMainStartTimestampMonotonic")
    facts[u] = {"state": d.get("ActiveState"), "restarts": int(d.get("NRestarts") or 0)}
    if d.get("ActiveState") != "active":
        alerts.append(f"{u} is {d.get('ActiveState')}")
    if facts[u]["restarts"] > 5:
        alerts.append(f"{u} has restarted {facts[u]['restarts']} times (a crash loop?)")

for u, path in DATA.items():
    p = os.path.join(ROOT, path)
    hours = (now - os.path.getmtime(p)) / 3600 if os.path.exists(p) else float("inf")
    facts[u]["data_hours_ago"] = round(hours, 1)
    if hours > DATA_HOURS:
        alerts.append(f"{u} wrote nothing to {path} for {hours:.0f} hours")

# lives: out/monitor/lives.json maps a life's log (out/<name>.jsonl) to the systemd unit that runs it
lives_file = os.path.join(ROOT, "out", "monitor", "lives.json")
lives = json.load(open(lives_file)) if os.path.exists(lives_file) else {}
for name, unit in lives.items():
    state = show(unit, "ActiveState").get("ActiveState")
    log = os.path.join(ROOT, "out", name + ".jsonl")
    last = open(log).readlines()[-1:] if os.path.exists(log) else []
    rec = json.loads(last[0]) if last else {}
    hours = (now - os.path.getmtime(log)) / 3600 if os.path.exists(log) else None
    facts[name] = {"unit": unit, "state": state, "age": rec.get("age"), "stage": rec.get("stage"), "verdict": rec.get("verdict"),
                   "hours_since_checkpoint": round(hours, 1) if hours is not None else None}
    if state == "active" and hours is not None and hours > LIFE_HOURS:
        alerts.append(f"life {name} is running but has not written a checkpoint for {hours:.1f} hours")
    if state != "active" and not rec.get("verdict"):
        alerts.append(f"life {name} stopped ({state}) without a verdict (killed? out of memory?)")

mem = {k: int(v.split()[0]) for k, v in (line.split(":") for line in open("/proc/meminfo")) if k in ("MemAvailable", "SwapFree")}
disk = shutil.disk_usage(ROOT)
facts["machine"] = {"memory_available_gb": round(mem["MemAvailable"] / 1e6, 1), "disk_free_gb": round(disk.free / 1e9, 1)}
if mem["MemAvailable"] < 1e6:
    alerts.append(f"memory available {mem['MemAvailable'] / 1e6:.1f} GB")
if disk.free < 10e9:
    alerts.append(f"disk free {disk.free / 1e9:.1f} GB")

os.makedirs(os.path.join(ROOT, "out", "monitor"), exist_ok=True)
stamp = time.strftime("%Y-%m-%d %H:%M")
json.dump({"when": stamp, "alerts": alerts, "facts": facts}, open(os.path.join(ROOT, "out", "monitor", "health.json"), "w"), indent=1)
if alerts:
    with open(os.path.join(ROOT, "out", "monitor", "alerts.log"), "a") as f:
        for a in alerts:
            f.write(f"[{stamp}] {a}\n")
print(f"[{stamp}] " + ("ALERT: " + "; ".join(alerts) if alerts else "all healthy") + " | " + json.dumps(facts))
sys.exit(1 if alerts else 0)
