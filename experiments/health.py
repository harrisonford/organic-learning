"""Health of everything that should be running (docs/REVIEW_2026-11-10.md, the ruler).

The data daemons were down for 3.5 days in November (2,520 restarts) and nobody noticed.
This check runs every hour (organic-health.timer) and before every status report:

  services   each expected systemd user unit is active and is not restarting in a loop
  data       each daemon's data file grew in the last DATA_HOURS hours (a daemon can be "active" and make nothing)
  life       if a life is running (organic-life, or any out/*.jsonl written in the last day by a live process),
             its log moved in the last LIFE_HOURS hours
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

life = show("organic-life", "ActiveState").get("ActiveState")
facts["organic-life"] = {"state": life}
if life == "active":
    logs = [os.path.join(ROOT, "out", f) for f in os.listdir(os.path.join(ROOT, "out")) if f.endswith(".jsonl")]
    newest = max(logs, key=os.path.getmtime)
    hours = (now - os.path.getmtime(newest)) / 3600
    facts["organic-life"].update(log=os.path.basename(newest), hours_ago=round(hours, 1))
    if hours > LIFE_HOURS:
        alerts.append(f"the life is active but {os.path.basename(newest)} has not moved for {hours:.1f} hours")
    last = open(newest).readlines()[-1:]
    if last and json.loads(last[0]).get("verdict"):
        facts["organic-life"]["verdict"] = json.loads(last[0])["verdict"]

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
