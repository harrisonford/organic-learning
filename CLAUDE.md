
## Memory (memoize)

This project's bitácora is mirrored in Kris's memoize brain (token: gitignored `.memoize-token`,
see the memoize skill). Recall `organic-learning` before substantial work; write decisions,
Kris's corrections and preferences, and findings (with `--outcome`) there, tagged
`organic-learning`, one self-contained sentence each. docs/LOG.md and the other docs stay as
the detailed local log.

## Status reports

Before any status report run `.venv/bin/python experiments/health.py` (daemons, data growth, lives listed in
`out/monitor/lives.json`, memory) and say what it shows. Long jobs go in `systemd-run --user` units (a session
that ends kills its background shells); add a life to `out/monitor/lives.json` when it starts. Claims need 3 seeds
and the 300-item tests. Start reading at docs/REVIEW_2026-11-10.md, then docs/AREAS.md and the end of docs/LOG.md.

## Memory budget (14 GB machine)

Three small lives take about 7 GB and grow. An offline probe that loads an ear and the visual cache takes 1.3–3.4 GB.
On 2026-11-10 four probes beside three lives made the kernel kill a life. Before starting anything, run `free -m`;
start probes with `systemd-run --user -p MemoryMax=... -p OOMScoreAdjust=900` so a probe dies before a life does;
keep at least 2 GB available.
