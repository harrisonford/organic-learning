"""The stream brain's first run: does it look, hear and sound at once, at what rhythms, at what cost? (docs/ASYNC.md)

Commons videos with their sound (experiments/world/commons_video.py, video_audio.py), streamed as the world gives
them: a frame every 100 ms, a row of the cochlea every 10 ms, into organic/stream.StreamBrain's 10 ms clock. Nothing
here is a learner yet except the sight-sound link. Measured: saccades (how many, the fixations' durations), sounds
made (how many, and how many while the world itself was loud), how often the eye moved without a new frame (driven by
its own race, not by the world), and the CPU time for each simulated second.

    VIDEOS=5 SECONDS=60 python experiments/stream_probe.py
"""
import json, os, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.cortex2 import Layer2  # noqa: E402
from organic.stream import StreamBrain  # noqa: E402

DIR = os.path.expanduser("~/datasets/commons_video")
VIDEOS, SECONDS, SEED = int(os.environ.get("VIDEOS", 5)), float(os.environ.get("SECONDS", 60)), int(os.environ.get("SEED", 0))
rng = np.random.default_rng(SEED)
heard = {e["page"]: e for e in map(json.loads, open(os.path.join(DIR, "audio.jsonl"))) if e["sound"] and e["rows"] >= 600}
videos = [v for v in map(json.loads, open(os.path.join(DIR, "videos.jsonl"))) if v["page"] in heard]
frames = np.memmap(os.path.join(DIR, "frames_64.u8"), np.uint8, "r").reshape(-1, 64, 64, 3)
sound = np.memmap(os.path.join(DIR, "audio_64.f16"), np.float16, "r").reshape(-1, 64)

cortex = pickle.load(open(f"out/bench/cortex_ph_d_s{SEED}.pkl", "rb"))
hearing = Layer2(1, 128, 8, 1, seed=SEED)  # the eye's rule on sound, developed briefly on other videos' sound
others = [heard[v["page"]] for v in videos[VIDEOS:VIDEOS + 400]]
clip = lambda a: (lambda s: np.asarray(sound[s: s + 100], np.float32)[..., None])(a["start"] + int(rng.integers(0, a["rows"] - 100)))  # noqa: E731
hearing.settle((clip(others[i % len(others)]) for i in range(400)), rng)
for n in range(2000):
    hearing.learn(clip(others[rng.integers(len(others))]), rng, rate=0.05 * (1 - n / 2000) + 0.005)
print(f"hearing developed on 2,000 seconds of other videos' sound", flush=True)

all_sacc, all_fix, all_sounds, loud_sounds, own_moves, sim, cpu = 0, [], 0, 0, 0, 0.0, 0.0
for v in [videos[i] for i in rng.permutation(VIDEOS)]:
    a = heard[v["page"]]
    brain = StreamBrain(cortex, hearing, seed=SEED)
    n_steps = int(min(SECONDS, v["frames"] / v["fps"], a["rows"] / 100) * 100)
    rows = np.asarray(sound[a["start"]: a["start"] + n_steps], np.float32)
    loud = rows.sum(axis=1) > np.percentile(rows.sum(axis=1), 50)
    last_sacc, fix, new_frame_at = 0.0, [], -1.0
    t0 = time.process_time()
    for s in range(n_steps):
        f = np.asarray(frames[v["start"] + s // 10], np.float32) / 255 if s % 10 == 0 else None
        if f is not None:
            new_frame_at = s * 0.01
        for e in brain.step(f, rows[s]):
            if e[0] == "saccade":
                fix.append(brain.t - last_sacc)
                last_sacc = brain.t
                own_moves += (brain.t - new_frame_at) > 0.05  # not in the 50 ms after a new frame
            else:
                all_sounds += 1
                loud_sounds += bool(loud[s])
    cpu += time.process_time() - t0
    sim += n_steps * 0.01
    all_sacc += len(fix)
    all_fix += fix[1:]
    print(f"{v['title'][5:60]:56} {n_steps / 100:5.1f}s: {len(fix)} saccades, fixations median {np.median(fix[1:] or [0]) * 1000:.0f} ms; "
          f"{sum(1 for e in brain.log if e[1] == 'sound')} sounds made; sight-sound link {np.abs(brain.assoc).mean() if brain.assoc is not None else 0:.2e}", flush=True)
fx = np.array(all_fix) * 1000
print(f"\n{VIDEOS} videos, {sim:.0f} simulated seconds: {all_sacc / sim:.2f} saccades a second; fixations median {np.median(fx):.0f} ms "
      f"(quartiles {np.percentile(fx, 25):.0f}-{np.percentile(fx, 75):.0f}); {own_moves / max(all_sacc, 1):.2f} of saccades not set off by a new frame; "
      f"{all_sounds / sim:.2f} sounds a second, {loud_sounds / max(all_sounds, 1):.2f} of them while the world was loud (chance 0.50); "
      f"CPU {cpu / sim:.2f} s for each simulated second", flush=True)
