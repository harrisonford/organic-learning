"""SpokenCOCO's recordings as the cochlea hears them (queue E 32; docs/WORLD.md 2026-11-26; docs/VISION.md 12:50).

Streams ~/datasets/spokencoco/SpokenCOCO.tar.gz (downloaded data: never extracted to disk, never run; this script is
run with `python -I`) and puts every recording through organic/cochlea.py: 64 bands, a row every 10 ms.

    out  ~/datasets/spokencoco/cochlea_64.f16   rows of 64 float16, appended
         ~/datasets/spokencoco/cochlea.jsonl    one line per recording: uttid, split, COCO image id, first row, rows,
                                                speaker, and the transcript (FOR MEASURING ONLY: the brain never gets it)

PART=val (the part fully downloaded first) or PART=train (after the download is whole). A recording already in the
index is skipped, so the script can be run again.
    PART=val .venv/bin/python -I experiments/world/spokencoco_cochlea.py
"""
import io, json, os, sys, tarfile, time, wave
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from organic.cochlea import Cochlea, RATE  # noqa: E402

DIR = os.path.expanduser("~/datasets/spokencoco")
PART = os.environ.get("PART", "val")
meta = {}
for split in ("val", "train"):
    for item in json.load(open(os.path.join(DIR, "meta", f"SpokenCOCO_{split}.json")))["data"]:
        image = int(item["image"].split("_")[-1][:-4])
        for c in item["captions"]:
            meta[c["wav"]] = (split, image, c["uttid"], c["speaker"], c["text"])
index, store = os.path.join(DIR, "cochlea.jsonl"), os.path.join(DIR, "cochlea_64.f16")
done = {json.loads(line)["uttid"] for line in open(index)} if os.path.exists(index) else set()
start = os.path.getsize(store) // (64 * 2) if os.path.exists(store) else 0
ear, n, t0 = Cochlea(), 0, time.time()
with tarfile.open(os.path.join(DIR, "SpokenCOCO.tar.gz"), "r|gz") as tar, open(store, "ab") as out, open(index, "a") as ix:
    try:
        for m in tar:
            tar.members = []  # a streaming tar keeps every member it has passed; 617,000 of them filled 900 MB (2026-11-26)
            name = m.name[len("SpokenCOCO/"):] if m.name.startswith("SpokenCOCO/") else m.name
            if not m.isfile() or not name.endswith(".wav") or name not in meta:
                continue
            split, image, uttid, speaker, text = meta[name]
            if split != PART:
                if PART == "val" and split == "train":
                    break  # the archive holds val first
                continue
            if uttid in done:
                continue
            with wave.open(io.BytesIO(tar.extractfile(m).read())) as w:
                if w.getframerate() != RATE or w.getsampwidth() != 2:
                    continue
                x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
                if w.getnchannels() > 1:
                    x = x.reshape(-1, w.getnchannels()).mean(axis=1)
            gram = ear.hear(x).astype(np.float16)
            out.write(gram.tobytes())
            ix.write(json.dumps({"uttid": uttid, "split": split, "image": image, "start": start, "rows": len(gram), "speaker": speaker, "text": text}) + "\n")
            start += len(gram)
            n += 1
            if n % 5000 == 0:
                out.flush(); ix.flush()
                print(f"{n} recordings ({time.time() - t0:.0f}s)", flush=True)
    except (EOFError, tarfile.ReadError) as e:  # the end of what has been downloaded so far
        print(f"stopped at the end of the downloaded part: {e}", flush=True)
print(f"{PART}: {n} recordings heard this run, {start} rows in all ({time.time() - t0:.0f}s)", flush=True)
