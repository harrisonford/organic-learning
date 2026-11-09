"""Local System One judges vs the hosted one (decisions-fast), on the same 240 answers (offline).

out/judge_study.json holds 240 of kid_s3's answers (W1, W2, P) with the local grade and the
acceptance probability from parallel/decisions-fast:free (experiments/judge_study.py). Each
local open-weight decision model is served by llama.cpp on the GPU (`llama serve`, endpoint
/v1/systemone, the same typed-decisions format), asked the same questions in the same
batches of 20, and compared: agreement with decisions-fast (both accept or both reject), the
correlation of probabilities, agreement with the local grader, and judgments per second.
    python experiments/local_judge_compare.py ggml-org/Kev-4B-GGUF:Q8_0 [more repos]
"""
import json, os, subprocess, sys, time, urllib.request
import numpy as np

LLAMA = os.path.expanduser("~/tools/llama.cpp/llama-b12229")
PORT = 8790
BATCH = int(os.environ.get("BATCH", 20))  # answers per request (1: one answer per state)
AGE = {"W1": "an 18-month-old", "W2": "a two-year-old", "P": "a preschooler (3-5 years)", "K": "a kindergarten child"}
rows = json.load(open("out/judge_study.json"))


def body(batch):
    state = {f"item{i}": {"child": AGE.get(o["stage"], "a young child"), "adult asked": o["question"], "adult's own answer": o["answers"][0],
                          "other answers the adult would accept": o["answers"][1:], "the child answered": o["said"] or "(nothing)"}
             for i, o in enumerate(batch)}
    questions = {f"item{i}": {"type": "noul", "instructions": f"For item{i}: would a caring adult accept what the child answered as a right "
                              "answer for a child of that age (meaning matters, not the exact words; a picture the judge cannot see is "
                              "described by the adult's own answer)?"} for i in range(len(batch))}
    return {"state": state, "questions": questions}


def ask(b):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/systemone", data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=600))
    return [float(d["answers"][f"item{i}"]["noul"]) for i in range(len(b["questions"]))]


def run(repo):
    env = dict(os.environ, LD_LIBRARY_PATH=LLAMA)
    log = open(f"out/tmp/llama_{repo.replace('/', '_').replace(':', '_')}.log", "w")
    srv = subprocess.Popen([f"{LLAMA}/llama", "serve", "-hf", repo, "--port", str(PORT), "-ngl", "99", "-b", "2048", "-ub", "2048", "-c", "8192", "-np", "4"], env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(1800):  # first use downloads the model
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)
                break
            except Exception:  # noqa: BLE001
                if srv.poll() is not None:
                    raise RuntimeError(f"server exited: see {log.name}")
                time.sleep(2)
        t0, local = time.time(), []
        for i in range(0, len(rows), BATCH):
            local += ask(body(rows[i : i + BATCH]))
        dt = time.time() - t0
        hosted = np.array([o["judge"] for o in rows])
        grader = np.array([o["local"] for o in rows])
        p = np.array(local)
        agree = np.mean((p > 0.5) == (hosted > 0.5))
        corr = float(np.corrcoef(p, hosted)[0, 1])
        print(f"{repo:32} batch {BATCH:>2} | agrees with decisions-fast {agree:.2f} | prob. correlation {corr:.2f} | agrees with grader {np.mean((p > 0.5) == (grader >= 0.5)):.2f}"
              f" | mean acceptance {p.mean():.2f} (hosted {hosted.mean():.2f}) | {len(rows) / dt:.1f} judgments/s", flush=True)
        for st in ("W1", "W2", "P"):
            m = np.array([o["stage"] == st for o in rows])
            print(f"    {st}: agrees {np.mean((p[m] > 0.5) == (hosted[m] > 0.5)):.2f}, acceptance {p[m].mean():.2f} vs hosted {hosted[m].mean():.2f}", flush=True)
        worst = np.argsort(-np.abs(p - hosted))[:4]
        for j in worst:
            o = rows[j]
            print(f"    disagree: {o['question'][:45]!r} -> {o['said']!r} (adult {o['answers'][:2]}) local {p[j]:.2f} hosted {hosted[j]:.2f}", flush=True)
        json.dump(local, open(f"out/tmp/judge_{repo.replace('/', '_').replace(':', '_')}_b{BATCH}.json", "w"))
    finally:
        srv.terminate()
        srv.wait(timeout=60)


for repo in sys.argv[1:]:
    try:
        run(repo)
    except Exception as e:  # noqa: BLE001
        print(f"{repo}: failed ({str(e)[:120]})", flush=True)
