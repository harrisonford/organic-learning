"""The visual cortex's code of every picture of a picture set, kept on disk (float16), in shards run side by side.
    python experiments/world/code_pictures.py cococrops SHARD NSHARDS   -> ~/datasets/cococrops/codes_g4.f16 (rows x 7168)
"""
import os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from organic.world import DATA  # noqa: E402

name, shard, nshards = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
pics = np.load(os.path.join(DATA, name, "pictures_64.npy"), mmap_mode="r")
cortex = pickle.load(open("data/visual_cortex.pkl", "rb"))
dim = sum(layer.kinds for layer in cortex.layers) * 16
path = os.path.join(DATA, name, "codes_g4.f16")
if not os.path.exists(path) or os.path.getsize(path) != len(pics) * dim * 2:
    np.memmap(path, np.float16, "w+", shape=(len(pics), dim)).flush()
out = np.memmap(path, np.float16, "r+", shape=(len(pics), dim))
for i in range(shard, len(pics), nshards):
    v = np.concatenate(cortex.code(np.asarray(pics[i], np.float32) / 255, grid=4))
    out[i] = v / (np.linalg.norm(v) + 1e-8)
    if (i // nshards) % 2000 == 0:
        out.flush()
        print(shard, i, flush=True)
out.flush()
print("done", shard, flush=True)
