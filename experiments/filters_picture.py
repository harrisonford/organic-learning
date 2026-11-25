"""A picture of a first layer's 256 filters as the eye's pixels see them (docs/img).

A filter of organic.cortex2.Layer2 answers the window after its decorrelation Z, so what it prefers among raw
windows is W @ Z (Z is symmetric). Each filter is drawn as its 5 x 5 colour patch, stretched to its own largest
absolute value (grey is zero), the filters in the order of their weights' length, longest first.
    python experiments/filters_picture.py out/bench/cortex_sailn_s0.pkl docs/img/sailn_filters_s0.png
"""
import pickle, sys
import numpy as np
from PIL import Image
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), ".."))

cortex = pickle.load(open(sys.argv[1], "rb"))
layer = cortex.layers[0]
W = np.asarray(layer.W[0], np.float64)
F = (W @ np.asarray(layer.Z[0], np.float64)).reshape(len(W), layer.window, layer.window, -1)
order = np.argsort(-np.linalg.norm(W, axis=1))
side, k, pad, zoom = int(np.ceil(np.sqrt(len(W)))), layer.window, 1, 6
sheet = np.full((side * (k + pad) + pad, side * (k + pad) + pad, 3), 255, np.uint8)
for n, i in enumerate(order):
    f = F[i] / (np.abs(F[i]).max() + 1e-12)
    r, c = divmod(n, side)
    sheet[pad + r * (k + pad): pad + r * (k + pad) + k, pad + c * (k + pad): pad + c * (k + pad) + k] = np.round(127.5 + 127.5 * f).astype(np.uint8)
Image.fromarray(sheet).resize((sheet.shape[1] * zoom, sheet.shape[0] * zoom), Image.NEAREST).save(sys.argv[2])
print(sys.argv[2], sheet.shape)
