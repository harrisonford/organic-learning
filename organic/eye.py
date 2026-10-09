"""The eye: a hard-wired organ that looks at a scene the way a primate eye does.

It is not trained. It does what the retina, the superior colliculus and early
visual cortex do before any learning is involved:

1. Saliency: find what stands out from the background (bright or colorful).
2. Saccades: fixate each salient blob, largest first.
3. For every fixation, split what reaches cortex into separate streams, as the
   primate visual system does:
   - form  (ventral / IT-like): a foveated view, centered on the object and
     scaled to fill the fovea, so the same shape looks the same anywhere.
   - color (V4-like): a ring of hue-tuned neurons plus saturation and
     brightness neurons, population coded.
   - where (dorsal / parietal-like): where the eye had to move to and how big
     the object was, population coded.
   - number (intraparietal-like): how many things were fixated, coded by
     log-Gaussian number neurons (Nieder & Miller 2003).

Two speeds, as in primates: a fast, coarse magnocellular glance at the whole
scene (low resolution, available almost at once) and the slow, fine
parvocellular detail that needs a saccade to each object. The eye only says
what it saw and when it arrives; it never decides which is good enough.

Population codes everywhere: each quantity is represented by many broadly
tuned neurons, so similar values give similar activity patterns.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image


def tuning(value, centers, width):
    """Gaussian tuning curves: how strongly each neuron responds to a value."""
    return np.exp(-0.5 * ((value - centers) / width) ** 2).astype(np.float32)


def ring_tuning(angle, n, width):
    """Tuning curves on a circle (for hue)."""
    centers = np.linspace(0, 2 * np.pi, n, endpoint=False)
    d = np.angle(np.exp(1j * (angle - centers)))
    return np.exp(-0.5 * (d / width) ** 2).astype(np.float32)


@dataclass
class Fixation:
    form: np.ndarray
    color: np.ndarray
    where: np.ndarray
    center: tuple
    size: float


class Eye:
    fovea = 12
    form_dim = 12 * 12 + 4 * 6 * 6
    color_dim = 16
    where_dim = 9 + 9 + 6
    number_dim = 8

    def __init__(self, saliency=0.35, min_blob=4):
        self.saliency = saliency
        self.min_blob = min_blob

    # ------------------------------------------------------------- saccades

    def salience_map(self, img):
        lum = img.mean(axis=2)
        sat = img.max(axis=2) - img.min(axis=2)
        return np.maximum(lum, sat)

    def blobs(self, img):
        """Connected salient regions, the targets of saccades (largest first)."""
        mask = self.salience_map(img) > self.saliency
        h, w = mask.shape
        seen = np.zeros_like(mask)
        found = []
        for y in range(h):
            for x in range(w):
                if not mask[y, x] or seen[y, x]:
                    continue
                stack, pix = [(y, x)], []
                seen[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    pix.append((cy, cx))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True
                                stack.append((ny, nx))
                found.append(np.array(pix))
        found = self._group(found)
        found = [p for p in found if len(p) >= self.min_blob]
        found.sort(key=len, reverse=True)
        return found

    @staticmethod
    def _group(blobs):
        """Gestalt closure: parts enclosed by another blob belong to it (eyes in a face)."""
        boxes = [(p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()) for p in blobs]
        owner = list(range(len(blobs)))
        for i, (y0, x0, y1, x1) in enumerate(boxes):
            for j, (Y0, X0, Y1, X1) in enumerate(boxes):
                if i != j and len(blobs[j]) > len(blobs[i]) and Y0 <= y0 and X0 <= x0 and y1 <= Y1 and x1 <= X1:
                    owner[i] = j
                    break
        for i in range(len(owner)):  # follow chains to the outermost blob
            while owner[owner[i]] != owner[i]:
                owner[i] = owner[owner[i]]
        groups = {}
        for i, o in enumerate(owner):
            groups.setdefault(o, []).append(blobs[i])
        return [np.concatenate(g) for g in groups.values()]

    def look(self, img):
        """Saccade to every salient object and return one Fixation per object."""
        img = np.asarray(img, np.float32)
        return [self._fixate(img, pix) for pix in self.blobs(img)]

    # --------------------------------------------------------------- streams

    def _fixate(self, img, pix):
        H, W = img.shape[:2]
        y0, x0 = pix.min(axis=0)
        y1, x1 = pix.max(axis=0) + 1
        side = max(y1 - y0, x1 - x0) + 2
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        # foveate: a square window centered on the object, scaled to the fovea
        box = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
        lum = Image.fromarray((img.mean(axis=2) * 255).astype(np.uint8))
        fov = np.asarray(lum.transform((self.fovea, self.fovea), Image.EXTENT, box, Image.BILINEAR), np.float32) / 255
        fov = fov - fov.mean()
        gx = np.zeros_like(fov)
        gy = np.zeros_like(fov)
        gx[:, 1:-1] = fov[:, 2:] - fov[:, :-2]
        gy[1:-1, :] = fov[2:, :] - fov[:-2, :]
        edges = [np.abs(gx), np.abs(gy), np.abs(gx + gy) / 1.4, np.abs(gx - gy) / 1.4]
        pooled = [e.reshape(6, 2, 6, 2).mean(axis=(1, 3)).ravel() for e in edges]
        form = np.concatenate([fov.ravel(), 2.0 * np.concatenate(pooled)])

        # color: average cone response over the object's own pixels
        rgb = img[pix[:, 0], pix[:, 1]].mean(axis=0)
        r, g, b = rgb
        hue = np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)  # from opponent channels
        sat = float(rgb.max() - rgb.min())
        color = np.concatenate(
            [
                ring_tuning(hue, 12, 0.45) * min(1.0, sat * 2.5),
                tuning(sat, np.array([0.0, 0.5]), 0.25),
                tuning(float(rgb.mean()), np.array([0.3, 0.9]), 0.25),
            ]
        )

        # where: the saccade's landing point and how big the object looked
        size = side / max(H, W)
        where = np.concatenate(
            [
                tuning(cx / W, np.linspace(0, 1, 9), 0.09),
                tuning(cy / H, np.linspace(0, 1, 9), 0.09),
                tuning(size, np.linspace(0.1, 0.7, 6), 0.08),
            ]
        )
        return Fixation(form, color, where, (cx / W, cy / H), size)

    gist_color_dim = 16
    gist_where_dim = 5 + 5 + 4
    gist_form_dim = 64

    def glance(self, img):
        """Magnocellular gist: one low-resolution look at the whole scene."""
        img = np.asarray(img, np.float32)
        H, W = img.shape[:2]
        low = img.reshape(8, H // 8, 8, W // 8, 3).mean(axis=(1, 3))  # 8x8 retina
        lum = low.mean(axis=2)
        sal = np.maximum(lum, low.max(axis=2) - low.min(axis=2))
        mask = sal > 0.6 * sal.max() if sal.max() > 0.15 else np.zeros_like(sal, bool)
        if mask.any():
            rgb = (low[mask] * sal[mask][:, None]).sum(axis=0) / sal[mask].sum()
            ys, xs = np.nonzero(mask)
            cy, cx = (ys.mean() + 0.5) / 8, (xs.mean() + 0.5) / 8
            extent = (max(np.ptp(ys), np.ptp(xs)) + 1) / 8
        else:
            rgb, cx, cy, extent = np.zeros(3, np.float32), 0.5, 0.5, 0.0
        r, g, b = rgb
        hue = np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)
        sat = float(rgb.max() - rgb.min())
        color = np.concatenate(
            [
                ring_tuning(hue, 12, 0.6) * min(1.0, sat * 2.5),
                tuning(sat, np.array([0.0, 0.5]), 0.3),
                tuning(float(rgb.mean()), np.array([0.3, 0.9]), 0.3),
            ]
        )
        where = np.concatenate(
            [
                tuning(cx, np.linspace(0, 1, 5), 0.15),
                tuning(cy, np.linspace(0, 1, 5), 0.15),
                tuning(extent, np.linspace(0.1, 0.7, 4), 0.15),
            ]
        )
        form = (lum - lum.mean()).ravel()
        return {"gist_color": color, "gist_where": where, "gist_form": form}

    def number(self, n_fixations):
        """Number neurons: log-scale tuning, so 1 vs 2 is clearer than 7 vs 8."""
        centers = np.log(np.arange(1, self.number_dim + 1))
        return tuning(np.log(max(n_fixations, 0.5)), centers, 0.25)
