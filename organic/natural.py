"""Natural images for the eye's development (photographs bundled with scikit-image).

The eye develops on the visual statistics of the natural world, not on the
task scenes, so the organ is not tailored to any task.
"""

import numpy as np
from PIL import Image

PHOTOS = ["astronaut", "brick", "camera", "chelsea", "coffee", "coins", "grass", "gravel", "moon", "rocket", "clock", "hubble_deep_field", "horse"]


def _load():
    import skimage.data as d

    out = []
    for name in PHOTOS:
        im = np.asarray(getattr(d, name)(), np.float32)
        if im.max() > 1.5:
            im = im / 255.0
        if im.ndim == 2:
            im = np.repeat(im[..., None], 3, axis=2)
        out.append(np.clip(im[..., :3], 0, 1))
    return out


_PHOTOS = None


def natural_views(rng, n, size=40):
    """n random crops (random scale, position, mirror) resized to the retina's world."""
    global _PHOTOS
    if _PHOTOS is None:
        _PHOTOS = _load()
    views = []
    for _ in range(n):
        im = _PHOTOS[rng.integers(len(_PHOTOS))]
        H, W = im.shape[:2]
        side = int(rng.uniform(40, min(H, W, 240)))
        y, x = rng.integers(0, H - side + 1), rng.integers(0, W - side + 1)
        crop = im[y : y + side, x : x + side]
        if rng.random() < 0.5:
            crop = crop[:, ::-1]
        views.append(np.asarray(Image.fromarray((crop * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR), np.float32) / 255)
    return views
