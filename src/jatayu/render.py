
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

PALETTE: dict[int, tuple[int, int, int]] = {
    0: (200, 198, 192),   # unclassified — neutral grey
    1: (33, 102, 172),    # water — blue
    2: (214, 96, 77),     # built-up — terracotta
    3: (146, 197, 222),   # flooded vegetation — pale blue
    4: (244, 226, 178),   # smooth bare — sand
}


def mask_to_png(classified: np.ndarray, out_path: str | Path) -> Path:
    h, w = classified.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for code, colour in PALETTE.items():
        rgb[classified == code] = colour
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out)
    return out
