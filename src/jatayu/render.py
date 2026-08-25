"""Rendering helpers for analysis outputs.

Rendering deliberately knows nothing about CRS or rasterio. Grounding produces
pixel coordinates, and this module visualises those coordinates without
attempting geographic conversion. The report layer remains responsible for
transforming pixel coordinates into WGS84.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PALETTE: dict[int, tuple[int, int, int]] = {
    0: (200, 198, 192),   # unclassified — neutral grey
    1: (33, 102, 172),    # water — blue
    2: (214, 96, 77),     # built-up — terracotta
    3: (146, 197, 222),   # flooded vegetation — pale blue
    4: (244, 226, 178),   # smooth bare — sand
}


def mask_to_png(
    classified: np.ndarray,
    out_path: str | Path,
) -> Path:
    """Render integer class codes as a simple categorical PNG."""
    if classified.ndim != 2:
        raise ValueError(
            f"classified mask must be 2-D, got shape {classified.shape}"
        )

    h, w = classified.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for code, colour in PALETTE.items():
        rgb[classified == code] = colour

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(rgb).save(out)

    return out


def _normalise_base_image(
    base: np.ndarray | None,
    height: int,
    width: int,
) -> Image.Image:
    """Convert an optional raster preview into an RGB PIL image."""
    if base is None:
        # Neutral background when the caller only needs geometry.
        return Image.new(
            "RGB",
            (width, height),
            (32, 32, 32),
        )

    array = np.asarray(base)

    if array.ndim == 2:
        array = np.repeat(
            array[..., None],
            3,
            axis=2,
        )

    if array.ndim != 3 or array.shape[:2] != (height, width):
        raise ValueError(
            "base image must have shape "
            f"({height}, {width}) or ({height}, {width}, 3); "
            f"got {array.shape}"
        )

    if array.shape[2] == 1:
        array = np.repeat(
            array,
            3,
            axis=2,
        )
    elif array.shape[2] != 3:
        raise ValueError(
            f"base image must have 1 or 3 channels, got {array.shape[2]}"
        )

    array = array.astype(np.float32)

    finite = np.isfinite(array)

    if not finite.any():
        array = np.zeros_like(array)
    else:
        lo = np.nanpercentile(array, 2)
        hi = np.nanpercentile(array, 98)

        if hi <= lo:
            hi = lo + 1.0

        array = np.clip(
            (array - lo) / (hi - lo),
            0.0,
            1.0,
        )

        array[~finite] = 0.0

    return Image.fromarray(
        np.round(array * 255).astype(np.uint8),
        mode="RGB",
    )


def boxes_to_png(
    boxes: list[tuple[int, int, int, int]],
    out_path: str | Path,
    *,
    shape: tuple[int, int],
    base: np.ndarray | None = None,
) -> Path:
    """Render pixel-coordinate bounding boxes over an optional raster preview.

    Coordinates are deliberately kept in image pixel space:

        (x_min, y_min, x_max, y_max)

    No CRS transformation belongs here. This prevents the common failure where
    a visually correct box is later interpreted as geographic coordinates.
    """
    height, width = shape

    if height <= 0 or width <= 0:
        raise ValueError(
            f"shape must contain positive dimensions, got {shape}"
        )

    image = _normalise_base_image(
        base,
        height,
        width,
    ).convert("RGBA")

    draw = ImageDraw.Draw(image)

    for index, box in enumerate(boxes, start=1):
        x_min, y_min, x_max, y_max = box

        if not (
            0 <= x_min < x_max <= width
            and 0 <= y_min < y_max <= height
        ):
            raise ValueError(
                f"box {box} is outside image bounds "
                f"(width={width}, height={height})"
            )

        # Two outlines make boxes visible over both dark and bright imagery.
        draw.rectangle(
            (x_min, y_min, x_max - 1, y_max - 1),
            outline=(255, 255, 255, 255),
            width=4,
        )
        draw.rectangle(
            (x_min, y_min, x_max - 1, y_max - 1),
            outline=(255, 40, 40, 255),
            width=2,
        )

        label_y = max(0, y_min - 18)

        draw.text(
            (x_min + 3, label_y),
            str(index),
            fill=(255, 255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 255),
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    image.save(out)

    return out
