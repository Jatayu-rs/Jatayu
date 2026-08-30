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


def mask_overlay_png(
    classified: np.ndarray,
    base_image: np.ndarray,
    out_path: str | Path,
    *,
    alpha: float = 0.55,
    palette: dict[int, tuple[int, int, int]] | None = None,
) -> Path:
    """Render class mask semi-transparently over a satellite RGB base.

    base_image: (bands, H, W) or (H, W, 3) array — first 3 bands used as RGB.
    alpha: overlay opacity, 0.0 = invisible, 1.0 = solid.
    palette: optional class→color override. Defaults to the module PALETTE
        (water/built-up/vegetation/bare). Pass a tool-specific palette when
        the class codes mean something else, e.g. crop-stress severity.
    """
    if classified.ndim != 2:
        raise ValueError(f"classified must be 2-D, got {classified.shape}")

    active_palette = palette or PALETTE
    h, w = classified.shape

    # Prepare base RGB
    base = np.asarray(base_image, dtype=np.float64)
    if base.ndim == 3 and base.shape[0] in (3, 4, 5):
        base = np.moveaxis(base[:3], 0, -1)
    if base.ndim == 2:
        base = np.stack([base] * 3, axis=-1)

    if base.shape[:2] != (h, w):
        from scipy.ndimage import zoom
        factors = (h / base.shape[0], w / base.shape[1], 1)
        base = zoom(base, factors, order=1)

    for c in range(3):
        ch = base[:, :, c]
        valid = ch[np.isfinite(ch)]
        if valid.size == 0:
            continue
        lo, hi = np.percentile(valid, [2, 98])
        if hi > lo:
            ch = np.clip((ch - lo) / (hi - lo) * 255, 0, 255)
        else:
            ch = np.zeros_like(ch)
        base[:, :, c] = ch

    base = np.nan_to_num(base, nan=0.0).astype(np.uint8)

    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    for code, colour in active_palette.items():
        if code == 0:
            continue  # class 0 stays transparent — healthy areas show raw imagery
        mask = classified == code
        overlay[mask, :3] = colour
        overlay[mask, 3] = int(alpha * 255)

    base_f = base.astype(np.float64)
    overlay_f = overlay[:, :, :3].astype(np.float64)
    alpha_arr = overlay[:, :, 3:4].astype(np.float64) / 255.0

    composited = base_f * (1 - alpha_arr) + overlay_f * alpha_arr
    composited = np.clip(composited, 0, 255).astype(np.uint8)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(composited).save(out)
    return out


def mask_to_png(
    classified: np.ndarray,
    out_path: str | Path,
    *,
    palette: dict[int, tuple[int, int, int]] | None = None,
) -> Path:
    """Render a class mask as flat palette colors, no base image required.

    Used as a fallback for callers (e.g. fusion) when compositing over a
    satellite RGB base isn't possible — for instance if the RGB bands
    couldn't be read or the base array is malformed. Unclassified pixels use
    the same neutral grey as ``PALETTE[0]`` so this reads consistently next
    to ``mask_overlay_png`` output rather than introducing a different
    background convention.
    """
    if classified.ndim != 2:
        raise ValueError(f"classified must be 2-D, got {classified.shape}")

    active_palette = palette or PALETTE
    h, w = classified.shape

    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for code, colour in active_palette.items():
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
