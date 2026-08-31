""" This is enhanced version of the previous render.py what it does
    [ ] gives heatmaps continuous index values 
    [ ] gives export geotiff option 
    [ ] masktopng and maskoverlaypng catogogrised class maps 
""""


from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
PALETTE: dict[int, tuple[int, int, int]] = {
    0: (200, 198, 192),   # unclassified — neutral grey
    1: (33, 102, 172),    # water — blue
    2: (214, 96, 77),     # built-up — terracotta
    3: (146, 197, 222),   # flooded vegetation — pale blue
    4: (244, 226, 178),   # smooth bare — sand
}

CROP_STRESS_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (34, 139, 34),     # healthy — forest green
    1: (255, 215, 0),     # mild stress — gold
    2: (255, 140, 0),     # moderate stress — dark orange
    3: (220, 20, 20),     # severe stress — red
}

CHANGE_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (180, 180, 180),   # no change — grey
    1: (34, 139, 34),     # increase — green
    2: (220, 20, 20),     # decrease — red
}

LULC_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (200, 198, 192),   # unknown
    1: (33, 102, 172),    # water
    2: (34, 139, 34),     # vegetation
    3: (214, 96, 77),     # built-up
    4: (244, 226, 178),   # bare soil
    5: (100, 180, 120),   # wet vegetation
}

# ---------------------------------------------------------------------------
# Continuous colormaps — for index heatmaps
# ---------------------------------------------------------------------------

def _ndvi_colormap(value: float) -> tuple[int, int, int]:
    """NDVI: red (stressed/bare) → yellow → green (healthy vegetation)."""
    v = max(0.0, min(1.0, value))
    if v < 0.5:
        # Red → Yellow
        t = v * 2
        return (int(220 - 40 * t), int(20 + 195 * t), 20)
    else:
        # Yellow → Green
        t = (v - 0.5) * 2
        return (int(180 - 146 * t), int(215 - 76 * t), int(20 + 14 * t))


def _water_colormap(value: float) -> tuple[int, int, int]:
    """Water index: light tan (dry) → light blue → deep blue (water)."""
    v = max(0.0, min(1.0, value))
    if v < 0.5:
        t = v * 2
        return (int(244 - 90 * t), int(226 - 80 * t), int(178 + 44 * t))
    else:
        t = (v - 0.5) * 2
        return (int(154 - 121 * t), int(146 - 44 * t), int(222 - 50 * t))


def _diverging_colormap(value: float) -> tuple[int, int, int]:
    """Change/anomaly: red (negative) → white (zero) → blue (positive)."""
    v = max(0.0, min(1.0, value))
    if v < 0.5:
        t = v * 2
        return (int(214 - 14 * t), int(96 + 159 * t), int(77 + 178 * t))
    else:
        t = (v - 0.5) * 2
        return (int(200 - 167 * t), int(255 - 153 * t), int(255 - 83 * t))


def _thermal_colormap(value: float) -> tuple[int, int, int]:
    """Temperature/LST: blue (cool) → yellow → red (hot)."""
    v = max(0.0, min(1.0, value))
    if v < 0.33:
        t = v * 3
        return (int(50 + 30 * t), int(50 + 150 * t), int(200 - 100 * t))
    elif v < 0.66:
        t = (v - 0.33) * 3
        return (int(80 + 175 * t), int(200 + 55 * t), int(100 - 80 * t))
    else:
        t = (v - 0.66) * 3
        return (int(255), int(255 - 175 * t), int(20))


COLORMAPS = {
    "ndvi": _ndvi_colormap,
    "vegetation": _ndvi_colormap,
    "water": _water_colormap,
    "mndwi": _water_colormap,
    "ndwi": _water_colormap,
    "change": _diverging_colormap,
    "anomaly": _diverging_colormap,
    "temperature": _thermal_colormap,
    "lst": _thermal_colormap,
}


def _apply_colormap(
    values: np.ndarray,
    colormap: str = "ndvi",
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Apply a continuous colormap to a 2D array. Returns (H, W, 3) uint8."""
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]

    if vmin is None:
        vmin = float(np.percentile(finite, 2)) if finite.size else 0.0
    if vmax is None:
        vmax = float(np.percentile(finite, 98)) if finite.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 0.01

    # Normalise to 0-1
    normalised = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)

    cmap_fn = COLORMAPS.get(colormap, _ndvi_colormap)
    h, w = arr.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for row in range(h):
        for col in range(w):
            if np.isfinite(arr[row, col]):
                rgb[row, col] = cmap_fn(normalised[row, col])
            else:
                rgb[row, col] = (40, 40, 40)  # nodata = dark

    return rgb


def _apply_colormap_fast(
    values: np.ndarray,
    colormap: str = "ndvi",
    vmin: float | None = None,
    vmax: float | None = None,
    n_steps: int = 256,
) -> np.ndarray:
    """Fast vectorised colormap using a lookup table."""
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]

    if vmin is None:
        vmin = float(np.percentile(finite, 2)) if finite.size else 0.0
    if vmax is None:
        vmax = float(np.percentile(finite, 98)) if finite.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 0.01

    cmap_fn = COLORMAPS.get(colormap, _ndvi_colormap)

    # Build LUT
    lut = np.zeros((n_steps, 3), dtype=np.uint8)
    for i in range(n_steps):
        lut[i] = cmap_fn(i / (n_steps - 1))

    # Map values to LUT indices
    normalised = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0 - 1e-9)
    indices = (normalised * n_steps).astype(np.int32)
    indices = np.clip(indices, 0, n_steps - 1)

    h, w = arr.shape
    rgb = lut[indices.ravel()].reshape(h, w, 3)

    # Set nodata pixels
    nodata = ~np.isfinite(arr)
    rgb[nodata] = (40, 40, 40)

    return rgb


# ---------------------------------------------------------------------------
# Base image preparation
# ---------------------------------------------------------------------------

def prepare_base_rgb(
    base_image: np.ndarray,
    target_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Convert any raster array to a percentile-stretched (H, W, 3) uint8 RGB."""
    base = np.asarray(base_image, dtype=np.float64)

    # Handle (bands, H, W) → (H, W, bands)
    if base.ndim == 3 and base.shape[0] in (3, 4, 5):
        base = np.moveaxis(base[:3], 0, -1)
    if base.ndim == 2:
        base = np.stack([base] * 3, axis=-1)
    if base.shape[2] > 3:
        base = base[:, :, :3]

    # Resize if needed
    if target_shape and base.shape[:2] != target_shape:
        from scipy.ndimage import zoom
        h, w = target_shape
        factors = (h / base.shape[0], w / base.shape[1], 1)
        base = zoom(base, factors, order=1)

    # Percentile stretch per channel
    for c in range(3):
        ch = base[:, :, c]
        valid = ch[np.isfinite(ch)]
        if valid.size == 0:
            continue
        lo, hi = np.percentile(valid, [2, 98])
        if hi > lo:
            base[:, :, c] = np.clip((ch - lo) / (hi - lo) * 255, 0, 255)
        else:
            base[:, :, c] = 0

    return np.nan_to_num(base, nan=0.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Heatmap rendering — continuous index values
# ---------------------------------------------------------------------------

def heatmap_png(
    values: np.ndarray,
    out_path: str | Path,
    *,
    colormap: str = "ndvi",
    vmin: float | None = None,
    vmax: float | None = None,
    label: str | None = None,
) -> Path:
    """Render continuous index values as a colormapped PNG."""
    rgb = _apply_colormap_fast(values, colormap, vmin, vmax)
    img = Image.fromarray(rgb)

    if label:
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), label, fill=(255, 255, 255),
                  stroke_width=2, stroke_fill=(0, 0, 0))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def heatmap_overlay_png(
    values: np.ndarray,
    base_image: np.ndarray,
    out_path: str | Path,
    *,
    colormap: str = "ndvi",
    alpha: float = 0.6,
    vmin: float | None = None,
    vmax: float | None = None,
    mask: np.ndarray | None = None,
    label: str | None = None,
) -> Path:
    """Render continuous index values semi-transparently over satellite RGB.

    mask: optional boolean array — only show heatmap where mask is True.
           Useful for showing NDVI only on vegetation pixels.
    """
    h, w = values.shape
    base = prepare_base_rgb(base_image, target_shape=(h, w))
    heat = _apply_colormap_fast(values, colormap, vmin, vmax)

    # Build alpha channel
    alpha_arr = np.full((h, w), alpha)
    alpha_arr[~np.isfinite(values)] = 0.0  # nodata stays transparent
    if mask is not None:
        alpha_arr[~mask] = 0.0  # unmasked areas show raw satellite

    # Composite
    alpha_3d = alpha_arr[:, :, np.newaxis]
    composited = base.astype(np.float64) * (1 - alpha_3d) + heat.astype(np.float64) * alpha_3d
    composited = np.clip(composited, 0, 255).astype(np.uint8)

    img = Image.fromarray(composited)

    if label:
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), label, fill=(255, 255, 255),
                  stroke_width=2, stroke_fill=(0, 0, 0))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


# ---------------------------------------------------------------------------
# Multi-panel rendering — side by side comparison
# ---------------------------------------------------------------------------

def multi_panel_png(
    panels: list[tuple[np.ndarray, str]],
    out_path: str | Path,
    *,
    max_width: int = 2048,
) -> Path:
    """Render multiple arrays side by side with labels.

    panels: list of (rgb_array_HxWx3, label_string)
    """
    if not panels:
        raise ValueError("At least one panel required")

    images = []
    for arr, label in panels:
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.dtype != np.uint8:
            arr = prepare_base_rgb(arr)
        img = Image.fromarray(arr)
        if label:
            draw = ImageDraw.Draw(img)
            draw.rectangle((0, img.height - 30, img.width, img.height), fill=(0, 0, 0, 180))
            draw.text((10, img.height - 25), label, fill=(255, 255, 255))
        images.append(img)

    # Resize all to same height
    min_h = min(img.height for img in images)
    resized = []
    for img in images:
        ratio = min_h / img.height
        new_w = int(img.width * ratio)
        resized.append(img.resize((new_w, min_h), Image.LANCZOS))

    total_w = sum(img.width for img in resized)
    if total_w > max_width:
        scale = max_width / total_w
        min_h = int(min_h * scale)
        resized = [img.resize((int(img.width * scale), min_h), Image.LANCZOS) for img in resized]
        total_w = sum(img.width for img in resized)

    canvas = Image.new("RGB", (total_w, min_h), (30, 30, 30))
    x = 0
    for img in resized:
        canvas.paste(img, (x, 0))
        x += img.width

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


# ---------------------------------------------------------------------------
# Colorbar / legend strip
# ---------------------------------------------------------------------------

def colorbar_png(
    out_path: str | Path,
    *,
    colormap: str = "ndvi",
    vmin: float = -1.0,
    vmax: float = 1.0,
    label: str = "NDVI",
    width: int = 40,
    height: int = 300,
) -> Path:
    """Render a vertical colorbar as a standalone PNG."""
    cmap_fn = COLORMAPS.get(colormap, _ndvi_colormap)
    bar = np.zeros((height, width, 3), dtype=np.uint8)

    for row in range(height):
        value = 1.0 - row / (height - 1)  # top = max, bottom = min
        color = cmap_fn(value)
        bar[row, :] = color

    # Add labels
    canvas_w = width + 80
    canvas = Image.new("RGB", (canvas_w, height + 40), (255, 255, 255))
    canvas.paste(Image.fromarray(bar), (10, 20))

    draw = ImageDraw.Draw(canvas)
    draw.text((width + 15, 15), f"{vmax:.1f}", fill=(0, 0, 0))
    draw.text((width + 15, height - 5), f"{vmin:.1f}", fill=(0, 0, 0))
    draw.text((10, 2), label, fill=(0, 0, 0))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


# ---------------------------------------------------------------------------
# Categorical masks — existing functions
# ---------------------------------------------------------------------------

def mask_overlay_png(
    classified: np.ndarray,
    base_image: np.ndarray,
    out_path: str | Path,
    *,
    alpha: float = 0.55,
    palette: dict[int, tuple[int, int, int]] | None = None,
) -> Path:
    """Render class mask semi-transparently over a satellite RGB base."""
    if classified.ndim != 2:
        raise ValueError(f"classified must be 2-D, got {classified.shape}")

    active_palette = palette or PALETTE
    h, w = classified.shape
    base = prepare_base_rgb(base_image, target_shape=(h, w))

    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    for code, colour in active_palette.items():
        if code == 0:
            continue
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
    """Render a class mask as flat palette colors, no base image required."""
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


# ---------------------------------------------------------------------------
# Bounding boxes
# ---------------------------------------------------------------------------

def boxes_to_png(
    boxes: list[tuple[int, int, int, int]],
    out_path: str | Path,
    *,
    shape: tuple[int, int],
    base: np.ndarray | None = None,
) -> Path:
    """Render pixel-coordinate bounding boxes over an optional raster preview."""
    height, width = shape

    if base is not None:
        base_rgb = prepare_base_rgb(base, target_shape=(height, width))
        image = Image.fromarray(base_rgb).convert("RGBA")
    else:
        image = Image.new("RGBA", (width, height), (32, 32, 32, 255))

    draw = ImageDraw.Draw(image)

    for index, box in enumerate(boxes, start=1):
        x_min, y_min, x_max, y_max = box
        x_min = max(0, min(x_min, width - 1))
        y_min = max(0, min(y_min, height - 1))
        x_max = max(0, min(x_max, width))
        y_max = max(0, min(y_max, height))

        draw.rectangle((x_min, y_min, x_max - 1, y_max - 1),
                        outline=(255, 255, 255, 255), width=4)
        draw.rectangle((x_min, y_min, x_max - 1, y_max - 1),
                        outline=(255, 40, 40, 255), width=2)

        label_y = max(0, y_min - 18)
        draw.text((x_min + 3, label_y), str(index),
                  fill=(255, 255, 255, 255), stroke_width=2,
                  stroke_fill=(0, 0, 0, 255))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    return out


# ---------------------------------------------------------------------------
# GeoTIFF export — preserves CRS for QGIS/GIS consumption
# ---------------------------------------------------------------------------

def export_geotiff(
    data: np.ndarray,
    out_path: str | Path,
    *,
    crs: str | None = None,
    transform: tuple | None = None,
    band_names: list[str] | None = None,
    nodata: float | None = None,
) -> Path:
    """Write analysis output as a georeferenced GeoTIFF.

    This is the only function in render.py that imports rasterio.
    All other rendering is pure PIL/NumPy.
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(data)

    # Determine shape and band count
    if arr.ndim == 2:
        bands, h, w = 1, arr.shape[0], arr.shape[1]
        arr = arr[np.newaxis, :, :]
    elif arr.ndim == 3 and arr.shape[0] <= 20:
        bands, h, w = arr.shape
    elif arr.ndim == 3:
        h, w, bands = arr.shape
        arr = np.moveaxis(arr, -1, 0)
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    # Determine dtype
    if arr.dtype in (np.float32, np.float64):
        dtype = "float32"
        arr = arr.astype(np.float32)
    elif arr.dtype == np.uint8:
        dtype = "uint8"
    else:
        dtype = "float32"
        arr = arr.astype(np.float32)

    profile = {
        "driver": "GTiff",
        "width": w,
        "height": h,
        "count": bands,
        "dtype": dtype,
        "compress": "lzw",
    }

    if crs:
        profile["crs"] = CRS.from_string(crs)
    if transform:
        if isinstance(transform, tuple) and len(transform) == 6:
            profile["transform"] = Affine(*transform)
        else:
            profile["transform"] = transform
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(out, "w", **profile) as dst:
        for i in range(bands):
            dst.write(arr[i], i + 1)
            if band_names and i < len(band_names):
                dst.set_band_description(i + 1, band_names[i])

    return out
