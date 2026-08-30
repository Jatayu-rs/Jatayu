"""Optical-SAR cross-modal analysis.

Physics:
* Water is radar-dark (smooth surface reflects away from sensor)
* Built-up is radar-bright (corner-reflector double-bounce)
* Vegetation identified with NDVI
* Radar-bright + spectrally vegetated = flooded vegetation or dense canopy
* MNDWI used instead of NDWI because turbid water reduces green-NIR contrast
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np

from jatayu.io.loader import read_bands
from jatayu.render import mask_to_png
from jatayu.schemas import (
    Evidence, ImageRef, Modality, TaskFamily, TaskName, ToolRequest, ToolResult,
)
from jatayu.tools.registry import register

MODEL_ID = "physics-adaptive-v2"

# Class IDs
CLS_UNCLASSIFIED = 0
CLS_WATER = 1
CLS_BUILTUP = 2
CLS_FLOODED_VEG = 3
CLS_SMOOTH_BARE = 4

# Rendering palette — matches render.py PALETTE
OVERLAY_COLORS = {
    1: (33, 102, 172),     # water — blue
    2: (214, 96, 77),      # built-up — terracotta
    3: (146, 197, 222),    # flooded vegetation — pale blue
    4: (244, 226, 178),    # smooth bare — sand
}

DEFAULTS = {
    "sar_water_db": -18.0,
    "sar_builtup_db": -2.0,
    "mndwi_threshold": 0.10,
    "ndbi_threshold": 0.06,
    "ndvi_threshold": 0.30,
}

MNDWI_FLOOR = 0.0

FULL_LEGEND = {
    "0": "unclassified",
    "1": "open water (radar + optical agree)",
    "2": "built-up (radar + optical agree)",
    "3": "flooded vegetation / dense canopy (radar-bright, spectrally vegetated)",
    "4": "smooth bare / radar shadow (radar-dark, not water)",
}


# ---------------------------------------------------------------------------
# Query interpretation
# ---------------------------------------------------------------------------

def _query_targets(query: str) -> tuple[set[int], str]:
    q = query.lower()
    targets: set[int] = set()

    water_terms = ("water", "river", "lake", "pond", "wetland", "flood", "flooded", "standing water")
    built_terms = ("built-up", "built up", "builtup", "urban", "building", "construction", "settlement", "city")
    veg_terms = ("vegetation", "forest", "trees", "crop", "canopy", "greenery")
    bare_terms = ("bare", "bare soil", "soil", "sand", "shadow")
    agreement_terms = ("agreement", "both", "optical and sar", "optical + sar", "jointly", "together")

    if any(t in q for t in water_terms):
        targets.add(CLS_WATER)
    if any(t in q for t in built_terms):
        targets.add(CLS_BUILTUP)
    if any(t in q for t in veg_terms):
        targets.add(CLS_FLOODED_VEG)
    if any(t in q for t in bare_terms):
        targets.add(CLS_SMOOTH_BARE)
    if any(t in q for t in agreement_terms) and not targets:
        targets.update({CLS_WATER, CLS_BUILTUP})

    if not targets:
        return {CLS_WATER, CLS_BUILTUP, CLS_FLOODED_VEG, CLS_SMOOTH_BARE}, "all"

    if targets == {CLS_WATER}: return targets, "water"
    if targets == {CLS_BUILTUP}: return targets, "builtup"
    if targets == {CLS_WATER, CLS_BUILTUP}: return targets, "agreement"
    return targets, "custom"


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def _nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(np.float32), b.astype(np.float32)
    denom = a + b
    out = np.zeros_like(denom)
    np.divide(a - b, denom, out=out, where=np.abs(denom) > 1e-6)
    return out


def _adaptive_thresholds(sar_db, mndwi, ndbi, ndvi) -> dict:
    return {
        "sar_water_db": float(np.nanpercentile(sar_db, 5)),
        "sar_builtup_db": float(np.nanpercentile(sar_db, 90)),
        "mndwi_threshold": max(MNDWI_FLOOR, float(np.nanpercentile(mndwi, 90))),
        "ndbi_threshold": float(np.nanpercentile(ndbi, 75)),
        "ndvi_threshold": float(np.nanpercentile(ndvi, 60)),
    }


def classify(*, sar_db, mndwi, ndbi, ndvi, thresholds=None):
    t = {**DEFAULTS, **(thresholds or {})}

    sar_dark = sar_db < t["sar_water_db"]
    sar_bright = sar_db > t["sar_builtup_db"]
    spec_water = mndwi > t["mndwi_threshold"]
    spec_built = ndbi > t["ndbi_threshold"]
    spec_veg = ndvi > t["ndvi_threshold"]

    out = np.full(sar_db.shape, CLS_UNCLASSIFIED, dtype=np.uint8)
    out[sar_dark & ~spec_water] = CLS_SMOOTH_BARE
    out[sar_bright & spec_veg] = CLS_FLOODED_VEG
    out[sar_bright & spec_built] = CLS_BUILTUP
    out[sar_dark & spec_water] = CLS_WATER

    invalid = ~(np.isfinite(sar_db) & np.isfinite(mndwi) & np.isfinite(ndbi) & np.isfinite(ndvi))
    out[invalid] = CLS_UNCLASSIFIED
    return out


def _focus(classified, targets):
    focused = np.full_like(classified, CLS_UNCLASSIFIED)
    for cls in targets:
        focused[classified == cls] = cls
    return focused


def _legend(targets):
    legend = {"0": "not selected / insufficient evidence"}
    for cls in sorted(targets):
        legend[str(cls)] = FULL_LEGEND[str(cls)]
    return legend


def _summarise(classified, targets, mode):
    total = max(classified.size, 1)
    water = float((classified == CLS_WATER).sum()) / total * 100
    built = float((classified == CLS_BUILTUP).sum()) / total * 100
    flooded = float((classified == CLS_FLOODED_VEG).sum()) / total * 100
    bare = float((classified == CLS_SMOOTH_BARE).sum()) / total * 100

    if mode == "water":
        return f"The optical-SAR fusion identifies {water:.1f}% of the scene as open water, supported by both spectral (MNDWI) and radar-dark evidence."
    elif mode == "builtup":
        return f"The fusion identifies {built:.1f}% as built-up, with strong radar backscatter and elevated NDBI providing cross-modal support."
    elif mode == "agreement":
        return f"Joint optical-SAR analysis finds {water:.1f}% open water and {built:.1f}% built-up surface where both modalities agree."
    else:
        return (f"Combining optical and SAR evidence: {water:.1f}% open water, {built:.1f}% built-up, "
                f"{flooded:.1f}% radar-bright vegetated surface, {bare:.1f}% radar-dark non-water surface.")


def _confidence(classified, targets):
    selected = np.isin(classified, list(targets))
    n = int(selected.sum())
    if n == 0:
        return 0.0
    direct = int((selected & np.isin(classified, [CLS_WATER, CLS_BUILTUP])).sum())
    score = direct / n if n else 0.0
    return float(max(0.0, min(1.0, 0.15 + 0.85 * score)))


# ---------------------------------------------------------------------------
# Composited overlay rendering
# ---------------------------------------------------------------------------

def _render_composited(classified, targets, base_rgb, out_path, alpha=0.55):
    """Render class overlay semi-transparently on satellite RGB."""
    h, w = classified.shape

    # Prepare base
    base = np.asarray(base_rgb, dtype=np.float64)
    if base.ndim == 3 and base.shape[0] in (3, 4, 5):
        base = np.moveaxis(base[:3], 0, -1)
    if base.ndim == 2:
        base = np.stack([base] * 3, axis=-1)

    # Resize if needed
    if base.shape[:2] != (h, w):
        from scipy.ndimage import zoom
        factors = (h / base.shape[0], w / base.shape[1], 1)
        base = zoom(base, factors, order=1)

    # Percentile stretch
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

    base = np.nan_to_num(base, nan=0.0).astype(np.float64)

    # Build overlay with alpha
    overlay_rgb = np.zeros((h, w, 3), dtype=np.float64)
    alpha_mask = np.zeros((h, w, 1), dtype=np.float64)

    for code, colour in OVERLAY_COLORS.items():
        if code not in targets:
            continue
        mask = classified == code
        overlay_rgb[mask] = colour
        alpha_mask[mask] = alpha

    # Composite
    composited = base * (1 - alpha_mask) + overlay_rgb * alpha_mask
    composited = np.clip(composited, 0, 255).astype(np.uint8)

    from PIL import Image
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(composited).save(out)
    return out


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def _split(images):
    optical = [i for i in images if i.modality.is_optical_family]
    sar = [i for i in images if i.modality is Modality.SAR]
    if len(optical) != 1 or len(sar) != 1:
        raise ValueError(
            f"Fusion needs exactly one optical and one SAR image; "
            f"got {len(optical)} optical and {len(sar)} SAR."
        )
    return optical[0], sar[0]


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@register(
    TaskName.FUSION,
    families={TaskFamily.CROSS_MODAL},
    description="Combines a co-registered optical image and a SAR image to "
                "identify surface types using independent physical evidence. "
                "Query-aware for water, built-up, vegetation, and bare surfaces.",
)
def run(req: ToolRequest) -> ToolResult:
    optical, sar = _split(req.images)
    decimate = int(req.params.get("decimate", 2))
    targets, mode = _query_targets(req.query)
    notes: list[str] = []

    # Read optical
    green, red, nir, swir = read_bands(optical, ["green", "red", "nir", "swir1"], decimate=decimate)

    # Read SAR
    (backscatter,) = read_bands(sar, ["vv"], decimate=decimate)

    # SAR calibration
    finite = backscatter[np.isfinite(backscatter)]
    if finite.size and finite.min() >= 0:
        sar_db = 10.0 * np.log10(np.where(backscatter > 0, backscatter, np.nan))
        notes.append("SAR converted from linear power to dB.")
    else:
        sar_db = backscatter

    # Optical indices
    mndwi = _nd(green, swir)
    ndbi = _nd(swir, nir)
    ndvi = _nd(nir, red)

    # Adaptive thresholds
    thresholds = _adaptive_thresholds(sar_db, mndwi, ndbi, ndvi)
    thresholds.update({k: v for k, v in req.params.items() if k in DEFAULTS})

    # Classify
    classified = classify(sar_db=sar_db, mndwi=mndwi, ndbi=ndbi, ndvi=ndvi, thresholds=thresholds)
    focused = _focus(classified, targets)
    answer = _summarise(classified, targets, mode)

    # Render — composite over satellite RGB
    out_dir = Path("data/samples/jatayu_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"fusion_{uuid4().hex[:8]}.png"

    try:
        # Read RGB for base image (red=band3, green=band2, blue=band1 from optical)
        blue_b, green_b, red_b = read_bands(optical, ["blue", "green", "red"], decimate=decimate)
        base_rgb = np.stack([red_b, green_b, blue_b])  # RGB order
        _render_composited(focused, targets, base_rgb, out_png, alpha=0.55)
        notes.append("Evidence rendered as semi-transparent overlay on satellite RGB.")
    except Exception as e:
        # Fallback to solid mask
        mask_to_png(focused, out_png)
        notes.append(f"Rendered as solid mask (RGB compositing failed: {e}).")

    legend = _legend(targets)

    # Coverage
    selected = np.isin(classified, list(targets))
    coverage = float(selected.sum()) / max(classified.size, 1)

    notes.insert(0, f"Query interpreted as '{mode}' analysis. Requested classes cover {coverage:.1%} of the scene.")
    notes.append("Thresholds derived from scene percentile distribution, not fixed global values.")
    notes.append("MNDWI (green-SWIR) used for water because green-NIR contrast collapses over turbid water.")

    if coverage < 0.01:
        notes.append("Very little of the scene matched the requested class.")

    # Caption
    captions = {
        "water": "Optical-SAR water agreement map",
        "builtup": "Optical-SAR built-up agreement map",
        "agreement": "Optical-SAR dual-signal agreement map",
    }
    caption = captions.get(mode, "Query-focused optical-SAR classification map")

    return ToolResult(
        answer=answer,
        evidence=Evidence(
            kind="mask",
            overlay_png=str(out_png),
            legend=legend,
            caption=caption,
        ),
        confidence=_confidence(classified, targets),
        confidence_method="query_conditioned_dual_signal_agreement",
        tool_name=TaskName.FUSION,
        model_id=MODEL_ID,
        params_used={"mode": mode, "decimate": decimate, **thresholds},
        notes=notes,
    )
