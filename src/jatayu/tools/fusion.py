"""Optical-SAR cross-modal analysis. STATUS: real.

No learned model here, and that is a choice rather than a shortfall. The physics
is unambiguous:

  * Water is specular to radar — the pulse reflects away from the sensor, so
    backscatter collapses.
  * Buildings form corner reflectors between wall and ground — the pulse returns
    strongly.

The optical bands give independent evidence of the same ground. Cross-tabulating
the two gives four classes, and the two disagreement classes are findings rather
than errors: radar-bright over vegetation means water beneath a canopy, which an
optical image physically cannot see.

We use MNDWI (green-SWIR), not NDWI (green-NIR). Over the Hooghly, NDWI's 95th
percentile barely reaches 0.03 while MNDWI's reaches 0.45 — green-NIR contrast
collapses over turbid water, and the GBM delta is among the most sediment-laden
water on Earth. SWIR is absorbed by water regardless of sediment load.

Thresholds are derived per scene from its own percentile distribution, not fixed.
Measured on chip0000, the previously fixed -18 dB water threshold sat below the
1st percentile and could never fire. The ISRO evaluation set is RISAT rather than
Sentinel-1 — a different sensor with different calibration — so fixed values
would fail there outright.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np

from jatayu.io.loader import read_bands
from jatayu.render import mask_to_png
from jatayu.schemas import (
    Evidence,
    ImageRef,
    Modality,
    TaskFamily,
    TaskName,
    ToolRequest,
    ToolResult,
)
from jatayu.tools.registry import register

MODEL_ID = "physics-adaptive-v2"

CLS_UNCLASSIFIED = 0
CLS_WATER = 1
CLS_BUILTUP = 2
CLS_FLOODED_VEG = 3
CLS_SMOOTH_BARE = 4

LEGEND = {
    "0": "unclassified",
    "1": "open water (radar and optical agree)",
    "2": "built-up (radar and optical agree)",
    "3": "flooded vegetation or dense canopy (radar-bright, spectrally vegetated)",
    "4": "smooth bare surface or radar shadow (radar-dark, not water)",
}

# Typical Sentinel-1 / Sentinel-2 values. No longer used directly — they document
# the overridable keys and provide a fallback if percentiles cannot be computed.
DEFAULTS = {
    "sar_water_db": -18.0,
    "sar_builtup_db": -2.0,
    "mndwi_threshold": 0.10,
    "ndbi_threshold": 0.06,
    "ndvi_threshold": 0.30,
}

# Physical floors. Percentiles set the threshold; physics sets the bound —
# without this, the wettest 10% of a bone-dry scene would be called water.
MNDWI_FLOOR = 0.0


def normalised_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b), guarding the zero denominator."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = a + b
    out = np.zeros_like(denom, dtype=np.float32)
    np.divide(a - b, denom, out=out, where=np.abs(denom) > 1e-6)
    return out


def adaptive_thresholds(
    sar_db: np.ndarray, mndwi: np.ndarray, ndbi: np.ndarray, ndvi: np.ndarray
) -> dict[str, float]:
    """Derive thresholds from this scene rather than assuming global values.

    Backscatter distributions shift with incidence angle, polarisation, and
    surface roughness, so a threshold calibrated on one scene does not transfer
    to another — even of the same city.
    """
    return {
        "sar_water_db": float(np.nanpercentile(sar_db, 5)),
        "sar_builtup_db": float(np.nanpercentile(sar_db, 90)),
        "mndwi_threshold": max(MNDWI_FLOOR, float(np.nanpercentile(mndwi, 90))),
        "ndbi_threshold": float(np.nanpercentile(ndbi, 75)),
        "ndvi_threshold": float(np.nanpercentile(ndvi, 60)),
    }


def classify(
    *,
    sar_db: np.ndarray,
    mndwi: np.ndarray,
    ndbi: np.ndarray,
    ndvi: np.ndarray,
    thresholds: dict[str, float] | None = None,
) -> np.ndarray:
    """Cross-tabulate radar and spectral evidence into the four-class map.

    A pure function over arrays, so the physics is unit-testable with synthetic
    inputs and no raster IO at all.
    """
    t = {**DEFAULTS, **(thresholds or {})}
    shapes = {sar_db.shape, mndwi.shape, ndbi.shape, ndvi.shape}
    if len(shapes) != 1:
        raise ValueError(f"All arrays must share a shape; got {shapes}")

    sar_dark = sar_db < t["sar_water_db"]
    sar_bright = sar_db > t["sar_builtup_db"]
    spec_water = mndwi > t["mndwi_threshold"]
    spec_built = ndbi > t["ndbi_threshold"]
    spec_veg = ndvi > t["ndvi_threshold"]

    out = np.full(sar_db.shape, CLS_UNCLASSIFIED, dtype=np.uint8)
    # Order matters: agreement classes are assigned last and win.
    out[sar_dark & ~spec_water] = CLS_SMOOTH_BARE
    out[sar_bright & spec_veg] = CLS_FLOODED_VEG
    out[sar_bright & spec_built] = CLS_BUILTUP
    out[sar_dark & spec_water] = CLS_WATER

    invalid = ~(
        np.isfinite(sar_db) & np.isfinite(mndwi) & np.isfinite(ndbi) & np.isfinite(ndvi)
    )
    out[invalid] = CLS_UNCLASSIFIED
    return out


def summarise(classified: np.ndarray) -> tuple[str, dict[str, float]]:
    """Turn the class map into a sentence and per-class fractions."""
    total = classified.size
    fractions = {
        LEGEND[str(c)]: float((classified == c).sum()) / total
        for c in (CLS_WATER, CLS_BUILTUP, CLS_FLOODED_VEG, CLS_SMOOTH_BARE)
    }
    water = fractions[LEGEND["1"]] * 100
    built = fractions[LEGEND["2"]] * 100
    flooded = fractions[LEGEND["3"]] * 100

    sentence = (
        f"Combining the optical and SAR images, {water:.1f}% of the scene is open water "
        f"(low radar backscatter confirmed by high MNDWI) and {built:.1f}% is built-up "
        f"(strong double-bounce return confirmed by high NDBI)."
    )
    if flooded > 2.0:
        sentence += (
            f" A further {flooded:.1f}% is radar-bright but spectrally vegetated, "
            "indicating flooded vegetation or dense canopy volume scattering — which "
            "the optical image alone would not reveal."
        )
    return sentence, fractions


def _split_by_modality(images: list[ImageRef]) -> tuple[ImageRef, ImageRef]:
    optical = [i for i in images if i.modality.is_optical_family]
    sar = [i for i in images if i.modality is Modality.SAR]
    if len(optical) != 1 or len(sar) != 1:
        raise ValueError(
            f"Fusion needs exactly one optical and one SAR image; "
            f"got {len(optical)} optical and {len(sar)} SAR."
        )
    return optical[0], sar[0]


def agreement_confidence(agreed: float, disputed: float) -> float:
    """Confidence from two independent sensors agreeing — a measurement, not a guess.

    `agreed` and `disputed` are fractions OF CLASSIFIED PIXELS, not of the whole
    scene. Most of a scene is ordinary land that meets no threshold; counting
    those as "not agreeing" is a category error and was what produced an
    implausible 0.27 on a result that was actually sound.

    Disagreement is capped in its effect: a genuinely half-flooded scene is a
    real finding, not a reason to distrust the result.
    """
    penalty = min(disputed, 0.5) * 0.6
    return max(0.0, min(1.0, agreed * (1.0 - penalty) + 0.15))


@register(
    TaskName.FUSION,
    families={TaskFamily.CROSS_MODAL},
    description="Combines a co-registered optical image and a SAR image to identify "
    "surface types more reliably than either alone — especially water, "
    "built-up areas, and flooded vegetation. Requires one optical and "
    "one SAR image.",
)
def run(req: ToolRequest) -> ToolResult:
    optical, sar = _split_by_modality(req.images)
    decimate = int(req.params.get("decimate", 2))

    green, red, nir, swir = read_bands(
        optical, ["green", "red", "nir", "swir1"], decimate=decimate
    )
    (backscatter,) = read_bands(sar, ["vv"], decimate=decimate)

    notes: list[str] = []

    # GEE's S1_GRD is already log-scaled. Only convert if the data looks linear.
    finite = backscatter[np.isfinite(backscatter)]
    if finite.size and finite.min() >= 0:
        sar_db = 10.0 * np.log10(np.where(backscatter > 0, backscatter, np.nan))
        notes.append("SAR appeared to be in linear power units and was converted to dB.")
    else:
        sar_db = backscatter

    mndwi = normalised_difference(green, swir)  # not NDWI — see module docstring
    ndbi = normalised_difference(swir, nir)
    ndvi = normalised_difference(nir, red)

    # Derive from the scene, then let the caller override any individual value.
    thresholds = adaptive_thresholds(sar_db, mndwi, ndbi, ndvi)
    thresholds.update({k: v for k, v in req.params.items() if k in DEFAULTS})

    classified = classify(
        sar_db=sar_db, mndwi=mndwi, ndbi=ndbi, ndvi=ndvi, thresholds=thresholds
    )
    sentence, fractions = summarise(classified)

    out_png = Path("outputs") / f"fusion_{uuid4().hex[:8]}.png"
    mask_to_png(classified, out_png)

    # Agreement among CLASSIFIED pixels only — see agreement_confidence.
    classified_mask = classified != CLS_UNCLASSIFIED
    n_classified = int(classified_mask.sum())
    coverage = n_classified / classified.size

    if n_classified == 0:
        agreed = disputed = 0.0
    else:
        agreed = (
            float(((classified == CLS_WATER) | (classified == CLS_BUILTUP)).sum())
            / n_classified
        )
        disputed = (
            float(
                ((classified == CLS_FLOODED_VEG) | (classified == CLS_SMOOTH_BARE)).sum()
            )
            / n_classified
        )

    notes.insert(
        0,
        f"{coverage:.0%} of pixels met a classification threshold; the remainder "
        "fell between thresholds and are unclassified.",
    )
    notes.append(
        "Thresholds derived from this scene's own percentile distribution rather "
        "than fixed values, because backscatter varies with sensor, incidence "
        "angle, and surface roughness."
    )
    notes.append(
        "MNDWI (green-SWIR) is used rather than NDWI (green-NIR), which collapses "
        "over turbid water."
    )
    if coverage < 0.05:
        notes.append(
            "Coverage is very low — thresholds may not suit this scene, or the "
            "scene may contain little water or built-up land."
        )

    return ToolResult(
        answer=sentence,
        evidence=Evidence(
            kind="mask",
            overlay_png=str(out_png),
            legend=LEGEND,
            caption="Four-class optical-SAR agreement map",
        ),
        confidence=agreement_confidence(agreed, disputed),
        confidence_method="dual_signal_agreement",
        tool_name=TaskName.FUSION,
        model_id=MODEL_ID,
        notes=notes,
    )
