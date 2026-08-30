"""Optical-SAR cross-modal analysis.

The fusion specialist combines independent optical and SAR evidence.

Unlike a fixed classifier, the specialist also interprets the user's query and
focuses the returned evidence on the classes actually requested.

Physics:

* Water is generally radar-dark because a smooth water surface reflects the
  radar pulse away from the sensor.
* Built-up surfaces can be radar-bright because walls and ground form strong
  double-bounce / corner-reflector responses.
* Vegetation can be spectrally identified with NDVI.
* Radar-bright + spectrally vegetated areas can indicate flooded vegetation or
  dense canopy volume scattering.
* Radar-dark surfaces that are not spectrally water-like are kept separate from
  actual water.

MNDWI (green-SWIR) is used instead of NDWI because turbid water can reduce the
usefulness of green-NIR contrast.
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

# ---------------------------------------------------------------------------
# Classification IDs
# ---------------------------------------------------------------------------

CLS_UNCLASSIFIED = 0
CLS_WATER = 1
CLS_BUILTUP = 2
CLS_FLOODED_VEG = 3
CLS_SMOOTH_BARE = 4


# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------

DEFAULTS = {
    "sar_water_db": -18.0,
    "sar_builtup_db": -2.0,
    "mndwi_threshold": 0.10,
    "ndbi_threshold": 0.06,
    "ndvi_threshold": 0.30,
}

MNDWI_FLOOR = 0.0


# ---------------------------------------------------------------------------
# Full legend
# ---------------------------------------------------------------------------

FULL_LEGEND = {
    "0": "unclassified",
    "1": "open water (radar and optical agree)",
    "2": "built-up (radar and optical agree)",
    "3": (
        "flooded vegetation or dense canopy "
        "(radar-bright, spectrally vegetated)"
    ),
    "4": (
        "smooth bare surface or radar shadow "
        "(radar-dark, not water)"
    ),
}


# ---------------------------------------------------------------------------
# Query interpretation
# ---------------------------------------------------------------------------


def _query_targets(query: str) -> tuple[set[int], str]:
    """Interpret the user's natural-language request.

    The Fusion tool remains deterministic: no LLM is required.

    Returns:
        (target_classes, analysis_mode)

    Modes:
        all
        water
        builtup
        vegetation
        bare
        agreement
        disagreement
    """

    q = query.lower()

    targets: set[int] = set()

    # Water ---------------------------------------------------------------
    water_terms = (
        "water",
        "river",
        "lake",
        "pond",
        "wetland",
        "water body",
        "waterbody",
        "flood",
        "flooded",
    )

    # Built-up ------------------------------------------------------------
    built_terms = (
        "built-up",
        "built up",
        "builtup",
        "urban",
        "building",
        "buildings",
        "construction",
        "settlement",
        "city",
        "urban area",
    )

    # Vegetation ----------------------------------------------------------
    vegetation_terms = (
        "vegetation",
        "vegetated",
        "forest",
        "trees",
        "tree cover",
        "crop",
        "cropland",
        "greenery",
        "canopy",
    )

    # Bare / shadow -------------------------------------------------------
    bare_terms = (
        "bare",
        "bare soil",
        "soil",
        "sand",
        "sandbar",
        "shadow",
    )

    # Agreement / joint evidence ----------------------------------------
    agreement_terms = (
        "agreement",
        "agree",
        "both modalities",
        "both sensors",
        "optical and sar",
        "optical + sar",
        "jointly",
        "joint",
        "confirmed by both",
        "supported by both",
    )

    disagreement_terms = (
        "disagreement",
        "disagree",
        "uncertainty",
        "uncertain",
        "ambiguous",
        "mismatch",
        "mismatches",
        "difference between",
    )

    if any(term in q for term in water_terms):
        targets.add(CLS_WATER)

    if any(term in q for term in built_terms):
        targets.add(CLS_BUILTUP)

    if any(term in q for term in vegetation_terms):
        targets.add(CLS_FLOODED_VEG)

    if any(term in q for term in bare_terms):
        targets.add(CLS_SMOOTH_BARE)

    # Explicit disagreement requests.
    if any(term in q for term in disagreement_terms):
        targets.update(
            {
                CLS_FLOODED_VEG,
                CLS_SMOOTH_BARE,
            }
        )

    # "Both modalities", "jointly", etc. means the user wants the
    # cross-modal agreement findings rather than a single class.
    if any(term in q for term in agreement_terms):
        if not targets:
            targets.update(
                {
                    CLS_WATER,
                    CLS_BUILTUP,
                }
            )

    # If nothing specific was requested, preserve the full analysis.
    if not targets:
        return (
            {
                CLS_WATER,
                CLS_BUILTUP,
                CLS_FLOODED_VEG,
                CLS_SMOOTH_BARE,
            },
            "all",
        )

    if targets == {CLS_WATER}:
        return targets, "water"

    if targets == {CLS_BUILTUP}:
        return targets, "builtup"

    if targets == {CLS_FLOODED_VEG}:
        return targets, "vegetation"

    if targets == {CLS_SMOOTH_BARE}:
        return targets, "bare"

    if targets == {CLS_WATER, CLS_BUILTUP}:
        return targets, "agreement"

    if targets.issubset(
        {
            CLS_FLOODED_VEG,
            CLS_SMOOTH_BARE,
        }
    ):
        return targets, "disagreement"

    return targets, "custom"


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


def normalised_difference(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Compute (a-b)/(a+b), safely handling zero denominators."""

    a = a.astype(np.float32)
    b = b.astype(np.float32)

    denom = a + b

    out = np.zeros_like(denom, dtype=np.float32)

    np.divide(
        a - b,
        denom,
        out=out,
        where=np.abs(denom) > 1e-6,
    )

    return out


def adaptive_thresholds(
    sar_db: np.ndarray,
    mndwi: np.ndarray,
    ndbi: np.ndarray,
    ndvi: np.ndarray,
) -> dict[str, float]:
    """Derive thresholds from the current scene."""

    return {
        "sar_water_db": float(
            np.nanpercentile(sar_db, 5)
        ),
        "sar_builtup_db": float(
            np.nanpercentile(sar_db, 90)
        ),
        "mndwi_threshold": max(
            MNDWI_FLOOR,
            float(np.nanpercentile(mndwi, 90)),
        ),
        "ndbi_threshold": float(
            np.nanpercentile(ndbi, 75)
        ),
        "ndvi_threshold": float(
            np.nanpercentile(ndvi, 60)
        ),
    }


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------


def classify(
    *,
    sar_db: np.ndarray,
    mndwi: np.ndarray,
    ndbi: np.ndarray,
    ndvi: np.ndarray,
    thresholds: dict[str, float] | None = None,
) -> np.ndarray:
    """Create the physical optical-SAR classification map."""

    t = {
        **DEFAULTS,
        **(thresholds or {}),
    }

    shapes = {
        sar_db.shape,
        mndwi.shape,
        ndbi.shape,
        ndvi.shape,
    }

    if len(shapes) != 1:
        raise ValueError(
            f"All arrays must share a shape; got {shapes}"
        )

    sar_dark = sar_db < t["sar_water_db"]
    sar_bright = sar_db > t["sar_builtup_db"]

    spec_water = mndwi > t["mndwi_threshold"]
    spec_built = ndbi > t["ndbi_threshold"]
    spec_veg = ndvi > t["ndvi_threshold"]

    out = np.full(
        sar_db.shape,
        CLS_UNCLASSIFIED,
        dtype=np.uint8,
    )

    # Radar-dark but not spectrally water-like.
    out[
        sar_dark & ~spec_water
    ] = CLS_SMOOTH_BARE

    # Radar-bright + spectrally vegetated.
    out[
        sar_bright & spec_veg
    ] = CLS_FLOODED_VEG

    # Radar-bright + spectrally built-up.
    out[
        sar_bright & spec_built
    ] = CLS_BUILTUP

    # Radar-dark + optical water.
    out[
        sar_dark & spec_water
    ] = CLS_WATER

    invalid = ~(
        np.isfinite(sar_db)
        & np.isfinite(mndwi)
        & np.isfinite(ndbi)
        & np.isfinite(ndvi)
    )

    out[invalid] = CLS_UNCLASSIFIED

    return out


# ---------------------------------------------------------------------------
# Query-focused evidence
# ---------------------------------------------------------------------------


def focus_classification(
    classified: np.ndarray,
    targets: set[int],
) -> np.ndarray:
    """Hide classes that were not requested by the user.

    This is the key difference from the old implementation.

    The physical classification is still computed for the entire scene, but
    the evidence returned to the frontend is focused on the user's request.
    """

    focused = np.full_like(
        classified,
        CLS_UNCLASSIFIED,
    )

    for cls in targets:
        focused[classified == cls] = cls

    return focused


# ---------------------------------------------------------------------------
# Dynamic legend
# ---------------------------------------------------------------------------


def focused_legend(
    targets: set[int],
) -> dict[str, str]:
    """Create a legend containing only classes relevant to the query."""

    legend = {
        "0": "not selected / insufficient evidence",
    }

    for cls in sorted(targets):
        legend[str(cls)] = FULL_LEGEND[str(cls)]

    return legend


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarise(
    classified: np.ndarray,
    targets: set[int],
    mode: str,
) -> tuple[str, dict[str, float]]:
    """Create a query-specific answer."""

    total = classified.size

    if total == 0:
        return (
            "The input scene contains no pixels to analyse.",
            {},
        )

    fractions = {
        FULL_LEGEND[str(c)]: float(
            (classified == c).sum()
        )
        / total
        for c in (
            CLS_WATER,
            CLS_BUILTUP,
            CLS_FLOODED_VEG,
            CLS_SMOOTH_BARE,
        )
    }

    water = fractions[FULL_LEGEND["1"]] * 100
    built = fractions[FULL_LEGEND["2"]] * 100
    flooded = fractions[FULL_LEGEND["3"]] * 100
    bare = fractions[FULL_LEGEND["4"]] * 100

    # ---------------------------------------------------------------
    # Query-specific responses
    # ---------------------------------------------------------------

    if mode == "water":
        answer = (
            f"The optical-SAR fusion identifies {water:.1f}% of the "
            "scene as open water. These pixels satisfy both the "
            "spectral water criterion (MNDWI) and radar-dark criterion, "
            "providing independent evidence from both modalities."
        )

    elif mode == "builtup":
        answer = (
            f"The optical-SAR fusion identifies {built:.1f}% of the "
            "scene as built-up. These pixels show strong radar "
            "backscatter together with elevated NDBI, giving "
            "cross-modal support for built-up surfaces."
        )

    elif mode == "vegetation":
        answer = (
            f"The fusion identifies {flooded:.1f}% of the scene as "
            "radar-bright, spectrally vegetated surface. This pattern "
            "can indicate flooded vegetation or dense canopy volume "
            "scattering that optical imagery alone cannot distinguish."
        )

    elif mode == "bare":
        answer = (
            f"The fusion identifies {bare:.1f}% of the scene as "
            "radar-dark but not optically water-like. These regions "
            "are classified as smooth bare surface or radar shadow "
            "rather than open water."
        )

    elif mode == "agreement":
        answer = (
            f"Joint optical-SAR analysis finds {water:.1f}% open water "
            f"and {built:.1f}% built-up surface where the two modalities "
            "provide supporting evidence. Water is supported by high "
            "MNDWI and low radar backscatter, while built-up regions "
            "are supported by high NDBI and strong radar return."
        )

    elif mode == "disagreement":
        answer = (
            f"The fusion identifies {flooded:.1f}% radar-bright "
            "spectrally vegetated surface and "
            f"{bare:.1f}% radar-dark non-water surface. These are "
            "the principal non-simple-agreement classes in the "
            "cross-modal interpretation."
        )

    else:
        answer = (
            f"Combining optical and SAR evidence, {water:.1f}% of the "
            f"scene is open water and {built:.1f}% is built-up. "
            f"Another {flooded:.1f}% is radar-bright but spectrally "
            "vegetated, while {bare:.1f}% is radar-dark and "
            "non-water-like."
        )

    return answer, fractions


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def _split_by_modality(
    images: list[ImageRef],
) -> tuple[ImageRef, ImageRef]:

    optical = [
        i
        for i in images
        if i.modality.is_optical_family
    ]

    sar = [
        i
        for i in images
        if i.modality is Modality.SAR
    ]

    if len(optical) != 1 or len(sar) != 1:
        raise ValueError(
            "Fusion needs exactly one optical and one SAR image; "
            f"got {len(optical)} optical and {len(sar)} SAR."
        )

    return optical[0], sar[0]


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def agreement_confidence(
    classified: np.ndarray,
    targets: set[int],
) -> float:
    """Estimate confidence for the requested evidence.

    Confidence is based on how much of the requested classification has
    coherent physical support, rather than blindly returning the same value
    for every query.
    """

    selected = np.isin(
        classified,
        list(targets),
    )

    n_selected = int(selected.sum())

    if n_selected == 0:
        return 0.0

    # Water and built-up are direct dual-signal agreement classes.
    direct_agreement = np.isin(
        classified,
        [CLS_WATER, CLS_BUILTUP],
    )

    direct = int(
        (selected & direct_agreement).sum()
    )

    # Vegetation/bare are weaker contextual classes.
    contextual = n_selected - direct

    score = (
        direct / n_selected
        if n_selected
        else 0.0
    )

    if contextual:
        score = max(
            score,
            0.65,
        )

    # Keep confidence bounded.
    return float(
        max(
            0.0,
            min(
                1.0,
                0.15 + 0.85 * score,
            ),
        )
    )


# ---------------------------------------------------------------------------
# Registered specialist
# ---------------------------------------------------------------------------


@register(
    TaskName.FUSION,
    families={TaskFamily.CROSS_MODAL},
    description=(
        "Combines a co-registered optical image and a SAR image "
        "to identify surface types using independent physical "
        "evidence. Query-aware for water, built-up areas, "
        "vegetation, bare surfaces and cross-modal agreement."
    ),
)
def run(req: ToolRequest) -> ToolResult:

    optical, sar = _split_by_modality(req.images)

    decimate = int(
        req.params.get(
            "decimate",
            2,
        )
    )

    # ---------------------------------------------------------------
    # Interpret query
    # ---------------------------------------------------------------

    targets, mode = _query_targets(req.query)

    # ---------------------------------------------------------------
    # Read imagery
    # ---------------------------------------------------------------

    green, red, nir, swir = read_bands(
        optical,
        [
            "green",
            "red",
            "nir",
            "swir1",
        ],
        decimate=decimate,
    )

    (backscatter,) = read_bands(
        sar,
        ["vv"],
        decimate=decimate,
    )

    notes: list[str] = []

    # ---------------------------------------------------------------
    # SAR calibration
    # ---------------------------------------------------------------

    finite = backscatter[
        np.isfinite(backscatter)
    ]

    if finite.size and finite.min() >= 0:

        sar_db = (
            10.0
            * np.log10(
                np.where(
                    backscatter > 0,
                    backscatter,
                    np.nan,
                )
            )
        )

        notes.append(
            "SAR appeared to be in linear power units "
            "and was converted to dB."
        )

    else:
        sar_db = backscatter

    # ---------------------------------------------------------------
    # Optical indices
    # ---------------------------------------------------------------

    mndwi = normalised_difference(
        green,
        swir,
    )

    ndbi = normalised_difference(
        swir,
        nir,
    )

    ndvi = normalised_difference(
        nir,
        red,
    )

    # ---------------------------------------------------------------
    # Adaptive thresholds
    # ---------------------------------------------------------------

    thresholds = adaptive_thresholds(
        sar_db,
        mndwi,
        ndbi,
        ndvi,
    )

    thresholds.update(
        {
            k: v
            for k, v in req.params.items()
            if k in DEFAULTS
        }
    )

    # ---------------------------------------------------------------
    # Full physical classification
    # ---------------------------------------------------------------

    classified = classify(
        sar_db=sar_db,
        mndwi=mndwi,
        ndbi=ndbi,
        ndvi=ndvi,
        thresholds=thresholds,
    )

    # ---------------------------------------------------------------
    # Focus output according to user's query
    # ---------------------------------------------------------------

    focused = focus_classification(
        classified,
        targets,
    )

    sentence, fractions = summarise(
        classified,
        targets,
        mode,
    )

    # ---------------------------------------------------------------
    # Evidence
    # ---------------------------------------------------------------

    out_png = (
        Path("outputs")
        / f"fusion_{uuid4().hex[:8]}.png"
    )

    mask_to_png(
        focused,
        out_png,
    )

    legend = focused_legend(
        targets,
    )

    # ---------------------------------------------------------------
    # Coverage
    # ---------------------------------------------------------------

    selected_mask = np.isin(
        classified,
        list(targets),
    )

    n_selected = int(
        selected_mask.sum()
    )

    coverage = (
        n_selected / classified.size
        if classified.size
        else 0.0
    )

    notes.insert(
        0,
        (
            f"Query interpreted as '{mode}' analysis. "
            f"Requested classes cover {coverage:.1%} of the scene."
        ),
    )

    notes.append(
        "Thresholds were derived from the current scene's "
        "percentile distribution rather than fixed global values."
    )

    notes.append(
        "MNDWI (green-SWIR) is used for water detection because "
        "green-NIR contrast can collapse over turbid water."
    )

    if coverage < 0.01:
        notes.append(
            "Very little of the scene matched the requested class; "
            "the result may therefore contain sparse evidence."
        )

    # ---------------------------------------------------------------
    # Query-aware caption
    # ---------------------------------------------------------------

    if mode == "water":
        caption = (
            "Optical-SAR water agreement mask"
        )

    elif mode == "builtup":
        caption = (
            "Optical-SAR built-up agreement mask"
        )

    elif mode == "vegetation":
        caption = (
            "Optical-SAR vegetated radar-bright mask"
        )

    elif mode == "bare":
        caption = (
            "Optical-SAR bare surface / radar-shadow mask"
        )

    elif mode == "agreement":
        caption = (
            "Optical-SAR dual-signal agreement map"
        )

    elif mode == "disagreement":
        caption = (
            "Optical-SAR non-simple-agreement map"
        )

    else:
        caption = (
            "Query-focused optical-SAR classification map"
        )

    # ---------------------------------------------------------------
    # Final result
    # ---------------------------------------------------------------

    return ToolResult(
        answer=sentence,
        evidence=Evidence(
            kind="mask",
            overlay_png=str(out_png),
            legend=legend,
            caption=caption,
        ),
        confidence=agreement_confidence(
            classified,
            targets,
        ),
        confidence_method=(
            "query_conditioned_dual_signal_agreement"
        ),
        tool_name=TaskName.FUSION,
        model_id=MODEL_ID,
        notes=notes,
    )
