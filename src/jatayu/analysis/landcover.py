"""
Physics-first land-cover classification.

Uses spectral indices from Jatayu's index registry.

This is a deterministic prototype classifier, not a trained land-cover model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jatayu.analysis.indices import IndexAnalyser


CLASS_UNKNOWN = 0
CLASS_WATER = 1
CLASS_VEGETATION = 2
CLASS_CROPLAND = 3
CLASS_BUILT_UP = 4
CLASS_BARE_FALLOW = 5
CLASS_WETLAND = 6


CLASS_NAMES = {
    CLASS_UNKNOWN: "unknown",
    CLASS_WATER: "water",
    CLASS_VEGETATION: "vegetation",
    CLASS_CROPLAND: "cropland",
    CLASS_BUILT_UP: "built_up",
    CLASS_BARE_FALLOW: "bare_or_fallow",
    CLASS_WETLAND: "wetland",
}


@dataclass(frozen=True)
class LandCoverResult:
    classification: np.ndarray
    class_names: dict[int, str]

    fractions: dict[str, float]

    confidence: float
    confidence_method: str


def _normalise(
    values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=np.float32,
    )

    result = np.zeros_like(
        values,
        dtype=np.float32,
    )

    finite = np.isfinite(values)

    if not np.any(finite):
        result[:] = np.nan
        return result

    low = np.percentile(
        values[finite],
        2,
    )

    high = np.percentile(
        values[finite],
        98,
    )

    if high <= low:
        result[finite] = 0.0
        return result

    result[finite] = np.clip(
        (
            values[finite] - low
        )
        / (
            high - low
        ),
        0.0,
        1.0,
    )

    result[~finite] = np.nan

    return result


def classify_landcover(
    bands: dict[str, np.ndarray],
    *,
    water_threshold: float = 0.20,
    vegetation_threshold: float = 0.35,
    built_threshold: float = 0.25,
    wetness_threshold: float = 0.30,
) -> LandCoverResult:
    """
    Classify a multispectral scene.

    Required bands depend on the indices used:

        blue
        green
        red
        nir
        swir1
    """

    analyser = IndexAnalyser()

    computed = analyser.compute(
        bands,
        [
            "MNDWI",
            "NDVI",
            "LSWI",
            "NDBI",
            "BSI",
        ],
    )

    mndwi = computed["MNDWI"]
    ndvi = computed["NDVI"]
    lswi = computed["LSWI"]
    ndbi = computed["NDBI"]
    bsi = computed["BSI"]

    classification = np.full(
        mndwi.shape,
        CLASS_UNKNOWN,
        dtype=np.uint8,
    )

    valid = (
        np.isfinite(mndwi)
        & np.isfinite(ndvi)
        & np.isfinite(lswi)
        & np.isfinite(ndbi)
        & np.isfinite(bsi)
    )

    # ------------------------------------------------------------
    # Water
    # ------------------------------------------------------------

    water = (
        valid
        & (mndwi >= water_threshold)
        & (ndvi < vegetation_threshold)
    )

    classification[water] = CLASS_WATER

    # ------------------------------------------------------------
    # Wetland / waterlogged
    # ------------------------------------------------------------

    wetland = (
        valid
        & ~water
        & (lswi >= wetness_threshold)
        & (ndvi >= 0.15)
    )

    classification[wetland] = CLASS_WETLAND

    # ------------------------------------------------------------
    # Built-up
    # ------------------------------------------------------------

    built_up = (
        valid
        & ~water
        & ~wetland
        & (ndbi >= built_threshold)
        & (ndvi < vegetation_threshold)
    )

    classification[built_up] = CLASS_BUILT_UP

    # ------------------------------------------------------------
    # Vegetation
    # ------------------------------------------------------------

    vegetation = (
        valid
        & ~water
        & ~wetland
        & ~built_up
        & (ndvi >= vegetation_threshold)
    )

    classification[vegetation] = CLASS_VEGETATION

    # ------------------------------------------------------------
    # Cropland
    # ------------------------------------------------------------

    cropland = (
        vegetation
        & (
            (lswi > 0.0)
            | (bsi < 0.15)
        )
    )

    classification[cropland] = CLASS_CROPLAND

    # ------------------------------------------------------------
    # Bare / fallow
    # ------------------------------------------------------------

    bare_fallow = (
        valid
        & ~water
        & ~wetland
        & ~built_up
        & ~cropland
        & (bsi >= 0.0)
    )

    classification[
        bare_fallow
    ] = CLASS_BARE_FALLOW

    # ------------------------------------------------------------
    # Fractions
    # ------------------------------------------------------------

    valid_pixels = np.count_nonzero(
        classification != CLASS_UNKNOWN
    )

    fractions: dict[str, float] = {}

    if valid_pixels > 0:
        for class_id, name in CLASS_NAMES.items():
            if class_id == CLASS_UNKNOWN:
                continue

            count = np.count_nonzero(
                classification == class_id
            )

            fractions[name] = float(
                count / valid_pixels
            )

    class_count = len(
        [
            value
            for value in fractions.values()
            if value > 0
        ]
    )

    confidence = round(
        min(
            0.90,
            0.55
            + 0.05 * class_count,
        ),
        2,
    )

    return LandCoverResult(
        classification=classification,
        class_names=CLASS_NAMES.copy(),
        fractions=fractions,
        confidence=confidence,
        confidence_method=(
            "deterministic_multi_index_landcover_rules"
        ),
    )


def class_mask(
    result: LandCoverResult,
    class_name: str,
) -> np.ndarray:
    """
    Extract one class mask by name.
    """

    matches = [
        class_id
        for class_id, name
        in result.class_names.items()
        if name == class_name
    ]

    if not matches:
        raise KeyError(
            f"Unknown land-cover class: {class_name}"
        )

    return (
        result.classification
        == matches[0]
    )


def summarise_landcover(
    result: LandCoverResult,
) -> str:
    """
    Human-readable land-cover summary.
    """

    if not result.fractions:
        return (
            "No valid land-cover classes "
            "could be identified."
        )

    ordered = sorted(
        result.fractions.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    parts = [
        f"{name.replace('_', ' ')} "
        f"{fraction * 100:.1f}%"
        for name, fraction
        in ordered
        if fraction > 0.01
    ]

    return (
        "Land-cover composition: "
        + ", ".join(parts)
        + "."
    )


__all__ = [
    "CLASS_UNKNOWN",
    "CLASS_WATER",
    "CLASS_VEGETATION",
    "CLASS_CROPLAND",
    "CLASS_BUILT_UP",
    "CLASS_BARE_FALLOW",
    "CLASS_WETLAND",
    "CLASS_NAMES",
    "LandCoverResult",
    "classify_landcover",
    "class_mask",
    "summarise_landcover",
]
