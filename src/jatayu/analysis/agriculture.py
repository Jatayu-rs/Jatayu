"""
Agricultural analysis for Jatayu.

Combines:
    - optical vegetation indices
    - moisture indices
    - SAR when available
    - rainfall
    - temperature

Designed for crop stress, sowing/fallow discrimination,
surface moisture change, and field-level screening.

This is a screening system, not a crop-yield model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jatayu.analysis.indices import IndexAnalyser


STRESS_HEALTHY = "healthy"
STRESS_WATCH = "watch"
STRESS_STRESSED = "stressed"
STRESS_SEVERE = "severe"

FIELD_SOWING = "prepared_for_sowing"
FIELD_FALLOW = "fallow"
FIELD_CROP = "standing_crop"
FIELD_WATERLOGGED = "waterlogged"


@dataclass(frozen=True)
class CropStressResult:
    ndvi: np.ndarray
    ndre: np.ndarray
    ndmi: np.ndarray
    lswi: np.ndarray

    stress_score: np.ndarray
    stress_class: np.ndarray

    mean_stress_score: float
    fraction_stressed: float

    confidence: float
    confidence_method: str


@dataclass(frozen=True)
class FieldPreparationResult:
    classification: np.ndarray

    fractions: dict[str, float]

    confidence: float
    confidence_method: str


def _zscore(
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

    mean = np.mean(
        values[finite]
    )

    std = np.std(
        values[finite]
    )

    if std < 1e-6:
        result[finite] = 0.0
    else:
        result[finite] = (
            values[finite] - mean
        ) / std

    result[~finite] = np.nan

    return result


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
        5,
    )

    high = np.percentile(
        values[finite],
        95,
    )

    if high <= low:
        result[finite] = 0.0
    else:
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


def analyse_crop_stress(
    bands: dict[str, np.ndarray],
    *,
    sar_moisture: np.ndarray | None = None,
    rainfall_mm: float | None = None,
    rainfall_baseline_mm: float | None = None,
    temperature_celsius: float | None = None,
    temperature_baseline_celsius: float | None = None,
) -> CropStressResult:
    """
    Estimate crop stress.

    Positive stress_score = healthier.

    Negative stress_score = more stressed.
    """

    analyser = IndexAnalyser()

    computed = analyser.compute(
        bands,
        [
            "NDVI",
            "NDRE",
            "NDMI",
            "LSWI",
        ],
    )

    ndvi = computed["NDVI"]
    ndre = computed["NDRE"]
    ndmi = computed["NDMI"]
    lswi = computed["LSWI"]

    valid = (
        np.isfinite(ndvi)
        & np.isfinite(ndre)
        & np.isfinite(ndmi)
        & np.isfinite(lswi)
    )

    # ------------------------------------------------------------
    # Relative optical signals
    # ------------------------------------------------------------

    ndvi_z = _zscore(ndvi)
    ndre_z = _zscore(ndre)
    ndmi_z = _zscore(ndmi)
    lswi_z = _zscore(lswi)

    # Vegetation and chlorophyll carry most weight.
    score = (
        0.35 * ndvi_z
        + 0.30 * ndre_z
        + 0.20 * ndmi_z
        + 0.15 * lswi_z
    )

    # ------------------------------------------------------------
    # SAR moisture
    # ------------------------------------------------------------

    if sar_moisture is not None:
        sar_moisture = np.asarray(
            sar_moisture,
            dtype=np.float32,
        )

        if sar_moisture.shape != score.shape:
            raise ValueError(
                "sar_moisture must have the same shape "
                "as optical indices."
            )

        sar_z = _zscore(
            sar_moisture
        )

        score += (
            0.15 * sar_z
        )

    # ------------------------------------------------------------
    # Rainfall context
    # ------------------------------------------------------------

    rainfall_signal = 0.0

    if (
        rainfall_mm is not None
        and rainfall_baseline_mm is not None
    ):
        rainfall_signal = (
            rainfall_mm
            - rainfall_baseline_mm
        ) / max(
            abs(rainfall_baseline_mm),
            1.0,
        )

        # Rainfall deficit increases stress.
        score -= np.float32(
            np.clip(
                rainfall_signal,
                -1.0,
                1.0,
            )
            * 0.20
        )

    # ------------------------------------------------------------
    # Temperature context
    # ------------------------------------------------------------

    if (
        temperature_celsius is not None
        and temperature_baseline_celsius is not None
    ):
        temperature_signal = (
            temperature_celsius
            - temperature_baseline_celsius
        )

        # Higher-than-baseline temperature increases stress.
        score -= np.float32(
            np.clip(
                temperature_signal / 10.0,
                -1.0,
                1.0,
            )
            * 0.20
        )

    score[~valid] = np.nan

    # ------------------------------------------------------------
    # Stress classes
    # ------------------------------------------------------------

    stress_class = np.full(
        score.shape,
        STRESS_WATCH,
        dtype=object,
    )

    stress_class[
        score >= 0.50
    ] = STRESS_HEALTHY

    stress_class[
        (score < 0.50)
        & (score >= -0.50)
    ] = STRESS_WATCH

    stress_class[
        (score < -0.50)
        & (score >= -1.00)
    ] = STRESS_STRESSED

    stress_class[
        score < -1.00
    ] = STRESS_SEVERE

    stress_class[
        ~valid
    ] = None

    finite_score = score[
        np.isfinite(score)
    ]

    mean_stress_score = (
        float(np.mean(finite_score))
        if finite_score.size
        else 0.0
    )

    stressed_pixels = (
        (stress_class == STRESS_STRESSED)
        | (stress_class == STRESS_SEVERE)
    )

    fraction_stressed = float(
        np.mean(
            stressed_pixels[valid]
        )
    ) if np.any(valid) else 0.0

    signals = 4

    if sar_moisture is not None:
        signals += 1

    if rainfall_mm is not None:
        signals += 1

    if temperature_celsius is not None:
        signals += 1

    confidence = round(
        min(
            0.95,
            0.45
            + signals * 0.06,
        ),
        2,
    )

    return CropStressResult(
        ndvi=ndvi,
        ndre=ndre,
        ndmi=ndmi,
        lswi=lswi,

        stress_score=score,
        stress_class=stress_class,

        mean_stress_score=mean_stress_score,
        fraction_stressed=fraction_stressed,

        confidence=confidence,
        confidence_method=(
            "multi_signal_crop_assessment_using_"
            "vegetation_chlorophyll_moisture_and_optional_weather_SAR"
        ),
    )


def classify_field_preparation(
    bands: dict[str, np.ndarray],
    *,
    sowing_ndvi_max: float = 0.35,
    sowing_bsi_min: float = 0.0,
    fallow_ndvi_max: float = 0.20,
    waterlogging_lswi_min: float = 0.30,
) -> FieldPreparationResult:
    """
    Distinguish:

        prepared_for_sowing
        fallow
        standing_crop
        waterlogged

    This is a spectral screening classifier.
    """

    analyser = IndexAnalyser()

    computed = analyser.compute(
        bands,
        [
            "NDVI",
            "BSI",
            "LSWI",
        ],
    )

    ndvi = computed["NDVI"]
    bsi = computed["BSI"]
    lswi = computed["LSWI"]

    valid = (
        np.isfinite(ndvi)
        & np.isfinite(bsi)
        & np.isfinite(lswi)
    )

    classification = np.full(
        ndvi.shape,
        FIELD_FALLOW,
        dtype=object,
    )

    # ------------------------------------------------------------
    # Standing crop
    # ------------------------------------------------------------

    standing_crop = (
        valid
        & (ndvi >= 0.35)
    )

    classification[
        standing_crop
    ] = FIELD_CROP

    # ------------------------------------------------------------
    # Waterlogged crop / fields
    # ------------------------------------------------------------

    waterlogged = (
        valid
        & ~standing_crop
        & (lswi >= waterlogging_lswi_min)
    )

    classification[
        waterlogged
    ] = FIELD_WATERLOGGED

    # ------------------------------------------------------------
    # Prepared soil
    # ------------------------------------------------------------

    prepared = (
        valid
        & ~standing_crop
        & ~waterlogged
        & (ndvi <= sowing_ndvi_max)
        & (bsi >= sowing_bsi_min)
    )

    classification[
        prepared
    ] = FIELD_SOWING

    # ------------------------------------------------------------
    # Fallow
    # ------------------------------------------------------------

    fallow = (
        valid
        & ~standing_crop
        & ~waterlogged
        & ~prepared
        & (ndvi <= fallow_ndvi_max)
    )

    classification[
        fallow
    ] = FIELD_FALLOW

    classification[
        ~valid
    ] = None

    # ------------------------------------------------------------
    # Fractions
    # ------------------------------------------------------------

    valid_count = np.count_nonzero(
        valid
    )

    fractions: dict[str, float] = {}

    if valid_count:
        for name in (
            FIELD_SOWING,
            FIELD_FALLOW,
            FIELD_CROP,
            FIELD_WATERLOGGED,
        ):
            fractions[name] = float(
                np.count_nonzero(
                    classification == name
                )
                / valid_count
            )

    confidence = round(
        min(
            0.90,
            0.55
            + 0.05
            * sum(
                fraction > 0
                for fraction
                in fractions.values()
            ),
        ),
        2,
    )

    return FieldPreparationResult(
        classification=classification,
        fractions=fractions,
        confidence=confidence,
        confidence_method=(
            "deterministic_NDVI_BSI_LSWI_field_preparation_rules"
        ),
    )


def summarise_crop_stress(
    result: CropStressResult,
) -> str:
    if result.mean_stress_score >= 0.5:
        status = "healthy"
    elif result.mean_stress_score >= -0.5:
        status = "under watch"
    elif result.mean_stress_score >= -1.0:
        status = "stressed"
    else:
        status = "severely stressed"

    return (
        f"The crop is classified as {status}. "
        f"Approximately {result.fraction_stressed * 100:.1f}% "
        f"of valid pixels show moderate-to-severe stress."
    )


__all__ = [
    "STRESS_HEALTHY",
    "STRESS_WATCH",
    "STRESS_STRESSED",
    "STRESS_SEVERE",
    "FIELD_SOWING",
    "FIELD_FALLOW",
    "FIELD_CROP",
    "FIELD_WATERLOGGED",
    "CropStressResult",
    "FieldPreparationResult",
    "analyse_crop_stress",
    "classify_field_preparation",
    "summarise_crop_stress",
]
