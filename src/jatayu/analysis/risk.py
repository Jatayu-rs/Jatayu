"""
Composite environmental risk analysis for Jatayu.

Combines independently derived physical signals.

Supported demo products:

    1. agricultural land conversion / recharge risk
    2. flood / waterlogging risk
    3. coastal salinity risk
    4. crop stress risk

All calculations are deterministic NumPy operations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


RISK_LOW = "low"
RISK_MODERATE = "moderate"
RISK_HIGH = "high"
RISK_CRITICAL = "critical"


@dataclass(frozen=True)
class RiskResult:
    score: np.ndarray
    level: np.ndarray

    mean_score: float
    high_risk_fraction: float

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


def _classify(
    score: np.ndarray,
) -> np.ndarray:
    level = np.full(
        score.shape,
        RISK_LOW,
        dtype=object,
    )

    level[
        (score >= 0.30)
        & (score < 0.60)
    ] = RISK_MODERATE

    level[
        (score >= 0.60)
        & (score < 0.80)
    ] = RISK_HIGH

    level[
        score >= 0.80
    ] = RISK_CRITICAL

    level[
        ~np.isfinite(score)
    ] = None

    return level


def _finish(
    score: np.ndarray,
    *,
    signals: int,
    method: str,
) -> RiskResult:
    score = np.asarray(
        score,
        dtype=np.float32,
    )

    level = _classify(
        score
    )

    finite = np.isfinite(score)

    if np.any(finite):
        mean_score = float(
            np.mean(
                score[finite]
            )
        )

        high_risk_fraction = float(
            np.mean(
                score[finite] >= 0.60
            )
        )
    else:
        mean_score = 0.0
        high_risk_fraction = 0.0

    confidence = round(
        min(
            0.95,
            0.35
            + signals * 0.10,
        ),
        2,
    )

    return RiskResult(
        score=score,
        level=level,
        mean_score=mean_score,
        high_risk_fraction=high_risk_fraction,
        confidence=confidence,
        confidence_method=method,
    )


def agricultural_recharge_risk(
    *,
    built_up_expansion: np.ndarray,
    agricultural_mask: np.ndarray,
    soil_infiltration: np.ndarray | None = None,
    waterlogging: np.ndarray | None = None,
) -> RiskResult:
    """
    Estimate where agricultural land conversion may compromise
    shallow-aquifer recharge.

    Inputs are expected to be normalised or convertible to [0, 1].

    Concept:

        built-up expansion
             × agricultural land
             × low infiltration
             + persistent waterlogging

    This is a RECHARGE-RISK PROXY.

    It does not directly observe groundwater or aquifer recharge.
    """

    expansion = _normalise(
        built_up_expansion
    )

    agriculture = np.asarray(
        agricultural_mask,
        dtype=np.float32,
    )

    if expansion.shape != agriculture.shape:
        raise ValueError(
            "built_up_expansion and agricultural_mask "
            "must have identical shapes."
        )

    score = (
        expansion
        * np.clip(
            agriculture,
            0.0,
            1.0,
        )
    )

    signals = 2

    if soil_infiltration is not None:
        infiltration = _normalise(
            soil_infiltration
        )

        if infiltration.shape != score.shape:
            raise ValueError(
                "soil_infiltration must match input shape."
            )

        # Low infiltration increases recharge risk.
        score += (
            0.30
            * (1.0 - infiltration)
            * agriculture
        )

        signals += 1

    if waterlogging is not None:
        waterlogging = _normalise(
            waterlogging
        )

        if waterlogging.shape != score.shape:
            raise ValueError(
                "waterlogging must match input shape."
            )

        score += (
            0.15
            * waterlogging
            * agriculture
        )

        signals += 1

    score = np.clip(
        score,
        0.0,
        1.0,
    )

    return _finish(
        score,
        signals=signals,
        method=(
            "recharge_risk_proxy_combining_"
            "built_up_conversion_agricultural_land_"
            "and_infiltration_context"
        ),
    )


def flood_risk(
    *,
    standing_water: np.ndarray,
    rainfall_anomaly: float | None = None,
    built_up: np.ndarray | None = None,
    low_elevation: np.ndarray | None = None,
) -> RiskResult:
    """
    Estimate urban/post-rainfall flood risk.
    """

    water = _normalise(
        standing_water
    )

    score = 0.65 * water

    signals = 1

    if rainfall_anomaly is not None:
        rainfall_signal = float(
            np.clip(
                rainfall_anomaly,
                0.0,
                3.0,
            )
            / 3.0
        )

        score += (
            0.20
            * rainfall_signal
        )

        signals += 1

    if built_up is not None:
        built = _normalise(
            built_up
        )

        if built.shape != score.shape:
            raise ValueError(
                "built_up must match standing_water shape."
            )

        score += (
            0.10
            * built
        )

        signals += 1

    if low_elevation is not None:
        elevation = _normalise(
            low_elevation
        )

        if elevation.shape != score.shape:
            raise ValueError(
                "low_elevation must match standing_water shape."
            )

        score += (
            0.15
            * (1.0 - elevation)
        )

        signals += 1

    score = np.clip(
        score,
        0.0,
        1.0,
    )

    return _finish(
        score,
        signals=signals,
        method=(
            "post_rainfall_flood_risk_proxy_using_"
            "standing_water_rainfall_built_up_and_elevation"
        ),
    )


def salinity_risk(
    *,
    salinity_proxy: np.ndarray,
    agricultural_mask: np.ndarray | None = None,
    waterlogging: np.ndarray | None = None,
) -> RiskResult:
    """
    Coastal agricultural salinity-risk proxy.
    """

    salinity = _normalise(
        salinity_proxy
    )

    score = salinity.copy()

    signals = 1

    if agricultural_mask is not None:
        agriculture = np.asarray(
            agricultural_mask,
            dtype=np.float32,
        )

        if agriculture.shape != score.shape:
            raise ValueError(
                "agricultural_mask must match salinity shape."
            )

        score *= np.clip(
            agriculture,
            0.0,
            1.0,
        )

        signals += 1

    if waterlogging is not None:
        wetness = _normalise(
            waterlogging
        )

        if wetness.shape != score.shape:
            raise ValueError(
                "waterlogging must match salinity shape."
            )

        score += (
            0.20
            * wetness
        )

        signals += 1

    score = np.clip(
        score,
        0.0,
        1.0,
    )

    return _finish(
        score,
        signals=signals,
        method=(
            "coastal_salinity_risk_proxy_combining_"
            "spectral_salinity_and_waterlogging"
        ),
    )


def crop_risk(
    *,
    stress_score: np.ndarray,
    waterlogging: np.ndarray | None = None,
    salinity: np.ndarray | None = None,
) -> RiskResult:
    """
    Convert crop stress signals into a risk surface.

    Higher crop stress -> higher risk.
    """

    stress = _normalise(
        -np.asarray(
            stress_score,
            dtype=np.float32,
        )
    )

    score = stress

    signals = 1

    if waterlogging is not None:
        wetness = _normalise(
            waterlogging
        )

        if wetness.shape != score.shape:
            raise ValueError(
                "waterlogging must match stress_score shape."
            )

        score += (
            0.20
            * wetness
        )

        signals += 1

    if salinity is not None:
        salt = _normalise(
            salinity
        )

        if salt.shape != score.shape:
            raise ValueError(
                "salinity must match stress_score shape."
            )

        score += (
            0.20
            * salt
        )

        signals += 1

    score = np.clip(
        score,
        0.0,
        1.0,
    )

    return _finish(
        score,
        signals=signals,
        method=(
            "crop_risk_combining_"
            "vegetation_stress_waterlogging_and_salinity"
        ),
    )


def summarise_risk(
    result: RiskResult,
    *,
    subject: str,
) -> str:
    if result.mean_score < 0.30:
        level = RISK_LOW
    elif result.mean_score < 0.60:
        level = RISK_MODERATE
    elif result.mean_score < 0.80:
        level = RISK_HIGH
    else:
        level = RISK_CRITICAL

    return (
        f"{subject} is classified as {level} risk. "
        f"High-to-critical risk covers "
        f"{result.high_risk_fraction * 100:.1f}% "
        f"of valid pixels."
    )


__all__ = [
    "RISK_LOW",
    "RISK_MODERATE",
    "RISK_HIGH",
    "RISK_CRITICAL",
    "RiskResult",
    "agricultural_recharge_risk",
    "flood_risk",
    "salinity_risk",
    "crop_risk",
    "summarise_risk",
]
