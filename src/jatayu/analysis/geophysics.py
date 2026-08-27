"""Multi-layer geophysical analysis engine.

Goes beyond spectral indices to couple satellite observations with climate
baselines and produce physically meaningful crop/environment assessments.

Four layers, each building on the previous:
  1. Spectral: NDVI, NDRE, MNDWI, etc. (from indices.py)
  2. Climate: rainfall and temperature anomalies from gridded datasets
  3. Surface: land surface temperature, soil moisture proxies
  4. Composite: fused stress/health scores for decision support

This module contains pure functions that operate on arrays and scalars.
No raster I/O — that stays in jatayu.io. No model weights — this is physics.

Data sources (all free, no API key):
- CHIRPS: daily/monthly precipitation, 0.05° resolution, 1981–present
- ERA5-Land: temperature, soil moisture (requires free CDS account)
- MODIS LST (MOD11A2): land surface temperature, 1km, 8-day
- IMD Gridded: India-specific daily rainfall, 0.25°
- ICRISAT: district-level crop statistics, 1966–2020
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Climate baseline and anomaly
# ---------------------------------------------------------------------------


class AnomalyLevel(str, Enum):
    """Human-readable severity for any anomaly."""

    SEVERE_DEFICIT = "severe_deficit"
    MODERATE_DEFICIT = "moderate_deficit"
    NORMAL = "normal"
    MODERATE_EXCESS = "moderate_excess"
    SEVERE_EXCESS = "severe_excess"


@dataclass(frozen=True)
class AnomalyResult:
    """The output of any anomaly computation — value, baseline, and interpretation."""

    variable: str
    current_value: float
    baseline_mean: float
    baseline_std: float
    anomaly_pct: float           # percentage deviation from mean
    z_score: float               # standard deviations from mean
    level: AnomalyLevel
    sentence: str                # human-readable, goes straight into the answer
    unit: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "current": round(self.current_value, 4),
            "baseline_mean": round(self.baseline_mean, 4),
            "baseline_std": round(self.baseline_std, 4),
            "anomaly_pct": round(self.anomaly_pct, 1),
            "z_score": round(self.z_score, 2),
            "level": self.level.value,
        }


def compute_anomaly(
    current: float,
    baseline_mean: float,
    baseline_std: float,
    variable_name: str,
    unit: str = "",
    *,
    deficit_threshold: float = -1.0,
    severe_deficit_threshold: float = -1.5,
    excess_threshold: float = 1.0,
    severe_excess_threshold: float = 1.5,
) -> AnomalyResult:
    """Compare a current observation against a historical baseline.

    Works for any variable: NDVI, rainfall, temperature, soil moisture.
    The z-score thresholds determine severity classification.
    """
    if baseline_std <= 0 or not np.isfinite(baseline_std):
        # Can't compute a meaningful anomaly without variance
        return AnomalyResult(
            variable=variable_name,
            current_value=current,
            baseline_mean=baseline_mean,
            baseline_std=0.0,
            anomaly_pct=0.0,
            z_score=0.0,
            level=AnomalyLevel.NORMAL,
            sentence=f"{variable_name} is {current:.2f}{unit} (no baseline variance available).",
            unit=unit,
        )

    z = (current - baseline_mean) / baseline_std
    pct = ((current - baseline_mean) / abs(baseline_mean) * 100) if baseline_mean != 0 else 0.0

    if z <= severe_deficit_threshold:
        level = AnomalyLevel.SEVERE_DEFICIT
    elif z <= deficit_threshold:
        level = AnomalyLevel.MODERATE_DEFICIT
    elif z >= severe_excess_threshold:
        level = AnomalyLevel.SEVERE_EXCESS
    elif z >= excess_threshold:
        level = AnomalyLevel.MODERATE_EXCESS
    else:
        level = AnomalyLevel.NORMAL

    # Build a sentence a district officer can read
    direction = "above" if pct > 0 else "below"
    sentence = (
        f"{variable_name} is {abs(pct):.0f}% {direction} the historical average "
        f"({current:.2f} vs {baseline_mean:.2f}{unit}). "
        f"This is classified as {level.value.replace('_', ' ')}."
    )

    return AnomalyResult(
        variable=variable_name,
        current_value=current,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        anomaly_pct=pct,
        z_score=z,
        level=level,
        sentence=sentence,
        unit=unit,
    )


def spatial_anomaly(
    current_raster: FloatArray,
    baseline_mean_raster: FloatArray,
    baseline_std_raster: FloatArray,
) -> FloatArray:
    """Per-pixel z-score anomaly map. Positive = above baseline, negative = below."""
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(
            baseline_std_raster > 0,
            (current_raster - baseline_mean_raster) / baseline_std_raster,
            0.0,
        )
    z[~np.isfinite(z)] = 0.0
    return z


# ---------------------------------------------------------------------------
# Rainfall analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RainfallContext:
    """Rainfall information for an area and time period."""

    total_mm: float
    baseline_mean_mm: float
    baseline_std_mm: float
    anomaly: AnomalyResult
    dry_spell_days: int | None = None  # consecutive days below 2.5mm


def analyse_rainfall(
    current_mm: float,
    baseline_mean_mm: float,
    baseline_std_mm: float,
    period_name: str = "this month",
) -> RainfallContext:
    """Interpret rainfall relative to climatological baseline."""
    anomaly = compute_anomaly(
        current_mm, baseline_mean_mm, baseline_std_mm,
        variable_name=f"Rainfall ({period_name})",
        unit=" mm",
        # Rainfall uses asymmetric thresholds: deficit matters more for crops
        deficit_threshold=-0.8,
        severe_deficit_threshold=-1.2,
        excess_threshold=1.2,
        severe_excess_threshold=2.0,
    )
    return RainfallContext(
        total_mm=current_mm,
        baseline_mean_mm=baseline_mean_mm,
        baseline_std_mm=baseline_std_mm,
        anomaly=anomaly,
    )


# ---------------------------------------------------------------------------
# Temperature and LST analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemperatureContext:
    """Temperature/LST information for an area."""

    current_celsius: float
    baseline_mean_celsius: float
    baseline_std_celsius: float
    anomaly: AnomalyResult
    heat_stress: bool  # above crop-specific threshold


def analyse_temperature(
    current_celsius: float,
    baseline_mean_celsius: float,
    baseline_std_celsius: float,
    *,
    heat_threshold_celsius: float = 35.0,
    variable_name: str = "Land surface temperature",
) -> TemperatureContext:
    """Interpret temperature relative to baseline with heat-stress flag."""
    anomaly = compute_anomaly(
        current_celsius, baseline_mean_celsius, baseline_std_celsius,
        variable_name=variable_name,
        unit="°C",
    )
    return TemperatureContext(
        current_celsius=current_celsius,
        baseline_mean_celsius=baseline_mean_celsius,
        baseline_std_celsius=baseline_std_celsius,
        anomaly=anomaly,
        heat_stress=current_celsius > heat_threshold_celsius,
    )


# ---------------------------------------------------------------------------
# Crop Stress Composite — the food security signal
# ---------------------------------------------------------------------------


class StressLevel(str, Enum):
    """Overall crop stress classification."""

    HEALTHY = "healthy"
    WATCH = "watch"
    STRESSED = "stressed"
    SEVERE = "severe"
    CRITICAL = "critical"


@dataclass(frozen=True)
class CropStressAssessment:
    """Fused multi-signal crop health assessment.

    This is the food-security product: it combines vegetation indices,
    rainfall, and temperature into a single actionable assessment with
    individually cited signals.
    """

    overall: StressLevel
    confidence: float
    ndvi_anomaly: AnomalyResult | None = None
    rainfall: RainfallContext | None = None
    temperature: TemperatureContext | None = None
    signals_used: int = 0
    summary: str = ""
    recommendations: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "overall_stress": self.overall.value,
            "confidence": round(self.confidence, 2),
            "signals_used": self.signals_used,
            "summary": self.summary,
        }
        if self.ndvi_anomaly:
            d["vegetation"] = self.ndvi_anomaly.to_dict()
        if self.rainfall:
            d["rainfall"] = self.rainfall.anomaly.to_dict()
        if self.temperature:
            d["temperature"] = self.temperature.anomaly.to_dict()
        return d


def assess_crop_stress(
    *,
    ndvi_current: float | None = None,
    ndvi_baseline_mean: float | None = None,
    ndvi_baseline_std: float | None = None,
    rainfall_mm: float | None = None,
    rainfall_baseline_mean: float | None = None,
    rainfall_baseline_std: float | None = None,
    temperature_celsius: float | None = None,
    temperature_baseline_mean: float | None = None,
    temperature_baseline_std: float | None = None,
    period_name: str = "current period",
) -> CropStressAssessment:
    """Fuse available signals into a crop stress assessment.

    Uses whatever signals are provided — one, two, or all three. Confidence
    scales with the number of independent signals: a single-index assessment
    is flagged as low-confidence, while a three-signal composite is strong.

    The fusion logic:
    - Each signal contributes a stress score from -2 (severe deficit) to +2 (excess)
    - Signals are weighted: NDVI (0.5), rainfall (0.3), temperature (0.2)
    - Cross-signal consistency raises confidence; contradiction lowers it
    """
    stress_scores: list[tuple[float, float]] = []  # (score, weight)
    signals = 0
    ndvi_anomaly = None
    rainfall_ctx = None
    temperature_ctx = None
    notes: list[str] = []

    # --- Vegetation signal ---
    if ndvi_current is not None and ndvi_baseline_mean is not None:
        ndvi_std = ndvi_baseline_std if ndvi_baseline_std and ndvi_baseline_std > 0 else 0.05
        ndvi_anomaly = compute_anomaly(
            ndvi_current, ndvi_baseline_mean, ndvi_std,
            variable_name=f"Vegetation greenness ({period_name})",
        )
        stress_scores.append((ndvi_anomaly.z_score, 0.5))
        signals += 1
        notes.append(ndvi_anomaly.sentence)

    # --- Rainfall signal ---
    if rainfall_mm is not None and rainfall_baseline_mean is not None:
        rain_std = rainfall_baseline_std if rainfall_baseline_std and rainfall_baseline_std > 0 else 20.0
        rainfall_ctx = analyse_rainfall(
            rainfall_mm, rainfall_baseline_mean, rain_std, period_name,
        )
        # Invert: rainfall deficit → negative stress (bad for crops)
        stress_scores.append((rainfall_ctx.anomaly.z_score, 0.3))
        signals += 1
        notes.append(rainfall_ctx.anomaly.sentence)

    # --- Temperature signal ---
    if temperature_celsius is not None and temperature_baseline_mean is not None:
        temp_std = temperature_baseline_std if temperature_baseline_std and temperature_baseline_std > 0 else 2.0
        temperature_ctx = analyse_temperature(
            temperature_celsius, temperature_baseline_mean, temp_std,
        )
        # Invert: higher temperature → more stress (negative for crops)
        stress_scores.append((-temperature_ctx.anomaly.z_score, 0.2))
        signals += 1
        notes.append(temperature_ctx.anomaly.sentence)
        if temperature_ctx.heat_stress:
            notes.append(
                f"Heat stress flag: {temperature_celsius:.0f}°C exceeds the 35°C threshold."
            )

    if signals == 0:
        return CropStressAssessment(
            overall=StressLevel.WATCH,
            confidence=0.0,
            summary="No data available for crop stress assessment.",
            signals_used=0,
        )

    # --- Fuse signals --------------------------------------------------------
    total_weight = sum(w for _, w in stress_scores)
    weighted_z = sum(z * w for z, w in stress_scores) / total_weight

    # Classify
    if weighted_z <= -1.5:
        overall = StressLevel.CRITICAL
    elif weighted_z <= -1.0:
        overall = StressLevel.SEVERE
    elif weighted_z <= -0.5:
        overall = StressLevel.STRESSED
    elif weighted_z <= 0.5:
        overall = StressLevel.HEALTHY
    else:
        overall = StressLevel.WATCH  # excess can also be problematic

    # Confidence: more signals = higher confidence, consistency matters
    base_confidence = min(signals / 3.0, 1.0)
    if signals >= 2:
        # Check consistency: all signals should point the same direction
        signs = [np.sign(z) for z, _ in stress_scores]
        consistent = len(set(signs)) == 1
        consistency_bonus = 0.15 if consistent else -0.1
    else:
        consistency_bonus = 0.0
    confidence = round(min(max(base_confidence * 0.7 + consistency_bonus + 0.1, 0.1), 0.95), 2)

    # --- Recommendations based on stress level ---
    recommendations = []
    if overall in (StressLevel.SEVERE, StressLevel.CRITICAL):
        recommendations.append("Immediate field inspection recommended.")
        if rainfall_ctx and rainfall_ctx.anomaly.level in (
            AnomalyLevel.SEVERE_DEFICIT, AnomalyLevel.MODERATE_DEFICIT
        ):
            recommendations.append("Consider supplemental irrigation if available.")
        if temperature_ctx and temperature_ctx.heat_stress:
            recommendations.append("Heat stress mitigation: mulching, adjusted irrigation timing.")
    elif overall == StressLevel.STRESSED:
        recommendations.append("Monitor closely over the next 7–14 days.")
        recommendations.append("Compare with adjacent blocks for spatial context.")

    summary = " ".join(notes)

    return CropStressAssessment(
        overall=overall,
        confidence=confidence,
        ndvi_anomaly=ndvi_anomaly,
        rainfall=rainfall_ctx,
        temperature=temperature_ctx,
        signals_used=signals,
        summary=summary,
        recommendations=recommendations,
    )


# ---------------------------------------------------------------------------
# Drought Index — SPI-style simplified
# ---------------------------------------------------------------------------


def simplified_drought_index(
    rainfall_mm: float,
    baseline_mean_mm: float,
    baseline_std_mm: float,
) -> tuple[float, str]:
    """Simplified Standardized Precipitation Index (SPI).

    Returns (spi_value, category). Negative = drought, positive = wet.
    Uses the same z-score as SPI but without the gamma distribution fitting
    that full SPI requires — honest about the simplification.
    """
    if baseline_std_mm <= 0:
        return 0.0, "insufficient_data"

    spi = (rainfall_mm - baseline_mean_mm) / baseline_std_mm

    if spi <= -2.0:
        category = "exceptional_drought"
    elif spi <= -1.5:
        category = "extreme_drought"
    elif spi <= -1.0:
        category = "severe_drought"
    elif spi <= -0.5:
        category = "moderate_drought"
    elif spi <= 0.5:
        category = "near_normal"
    elif spi <= 1.0:
        category = "moderately_wet"
    elif spi <= 1.5:
        category = "very_wet"
    else:
        category = "extremely_wet"

    return round(spi, 2), category


# ---------------------------------------------------------------------------
# Water Balance — evapotranspiration proxy
# ---------------------------------------------------------------------------


def simple_water_balance(
    rainfall_mm: float,
    temperature_celsius: float,
    *,
    days: int = 30,
) -> dict[str, float]:
    """Thornthwaite-style potential evapotranspiration estimate.

    A crude but physically grounded water balance. PET is estimated from
    temperature alone — accurate enough for monthly anomaly detection, not
    for irrigation scheduling.
    """
    # Thornthwaite PET approximation (mm/month)
    if temperature_celsius <= 0:
        pet = 0.0
    else:
        # Simplified: PET ≈ 16 * (10 * T / I)^a where I is heat index
        # For a quick prototype, use the Hamon approximation instead
        # PET ≈ 0.55 * (days) * (temperature/12)^2 mm/month
        pet = 0.55 * days * (temperature_celsius / 12.0) ** 2

    balance = rainfall_mm - pet

    return {
        "rainfall_mm": round(rainfall_mm, 1),
        "pet_mm": round(pet, 1),
        "balance_mm": round(balance, 1),
        "deficit": balance < 0,
        "deficit_mm": round(abs(min(balance, 0)), 1),
    }


__all__ = [
    "AnomalyLevel",
    "AnomalyResult",
    "CropStressAssessment",
    "RainfallContext",
    "StressLevel",
    "TemperatureContext",
    "analyse_rainfall",
    "analyse_temperature",
    "assess_crop_stress",
    "compute_anomaly",
    "simple_water_balance",
    "simplified_drought_index",
    "spatial_anomaly",
]
