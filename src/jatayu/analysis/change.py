
"""
Temporal change analysis.

Pure NumPy implementation:
    delta = after - before

Produces signed change masks:
    0 = no meaningful change
    1 = increase
    2 = decrease

The threshold is based on absolute change magnitude rather than simply
taking upper/lower percentiles of the signed distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChangeResult:
    delta: np.ndarray
    mask: np.ndarray
    threshold: float

    increase_fraction: float
    decrease_fraction: float
    changed_fraction: float

    before_mean: float
    after_mean: float
    mean_delta: float

    direction: str
    magnitude: float

    confidence: float
    confidence_method: str


def compute_delta(
    before: np.ndarray,
    after: np.ndarray,
) -> np.ndarray:
    """Return after - before while requiring identical shapes."""

    before = np.asarray(before, dtype=np.float32)
    after = np.asarray(after, dtype=np.float32)

    if before.shape != after.shape:
        raise ValueError(
            f"before and after must have identical shapes; "
            f"got {before.shape} and {after.shape}"
        )

    return after - before


def signed_change_mask(
    delta: np.ndarray,
    *,
    percentile: float = 95.0,
    floor: float = 0.02,
    direction: str = "any",
) -> tuple[np.ndarray, float]:
    """
    Create a signed change mask.

    0 = no change
    1 = increase
    2 = decrease

    The threshold is:

        max(percentile(|delta|), floor)

    This avoids automatically declaring ~10% of a scene changed merely
    because the distribution has upper/lower tails.
    """

    if direction not in {"any", "increase", "decrease"}:
        raise ValueError(
            "direction must be 'any', 'increase', or 'decrease'"
        )

    delta = np.asarray(delta, dtype=np.float32)
    finite = np.isfinite(delta)

    if not np.any(finite):
        return np.zeros(delta.shape, dtype=np.uint8), float(floor)

    magnitude = np.abs(delta[finite])
    threshold = max(
        float(np.percentile(magnitude, percentile)),
        float(floor),
    )

    mask = np.zeros(delta.shape, dtype=np.uint8)

    if direction in {"any", "increase"}:
        mask[finite & (delta >= threshold)] = 1

    if direction in {"any", "decrease"}:
        mask[finite & (delta <= -threshold)] = 2

    return mask, threshold


def _direction(mean_delta: float, threshold: float) -> str:
    if abs(mean_delta) < threshold:
        return "stable"
    return "increase" if mean_delta > 0 else "decrease"


def analyse_change(
    before: np.ndarray,
    after: np.ndarray,
    *,
    percentile: float = 95.0,
    floor: float = 0.02,
    direction: str = "any",
) -> ChangeResult:
    """Complete temporal change analysis."""

    delta = compute_delta(before, after)

    finite = np.isfinite(delta)

    if not np.any(finite):
        raise ValueError("No finite pixels available for change analysis.")

    mask, threshold = signed_change_mask(
        delta,
        percentile=percentile,
        floor=floor,
        direction=direction,
    )

    n = int(np.count_nonzero(finite))

    increase_fraction = float(np.count_nonzero(mask == 1) / n)
    decrease_fraction = float(np.count_nonzero(mask == 2) / n)
    changed_fraction = increase_fraction + decrease_fraction

    before_arr = np.asarray(before, dtype=np.float32)
    after_arr = np.asarray(after, dtype=np.float32)

    before_mean = float(np.nanmean(before_arr))
    after_mean = float(np.nanmean(after_arr))
    mean_delta = after_mean - before_mean

    change_direction = _direction(mean_delta, threshold)

    magnitude = float(np.nanpercentile(np.abs(delta[finite]), 95))

    # Confidence is based on physical signal strength + spatial coverage.
    coverage_signal = min(changed_fraction / 0.15, 1.0)
    strength_signal = min(magnitude / max(threshold, 1e-6), 1.0)

    confidence = round(
        min(
            0.95,
            0.45 * coverage_signal
            + 0.50 * strength_signal
            + 0.05,
        ),
        2,
    )

    return ChangeResult(
        delta=delta,
        mask=mask,
        threshold=threshold,
        increase_fraction=increase_fraction,
        decrease_fraction=decrease_fraction,
        changed_fraction=changed_fraction,
        before_mean=before_mean,
        after_mean=after_mean,
        mean_delta=mean_delta,
        direction=change_direction,
        magnitude=magnitude,
        confidence=confidence,
        confidence_method=(
            "temporal_change_signal_combining_spatial_coverage"
            "_and_absolute_delta_magnitude"
        ),
    )


def summarise_change(
    result: ChangeResult,
    *,
    observable: str,
    target: str = "surface",
) -> str:
    """Create a human-readable explanation."""

    if result.direction == "stable":
        return (
            f"No substantial {observable} change was detected in the "
            f"{target}. Mean change was {result.mean_delta:+.3f}."
        )

    return (
        f"{observable} {result.direction} detected across "
        f"{result.changed_fraction * 100:.1f}% of the valid scene. "
        f"Mean change was {result.mean_delta:+.3f}; "
        f"change threshold was {result.threshold:.3f}."
    )
