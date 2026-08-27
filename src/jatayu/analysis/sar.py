"""
SAR observables and analysis.

Designed primarily for Sentinel-1-like VV/VH inputs.

The functions operate on arrays and do not perform raster IO.

Public API:
    vv_vh_ratio()
    vh_vv_ratio()
    rvi()
    log_backscatter_difference()
    sar_water_suppression()
    sar_features()
    analyse_sar()
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1e-6


# ============================================================================
# RESULT TYPES
# ============================================================================


@dataclass(frozen=True)
class SARResult:
    vv: np.ndarray | None
    vh: np.ndarray | None
    vv_vh_ratio: np.ndarray | None
    vh_vv_ratio: np.ndarray | None
    rvi: np.ndarray | None

    mean_vv: float | None
    mean_vh: float | None


# ============================================================================
# BASIC UTILITIES
# ============================================================================


def _linearise(value: np.ndarray) -> np.ndarray:
    """
    Convert SAR backscatter from dB to linear power.

        linear = 10 ** (dB / 10)
    """

    value = np.asarray(value, dtype=np.float32)

    return np.power(
        10.0,
        value / 10.0,
    )


def _looks_like_db(arr: np.ndarray) -> bool:
    """
    Conservative heuristic for detecting Sentinel-1-style dB data.

    Typical Sentinel-1 sigma0 values are negative in dB.
    """

    arr = np.asarray(arr, dtype=np.float32)

    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        return False

    return float(np.nanmedian(finite)) < 0.0


def _validate_pair(
    vv: np.ndarray,
    vh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    vv = np.asarray(
        vv,
        dtype=np.float32,
    )

    vh = np.asarray(
        vh,
        dtype=np.float32,
    )

    if vv.shape != vh.shape:
        raise ValueError(
            "VV and VH must have identical shapes."
        )

    return vv, vh


# ============================================================================
# SAR INDICES
# ============================================================================


def vv_vh_ratio(
    vv: np.ndarray,
    vh: np.ndarray,
    *,
    input_db: bool | None = None,
) -> np.ndarray:
    """
    Compute VV/VH backscatter power ratio.
    """

    vv, vh = _validate_pair(
        vv,
        vh,
    )

    if input_db is None:
        input_db = _looks_like_db(vv)

    if input_db:
        vv = _linearise(vv)
        vh = _linearise(vh)

    return vv / (
        vh + EPS
    )


def vh_vv_ratio(
    vv: np.ndarray,
    vh: np.ndarray,
    *,
    input_db: bool | None = None,
) -> np.ndarray:
    """
    Compute VH/VV backscatter power ratio.
    """

    ratio = vv_vh_ratio(
        vv,
        vh,
        input_db=input_db,
    )

    return 1.0 / (
        ratio + EPS
    )


def rvi(
    vv: np.ndarray,
    vh: np.ndarray,
    *,
    input_db: bool | None = None,
) -> np.ndarray:
    """
    Radar Vegetation Index approximation.

        RVI = 4 * VH / (VV + VH)

    Higher values generally indicate stronger
    volume scattering / vegetation structure.
    """

    vv_arr, vh_arr = _validate_pair(
        vv,
        vh,
    )

    if input_db is None:
        input_db = _looks_like_db(
            vv_arr
        )

    if input_db:
        vv_arr = _linearise(
            vv_arr
        )

        vh_arr = _linearise(
            vh_arr
        )

    denominator = (
        vv_arr
        + vh_arr
        + EPS
    )

    return (
        4.0
        * vh_arr
        / denominator
    )


# ============================================================================
# TEMPORAL SAR CHANGE
# ============================================================================


def log_backscatter_difference(
    before: np.ndarray,
    after: np.ndarray,
    *,
    input_db: bool = True,
) -> np.ndarray:
    """
    Calculate temporal SAR backscatter change.

    For dB:

        delta = after_dB - before_dB

    For linear power, both arrays are converted to dB first.
    """

    before = np.asarray(
        before,
        dtype=np.float32,
    )

    after = np.asarray(
        after,
        dtype=np.float32,
    )

    if before.shape != after.shape:
        raise ValueError(
            "Before and after SAR arrays must match."
        )

    if input_db:
        return after - before

    before_db = (
        10.0
        * np.log10(
            np.maximum(
                before,
                EPS,
            )
        )
    )

    after_db = (
        10.0
        * np.log10(
            np.maximum(
                after,
                EPS,
            )
        )
    )

    return after_db - before_db


def sar_water_suppression(
    before: np.ndarray,
    after: np.ndarray,
    *,
    threshold_db: float = -2.0,
) -> np.ndarray:
    """
    Detect strong negative SAR change.

    Returns:

        0 = not detected
        1 = strong suppression

    Negative SAR change can be associated with increased
    surface water under appropriate acquisition geometry.
    """

    delta = log_backscatter_difference(
        before,
        after,
        input_db=True,
    )

    mask = np.zeros(
        delta.shape,
        dtype=np.uint8,
    )

    finite = np.isfinite(
        delta
    )

    mask[
        finite
        & (
            delta <= threshold_db
        )
    ] = 1

    return mask


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================


def sar_features(
    vv: np.ndarray | None = None,
    vh: np.ndarray | None = None,
    *,
    input_db: bool | None = None,
) -> SARResult:
    """
    Compute all available SAR observables.
    """

    if vv is None and vh is None:
        raise ValueError(
            "At least VV or VH must be supplied."
        )

    ratio = None
    reverse = None
    vegetation = None

    if vv is not None and vh is not None:

        ratio = vv_vh_ratio(
            vv,
            vh,
            input_db=input_db,
        )

        reverse = 1.0 / (
            ratio + EPS
        )

        vegetation = rvi(
            vv,
            vh,
            input_db=input_db,
        )

    return SARResult(
        vv=vv,
        vh=vh,
        vv_vh_ratio=ratio,
        vh_vv_ratio=reverse,
        rvi=vegetation,

        mean_vv=(
            float(np.nanmean(vv))
            if vv is not None
            else None
        ),

        mean_vh=(
            float(np.nanmean(vh))
            if vh is not None
            else None
        ),
    )


# ============================================================================
# SAFE HELPERS
# ============================================================================


def _safe_mean(
    arr: np.ndarray | None,
) -> float | None:

    if arr is None:
        return None

    arr = np.asarray(
        arr,
        dtype=np.float32,
    )

    finite = np.isfinite(
        arr
    )

    if not np.any(finite):
        return None

    return float(
        np.nanmean(arr)
    )


def _safe_fraction(
    mask: np.ndarray,
) -> float:

    mask = np.asarray(
        mask,
        dtype=bool,
    )

    if mask.size == 0:
        return 0.0

    return float(
        np.mean(mask)
    )


# ============================================================================
# HIGH-LEVEL ANALYSIS
# ============================================================================


def analyse_sar(
    vv: np.ndarray | None = None,
    vh: np.ndarray | None = None,
    *,
    before_vv: np.ndarray | None = None,
    after_vv: np.ndarray | None = None,
    before_vh: np.ndarray | None = None,
    after_vh: np.ndarray | None = None,
    query: str = "",
    input_db: bool | None = None,
    water_threshold_db: float = -2.0,
) -> dict:
    """
    High-level SAR analysis entry point.

    Supports:

    1. Single SAR observation

        analyse_sar(
            vv=vv,
            vh=vh,
            query="Is my wheat field stressed?"
        )

    2. Temporal VV comparison

        analyse_sar(
            before_vv=before_vv,
            after_vv=after_vv,
            query="Where is the standing water?"
        )

    3. Temporal VV + VH comparison

        analyse_sar(
            before_vv=before_vv,
            after_vv=after_vv,
            before_vh=before_vh,
            after_vh=after_vh,
        )

    No raster IO is performed here.
    """

    results: dict[str, dict] = {}

    # ========================================================================
    # SINGLE-DATE FEATURES
    # ========================================================================

    if vv is not None or vh is not None:

        features = sar_features(
            vv=vv,
            vh=vh,
            input_db=input_db,
        )

        if vv is not None:

            results["VV"] = {
                "array": np.asarray(
                    vv,
                    dtype=np.float32,
                ),
                "mean": _safe_mean(
                    vv
                ),
                "interpretation": (
                    "SAR VV backscatter"
                ),
            }

        if vh is not None:

            results["VH"] = {
                "array": np.asarray(
                    vh,
                    dtype=np.float32,
                ),
                "mean": _safe_mean(
                    vh
                ),
                "interpretation": (
                    "SAR VH backscatter"
                ),
            }

        if features.vv_vh_ratio is not None:

            results["VV_VH_RATIO"] = {
                "array": features.vv_vh_ratio,
                "mean": _safe_mean(
                    features.vv_vh_ratio
                ),
                "interpretation": (
                    "relative VV to VH scattering"
                ),
            }

        if features.rvi is not None:

            results["RVI"] = {
                "array": features.rvi,
                "mean": _safe_mean(
                    features.rvi
                ),
                "interpretation": (
                    "vegetation structure / "
                    "volume scattering"
                ),
            }

    # ========================================================================
    # TEMPORAL VV CHANGE
    # ========================================================================

    if before_vv is not None or after_vv is not None:

        if before_vv is None or after_vv is None:
            raise ValueError(
                "Both before_vv and after_vv "
                "must be supplied for temporal SAR analysis."
            )

        vv_delta = log_backscatter_difference(
            before_vv,
            after_vv,
            input_db=(
                True
                if input_db is None
                else input_db
            ),
        )

        finite = np.isfinite(
            vv_delta
        )

        negative = (
            finite
            & (
                vv_delta <= water_threshold_db
            )
        )

        positive = (
            finite
            & (
                vv_delta >= abs(
                    water_threshold_db
                )
            )
        )

        valid_count = max(
            int(np.count_nonzero(finite)),
            1,
        )

        negative_fraction = float(
            np.count_nonzero(
                negative
            )
            / valid_count
        )

        positive_fraction = float(
            np.count_nonzero(
                positive
            )
            / valid_count
        )

        results["VV_CHANGE"] = {
            "array": vv_delta,

            "mean": _safe_mean(
                vv_delta
            ),

            "negative_fraction": (
                negative_fraction
            ),

            "positive_fraction": (
                positive_fraction
            ),

            "water_suppression_fraction": (
                negative_fraction
            ),

            "interpretation": (
                "negative SAR backscatter change "
                "can indicate increased surface water "
                "under suitable acquisition geometry"
            ),
        }

    # ========================================================================
    # TEMPORAL VH CHANGE
    # ========================================================================

    if before_vh is not None or after_vh is not None:

        if before_vh is None or after_vh is None:
            raise ValueError(
                "Both before_vh and after_vh "
                "must be supplied for temporal SAR analysis."
            )

        vh_delta = log_backscatter_difference(
            before_vh,
            after_vh,
            input_db=(
                True
                if input_db is None
                else input_db
            ),
        )

        finite = np.isfinite(
            vh_delta
        )

        negative = (
            finite
            & (
                vh_delta <= water_threshold_db
            )
        )

        positive = (
            finite
            & (
                vh_delta >= abs(
                    water_threshold_db
                )
            )
        )

        valid_count = max(
            int(np.count_nonzero(finite)),
            1,
        )

        results["VH_CHANGE"] = {
            "array": vh_delta,

            "mean": _safe_mean(
                vh_delta
            ),

            "negative_fraction": float(
                np.count_nonzero(
                    negative
                )
                / valid_count
            ),

            "positive_fraction": float(
                np.count_nonzero(
                    positive
                )
                / valid_count
            ),

            "interpretation": (
                "temporal VH backscatter change"
            ),
        }

    # ========================================================================
    # WATER / FLOOD SIGNAL
    # ========================================================================

    if (
        before_vv is not None
        and after_vv is not None
    ):

        water_mask = sar_water_suppression(
            before_vv,
            after_vv,
            threshold_db=water_threshold_db,
        )

        water_fraction = float(
            np.mean(
                water_mask == 1
            )
        )

        results["WATER_SUPPRESSION"] = {
            "array": water_mask,

            "fraction": water_fraction,

            "threshold_db": (
                water_threshold_db
            ),

            "interpretation": (
                "relative SAR suppression after the "
                "event; possible surface-water increase"
            ),
        }

    # ========================================================================
    # HIGH-LEVEL SUMMARY
    # ========================================================================

    mean_vv_change = None

    if "VV_CHANGE" in results:

        mean_vv_change = results[
            "VV_CHANGE"
        ]["mean"]

    # ------------------------------------------------------------------------
    # Water suppression gets priority because the query may specifically
    # concern standing water / flooding.
    # ------------------------------------------------------------------------

    if "WATER_SUPPRESSION" in results:

        water_fraction = results[
            "WATER_SUPPRESSION"
        ]["fraction"]

        if water_fraction > 0:

            summary = (
                "Potential standing-water expansion "
                f"detected across "
                f"{water_fraction * 100:.1f}% "
                "of the scene based on SAR "
                "backscatter suppression."
            )

        else:

            summary = (
                "No significant SAR "
                "water-suppression regions "
                "were detected."
            )

    elif mean_vv_change is None:

        summary = (
            "Computed available SAR observables."
        )

    elif mean_vv_change <= water_threshold_db:

        summary = (
            "SAR backscatter decreased after "
            "the event, indicating a potential "
            "increase in surface water or "
            "reduced scattering."
        )

    elif mean_vv_change >= abs(
        water_threshold_db
    ):

        summary = (
            "SAR backscatter increased after "
            "the event, indicating increased "
            "surface scattering or structural change."
        )

    else:

        summary = (
            "No strong scene-wide SAR "
            "backscatter change was detected."
        )

    # ========================================================================
    # CONFIDENCE
    # ========================================================================

    confidence = 0.50

    if "WATER_SUPPRESSION" in results:

        fraction = results[
            "WATER_SUPPRESSION"
        ]["fraction"]

        # More spatially coherent suppression
        # gives stronger deterministic confidence.
        confidence = min(
            0.95,
            0.50
            + min(
                fraction / 0.20,
                1.0,
            )
            * 0.35,
        )

    elif results:

        # Single-date SAR observables are useful,
        # but cannot establish a temporal event.
        confidence = 0.65

    return {
        "query": query,

        "results": results,

        "summary": summary,

        "confidence": round(
            float(confidence),
            2,
        ),

        "confidence_method": (
            "deterministic SAR signal strength "
            "and spatial coverage"
        ),
    }


# ============================================================================
# PUBLIC API
# ============================================================================


__all__ = [
    "SARResult",
    "vv_vh_ratio",
    "vh_vv_ratio",
    "rvi",
    "log_backscatter_difference",
    "sar_water_suppression",
    "sar_features",
    "analyse_sar",
]
