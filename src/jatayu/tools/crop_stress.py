"""Crop stress assessment — "Is my wheat field stressed?"

Computes available vegetation indices from the image, produces a stress
score, and generates a classified stress map. Gracefully skips indices
whose required bands are missing rather than crashing.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from jatayu.analysis.indices import INDEX_REGISTRY, IndexAnalyser
from jatayu.io.loader import read_bands
from jatayu.render import mask_to_png
from jatayu.schemas import (
    Evidence, TaskFamily, TaskName, ToolRequest, ToolResult,
)
from jatayu.tools.registry import register

_OUTPUT_DIR = Path("data/samples/jatayu_outputs")
_ANALYSER = IndexAnalyser()

# Indices to try, in order of importance for crop stress
_STRESS_INDICES = ["NDVI", "NDMI", "NDRE", "LSWI"]

# Baseline values — conservative estimates for Indian agricultural scenes.
# For production, these would come from per-district historical time series.
_BASELINES = {
    "NDVI": 0.50,
    "NDMI": 0.20,
    "NDRE": 0.30,
    "LSWI": 0.15,
}

# Weight each index contributes to the composite stress score
_WEIGHTS = {
    "NDVI": 0.45,
    "NDMI": 0.25,
    "NDRE": 0.20,
    "LSWI": 0.10,
}


@register(
    TaskName.CROP_STRESS,
    families={TaskFamily.SINGLE_IMAGE},
    description="Assesses crop health and stress using available spectral indices "
                "(NDVI, NDMI, NDRE, LSWI). Gracefully skips indices whose bands "
                "are unavailable. Returns a stress map with signal breakdown.",
)
def run(req: ToolRequest) -> ToolResult:
    t0 = perf_counter()
    notes: list[str] = []
    image = req.images[0]

    decimate = max(1, max(image.width, image.height) // 1024)
    if decimate > 1:
        notes.append(f"decimate={decimate}")

    # --- probe which indices we can compute ----------------------------------
    computed_indices: dict[str, np.ndarray] = {}

    for idx_name in _STRESS_INDICES:
        definition = INDEX_REGISTRY[idx_name]
        required = list(definition.required_bands)
        try:
            arrays = read_bands(image, required, decimate=decimate)
            band_dict = dict(zip(required, arrays))
            result = _ANALYSER.compute(band_dict, [idx_name])
            computed_indices[idx_name] = result[idx_name]
        except (ValueError, KeyError):
            notes.append(f"{idx_name} skipped — missing required bands {required}.")

    if not computed_indices:
        return ToolResult(
            answer="Cannot assess crop stress — no vegetation indices could be computed from the available bands.",
            confidence=0.0,
            confidence_method="not_attempted",
            tool_name=TaskName.CROP_STRESS,
            model_id="physics_crop_stress_v1",
            abstained=True,
            notes=notes,
        )

    notes.append(f"indices_computed={list(computed_indices.keys())}")

    # --- compute per-index deficit and composite stress ----------------------
    # Deficit = how much below baseline. Positive = stressed, negative = healthy.
    shape = next(iter(computed_indices.values())).shape
    composite = np.zeros(shape, dtype=np.float64)
    total_weight = 0.0
    signal_details = []

    for idx_name, values in computed_indices.items():
        baseline = _BASELINES.get(idx_name, 0.0)
        mean_val = float(np.nanmean(values))
        deficit = baseline - mean_val  # positive = below baseline = stressed

        weight = _WEIGHTS.get(idx_name, 0.1)
        # Per-pixel deficit, clipped to [0, 1]
        pixel_deficit = np.clip((baseline - values) / max(abs(baseline), 0.01), 0.0, 1.0)
        composite += weight * pixel_deficit
        total_weight += weight

        pct = (deficit / baseline * 100) if baseline != 0 else 0
        direction = "below" if deficit > 0 else "above"
        signal_details.append(
            f"{idx_name} is {mean_val:.3f} ({abs(pct):.0f}% {direction} baseline of {baseline:.2f})"
        )
        notes.append(f"mean_{idx_name}={mean_val:.3f}")

    # Normalise by actual weight used
    if total_weight > 0:
        composite = composite / total_weight

    composite = np.clip(composite, 0.0, 1.0)
    mean_stress = float(np.nanmean(composite))

    # --- classify into stress levels -----------------------------------------
    stress_class = np.zeros(shape, dtype=np.uint8)
    stress_class[composite >= 0.20] = 1  # mild
    stress_class[composite >= 0.40] = 2  # moderate
    stress_class[composite >= 0.65] = 3  # severe

    n_valid = int(np.sum(np.isfinite(composite)))
    if n_valid == 0:
        n_valid = 1  # avoid division by zero

    frac_healthy = float(np.sum(stress_class == 0)) / n_valid
    frac_mild = float(np.sum(stress_class == 1)) / n_valid
    frac_moderate = float(np.sum(stress_class == 2)) / n_valid
    frac_severe = float(np.sum(stress_class == 3)) / n_valid
    frac_stressed = frac_mild + frac_moderate + frac_severe

    notes.append(f"mean_composite_stress={mean_stress:.3f}")
    notes.append(f"fraction_stressed={frac_stressed:.3f}")

    # --- build the answer ----------------------------------------------------
    if mean_stress >= 0.65:
        severity = "CRITICAL"
        description = "Critical crop stress — immediate field inspection recommended."
    elif mean_stress >= 0.40:
        severity = "HIGH"
        description = "Significant crop stress detected across the scene."
    elif mean_stress >= 0.20:
        severity = "MODERATE"
        description = "Moderate crop stress — some areas show reduced vigour."
    else:
        severity = "LOW"
        description = "Crop vegetation appears generally healthy."

    answer = (
        f"Crop stress level: {severity}. {description} "
        f"Analysis based on {len(computed_indices)} spectral indices: "
        f"{'; '.join(signal_details)}. "
        f"Healthy: {frac_healthy:.0%}, mild stress: {frac_mild:.0%}, "
        f"moderate: {frac_moderate:.0%}, severe: {frac_severe:.0%}."
    )

    # --- render the stress map -----------------------------------------------
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(image.path).stem
    overlay_path = mask_to_png(stress_class, str(_OUTPUT_DIR / f"{stem}_crop_stress.png"))

    legend = {
        "0": "healthy",
        "1": "mild stress",
        "2": "moderate stress",
        "3": "severe stress",
    }

    # --- confidence ----------------------------------------------------------
    # More indices = higher confidence; severe stress is easier to trust
    index_coverage = len(computed_indices) / len(_STRESS_INDICES)
    signal_strength = min(mean_stress / 0.5, 1.0)
    confidence = round(min(0.95, 0.3 + 0.4 * index_coverage + 0.2 * signal_strength), 2)

    elapsed = int((perf_counter() - t0) * 1000)

    return ToolResult(
        answer=answer,
        evidence=Evidence(
            kind="mask",
            overlay_png=str(overlay_path),
            legend=legend,
            caption=f"Crop stress assessment using {', '.join(computed_indices.keys())}.",
        ),
        confidence=confidence,
        confidence_method=f"multi_index_deficit_from_{len(computed_indices)}_indices",
        tool_name=TaskName.CROP_STRESS,
        model_id="physics_crop_stress_v1",
        params_used={
            "indices": list(computed_indices.keys()),
            "baselines": {k: v for k, v in _BASELINES.items() if k in computed_indices},
            "decimate": decimate,
        },
        latency_ms=elapsed,
        abstained=False,
        notes=notes,
    )
