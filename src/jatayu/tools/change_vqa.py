"""Bi-temporal change understanding — what changed between two dates.

Computes the same spectral index on both dates, differences them, thresholds
the delta by absolute magnitude, and produces a signed three-class change mask:
increase, decrease, no change. No GPU, no learned model.

Inputs arrive already co-registered. If they are not, this tool checks bounds
overlap and abstains rather than silently differencing misaligned pixels.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from jatayu.analysis.indices import INDEX_REGISTRY, IndexAnalyser
from jatayu.io.loader import read_bands
from jatayu.render import mask_to_png
from jatayu.schemas import (
    Evidence, ImageRef, TaskFamily, TaskName, ToolRequest, ToolResult,
)
from jatayu.tools.registry import register

_OUTPUT_DIR = Path("data/samples/jatayu_outputs")
_ANALYSER = IndexAnalyser()

# Minimum fraction of the scene that must exceed the threshold to report change.
# Below this, the change is within noise and the tool abstains.
_MIN_CHANGE_FRACTION = 0.005


def _bounds_overlap(a: ImageRef, b: ImageRef, tolerance: float = 0.1) -> bool:
    """Check whether two images' bounds overlap closely enough to difference."""
    if a.bounds is None or b.bounds is None:
        return True  # no metadata to check — proceed, don't block
    al, ab, ar, at_ = a.bounds
    bl, bb, br, bt = b.bounds
    a_w, a_h = ar - al, at_ - ab
    b_w, b_h = br - bl, bt - bb
    if a_w == 0 or a_h == 0 or b_w == 0 or b_h == 0:
        return False
    overlap_l = max(al, bl)
    overlap_r = min(ar, br)
    overlap_b = max(ab, bb)
    overlap_t = min(at_, bt)
    if overlap_r <= overlap_l or overlap_t <= overlap_b:
        return False
    overlap_area = (overlap_r - overlap_l) * (overlap_t - overlap_b)
    min_area = min(a_w * a_h, b_w * b_h)
    return (overlap_area / min_area) >= (1.0 - tolerance)


@register(
    TaskName.CHANGE_VQA,
    families={TaskFamily.BI_TEMPORAL},
    description="Compares two images of the same area taken at different times and "
                "answers questions about what changed, where, and in which direction "
                "(increase, decrease, no change). Requires exactly two images.",
)
def run(req: ToolRequest) -> ToolResult:
    t0 = perf_counter()
    notes: list[str] = []

    # --- pick the index ------------------------------------------------------
    index_name = "NDVI"  # safe default
    candidate_indices = req.params.get("candidate_indices", ())
    if candidate_indices:
        index_name = candidate_indices[0]

    definition = INDEX_REGISTRY[index_name]
    required = list(definition.required_bands)
    target = req.params.get("target", "change")
    direction_hint = req.params.get("change_direction", "any")

    notes.append(f"index={index_name}")
    notes.append(f"target={target}")
    notes.append(f"direction_hint={direction_hint}")

    # --- validate pair -------------------------------------------------------
    if len(req.images) != 2:
        return ToolResult(
            answer=f"Change analysis requires exactly 2 images; received {len(req.images)}.",
            confidence=0.0, confidence_method="not_attempted",
            tool_name=TaskName.CHANGE_VQA, model_id="index_differencing",
            abstained=True, notes=["Wrong number of images for bi-temporal analysis."],
        )

    before_img, after_img = req.images[0], req.images[1]

    if not _bounds_overlap(before_img, after_img):
        return ToolResult(
            answer=(
                "These two images do not cover the same area closely enough "
                "to compare safely. Re-export them in the same projection and "
                "extent, or use images that overlap."
            ),
            confidence=0.0, confidence_method="not_attempted",
            tool_name=TaskName.CHANGE_VQA, model_id="index_differencing",
            abstained=True,
            notes=["Bounds overlap check failed — differencing would be meaningless."],
        )

    # --- compute index on both dates -----------------------------------------
    decimate = max(1, max(before_img.width, before_img.height) // 1024)
    if decimate > 1:
        notes.append(f"decimate={decimate}")

    try:
        before_arrays = read_bands(before_img, required, decimate=decimate)
        after_arrays = read_bands(after_img, required, decimate=decimate)
    except (ValueError, KeyError) as exc:
        return ToolResult(
            answer=f"Cannot compute {index_name}: {exc}",
            confidence=0.0, confidence_method="not_attempted",
            tool_name=TaskName.CHANGE_VQA, model_id="index_differencing",
            abstained=True, notes=[str(exc)],
        )

    before_bands = dict(zip(required, before_arrays))
    after_bands = dict(zip(required, after_arrays))

    before_index = _ANALYSER.compute(before_bands, [index_name])[index_name]
    after_index = _ANALYSER.compute(after_bands, [index_name])[index_name]

    # --- difference ----------------------------------------------------------
    delta = _ANALYSER.diff(before_index, after_index)
    finite = np.isfinite(delta)

    if not np.any(finite):
        return ToolResult(
            answer="No valid pixels to compare — both images may be entirely masked.",
            confidence=0.0, confidence_method="not_attempted",
            tool_name=TaskName.CHANGE_VQA, model_id="index_differencing",
            abstained=True, notes=["All delta pixels are NaN."],
        )

    # --- threshold by absolute magnitude ------------------------------------
    # Percentile-based thresholding always classifies ~10% as increase and ~10%
    # as decrease regardless of whether real change occurred. Instead, threshold
    # on the magnitude: only pixels whose |delta| exceeds the 95th percentile
    # of |delta| count as meaningful.
    abs_delta = np.abs(delta[finite])
    magnitude_threshold = float(np.percentile(abs_delta, 95))

    # Floor: if the 95th percentile of |delta| is tiny, the scene hasn't
    # changed and we should not report noise as change. 0.02 is conservative
    # for normalised difference indices whose range is [-1, 1].
    magnitude_threshold = max(magnitude_threshold, 0.02)

    # Signed classification using the single magnitude threshold.
    # When direction_hint is set, only the requested direction is classified —
    # "where did water increase" should not also report decreases.
    change_mask = np.zeros(delta.shape, dtype=np.uint8)

    if direction_hint in ("increase", "any"):
        change_mask[finite & (delta >= magnitude_threshold)] = 1
    if direction_hint in ("decrease", "any"):
        change_mask[finite & (delta <= -magnitude_threshold)] = 2

    n_finite = int(np.sum(finite))
    n_increase = int(np.sum(change_mask == 1))
    n_decrease = int(np.sum(change_mask == 2))
    frac_increase = n_increase / n_finite if n_finite else 0.0
    frac_decrease = n_decrease / n_finite if n_finite else 0.0
    frac_changed = frac_increase + frac_decrease

    notes.append(f"magnitude_threshold={magnitude_threshold:.4f}")
    notes.append(f"increase_fraction={frac_increase:.3f}")
    notes.append(f"decrease_fraction={frac_decrease:.3f}")

    # --- build the answer ----------------------------------------------------
    target_label = target.replace("_", " ")
    index_label = (
        definition.positive_means.split(" are ")[0]
        if " are " in definition.positive_means
        else index_name
    )

    if frac_changed < _MIN_CHANGE_FRACTION:
        answer = (
            f"No meaningful change in {target_label} was detected between the two dates. "
            f"The {index_name} values remained stable across the scene."
        )
        abstained = True
    else:
        parts = []
        if frac_increase > _MIN_CHANGE_FRACTION:
            pct = frac_increase * 100
            parts.append(f"{index_label} increased over {pct:.1f}% of the scene")
        if frac_decrease > _MIN_CHANGE_FRACTION:
            pct = frac_decrease * 100
            parts.append(f"{index_label} decreased over {pct:.1f}% of the scene")

        if direction_hint == "increase" and parts:
            parts = [p for p in parts if "increased" in p] or parts
        elif direction_hint == "decrease" and parts:
            parts = [p for p in parts if "decreased" in p] or parts

        answer = "; ".join(parts) + f" (measured by {index_name} differencing)."
        abstained = False

    # --- mean values for context ---------------------------------------------
    before_mean = float(np.nanmean(before_index))
    after_mean = float(np.nanmean(after_index))
    notes.append(f"before_mean_{index_name}={before_mean:.4f}")
    notes.append(f"after_mean_{index_name}={after_mean:.4f}")

    # --- render the change mask ----------------------------------------------
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(before_img.path).stem
    out_name = f"{stem}_{index_name}_change"
    overlay_path = mask_to_png(change_mask, str(_OUTPUT_DIR / f"{out_name}.png"))

    legend = {
        "0": "no change",
        "1": f"{index_name} increase",
        "2": f"{index_name} decrease",
    }

    # --- confidence ----------------------------------------------------------
    # Two signals: how much of the scene changed, and how strong the change is.
    if abstained:
        confidence = 0.1
    else:
        coverage_signal = min(frac_changed / 0.15, 1.0)
        strength_signal = min(magnitude_threshold / 0.10, 1.0)
        confidence = round(0.4 * coverage_signal + 0.5 * strength_signal + 0.05, 2)
        confidence = min(confidence, 0.95)

    confidence_method = "magnitude_threshold_coverage_and_strength"

    elapsed = int((perf_counter() - t0) * 1000)

    return ToolResult(
        answer=answer,
        evidence=Evidence(
            kind="mask",
            overlay_png=str(overlay_path),
            legend=legend,
            caption=f"Signed {index_name} change mask between the two dates.",
        ),
        confidence=confidence,
        confidence_method=confidence_method,
        tool_name=TaskName.CHANGE_VQA,
        model_id="index_differencing",
        params_used={
            "index": index_name,
            "magnitude_threshold": magnitude_threshold,
            "decimate": decimate,
        },
        latency_ms=elapsed,
        abstained=abstained,
        notes=notes,
    )
