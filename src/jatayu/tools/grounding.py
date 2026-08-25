"""Text-guided localisation of regions in a single satellite image.

Grounding deliberately uses physics-first spectral indices rather than a
learned detector. The query selects an appropriate index, the index produces
a scene-adaptive binary mask, connected components turn pixels into candidate
objects, and object-level features rank those candidates.

All bounding boxes returned by this module are PIXEL coordinates of the
referenced image:

    (x_min, y_min, x_max, y_max)

The report/geospatial layer is responsible for converting them through the
raster transform into geographic coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal
import re

import numpy as np
from scipy import ndimage

from jatayu.analysis.indices import INDEX_REGISTRY, IndexAnalyser
from jatayu.io.loader import read_bands
from jatayu.render import boxes_to_png
from jatayu.schemas import (
    Evidence,
    TaskFamily,
    TaskName,
    ToolRequest,
    ToolResult,
)
from jatayu.tools.registry import register


_MIN_COMPONENT_PIXELS = 9
_CONNECTIVITY = np.ones((3, 3), dtype=np.uint8)

_AREA_WEIGHT = 0.40
_STRENGTH_WEIGHT = 0.25
_COMPACTNESS_WEIGHT = 0.20
_EDGE_WEIGHT = 0.15

_MIN_SCORE_FOR_MATCH = 0.10


@dataclass(frozen=True, slots=True)
class CandidateRegion:
    """Object-level measurements used to rank a connected component."""

    label: int
    bbox: tuple[int, int, int, int]
    area: int
    mean_strength: float
    compactness: float
    edge_fraction: float
    centroid_x: float
    centroid_y: float
    score: float = 0.0


@dataclass(frozen=True, slots=True)
class QueryIntent:
    """Small deterministic representation of grounding language."""

    selection: Literal["largest", "all", "single"] = "single"
    direction: Literal["north", "south", "east", "west"] | None = None


def _parse_intent(query: str) -> QueryIntent:
    """Extract only spatial/superlative language that changes object selection.

    This is intentionally deterministic. The router has already converted the
    natural-language query to English, so grounding should not introduce
    another LLM dependency merely to interpret words such as "largest".
    """
    text = query.lower()

    if re.search(r"\b(all|every|each)\b", text):
        selection: Literal["largest", "all", "single"] = "all"
    elif re.search(
        r"\b(largest|biggest|main|major|most extensive)\b",
        text,
    ):
        selection = "largest"
    else:
        selection = "single"

    direction: Literal["north", "south", "east", "west"] | None = None

    if re.search(r"\b(north|northern|top)\b", text):
        direction = "north"
    elif re.search(r"\b(south|southern|bottom)\b", text):
        direction = "south"
    elif re.search(r"\b(east|eastern|right)\b", text):
        direction = "east"
    elif re.search(r"\b(west|western|left)\b", text):
        direction = "west"

    return QueryIntent(
        selection=selection,
        direction=direction,
    )


def _normalise(values: np.ndarray) -> np.ndarray:
    """Min-max normalise finite values while remaining stable for constants."""
    result = np.zeros_like(
        values,
        dtype=np.float64,
    )

    finite = np.isfinite(values)

    if not finite.any():
        return result

    minimum = np.nanmin(values)
    maximum = np.nanmax(values)

    if maximum <= minimum:
        result[finite] = 1.0
        return result

    result[finite] = (
        values[finite] - minimum
    ) / (
        maximum - minimum
    )

    return result


def _component_features(
    labels: np.ndarray,
    index: np.ndarray,
    label: int,
) -> CandidateRegion:
    """Calculate shape, spectral and spatial features for one object."""
    rows, cols = np.where(labels == label)

    area = int(rows.size)

    y_min = int(rows.min())
    y_max = int(rows.max()) + 1
    x_min = int(cols.min())
    x_max = int(cols.max()) + 1

    bbox_area = max(
        1,
        (x_max - x_min) * (y_max - y_min),
    )

    # Compactness based on area / bounding-box area. A compact circular object
    # gets a higher value than a thin or highly fragmented object.
    compactness = min(
        1.0,
        area / bbox_area,
    )

    values = index[labels == label]
    finite = values[np.isfinite(values)]

    mean_strength = (
        float(np.mean(finite))
        if finite.size
        else float("nan")
    )

    return CandidateRegion(
        label=label,
        bbox=(x_min, y_min, x_max, y_max),
        area=area,
        mean_strength=mean_strength,
        compactness=compactness,
        edge_fraction=0.0,
        centroid_x=float(np.mean(cols)),
        centroid_y=float(np.mean(rows)),
    )


def _add_edge_fraction(
    candidate: CandidateRegion,
    image_shape: tuple[int, int],
    labels: np.ndarray,
) -> CandidateRegion:
    """Measure how much of an object touches the image boundary."""
    height, width = image_shape

    x_min, y_min, x_max, y_max = candidate.bbox

    edge_mask = (
        (x_min == 0)
        | (y_min == 0)
        | (x_max == width)
        | (y_max == height)
    )

    if not edge_mask:
        return candidate

    rows, cols = np.where(labels == candidate.label)

    touching = (
        (rows == 0)
        | (rows == height - 1)
        | (cols == 0)
        | (cols == width - 1)
    )

    edge_fraction = float(
        np.mean(touching)
    )

    return CandidateRegion(
        label=candidate.label,
        bbox=candidate.bbox,
        area=candidate.area,
        mean_strength=candidate.mean_strength,
        compactness=candidate.compactness,
        edge_fraction=edge_fraction,
        centroid_x=candidate.centroid_x,
        centroid_y=candidate.centroid_y,
        score=candidate.score,
    )


def _rank_candidates(
    candidates: list[CandidateRegion],
    image_shape: tuple[int, int],
    direction: str | None,
) -> list[CandidateRegion]:
    """Rank objects using area, spectral strength, shape and edge completeness.

    Area is deliberately not the sole criterion.

    - 40% area: large real objects should generally outrank speckle.
    - 25% index strength: spectrally convincing objects should win ties.
    - 20% compactness: rejects thin/fragmented false positives.
    - 15% edge completeness: objects touching the frame are penalised because
      their observed area is probably only a truncated part of the object.

    A directional query adds a deterministic spatial score after the physical
    object score. This allows "water body in the north" to differ from
    "largest water body".
    """
    if not candidates:
        return []

    height, width = image_shape

    areas = np.array(
        [candidate.area for candidate in candidates],
        dtype=np.float64,
    )

    strengths = np.array(
        [
            candidate.mean_strength
            for candidate in candidates
        ],
        dtype=np.float64,
    )

    compactness = np.array(
        [
            candidate.compactness
            for candidate in candidates
        ],
        dtype=np.float64,
    )

    # Edge completeness means "not touching the edge".
    edge_completeness = 1.0 - np.array(
        [
            candidate.edge_fraction
            for candidate in candidates
        ],
        dtype=np.float64,
    )

    area_score = _normalise(areas)
    strength_score = _normalise(strengths)
    compactness_score = compactness
    edge_score = edge_completeness

    score = (
        _AREA_WEIGHT * area_score
        + _STRENGTH_WEIGHT * strength_score
        + _COMPACTNESS_WEIGHT * compactness_score
        + _EDGE_WEIGHT * edge_score
    )

    if direction:
        x = np.array(
            [candidate.centroid_x / width for candidate in candidates],
            dtype=np.float64,
        )
        y = np.array(
            [candidate.centroid_y / height for candidate in candidates],
            dtype=np.float64,
        )

        if direction == "north":
            spatial = 1.0 - y
        elif direction == "south":
            spatial = y
        elif direction == "west":
            spatial = 1.0 - x
        else:
            spatial = x

        # Direction modifies ranking rather than completely overriding
        # physical evidence.
        score = 0.75 * score + 0.25 * spatial

    ranked: list[CandidateRegion] = []

    for candidate, candidate_score in zip(
        candidates,
        score,
        strict=True,
    ):
        ranked.append(
            CandidateRegion(
                label=candidate.label,
                bbox=candidate.bbox,
                area=candidate.area,
                mean_strength=candidate.mean_strength,
                compactness=candidate.compactness,
                edge_fraction=candidate.edge_fraction,
                centroid_x=candidate.centroid_x,
                centroid_y=candidate.centroid_y,
                score=float(candidate_score),
            )
        )

    ranked.sort(
        key=lambda candidate: candidate.score,
        reverse=True,
    )

    return ranked


def _make_overlay_path(
    req: ToolRequest,
    index_name: str,
) -> Path:
    """Create a deterministic local output path for the grounding overlay."""
    source = Path(req.images[0].path)

    output_dir = Path(
        req.params.get(
            "output_dir",
            source.parent / "jatayu_outputs",
        )
    )

    return output_dir / f"{source.stem}_{index_name.lower()}_grounding.png"


def _abstained(
    query: str,
    reason: str,
    *,
    latency_ms: int,
    notes: list[str] | None = None,
) -> ToolResult:
    """Construct an explicit abstention using the existing ToolResult contract."""
    all_notes = [
        "abstained=True",
        reason,
    ]

    if notes:
        all_notes.extend(notes)

    return ToolResult(
        answer=(
            "I could not reliably localise the requested region. "
            f"{reason}"
        ),
        evidence=Evidence(
            kind="none",
            caption=None,
        ),
        confidence=0.0,
        confidence_method=f"abstained: {reason}",
        tool_name=TaskName.GROUNDING,
        model_id="spectral-index-connected-components-v1",
        latency_ms=latency_ms,
        notes=all_notes,
    )


def _choose_index(query: str) -> str | None:
    """Select an index only when the query has a supported physical target.

    Grounding must abstain rather than use a semantically unrelated index.
    This is especially important for targets such as snow when NDSI is not
    registered: NDVI, for example, is not a valid substitute for snow extent.
    """
    text = query.lower()

    unsupported_targets = {
        "snow": ("snow", "snowfall", "snow cover", "snow-covered"),
        "ice": ("ice", "ice cover", "sea ice"),
    }

    for target, terms in unsupported_targets.items():
        if any(term in text for term in terms):
            return None

    analyser = IndexAnalyser()
    candidates = analyser.select_indices(query)

    if not candidates:
        return None

    return candidates[0]
def _load_index(
    req: ToolRequest,
    index_name: str,
    *,
    decimate: int,
) -> np.ndarray:
    """Read only the bands required by the selected index."""
    definition = INDEX_REGISTRY[index_name]

    arrays = read_bands(
        req.images[0],
        list(definition.required_bands),
        decimate=decimate,
    )

    bands = {
        band: array
        for band, array in zip(
            definition.required_bands,
            arrays,
            strict=True,
        )
    }

    return IndexAnalyser().compute(
        bands,
        [index_name],
    )[index_name]


def _find_candidates(
    index: np.ndarray,
    index_name: str,
    *,
    min_component_pixels: int,
) -> list[CandidateRegion]:
    """Threshold an index and turn surviving pixels into connected objects."""
    analyser = IndexAnalyser()

    mask, metadata = analyser.classify(
        index_name,
        index,
    )

    if metadata["floor_disabled"]:
        return []

    labels, count = ndimage.label(
        mask,
        structure=_CONNECTIVITY,
    )

    if count == 0:
        return []

    objects = ndimage.find_objects(labels)

    candidates: list[CandidateRegion] = []

    for label, component_slice in enumerate(
        objects,
        start=1,
    ):
        if component_slice is None:
            continue

        area = int(
            np.count_nonzero(
                labels[component_slice] == label
            )
        )

        if area < min_component_pixels:
            continue

        candidate = _component_features(
            labels,
            index,
            label,
        )

        candidate = _add_edge_fraction(
            candidate,
            index.shape,
            labels,
        )

        candidates.append(candidate)

    return candidates


def _scale_bbox(
    bbox: tuple[int, int, int, int],
    decimate: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Convert a decimated-read bbox back to original image pixel coordinates."""
    x_min, y_min, x_max, y_max = bbox

    scaled = (
        x_min * decimate,
        y_min * decimate,
        min(x_max * decimate, width),
        min(y_max * decimate, height),
    )

    if scaled[2] <= scaled[0] or scaled[3] <= scaled[1]:
        raise ValueError(
            f"Scaled bbox became invalid: {bbox} -> {scaled}"
        )

    return scaled


@register(
    TaskName.GROUNDING,
    families={TaskFamily.SINGLE_IMAGE},
    description=(
        "Localises regions referred to by a query using spectral indices and "
        "connected components. Use when the user asks to highlight, locate, "
        "box, or identify a specific region or object in an image."
    ),
)
def run(req: ToolRequest) -> ToolResult:
    """Localise query-relevant regions in one raster."""
    started = perf_counter()

    if not req.images:
        return _abstained(
            req.query,
            "No image was supplied.",
            latency_ms=0,
        )

    image = req.images[0]

    try:
        decimate = int(
            req.params.get(
                "decimate",
                max(
                    1,
                    int(
                        max(
                            image.width,
                            image.height,
                        )
                        / 4000
                    ),
                ),
            )
        )
    except (TypeError, ValueError):
        decimate = 1

    decimate = max(1, decimate)

    try:
        min_component_pixels = int(
            req.params.get(
                "min_component_pixels",
                _MIN_COMPONENT_PIXELS,
            )
        )
    except (TypeError, ValueError):
        min_component_pixels = _MIN_COMPONENT_PIXELS

    min_component_pixels = max(
        1,
        min_component_pixels,
    )

    index_name = _choose_index(req.query)

    if index_name is None:
        elapsed = int(
            (perf_counter() - started) * 1000
        )

        return _abstained(
            req.query,
            "The query did not map to a supported spectral index.",
            latency_ms=elapsed,
        )

    definition = INDEX_REGISTRY[index_name]

    try:
        index = _load_index(
            req,
            index_name,
            decimate=decimate,
        )
    except (KeyError, ValueError) as exc:
        elapsed = int(
            (perf_counter() - started) * 1000
        )

        return _abstained(
            req.query,
            (
                f"The selected index {index_name} cannot be computed from "
                f"this scene: {exc}"
            ),
            latency_ms=elapsed,
        )

    if not np.isfinite(index).any():
        elapsed = int(
            (perf_counter() - started) * 1000
        )

        return _abstained(
            req.query,
            (
                f"The selected index {index_name} contains no valid pixels "
                "in this scene."
            ),
            latency_ms=elapsed,
        )

    candidates = _find_candidates(
        index,
        index_name,
        min_component_pixels=min_component_pixels,
    )

    if not candidates:
        elapsed = int(
            (perf_counter() - started) * 1000
        )

        return _abstained(
            req.query,
            (
                f"No region exceeded the physical threshold for {index_name} "
                f"and the minimum component size of "
                f"{min_component_pixels} pixels."
            ),
            latency_ms=elapsed,
        )

    intent = _parse_intent(req.query)

    ranked = _rank_candidates(
        candidates,
        index.shape,
        intent.direction,
    )

    if not ranked:
        elapsed = int(
            (perf_counter() - started) * 1000
        )

        return _abstained(
            req.query,
            "No candidate region survived object-level ranking.",
            latency_ms=elapsed,
        )

    if ranked[0].score < _MIN_SCORE_FOR_MATCH:
        elapsed = int(
            (perf_counter() - started) * 1000
        )

        return _abstained(
            req.query,
            "Candidate regions were too weak to support a reliable match.",
            latency_ms=elapsed,
        )

    if intent.selection == "all":
        selected = ranked
    else:
        selected = ranked[:1]

    boxes = [
        _scale_bbox(
            candidate.bbox,
            decimate,
            image.width,
            image.height,
        )
        for candidate in selected
    ]

    # A grounding overlay is rendered at the decimated analysis resolution.
    # The boxes are first scaled back to original pixel coordinates, then
    # converted back to overlay coordinates so that the evidence image and
    # reported boxes remain internally consistent.
    overlay_boxes = [
        (
            box[0] // decimate,
            box[1] // decimate,
            max(
                box[0] // decimate + 1,
                box[2] // decimate,
            ),
            max(
                box[1] // decimate + 1,
                box[3] // decimate,
            ),
        )
        for box in boxes
    ]

    output_path = _make_overlay_path(
        req,
        index_name,
    )

    try:
        boxes_to_png(
            overlay_boxes,
            output_path,
            shape=index.shape,
        )
    except (OSError, ValueError) as exc:
        elapsed = int(
            (perf_counter() - started) * 1000
        )

        return _abstained(
            req.query,
            f"The region was found, but the evidence overlay failed: {exc}",
            latency_ms=elapsed,
        )

    best = ranked[0]

    strength = (
        best.mean_strength
        if np.isfinite(best.mean_strength)
        else 0.0
    )

    confidence = float(
        np.clip(
            0.45
            + 0.25 * best.score
            + 0.20 * best.compactness
            + 0.10 * np.clip(abs(strength), 0.0, 1.0),
            0.0,
            0.95,
        )
    )

    if intent.selection == "all":
        description = (
            f"Found {len(selected)} candidate "
            f"{index_name} region(s)."
        )
    elif intent.selection == "largest":
        description = (
            f"Localised the largest relevant {index_name} region."
        )
    else:
        description = (
            f"Localised the most relevant {index_name} region."
        )

    if intent.direction:
        description += (
            f" The ranking was adjusted for the requested "
            f"{intent.direction} direction."
        )

    elapsed = int(
        (perf_counter() - started) * 1000
    )

    return ToolResult(
        answer=description,
        evidence=Evidence(
            kind="bbox",
            overlay_png=str(output_path),
            legend={
                "red box": f"candidate {index_name} region",
            },
            caption=(
                "Bounding boxes are pixel coordinates in the referenced image. "
                "Geographic conversion is performed by the report layer."
            ),
        ),
        confidence=confidence,
        confidence_method=(
            "deterministic object-ranking score combining area, "
            "spectral strength, compactness and edge completeness"
        ),
        tool_name=TaskName.GROUNDING,
        model_id="spectral-index-connected-components-v1",
        latency_ms=elapsed,
        notes=[
            f"index={index_name}",
            f"components_after_filter={len(ranked)}",
            f"minimum_component_pixels={min_component_pixels}",
            f"decimate={decimate}",
            (
                "Minimum component size is 9 pixels by default to suppress "
                "isolated index/SAR speckle while retaining small objects."
            ),
            (
                "Bounding boxes are original-image pixel coordinates, "
                "not geographic coordinates."
            ),
            (
                "Physical floor was enforced before connected-component "
                "ranking."
            ),
        ],
    )


__all__ = [
    "CandidateRegion",
    "QueryIntent",
    "run",
]
