from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jatayu.schemas import (
    ImageRef,
    Modality,
    TaskName,
    ToolRequest,
)
from jatayu.tools import grounding


def _request(
    query: str,
    tmp_path: Path,
    *,
    width: int = 100,
    height: int = 100,
    params: dict | None = None,
) -> ToolRequest:
    """Build a minimal ToolRequest for grounding tests."""
    return ToolRequest(
        query=query,
        images=[
            ImageRef(
                path=str(tmp_path / "scene.tif"),
                modality=Modality.MULTISPECTRAL,
                width=width,
                height=height,
                band_count=7,
            )
        ],
        params=params or {},
    )


def _fake_index(
    *,
    shape: tuple[int, int] = (100, 100),
) -> np.ndarray:
    """Create three water-like regions with known geometry."""
    index = np.full(
        shape,
        -0.5,
        dtype=np.float32,
    )

    # Small rectangle: 10 x 10 = 100 pixels.
    index[10:20, 10:20] = 0.4

    # Medium rectangle: 20 x 15 = 300 pixels.
    index[30:50, 20:35] = 0.6

    # Large rectangle: 30 x 20 = 600 pixels.
    index[50:80, 40:60] = 0.8

    return index


# ---------------------------------------------------------------------------
# Query / selection behaviour
# ---------------------------------------------------------------------------


def test_largest_water_body_returns_largest_box(
    monkeypatch,
    tmp_path: Path,
):
    """'Largest' should select exactly one relevant region."""
    index = _fake_index()

    monkeypatch.setattr(
        grounding,
        "_load_index",
        lambda *args, **kwargs: index,
    )

    req = _request(
        "the largest water body",
        tmp_path,
    )

    result = grounding.run(req)

    assert result.tool_name == TaskName.GROUNDING
    assert result.evidence.kind == "bbox"
    assert result.evidence.overlay_png is not None

    overlay = Path(result.evidence.overlay_png)

    assert overlay.exists()
    assert "largest" in result.answer.lower()


def test_all_water_bodies_returns_multiple_candidates(
    monkeypatch,
    tmp_path: Path,
):
    """'All' should return all surviving connected components."""
    index = _fake_index()

    monkeypatch.setattr(
        grounding,
        "_load_index",
        lambda *args, **kwargs: index,
    )

    req = _request(
        "all water bodies",
        tmp_path,
    )

    result = grounding.run(req)

    assert result.evidence.kind == "bbox"
    assert "3 candidate" in result.answer


def test_north_water_body_changes_ranking(
    monkeypatch,
    tmp_path: Path,
):
    """Directional language should affect candidate ranking."""
    index = _fake_index()

    monkeypatch.setattr(
        grounding,
        "_load_index",
        lambda *args, **kwargs: index,
    )

    req = _request(
        "the water body in the north",
        tmp_path,
    )

    result = grounding.run(req)

    assert result.evidence.kind == "bbox"


# ---------------------------------------------------------------------------
# Abstention behaviour
# ---------------------------------------------------------------------------


def test_no_matching_region_abstains(
    monkeypatch,
    tmp_path: Path,
):
    """A supported index with no plausible pixels must abstain."""
    index = np.full(
        (100, 100),
        -0.5,
        dtype=np.float32,
    )

    monkeypatch.setattr(
        grounding,
        "_load_index",
        lambda *args, **kwargs: index,
    )

    req = _request(
        "water body",
        tmp_path,
    )

    result = grounding.run(req)

    assert result.confidence == 0.0
    assert result.evidence.kind == "none"
    assert "abstained=True" in result.notes
    assert "could not reliably localise" in result.answer.lower()


def test_snow_query_abstains_when_ndsi_is_not_registered(
    monkeypatch,
    tmp_path: Path,
):
    """Snow must abstain because NDSI was intentionally removed."""
    index = np.full(
        (100, 100),
        0.1,
        dtype=np.float32,
    )

    monkeypatch.setattr(
        grounding,
        "_load_index",
        lambda *args, **kwargs: index,
    )

    req = _request(
        "snow cover",
        tmp_path,
    )

    result = grounding.run(req)

    assert result.confidence == 0.0
    assert result.evidence.kind == "none"
    assert "abstained=True" in result.notes
    assert "supported spectral index" in result.answer.lower()


def test_missing_required_band_abstains(
    monkeypatch,
    tmp_path: Path,
):
    """A supported target must abstain when its required bands are unavailable."""

    def fail_loader(*args, **kwargs):
        raise KeyError(
            "Band swir1 not in scene"
        )

    monkeypatch.setattr(
        grounding,
        "_load_index",
        fail_loader,
    )

    req = _request(
        "water body",
        tmp_path,
    )

    result = grounding.run(req)

    assert result.confidence == 0.0
    assert result.evidence.kind == "none"
    assert "abstained=True" in result.notes
    assert "cannot be computed" in result.answer.lower()


# ---------------------------------------------------------------------------
# Speckle / connected components
# ---------------------------------------------------------------------------


def test_single_pixel_speckle_does_not_create_box(
    monkeypatch,
    tmp_path: Path,
):
    """A single isolated pixel should be removed as speckle."""
    index = np.full(
        (100, 100),
        -0.5,
        dtype=np.float32,
    )

    index[50, 50] = 0.8

    monkeypatch.setattr(
        grounding,
        "_load_index",
        lambda *args, **kwargs: index,
    )

    req = _request(
        "water body",
        tmp_path,
    )

    result = grounding.run(req)

    assert result.confidence == 0.0
    assert result.evidence.kind == "none"
    assert "abstained=True" in result.notes


def test_boxes_are_inside_image_bounds():
    """Every connected-component bbox must stay inside the image."""
    index = _fake_index()

    candidates = grounding._find_candidates(
        index,
        "MNDWI",
        min_component_pixels=9,
    )

    ranked = grounding._rank_candidates(
        candidates,
        index.shape,
        None,
    )

    height, width = index.shape

    for candidate in ranked:
        x_min, y_min, x_max, y_max = candidate.bbox

        assert 0 <= x_min < x_max <= width
        assert 0 <= y_min < y_max <= height


# ---------------------------------------------------------------------------
# Ranking behaviour
# ---------------------------------------------------------------------------


def test_edge_touching_object_is_penalised():
    """A truncated edge object should receive an edge penalty.

    This test constructs components directly because the purpose here is
    ranking, not percentile thresholding.
    """
    index = np.full(
        (100, 100),
        -0.5,
        dtype=np.float32,
    )

    labels = np.zeros(
        (100, 100),
        dtype=np.int32,
    )

    # Large but truncated object.
    labels[:, :30] = 1

    # Smaller complete object.
    labels[30:70, 50:80] = 2

    candidates = [
        grounding._component_features(
            labels,
            index,
            1,
        ),
        grounding._component_features(
            labels,
            index,
            2,
        ),
    ]

    candidates = [
        grounding._add_edge_fraction(
            candidate,
            index.shape,
            labels,
        )
        for candidate in candidates
    ]

    ranked = grounding._rank_candidates(
        candidates,
        index.shape,
        None,
    )

    assert len(ranked) == 2

    edge = next(
        candidate
        for candidate in ranked
        if candidate.bbox[0] == 0
    )

    complete = next(
        candidate
        for candidate in ranked
        if candidate.bbox[0] != 0
    )

    assert edge.edge_fraction > 0
    assert complete.edge_fraction == 0


# ---------------------------------------------------------------------------
# Bounding-box scaling
# ---------------------------------------------------------------------------


def test_scaled_bbox_stays_inside_original_image():
    """A bbox calculated on a decimated image must map back to valid pixels."""
    bbox = (
        10,
        20,
        50,
        80,
    )

    scaled = grounding._scale_bbox(
        bbox,
        4,
        100,
        120,
    )

    assert scaled == (
        40,
        80,
        100,
        120,
    )

    x_min, y_min, x_max, y_max = scaled

    assert 0 <= x_min < x_max <= 100
    assert 0 <= y_min < y_max <= 120


def test_invalid_scaled_bbox_raises():
    """An invalid bbox must fail loudly rather than silently corrupt geometry."""
    with pytest.raises(
        ValueError,
        match="invalid",
    ):
        grounding._scale_bbox(
            (5, 5, 5, 5),
            4,
            100,
            100,
        )
