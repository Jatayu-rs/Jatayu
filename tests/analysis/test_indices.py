import numpy as np
import pytest

from jatayu.analysis.indices import INDEX_REGISTRY, IndexAnalyser


BANDS = {
    "blue": np.array([[0.1, 0.12]]),
    "green": np.array([[0.2, 0.22]]),
    "red": np.array([[0.3, 0.32]]),
    "rededge1": np.array([[0.4, 0.42]]),
    "nir": np.array([[0.6, 0.62]]),
    "swir1": np.array([[0.2, 0.21]]),
    "swir2": np.array([[0.1, 0.11]]),
}


def test_every_registered_formula_has_a_hand_computed_value():
    analyser = IndexAnalyser()

    scalar_bands = {
        key: np.array([value[0, 0]])
        for key, value in BANDS.items()
    }

    values = analyser.compute(
        scalar_bands,
        list(INDEX_REGISTRY),
    )

    expected = {
        "MNDWI": 0.0,
        "NDWI": -0.5,
        "LSWI": 0.5,
        "NDVI": 1 / 3,
        "EVI": (
            2.5 * 0.3
            / (0.6 + 1.8 - 0.75 + 1)
        ),
        "SAVI": 1.5 * 0.3 / 1.4,
        "NDRE": 0.2,
        "CMR": 1 / 3 - (-0.4 / 0.8),
        "NDMI": 0.5,
        "NDBI": -0.5,
        "UI": -5 / 7,
        "BAI": 1 / (0.2**2 + 0.54**2),
        "BSI": -0.2 / 1.2,
        "RI": 0.3**2 / (0.1 * 0.2**3),
        "CLAY_RATIO": 2.0,
        "FERROUS_RATIO": 1 / 3,
        "NBR": 0.5 / 0.7,
        "NBR2": 1 / 3,
        "NDTI": 0.1 / 0.5,
        "CHLOROPHYLL_RATIO": 0.5,
        "SALINITY_PROXY": -1 / 3,
        
    }

    assert set(values) == set(expected)

    for name, want in expected.items():
        np.testing.assert_allclose(
            values[name],
            want,
            rtol=1e-12,
            atol=1e-12,
        )


@pytest.mark.parametrize(
    "name",
    [
        "MNDWI",
        "NDWI",
        "LSWI",
        "NDVI",
        "NDRE",
        "NDMI",
        "NDBI",
        "UI",
        "BSI",
        "NBR",
        "NBR2",
        "NDTI",
        "SALINITY_PROXY",
    
    ],
)
def test_normalised_indices_stay_in_range_for_physical_finite_inputs(name):
    rng = np.random.default_rng(42)

    bands = {
        key: rng.uniform(
            0.001,
            1.0,
            size=(64, 64),
        )
        for key in BANDS
    }

    arr = IndexAnalyser().compute(
        bands,
        [name],
    )[name]

    assert np.all(np.isfinite(arr))
    assert np.all(arr >= -1.0)
    assert np.all(arr <= 1.0)


def test_zero_denominator_is_zero_not_inf_or_nan():
    analyser = IndexAnalyser()

    bands = {
        key: np.ones((2, 2))
        for key in BANDS
    }

    bands["green"][:] = 0
    bands["nir"][:] = 0

    result = analyser.compute(
        bands,
        ["NDWI"],
    )["NDWI"]

    assert np.array_equal(
        result,
        np.zeros((2, 2)),
    )
    assert np.all(np.isfinite(result))


def test_all_nan_band_propagates_nan():
    analyser = IndexAnalyser()

    bands = {
        key: value.copy()
        for key, value in BANDS.items()
    }

    bands["nir"][:] = np.nan

    for name in (
        "NDVI",
        "NDMI",
        "NDBI",
        "UI",
        "NDRE",
    ):
        result = analyser.compute(
            bands,
            [name],
        )[name]

        assert np.all(np.isnan(result))


def test_compute_rejects_shape_mismatch():
    bands = {
        key: value.copy()
        for key, value in BANDS.items()
    }

    bands["nir"] = np.zeros((2, 2))

    with pytest.raises(
        ValueError,
        match="identical shapes",
    ):
        IndexAnalyser().compute(
            bands,
            ["NDVI"],
        )


def test_compute_rejects_missing_required_band():
    bands = {
        key: value
        for key, value in BANDS.items()
        if key != "swir1"
    }

    with pytest.raises(
        ValueError,
        match="missing",
    ):
        IndexAnalyser().compute(
            bands,
            ["MNDWI"],
        )


def test_classify_disables_positive_class_below_floor():
    arr = np.full(
        (10, 10),
        0.05,
    )

    mask, metadata = IndexAnalyser().classify(
        "NDVI",
        arr,
    )

    assert not mask.any()
    assert metadata["floor_disabled"] is True
    assert "physical floor" in metadata["note"]


def test_classify_disables_negative_class_above_floor():
    arr = np.full(
        (10, 10),
        -0.05,
    )

    mask, metadata = IndexAnalyser().classify(
        "NBR",
        arr,
    )

    assert not mask.any()
    assert metadata["floor_disabled"] is True
    assert "physical floor" in metadata["note"]


def test_classify_uses_floor_as_bound_and_reports_fraction():
    arr = np.linspace(
        0.0,
        0.8,
        100,
    ).reshape(10, 10)

    mask, metadata = IndexAnalyser().classify(
        "NDVI",
        arr,
        percentile=90,
    )

    assert metadata["floor_disabled"] is False
    assert metadata["threshold"] >= 0.1
    assert metadata["fraction_selected"] == pytest.approx(
        mask.mean()
    )


def test_classify_negative_polarity_selects_low_tail():
    arr = np.linspace(
        -0.8,
        0.8,
        100,
    ).reshape(10, 10)

    mask, metadata = IndexAnalyser().classify(
        "NBR",
        arr,
        percentile=90,
    )

    assert metadata["floor_disabled"] is False
    assert metadata["threshold"] <= -0.1
    assert np.all(
        arr[mask] <= metadata["threshold"]
    )


def test_all_nan_classification_is_disabled_without_warning():
    arr = np.full(
        (3, 3),
        np.nan,
    )

    mask, metadata = IndexAnalyser().classify(
        "NDVI",
        arr,
    )

    assert not mask.any()
    assert metadata["floor_disabled"] is True
    assert metadata["threshold"] is None


def test_diff_identity_is_exactly_zero():
    x = np.arange(
        100,
        dtype=np.float64,
    ).reshape(10, 10)

    result = IndexAnalyser().diff(
        x,
        x,
    )

    assert np.array_equal(
        result,
        np.zeros_like(x),
    )


def test_diff_rejects_shape_mismatch():
    with pytest.raises(
        ValueError,
        match="identical shapes",
    ):
        IndexAnalyser().diff(
            np.zeros((2, 2)),
            np.zeros((2, 3)),
        )


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("is my paddy stressed", "NDRE"),
        ("mangrove health", "CMR"),
        ("illegal mining", "FERROUS_RATIO"),
        ("burn scar", "NBR"),
        ("how turbid is the lagoon", "NDTI"),
        ("map river water", "MNDWI"),
        ("crop moisture", "NDMI"),
        ("urban expansion", "NDBI"),
        ("clay minerals", "CLAY_RATIO"),
        
        ("bare soil", "BSI"),
        ("iron ore exploration", "FERROUS_RATIO"),
    ],
)
def test_select_indices_realistic_queries(
    query,
    expected,
):
    results = IndexAnalyser().select_indices(query)

    assert results
    assert expected in results[:3]


def test_select_indices_is_empty_for_empty_query():
    assert IndexAnalyser().select_indices("   ") == []

def test_ratio_indices_are_scale_invariant_or_reject_dn():
    """Normalised differences cancel DN scaling. Power ratios and indices with
    absolute constants do not — they must reject unscaled input rather than
    returning a silently meaningless number."""
    refl = {k: v.copy() for k, v in BANDS.items()}
    dn = {k: v * 10000 for k, v in BANDS.items()}
    analyser = IndexAnalyser()

    for name in ("NDVI", "MNDWI", "CLAY_RATIO", "FERROUS_RATIO"):
        a = analyser.compute(refl, [name])[name]
        b = analyser.compute(dn, [name])[name]
        np.testing.assert_allclose(a, b, rtol=1e-9)

    for name in ("RI", "BAI"):
        with pytest.raises(ValueError, match="reflectance"):
            analyser.compute(dn, [name])
def test_unknown_index_and_invalid_percentile_are_rejected():
    analyser = IndexAnalyser()

    with pytest.raises(
        KeyError,
        match="Unknown spectral index",
    ):
        analyser.compute(
            BANDS,
            ["NOT_AN_INDEX"],
        )

    with pytest.raises(
        KeyError,
        match="Unknown spectral index",
    ):
        analyser.classify(
            "NOT_AN_INDEX",
            np.ones((2, 2)),
        )

    with pytest.raises(
        ValueError,
        match="percentile",
    ):
        analyser.classify(
            "NDVI",
            np.ones((2, 2)),
            percentile=0,
        )
