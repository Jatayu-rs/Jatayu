import numpy as np

from jatayu.analysis.change import (
    analyse_change,
    signed_change_mask,
)

from jatayu.analysis.sar import (
    rvi,
    vv_vh_ratio,
    log_backscatter_difference,
)

from jatayu.analysis.landcover import (
    classify_landcover,
)

from jatayu.analysis.agriculture import (
    crop_stress_map,
    classify_field_state,
)

from jatayu.analysis.risk import (
    composite_risk,
)


def test_change_detects_real_increase():
    before = np.zeros((10, 10), dtype=np.float32)

    after = np.zeros((10, 10), dtype=np.float32)

    # 25% of the scene changes strongly.
    after[:5, :5] = 0.5

    result = analyse_change(
        before,
        after,
        percentile=95,
        floor=0.02,
    )

    assert result.changed_fraction > 0
    assert result.increase_fraction > 0
    assert result.decrease_fraction == 0
    assert result.mean_delta > 0


def test_change_detects_decrease():
    before = np.ones((10, 10), dtype=np.float32)

    after = np.ones((10, 10), dtype=np.float32)

    after[:5, :5] = 0.2

    result = analyse_change(
        before,
        after,
        percentile=95,
        floor=0.02,
    )

    assert result.decrease_fraction > 0
    assert result.mean_delta < 0


def test_change_rejects_shape_mismatch():
    before = np.zeros((10, 10))
    after = np.zeros((20, 20))

    try:
        analyse_change(before, after)
        assert False
    except ValueError:
        assert True


def test_sar_ratio():
    vv = np.full((10, 10), -10.0)
    vh = np.full((10, 10), -15.0)

    ratio = vv_vh_ratio(
        vv,
        vh,
        input_db=True,
    )

    assert ratio.shape == vv.shape
    assert np.all(np.isfinite(ratio))
    assert np.nanmean(ratio) > 1


def test_rvi():
    vv = np.full((10, 10), -10.0)
    vh = np.full((10, 10), -15.0)

    result = rvi(
        vv,
        vh,
        input_db=True,
    )

    assert result.shape == vv.shape
    assert np.all(np.isfinite(result))


def test_sar_temporal_difference():
    before = np.full((10, 10), -10.0)
    after = np.full((10, 10), -15.0)

    delta = log_backscatter_difference(
        before,
        after,
        input_db=True,
    )

    assert np.allclose(delta, -5.0)


def test_landcover():
    shape = (20, 20)

    blue = np.ones(shape) * 0.1
    green = np.ones(shape) * 0.1
    red = np.ones(shape) * 0.1
    nir = np.ones(shape) * 0.6
    swir1 = np.ones(shape) * 0.2

    result = classify_landcover(
        blue=blue,
        green=green,
        red=red,
        nir=nir,
        swir1=swir1,
    )

    assert result.classification.shape == shape
    assert result.confidence >= 0
    assert result.confidence <= 1


def test_crop_stress():
    ndvi = np.full((20, 20), 0.35)
    baseline = np.full((20, 20), 0.60)

    result = crop_stress_map(
        ndvi=ndvi,
        ndvi_baseline=baseline,
        rainfall_deficit=np.full((20, 20), 0.8),
        temperature_excess=np.full((20, 20), 0.7),
        sar_moisture_stress=np.full((20, 20), 0.8),
    )

    assert result.mean_stress > 0
    assert result.critical_fraction >= 0
    assert result.confidence > 0


def test_sowing_fallow_classification():
    shape = (10, 10)

    ndvi = np.full(shape, 0.15)
    ndmi = np.full(shape, 0.00)
    bsi = np.full(shape, 0.20)

    result = classify_field_state(
        ndvi=ndvi,
        ndmi=ndmi,
        bsi=bsi,
    )

    assert result.shape == shape
    assert np.any(result == 2)


def test_risk_engine():
    shape = (10, 10)

    result = composite_risk(
        signals={
            "crop_stress": np.full(shape, 0.9),
            "rainfall_deficit": np.full(shape, 0.8),
            "heat_stress": np.full(shape, 0.7),
        }
    )

    assert result.mean_score > 0.5
    assert result.high_fraction > 0
    assert result.critical_fraction > 0
    assert result.confidence > 0
