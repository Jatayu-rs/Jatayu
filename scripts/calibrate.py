"""Look at the real distributions before choosing any threshold.

Usage:
    python scripts/calibrate.py <optical.tif> <sar.tif>
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from jatayu.io.loader import read_bands
from jatayu.schemas import ImageRef, Modality

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

opt_path, sar_path = Path(sys.argv[1]), Path(sys.argv[2])
stem = opt_path.stem[:40]


def ref(path, modality):
    import rasterio
    with rasterio.open(path) as src:
        return ImageRef(
            path=str(path), modality=modality,
            crs=str(src.crs), bounds=tuple(src.bounds),
            width=src.width, height=src.height, band_count=src.count,
        )


optical = ref(opt_path, Modality.MULTISPECTRAL)
sar = ref(sar_path, Modality.SAR)

# read_bands resolves B3 -> green, B11 -> swir1 via BAND_ALIASES
green, red, nir, swir = read_bands(optical, ["green", "red", "nir", "swir1"])
(vv,) = read_bands(sar, ["vv"])


def nd(a, b):
    denom = a + b
    out = np.zeros_like(denom, dtype=np.float32)
    np.divide(a - b, denom, out=out, where=np.abs(denom) > 1e-6)
    return out


mndwi = nd(green, swir)
ndbi = nd(swir, nir)
ndvi = nd(nir, red)

print(f"\n=== {opt_path.name} ===")
print(f"raw band ranges (checking DN vs reflectance):")
for name, arr in [("green", green), ("nir", nir), ("swir1", swir)]:
    print(f"  {name:6s} min={np.nanmin(arr):10.3f}  max={np.nanmax(arr):10.3f}")

print("\n          " + "  ".join(f"{q:>8}" for q in ["1%", "5%", "25%", "50%", "75%", "95%", "99%"]))
for name, arr in [("SAR dB", vv), ("MNDWI", mndwi), ("NDBI", ndbi), ("NDVI", ndvi)]:
    p = np.nanpercentile(arr, [1, 5, 25, 50, 75, 95, 99])
    print(f"{name:8s} " + "  ".join(f"{v:8.3f}" for v in p))


def save(arr, lo, hi, name):
    disp = np.nan_to_num(np.clip((arr - lo) / (hi - lo), 0, 1))
    Image.fromarray((disp * 255).astype("uint8")).save(OUT / name)


save(vv, -25.0, 5.0, f"{stem}_sar.png")
save(mndwi, -0.6, 0.6, f"{stem}_mndwi.png")
print(f"\nwrote previews to {OUT}")
