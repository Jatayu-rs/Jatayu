"""Look at the real distributions before choosing any threshold."""

from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data" / "samples"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

D = 4  # decimation factor

with rasterio.open(SAMPLES / "kolkata_optical.tif") as src:
    shape = (src.height // D, src.width // D)
    green = src.read(2, out_shape=shape).astype("float32")
    red   = src.read(3, out_shape=shape).astype("float32")
    nir   = src.read(4, out_shape=shape).astype("float32")
    swir  = src.read(5, out_shape=shape).astype("float32")

with rasterio.open(SAMPLES / "kolkata_sar.tif") as src:
    sar = src.read(1, out_shape=shape).astype("float32")  # already dB


def nd(a, b):
    denom = a + b
    out = np.zeros_like(denom)
    np.divide(a - b, denom, out=out, where=np.abs(denom) > 1e-6)
    return out


ndwi  = nd(green, nir)    # McFeeters — fails on turbid water
mndwi = nd(green, swir)   # Xu — survives sediment load
ndbi  = nd(swir, nir)
ndvi  = nd(nir, red)

print("          " + "  ".join(f"{q:>7}" for q in ["1%", "5%", "25%", "50%", "75%", "95%", "99%"]))
for name, arr in [("SAR dB", sar), ("NDWI", ndwi), ("MNDWI", mndwi), ("NDBI", ndbi), ("NDVI", ndvi)]:
    p = np.nanpercentile(arr, [1, 5, 25, 50, 75, 95, 99])
    print(f"{name:8s} " + "  ".join(f"{v:7.3f}" for v in p))


def save_preview(arr, lo, hi, name):
    disp = np.nan_to_num(np.clip((arr - lo) / (hi - lo), 0, 1))
    Image.fromarray((disp * 255).astype("uint8")).save(OUT / name)
    print("wrote", OUT / name)


save_preview(sar, -25.0, 0.0, "sar_preview.png")
save_preview(mndwi, -0.6, 0.6, "mndwi_preview.png")
save_preview(ndwi, -0.6, 0.6, "ndwi_preview.png")
