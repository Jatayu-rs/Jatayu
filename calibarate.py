

import numpy as np
import rasterio
from PIL import Image

D = 4  # decimation factor — read every 4th pixel, fast enough to iterate

with rasterio.open("data/samples/kolkata_optical.tif") as src:
    shape = (src.height // D, src.width // D)
    green = src.read(2, out_shape=shape).astype("float32")
    red   = src.read(3, out_shape=shape).astype("float32")
    nir   = src.read(4, out_shape=shape).astype("float32")
    swir  = src.read(5, out_shape=shape).astype("float32")

with rasterio.open("data/samples/kolkata_sar.tif") as src:
    sar = src.read(1, out_shape=shape).astype("float32")   # already dB


def nd(a, b):
    denom = a + b
    out = np.zeros_like(denom)
    np.divide(a - b, denom, out=out, where=np.abs(denom) > 1e-6)
    return out


ndwi = nd(green, nir)     # water high
ndbi = nd(swir, nir)      # built-up high
ndvi = nd(nir, red)       # vegetation high

for name, arr in [("SAR dB", sar), ("NDWI", ndwi), ("NDBI", ndbi), ("NDVI", ndvi)]:
    p = np.nanpercentile(arr, [1, 5, 25, 50, 75, 95, 99])
    print(f"{name:8s} " + "  ".join(f"{v:7.3f}" for v in p))
print("          " + "  ".join(f"{q:>7}" for q in ["1%", "5%", "25%", "50%", "75%", "95%", "99%"]))

# Quick visual: is water dark and city bright where you expect?
disp = np.clip((sar + 25) / 25, 0, 1)      # stretch -25..0 dB to 0..1
Image.fromarray((disp * 255).astype("uint8")).save("outputs/sar_preview.png")
print("\nwrote outputs/sar_preview.png")
