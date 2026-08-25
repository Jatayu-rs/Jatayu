"""GeoTIFF reading.

All geospatial IO in the system passes through here. No direct rasterio or GDAL
calls exist elsewhere.
"""

from __future__ import annotations

import numpy as np
import rasterio

from jatayu.schemas import ImageRef

# Different exports name the same physical band differently. Sentinel-2 products
# use B2/B3/B4/B8/B11; our earlier GEE export used plain names. The ISRO
# evaluation set will have its own convention — add to this map rather than
# changing call sites.
BAND_ALIASES = {
    "b2": "blue",
    "b3": "green",
    "b4": "red",
    "b8": "nir",
    "b8a": "nir",
    "b11": "swir1",
    "b12": "swir2",
}


def _canonical(name: str | None) -> str:
    """Resolve a band description to our internal vocabulary."""
    n = (name or "").strip().lower()
    return BAND_ALIASES.get(n, n)


def read_bands(
    img: ImageRef, names: list[str], *, decimate: int = 1
) -> list[np.ndarray]:
    """Read named bands as float32 arrays.

    Band names are resolved through BAND_ALIASES, so both 'green' and 'B3' work.
    `decimate=4` reads every 4th pixel — necessary because some rasters here are
    tens of thousands of pixels wide.
    """
    wanted = [_canonical(n) for n in names]

    with rasterio.open(img.path) as src:
        available = [_canonical(d) for d in (src.descriptions or [])]
        shape = (
            (src.height // decimate, src.width // decimate) if decimate > 1 else None
        )

        indices = []
        for original, name in zip(names, wanted, strict=True):
            if name not in available:
                raise KeyError(
                    f"Band {original!r} (resolved to {name!r}) not in {img.path}. "
                    f"Available (canonical): {available or 'unnamed'}. "
                    f"Raw descriptions: {list(src.descriptions)}. "
                    "Add an entry to BAND_ALIASES if this is a new convention."
                )
            indices.append(available.index(name) + 1)  # rasterio is 1-indexed

        arrays = [
            src.read(i, out_shape=shape).astype("float32")
            if shape
            else src.read(i).astype("float32")
            for i in indices
        ]
        nodata = src.nodata

    if nodata is not None:
        arrays = [np.where(a == nodata, np.nan, a) for a in arrays]
    return arrays
