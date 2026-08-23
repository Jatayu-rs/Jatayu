"""GeoTIFF reading."""

from __future__ import annotations

import numpy as np
import rasterio

from jatayu.schemas import ImageRef


def read_bands(
    img: ImageRef, names: list[str], *, decimate: int = 1
) -> list[np.ndarray]:
    """Read named bands as float32.

    Band names come from the file's own descriptions — GEE writes them, so
    'green', 'swir1', 'vv' resolve directly. `decimate=4` reads every 4th pixel,
    which matters because rasters here can be tens of thousands of pixels wide.
    """
    with rasterio.open(img.path) as src:
        available = [d.lower() if d else "" for d in (src.descriptions or [])]
        shape = (src.height // decimate, src.width // decimate) if decimate > 1 else None

        indices = []
        for name in names:
            if name not in available:
                raise KeyError(
                    f"Band {name!r} not in {img.path}. Available: {available or 'unnamed'}. "
                    "Re-export with band descriptions, or pass explicit indices."
                )
            indices.append(available.index(name) + 1)  # rasterio is 1-indexed

        arrays = [
            src.read(i, out_shape=shape).astype("float32") if shape
            else src.read(i).astype("float32")
            for i in indices
        ]
        nodata = src.nodata

    if nodata is not None:
        arrays = [np.where(a == nodata, np.nan, a) for a in arrays]
    return arrays
