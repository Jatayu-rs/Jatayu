
import sys
from pathlib import Path

import numpy as np
import rasterio

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "data/samples")

for path in sorted(folder.glob("*.tif")):
    with rasterio.open(path) as src:
        print(f"\n=== {path.name} ===")
        print(f"  size    {src.width} x {src.height}")
        print(f"  bands   {src.count}  names={src.descriptions}")
        print(f"  dtype   {src.dtypes[0]}")
        print(f"  crs     {src.crs}")
        print(f"  bounds  {tuple(round(b, 4) for b in src.bounds)}")
        print(f"  nodata  {src.nodata}")
        if src.tags():
            print(f"  tags    {src.tags()}")
        for i in range(1, src.count + 1):
            a = src.read(i).astype("float32")
            a = a[np.isfinite(a)]
            if src.nodata is not None:
                a = a[a != src.nodata]
            if a.size:
                print(f"    band {i}: min={a.min():.3f} max={a.max():.3f} mean={a.mean():.3f}")
