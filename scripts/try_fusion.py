import sys
from pathlib import Path

from jatayu.schemas import ImageRef, Modality, ToolRequest
from jatayu.tools import fusion

import rasterio


def ref(path: Path, modality: Modality) -> ImageRef:
    with rasterio.open(path) as src:
        return ImageRef(
            path=str(path), modality=modality,
            crs=str(src.crs), bounds=tuple(src.bounds),
            width=src.width, height=src.height, band_count=src.count,
        )


opt_path, sar_path = Path(sys.argv[1]), Path(sys.argv[2])

req = ToolRequest(
    query="Use the optical and SAR images together to identify built-up and water regions.",
    images=[ref(opt_path, Modality.MULTISPECTRAL), ref(sar_path, Modality.SAR)],
    params={"decimate": 1},   # chips are only ~278px; don't decimate
)

result = fusion.run(req)
print(result.answer)
print(f"\nconfidence {result.confidence:.2f} via {result.confidence_method}")
print("overlay   ", result.evidence.overlay_png)
for n in result.notes:
    print("  note:", n)
