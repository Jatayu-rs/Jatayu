from pathlib import Path

from jatayu.io.loader import read_bands  # noqa: F401 — ensures imports resolve
from jatayu.schemas import ImageRef, Modality, ToolRequest
from jatayu.tools import fusion

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "data" / "samples"


def ref(name, modality, w, h):
    return ImageRef(path=str(S / name), modality=modality, width=w, height=h)


req = ToolRequest(
    query="Use the optical and SAR images together to identify built-up and water regions.",
    images=[
        ref("kolkata_optical.tif", Modality.MULTISPECTRAL, 2075, 1902),
        ref("kolkata_sar.tif", Modality.SAR, 2075, 1902),
    ],
)

result = fusion.run(req)
print(result.answer)
print(f"\nconfidence {result.confidence:.2f} via {result.confidence_method}")
print("overlay   ", result.evidence.overlay_png)
for n in result.notes:
    print("  note:", n)
