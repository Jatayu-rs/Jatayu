"""Optical-SAR cross-modal analysis. STATUS: stub, but this one becomes real today.

Unlike the other three, fusion needs no learned model. Water is specular to
radar so backscatter collapses; buildings form corner reflectors so it spikes.
Cross-tabulated with NDWI and NDBI from the optical bands, that gives a real
four-class map. Replaced with the physics implementation later this session.
"""

from jatayu.schemas import Evidence, TaskFamily, TaskName, ToolRequest, ToolResult
from jatayu.tools.registry import register


@register(
    TaskName.FUSION,
    families={TaskFamily.CROSS_MODAL},
    description="Combines a co-registered optical image and a SAR image to identify "
                "surface types more reliably than either alone — especially water, "
                "built-up areas, and flooded vegetation. Requires one optical and "
                "one SAR image.",
)
def run(req: ToolRequest) -> ToolResult:
    return ToolResult(
        answer=(
            "[stub] Combining optical and SAR, 12% of the scene is open water and "
            f"31% is built-up. (asked: {req.query!r})"
        ),
        evidence=Evidence(kind="mask", caption="Stub agreement map"),
        confidence=0.5,
        confidence_method="stub",
        tool_name=TaskName.FUSION,
        model_id="stub",
        notes=["Stubbed tool — no pixels were analysed."],
    )
