"""Single-image question answering. STATUS: stub."""

from jatayu.schemas import Evidence, TaskFamily, TaskName, ToolRequest, ToolResult
from jatayu.tools.registry import register


@register(
    TaskName.VQA,
    families={TaskFamily.SINGLE_IMAGE},
    description="Answers a factual question about one image: presence, count, "
                "extent, land-cover type. Use when the user asks what is in an image.",
)
def run(req: ToolRequest) -> ToolResult:
    return ToolResult(
        answer=f"[stub] Three water bodies are visible. (asked: {req.query!r})",
        evidence=Evidence(kind="none"),
        confidence=0.5,
        confidence_method="stub",
        tool_name=TaskName.VQA,
        model_id="stub",
        notes=["Stubbed tool — this answer is hardcoded."],
    )
