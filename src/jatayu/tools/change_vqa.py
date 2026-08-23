"""Bi-temporal change understanding — what changed between two dates. STATUS: stub.

Inputs arrive already co-registered. If they are not, validation should have
rejected them before reaching here; do not silently re-align inside this tool.
"""

from jatayu.schemas import Evidence, TaskFamily, TaskName, ToolRequest, ToolResult
from jatayu.tools.registry import register


@register(
    TaskName.CHANGE_VQA,
    families={TaskFamily.BI_TEMPORAL},
    description="Compares two images of the same area taken at different times and "
                "answers questions about what changed, where, and in which direction "
                "(increase, decrease, no change). Requires exactly two images.",
)
def run(req: ToolRequest) -> ToolResult:
    return ToolResult(
        answer=(
            "[stub] Built-up area increased in the north-east; vegetation cover "
            f"decreased correspondingly. (asked: {req.query!r})"
        ),
        evidence=Evidence(kind="none", caption="Stub result — no change map computed"),
        confidence=0.5,
        confidence_method="stub",
        tool_name=TaskName.CHANGE_VQA,
        model_id="stub",
        notes=["Stubbed tool — this answer is hardcoded."],
    )
