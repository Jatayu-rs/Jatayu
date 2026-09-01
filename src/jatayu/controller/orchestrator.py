"""Main Jatayu execution pipeline.

Pipeline:
validate -> classify -> interpret -> route -> execute -> assemble

--------------------------------------------------------------------------------
CHANGELOG (see schemas.py changelog too — this patch depends on that one):
  - request_id is now generated at the top of run() if the caller doesn't
    supply one, threaded into the ToolRequest passed to the tool, and set on
    every QueryResponse returned (previously the field existed on both
    schemas but was never actually connected).
  - The success path now calls QueryResponse.from_tool_result(...) instead of
    manually re-listing every field, so ToolResult/QueryResponse field changes
    only need updating in one place.
  - validation.ok now reflects whether the *input* was valid, not whether the
    pipeline succeeded. A tool crashing during execution is not an input
    validation failure — the images were fine — so that branch now returns
    ValidationReport(ok=True) instead of ok=False. The "no images" and
    "ambiguous task family" branches, which genuinely are about invalid/
    unclassifiable input, now populate a ValidationIssue explaining why,
    instead of setting ok=False with an empty issues list.
  - Intent-parsing trace entries now use stage="interpret" (added to the
    TraceStep.stage Literal in schemas.py) instead of "classify", so the
    trace matches the six-stage pipeline this docstring describes.
  - Refusal responses are now built through a single _refuse() helper instead
    of four separate inline QueryResponse(...) calls, so the four failure
    paths can't silently drift out of sync with each other or with the
    success path.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import time
import uuid

from jatayu.controller.gate import AmbiguousInputError, classify_family
from jatayu.controller.intent import parse_intent
from jatayu.controller.router import default_router
from jatayu.schemas import (
    Evidence,
    ImageRef,
    QueryResponse,
    Severity,
    TaskFamily,
    TaskName,
    ToolRequest,
    TraceStep,
    ValidationIssue,
    ValidationReport,
)
from jatayu.tools.registry import get_tool


class Orchestrator:
    """Coordinates deterministic classification, intent parsing, routing and tools."""

    def __init__(self, router=None):
        self.router = router or default_router()

    def run(self, query: str, images: list[ImageRef], request_id: str | None = None) -> QueryResponse:
        started_at = time.perf_counter()
        request_id = request_id or uuid.uuid4().hex

        steps: list[TraceStep] = []
        last = started_at

        def note(stage: str, detail: str, model_id: str | None = None) -> None:
            nonlocal last
            now = time.perf_counter()
            steps.append(
                TraceStep(
                    step=len(steps) + 1,
                    stage=stage,
                    detail=detail,
                    model_id=model_id,
                    duration_ms=int((now - last) * 1000),
                )
            )
            last = now

        def elapsed_ms() -> int:
            return int((time.perf_counter() - started_at) * 1000)

        def refuse(
            answer: str,
            *,
            confidence_method: str,
            task_family: TaskFamily,
            validation: ValidationReport,
            tools_used: list[TaskName] | None = None,
        ) -> QueryResponse:
            return QueryResponse(
                answer=answer,
                evidence=Evidence(kind="none"),
                confidence=0.0,
                confidence_method=confidence_method,
                task_family=task_family,
                tools_used=tools_used or [],
                trace=steps,
                validation=validation,
                total_latency_ms=elapsed_ms(),
                request_id=request_id,
            )

        # ---------------------------------------------------------------
        # 1. VALIDATE
        # ---------------------------------------------------------------

        note("validate", f"Received {len(images)} input(s).")

        if not images:
            note("validate", "No images were supplied.")
            validation = ValidationReport(
                ok=False,
                issues=[
                    ValidationIssue(
                        severity=Severity.ERROR,
                        code="no_images",
                        message="No images were supplied.",
                    )
                ],
            )
            return refuse(
                "I cannot analyse the request because no image was provided.",
                confidence_method="not_attempted",
                task_family=TaskFamily.SINGLE_IMAGE,
                validation=validation,
            )

        # ---------------------------------------------------------------
        # 2. CLASSIFY TASK FAMILY
        # ---------------------------------------------------------------

        try:
            family = classify_family(images)
        except AmbiguousInputError as exc:
            note("classify", f"Could not determine the task: {exc}")
            validation = ValidationReport(
                ok=False,
                issues=[
                    ValidationIssue(
                        severity=Severity.ERROR,
                        code="ambiguous_task_family",
                        message=str(exc),
                    )
                ],
            )
            return refuse(
                f"I cannot analyse these inputs. {exc}",
                confidence_method="not_attempted",
                task_family=TaskFamily.SINGLE_IMAGE,
                validation=validation,
            )

        note("classify", f"Interpreted as a {family.value.replace('_', '-')} task.")

        # ---------------------------------------------------------------
        # 3. INTERPRET QUERY
        # ---------------------------------------------------------------

        intent = parse_intent(query, family)

        # FIXED: was logged under "classify"; this is the "interpret" stage
        # the module docstring describes, and now has its own Literal value.
        note("interpret", intent.trace_detail())

        if not intent.is_understood:
            return refuse(
                "I could not determine what you are asking about. "
                "Try rephrasing with a specific target — for example, "
                "'highlight the water body' or 'has vegetation changed'.",
                confidence_method="not_attempted",
                task_family=family,
                # The images themselves were valid; it's the query that
                # wasn't understood. That's worth a WARNING-level issue for
                # audit purposes, but it doesn't make the input invalid.
                validation=ValidationReport(
                    ok=True,
                    issues=[
                        ValidationIssue(
                            severity=Severity.WARNING,
                            code="query_not_understood",
                            message="The query could not be mapped to a specific task.",
                        )
                    ],
                ),
            )

        # ---------------------------------------------------------------
        # 4. ROUTE
        # ---------------------------------------------------------------

        task = intent.task
        note("route", f"Selected the {task.value} specialist.")

        tool_params: dict = {}
        if intent.candidate_indices:
            tool_params["candidate_indices"] = intent.candidate_indices
        if intent.answer_type:
            tool_params["answer_type"] = intent.answer_type.value
        if intent.change_direction:
            tool_params["change_direction"] = intent.change_direction.value
        if intent.target:
            tool_params["target"] = intent.target.value

        # ---------------------------------------------------------------
        # 5. EXECUTE SPECIALIST TOOL
        # ---------------------------------------------------------------

        try:
            tool = get_tool(task)
            result = tool(
                ToolRequest(
                    query=query,
                    images=images,
                    params=tool_params,
                    request_id=request_id,
                )
            )
        except Exception as exc:
            # Never let one specialist crash the entire demo. The exception
            # detail goes into the trace (for researchers/audit) but never
            # into the user-facing `answer`.
            note("execute", f"{task.value} failed: {type(exc).__name__}: {exc}")
            return refuse(
                f"Jatayu could not safely complete this analysis. "
                f"The {task.value} specialist encountered an error.",
                confidence_method="execution_error",
                task_family=family,
                tools_used=[task],
                # FIXED: was ValidationReport(ok=False). The input images
                # were valid — the tool crashed during execution, which is a
                # different failure mode than "the input was invalid," and
                # a caller branching on validation.ok to decide "ask the
                # user to fix their input" vs. "this is our bug" needs that
                # distinction to be accurate.
                validation=ValidationReport(ok=True),
            )

        note("execute", f"Ran {task.value}.", model_id=result.model_id)

        # ---------------------------------------------------------------
        # 6. COMBINE / ASSEMBLE
        # ---------------------------------------------------------------

        combine_detail = f"Confidence {result.confidence:.2f} via {result.confidence_method}."
        if result.abstained:
            combine_detail += " Tool abstained for lack of sufficient evidence."
        note("combine", combine_detail)

        # ---------------------------------------------------------------
        # 7. FINAL API RESPONSE
        # ---------------------------------------------------------------

        # FIXED: previously reconstructed every field manually here instead
        # of using the classmethod already defined for exactly this purpose.
        return QueryResponse.from_tool_result(
            result,
            task_family=family,
            trace=steps,
            validation=ValidationReport(ok=True),
            total_latency_ms=elapsed_ms(),
            request_id=request_id,
        )
