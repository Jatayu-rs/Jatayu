"""Main Jatayu execution pipeline.

Pipeline:
validate -> classify -> interpret -> route -> execute -> assemble
"""

from __future__ import annotations

import time

from jatayu.controller.gate import AmbiguousInputError, classify_family
from jatayu.controller.intent import parse_intent
from jatayu.controller.router import default_router
from jatayu.schemas import (
    Evidence,
    ImageRef,
    QueryResponse,
    TaskFamily,
    ToolRequest,
    TraceStep,
    ValidationReport,
)
from jatayu.tools.registry import get_tool


class Orchestrator:
    """Coordinates deterministic classification, intent parsing, routing and tools."""

    def __init__(self, router=None):
        self.router = router or default_router()

    def run(self, query: str, images: list[ImageRef]) -> QueryResponse:
        started_at = time.perf_counter()

        steps: list[TraceStep] = []
        last = started_at

        def note(
            stage: str,
            detail: str,
            model_id: str | None = None,
        ) -> None:
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

        # ---------------------------------------------------------------
        # 1. VALIDATE
        # ---------------------------------------------------------------

        note(
            "validate",
            f"Received {len(images)} input(s).",
        )

        if not images:
            validation = ValidationReport(ok=False)

            note(
                "validate",
                "No images were supplied.",
            )

            return QueryResponse(
                answer="I cannot analyse the request because no image was provided.",
                evidence=Evidence(kind="none"),
                confidence=0.0,
                confidence_method="not_attempted",
                task_family=TaskFamily.SINGLE_IMAGE,
                tools_used=[],
                trace=steps,
                validation=validation,
                total_latency_ms=int(
                    (time.perf_counter() - started_at) * 1000
                ),
            )

        # ---------------------------------------------------------------
        # 2. CLASSIFY TASK FAMILY
        # ---------------------------------------------------------------

        try:
            family = classify_family(images)

        except AmbiguousInputError as exc:
            note(
                "classify",
                f"Could not determine the task: {exc}",
            )

            return QueryResponse(
                answer=f"I cannot analyse these inputs. {exc}",
                evidence=Evidence(kind="none"),
                confidence=0.0,
                confidence_method="not_attempted",
                task_family=TaskFamily.SINGLE_IMAGE,
                tools_used=[],
                trace=steps,
                validation=ValidationReport(ok=False),
                total_latency_ms=int(
                    (time.perf_counter() - started_at) * 1000
                ),
            )

        note(
            "classify",
            f"Interpreted as a {family.value.replace('_', '-')} task.",
        )

        # ---------------------------------------------------------------
        # 3. INTERPRET QUERY
        # ---------------------------------------------------------------

        intent = parse_intent(query, family)

        note(
            "classify",
            intent.trace_detail(),
        )

        if not intent.is_understood:
            return QueryResponse(
                answer=(
                    "I could not determine what you are asking about. "
                    "Try rephrasing with a specific target — for example, "
                    "'highlight the water body' or "
                    "'has vegetation changed'."
                ),
                evidence=Evidence(kind="none"),
                confidence=0.0,
                confidence_method="not_attempted",
                task_family=family,
                tools_used=[],
                trace=steps,
                validation=ValidationReport(ok=True),
                total_latency_ms=int(
                    (time.perf_counter() - started_at) * 1000
                ),
            )

        # ---------------------------------------------------------------
        # 4. ROUTE
        # ---------------------------------------------------------------

        task = intent.task

        note(
            "route",
            f"Selected the {task.value} specialist.",
        )

        # Build parameters for the specialist tool.
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
                )
            )

        except Exception as exc:
            # Never let one specialist crash the entire demo.
            note(
                "execute",
                f"{task.value} failed: {type(exc).__name__}: {exc}",
            )

            return QueryResponse(
                answer=(
                    "Jatayu could not safely complete this analysis. "
                    f"The {task.value} specialist encountered an error."
                ),
                evidence=Evidence(kind="none"),
                confidence=0.0,
                confidence_method="execution_error",
                task_family=family,
                tools_used=[task],
                trace=steps,
                validation=ValidationReport(ok=False),
                total_latency_ms=int(
                    (time.perf_counter() - started_at) * 1000
                ),
            )

        note(
            "execute",
            f"Ran {task.value}.",
            model_id=result.model_id,
        )

        # ---------------------------------------------------------------
        # 6. COMBINE / ASSEMBLE
        # ---------------------------------------------------------------

        combine_detail = (
            f"Confidence {result.confidence:.2f} "
            f"via {result.confidence_method}."
        )

        if result.abstained:
            combine_detail += (
                " Tool abstained for lack of sufficient evidence."
            )

        note(
            "combine",
            combine_detail,
        )

        total_latency_ms = int(
            (time.perf_counter() - started_at) * 1000
        )

        # ---------------------------------------------------------------
        # 7. FINAL API RESPONSE
        # ---------------------------------------------------------------

        return QueryResponse(
            answer=result.answer,
            evidence=result.evidence,
            confidence=result.confidence,
            confidence_method=result.confidence_method,
            task_family=family,
            tools_used=[task],
            trace=steps,
            validation=ValidationReport(ok=True),
            total_latency_ms=total_latency_ms,
        )
