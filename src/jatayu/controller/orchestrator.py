"""validate -> classify -> interpret -> route -> execute -> assemble."""
from __future__ import annotations

import time

from jatayu.controller.gate import AmbiguousInputError, classify_family
from jatayu.controller.intent import parse_intent
from jatayu.controller.router import RuleRouter
from jatayu.schemas import (
    Evidence, ImageRef, QueryResponse, TaskFamily, ToolRequest, TraceStep,
    ValidationReport,
)
from jatayu.tools.registry import get_tool


class Orchestrator:
    def __init__(self, router=None):
        self.router = router or RuleRouter()

    def run(self, query: str, images: list[ImageRef]) -> QueryResponse:
        steps: list[TraceStep] = []
        last = time.perf_counter()

        def note(stage, detail, model_id=None):
            nonlocal last
            now = time.perf_counter()
            steps.append(TraceStep(
                step=len(steps) + 1, stage=stage, detail=detail,
                model_id=model_id, duration_ms=int((now - last) * 1000),
            ))
            last = now

        note("validate", f"Received {len(images)} input(s).")

        try:
            family = classify_family(images)
        except AmbiguousInputError as exc:
            note("classify", f"Could not determine the task: {exc}")
            return QueryResponse(
                answer=f"I cannot analyse these inputs. {exc}",
                evidence=Evidence(kind="none"), confidence=0.0,
                confidence_method="not_attempted",
                task_family=TaskFamily.SINGLE_IMAGE, tools_used=[], trace=steps,
                validation=ValidationReport(ok=False),
            )

        note("classify", f"Interpreted as a {family.value.replace('_', '-')} task.")

        intent = parse_intent(query, family)
        note("classify", intent.trace_detail())

        if not intent.is_understood:
            return QueryResponse(
                answer=(
                    "I could not determine what you are asking about. "
                    "Try rephrasing with a specific target — for example, "
                    "'highlight the water body' or 'has vegetation changed'."
                ),
                evidence=Evidence(kind="none"), confidence=0.0,
                confidence_method="not_attempted",
                task_family=family, tools_used=[], trace=steps,
                validation=ValidationReport(ok=True),
            )

        task = intent.task
        note("route", f"Selected the {task.value} specialist.")

        tool_params = {}
        if intent.candidate_indices:
            tool_params["candidate_indices"] = intent.candidate_indices
        if intent.answer_type:
            tool_params["answer_type"] = intent.answer_type.value
        if intent.change_direction:
            tool_params["change_direction"] = intent.change_direction.value
        if intent.target:
            tool_params["target"] = intent.target.value

        result = get_tool(task)(ToolRequest(query=query, images=images, params=tool_params))
        note("execute", f"Ran {task.value}.", model_id=result.model_id)
        note("combine",
             f"Confidence {result.confidence:.2f} via {result.confidence_method}."
             + (" Tool abstained for lack of evidence." if result.abstained else ""))

        return QueryResponse(
            answer=result.answer, evidence=result.evidence,
            confidence=result.confidence,
            confidence_method=result.confidence_method,
            task_family=family, tools_used=[task], trace=steps,
            validation=ValidationReport(ok=True),
        )
