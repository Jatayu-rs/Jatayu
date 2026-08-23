"""validate -> classify -> route -> execute -> assemble."""

from __future__ import annotations

import time

from jatayu.controller.gate import AmbiguousInputError, classify_family
from jatayu.controller.router import RuleRouter
from jatayu.schemas import (
    Evidence, ImageRef, QueryResponse, TaskFamily, ToolRequest, TraceStep,
)
from jatayu.tools.registry import get_tool


class Orchestrator:
    def __init__(self, router=None):
        self.router = router or RuleRouter()

    def run(self, query: str, images: list[ImageRef]) -> QueryResponse:
        steps: list[TraceStep] = []
        last = time.perf_counter()

        def note(stage: str, detail: str, model_id: str | None = None) -> None:
            nonlocal last
            now = time.perf_counter()
            steps.append(TraceStep(step=len(steps) + 1, stage=stage, detail=detail,
                                   model_id=model_id,
                                   duration_ms=int((now - last) * 1000)))
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
            )

        note("classify", f"Interpreted as a {family.value.replace('_', '-')} task.")

        task = self.router.select(query, family)
        note("route", f"Selected the {task.value} specialist.")

        result = get_tool(task)(ToolRequest(query=query, images=images))
        note("execute", f"Ran {task.value}.", model_id=result.model_id)
        note("combine", f"Confidence {result.confidence:.2f} via {result.confidence_method}.")

        return QueryResponse(
            answer=result.answer, evidence=result.evidence,
            confidence=result.confidence, confidence_method=result.confidence_method,
            task_family=family, tools_used=[task], trace=steps,
        )
