"""Name -> tool lookup. The controller asks this, never a tool module directly."""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import wraps

from jatayu.schemas import TaskFamily, TaskName, ToolRequest, ToolResult

ToolFn = Callable[[ToolRequest], ToolResult]

REGISTRY: dict[TaskName, ToolFn] = {}
FAMILIES: dict[TaskName, set[TaskFamily]] = {}
DESCRIPTIONS: dict[TaskName, str] = {}


def register(name: TaskName, *, families: set[TaskFamily], description: str):
    def decorator(fn: ToolFn) -> ToolFn:
        @wraps(fn)
        def wrapper(req: ToolRequest) -> ToolResult:
            start = time.perf_counter()
            result = fn(req)
            ms = int((time.perf_counter() - start) * 1000)
            return result.model_copy(update={"latency_ms": ms})

        REGISTRY[name] = wrapper
        FAMILIES[name] = families
        DESCRIPTIONS[name] = description
        return wrapper

    return decorator


def get_tool(name: TaskName) -> ToolFn:
    if name not in REGISTRY:
        raise KeyError(f"No tool for {name.value!r}. Did you import jatayu.tools?")
    return REGISTRY[name]


def tools_for_family(family: TaskFamily) -> list[TaskName]:
    return [n for n, f in FAMILIES.items() if family in f]
