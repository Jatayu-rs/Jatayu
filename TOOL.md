# The tool contract

The most important document in this repository. If you read one thing before
writing code, read this.

Source of truth: [`src/jatayu/schemas.py`](../src/jatayu/schemas.py). This page
explains it; the code defines it.

---

## Why it exists

Without a contract, coordination between six people scales with *pairs* of
people — fifteen conversations that all have to stay in sync. The frontend
engineer asks the model engineer what shape the output is; the model engineer
hasn't built it yet; the frontend engineer guesses. Three people guess three
different shapes. Week five is spent writing translation glue that nobody owns
and nobody tested.

With a contract, everyone coordinates with *the document* instead. Five
workstreams run simultaneously from day one:

- The UI is built against `ToolResult`, needing no GPU and no models.
- The report generator is built against `ToolResult`.
- The evaluation harness scores `ToolResult` against benchmark answers.
- The controller routes to things that return `ToolResult`.
- The model engineers build things that produce `ToolResult`.

This pattern is called **contract-first design**, and building the whole system
end to end on fakes before filling them in is a **walking skeleton**. Both are
standard practice, worth naming if a judge asks about engineering process.

---

## The interface

```python
def run(req: ToolRequest) -> ToolResult: ...
```

That is the whole thing. No tool knows about the controller, the API, the UI, or
any other tool.

---

## Types

### `ImageRef` — a validated image

Produced by `jatayu.io.loader.load()`. Frozen, so tools cannot mutate their
inputs. Carries everything the controller routes on, so a tool never has to
reopen a file to find out how many bands it has.

Key fields: `path`, `modality`, `crs`, `bounds`, `transform`, `acquired`,
`width`, `height`, `band_count`, `band_names`, `dtype`, `nodata`.

`crs` and `transform` are `None` for benchmark PNG/JPEG inputs, which are not
georeferenced. Use `img.is_georeferenced` rather than checking fields yourself.

### `ToolRequest` — the input

```python
ToolRequest(query=..., images=[...], params={...})
```

`params` carries permitted task parameters. **Tools must ignore unknown keys
rather than raising**, so the router can pass a superset without knowing each
tool's exact signature.

`original_query` holds the pre-translation text when the user wrote in a language
other than English.

### `Evidence` — visual backing

`kind` is one of `bbox`, `mask`, `overlay`, `table`, `none`.

`kind="none"` is legitimate and expected. A yes/no VQA answer has no region to
point at. It is not a failure state and the UI must not treat it as one.

`BoundingBox` coordinates are **pixel space of the referenced image** — not
normalised, not geographic. The report layer converts to WGS84 using the image
transform. Grounding models emit normalised 0–1000 coordinates as text; rescale
them at the tool boundary, not downstream.

### `ToolResult` — the output

```python
ToolResult(
    answer="...",                          # what the user reads
    evidence=Evidence(...),                # what backs it up
    confidence=0.82,                       # 0.0-1.0
    confidence_method="answer_softmax",    # how that number was computed
    tool_name=TaskName.VQA,
    model_id="jatayu/rsvqa-lora-v3",
    params_used={...},
    abstained=False,
    notes=["..."],                         # caveats shown to the user
)
```

Two fields deserve attention.

**`confidence_method` is mandatory.** Schema-enforced, and there is a test for
it. A confidence number without a stated method is exactly what we promised not
to ship. If you cannot name the method, do not invent a number — set
`abstained=True`.

**`abstained=True` is a correct outcome, not an error.** Saying "I cannot tell
from this image" is one of our stated differentiators. The controller and UI
handle it as a valid result.

`latency_ms` is filled in automatically by the `@timed` decorator. Do not set it
by hand.

### `TraceStep` and `QueryResponse`

The trace is what judges evaluate — selected task, tool and model names,
parameters, outputs. Internal LLM reasoning text is explicitly not required and
not scored, so we do not log it. Keep each `detail` factual and short enough to
render in a UI panel.

---

## Rules

1. Every tool is `run(req: ToolRequest) -> ToolResult`. No tool-specific return
   types, ever.
2. Tools ignore unknown `params` keys.
3. Tools do not mutate `ImageRef` (it is frozen; they cannot).
4. Tools do not import `jatayu.controller`, `jatayu.api`, or each other.
5. `schemas.py` imports nothing from `jatayu`. It sits at the bottom of the graph.
6. Changing `schemas.py` requires a team announcement and the orchestration
   lead's approval.

---

## Adding a tool

```python
# src/jatayu/tools/my_task.py
from jatayu.schemas import Evidence, TaskFamily, TaskName, ToolRequest, ToolResult
from jatayu.tools.base import timed
from jatayu.tools.registry import register


@register(
    TaskName.MY_TASK,
    families={TaskFamily.SINGLE_IMAGE},
    description="Written for the router LLM. Say what question this answers.",
)
@timed
def run(req: ToolRequest) -> ToolResult:
    return ToolResult(
        answer="...",
        evidence=Evidence(kind="none"),
        confidence=0.5,
        confidence_method="stub",
        tool_name=TaskName.MY_TASK,
        model_id="stub",
    )
```

Add the import to `src/jatayu/tools/__init__.py`. That is all — the parametrised
contract tests pick it up automatically, the router can select it, and the UI and
report generator already handle its output.

**Merge the stub first, then the implementation.** The stub unblocks everyone
else immediately; the real model can take three more weeks without anyone
waiting.
