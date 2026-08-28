"""The tool contract.

This is the most important file in the repository. Every module in Jatayu — the
controller, the tools, the API, the UI, the report generator, the evaluation
harness — agrees on the types defined here and on nothing else.

Rules:
  1. Every specialist tool is `run(req: ToolRequest) -> ToolResult`. No exceptions,
     no tool-specific return types.
  2. Changing anything in this file affects all six of us. Flag it in the team chat
     and tag the orchestration lead on the PR.
  3. Nothing in this file may import from anywhere else in `jatayu`. It sits at the
     bottom of the dependency graph.

See docs/tool-contract.md for the prose explanation.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Modality(str, Enum):
    """Sensor type of an input image."""

    OPTICAL = "optical"
    MULTISPECTRAL = "multispectral"
    SAR = "sar"
    UNKNOWN = "unknown"

    @property
    def is_optical_family(self) -> bool:
        """True for anything carrying spectral information (not radar)."""
        return self in (Modality.OPTICAL, Modality.MULTISPECTRAL)


class TaskFamily(str, Enum):
    """Coarse task family, decided deterministically from the inputs alone.

    The controller's first stage picks one of these by counting and inspecting
    images — no LLM involved. The second stage picks a specific tool within it.
    """

    SINGLE_IMAGE = "single_image"
    BI_TEMPORAL = "bi_temporal"
    CROSS_MODAL = "cross_modal"


class TaskName(str, Enum):
    """The specific task a tool performs. One tool per name."""

    VQA = "vqa"
    GROUNDING = "grounding"
    CAPTIONING = "captioning"
    CHANGE_VQA = "change_vqa"
    FUSION = "fusion"


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


class ImageRef(BaseModel):
    """A validated, on-disk image plus the metadata the controller routes on.

    Produced by `jatayu.io.loader`. Tools receive these already validated — a tool
    should never have to re-open a file to find out how many bands it has.
    """

    model_config = ConfigDict(frozen=True)

    path: str = Field(description="Absolute or repo-relative path to the raster on disk")
    modality: Modality = Modality.UNKNOWN

    # Geospatial metadata. None for benchmark PNG/JPEG inputs, which are ungeoreferenced.
    crs: str | None = Field(default=None, description="CRS as an authority string, e.g. 'EPSG:32643'")
    bounds: tuple[float, float, float, float] | None = Field(
        default=None, description="(left, bottom, right, top) in CRS units"
    )
    transform: tuple[float, float, float, float, float, float] | None = Field(
        default=None, description="Affine transform coefficients (a, b, c, d, e, f)"
    )
    acquired: datetime | None = Field(default=None, description="Acquisition timestamp if known")

    # Raster properties.
    width: int
    height: int
    band_count: int
    band_names: list[str] = Field(default_factory=list)
    dtype: str = "uint8"
    nodata: float | None = None

    @property
    def is_georeferenced(self) -> bool:
        return self.crs is not None and self.transform is not None


class ToolRequest(BaseModel):
    """Everything a specialist tool receives. Tools get nothing else."""

    query: str = Field(description="The user's natural-language query, in English")
    images: list[ImageRef] = Field(min_length=1)
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Permitted task parameters only. Tools must ignore unknown keys "
        "rather than raising, so the router can pass a superset.",
    )
    original_query: str | None = Field(
        default=None, description="Pre-translation query, if the user wrote in another language"
    )
    request_id: str | None = None


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


class BoundingBox(BaseModel):
    """A box in pixel coordinates of the referenced image.

    Pixel space, not normalised, not geographic. The report layer converts to
    geographic coordinates using the image transform when one exists.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    label: str | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    image_index: int = Field(default=0, description="Which image in ToolRequest.images this refers to")

    @field_validator("x_max")
    @classmethod
    def _x_ordered(cls, v: float, info: Any) -> float:
        if "x_min" in info.data and v < info.data["x_min"]:
            raise ValueError("x_max must be >= x_min")
        return v


class Evidence(BaseModel):
    """The visual backing for an answer.

    `kind == "none"` is legitimate and expected — a yes/no VQA answer has no
    region to point at. It is not a failure state.
    """

    kind: Literal["bbox", "mask", "overlay", "table", "none"] = "none"
    boxes: list[BoundingBox] = Field(default_factory=list)
    mask_path: str | None = Field(default=None, description="Single-band raster, 0/1 or class-indexed")
    overlay_png: str | None = Field(default=None, description="Rendered PNG for display in the UI")
    geojson: dict[str, Any] | None = Field(
        default=None, description="Georeferenced geometries for the downloadable report"
    )
    legend: dict[str, str] = Field(
        default_factory=dict, description="Class value -> human-readable label, for masks"
    )
    caption: str | None = Field(default=None, description="Short description of what is shown")


class ToolResult(BaseModel):
    """What every specialist tool returns. The single output type in the system."""

    # --- the answer ---
    answer: str = Field(description="Natural-language answer shown to the user")
    evidence: Evidence = Field(default_factory=Evidence)

    # --- confidence ---
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_method: str = Field(
        description="How `confidence` was computed, e.g. 'answer_softmax', "
        "'dual_signal_agreement', 'stub'. Never report a number we cannot explain."
    )

    # --- provenance, for the auditable execution trace ---
    tool_name: TaskName
    model_id: str = Field(description="HF repo id, local checkpoint path, or 'stub'")
    params_used: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = 0

    # --- honesty ---
    abstained: bool = Field(
        default=False,
        description="True when the tool declined to answer for lack of evidence. "
        "Abstaining is a correct outcome, not an error.",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Caveats surfaced to the user, e.g. 'SAR image resampled to optical grid'",
    )


# ---------------------------------------------------------------------------
# Validation and trace
# ---------------------------------------------------------------------------


class ValidationIssue(BaseModel):
    severity: Severity
    code: str = Field(description="Stable machine-readable code, e.g. 'crs_mismatch'")
    message: str = Field(description="Human-readable, shown to the user")
    image_index: int | None = None


class ValidationReport(BaseModel):
    ok: bool
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]


class TraceStep(BaseModel):
    """One observable step in the execution trace.

    Only this is evaluated by the judges — internal LLM reasoning text is neither
    required nor scored. Keep it factual.
    """

    step: int
    stage: Literal["validate", "classify", "route", "execute", "combine"]
    detail: str
    tool_name: TaskName | None = None
    model_id: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    duration_ms: int = 0


class QueryResponse(BaseModel):
    """The complete API response: answer, evidence, and how we got there."""

    answer: str
    evidence: Evidence
    confidence: float
    confidence_method: str

    task_family: TaskFamily
    tools_used: list[TaskName] = Field(default_factory=list)
    trace: list[TraceStep] = Field(default_factory=list)
    validation: ValidationReport

    total_latency_ms: int = 0
    request_id: str | None = None

    @classmethod
    def from_tool_result(
        cls,
        result: ToolResult,
        *,
        task_family: TaskFamily,
        trace: list[TraceStep],
        validation: ValidationReport,
        total_latency_ms: int = 0,
        request_id: str | None = None,
    ) -> QueryResponse:
        return cls(
            answer=result.answer,
            evidence=result.evidence,
            confidence=result.confidence,
            confidence_method=result.confidence_method,
            task_family=task_family,
            tools_used=[result.tool_name],
            trace=trace,
            validation=validation,
            total_latency_ms=total_latency_ms,
            request_id=request_id,
        )