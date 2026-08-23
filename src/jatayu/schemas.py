"""The tool contract. Everything depends on this file; it depends on nothing."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Modality(str, Enum):
    OPTICAL = "optical"
    MULTISPECTRAL = "multispectral"
    SAR = "sar"
    UNKNOWN = "unknown"

    @property
    def is_optical_family(self) -> bool:
        return self in (Modality.OPTICAL, Modality.MULTISPECTRAL)


class TaskFamily(str, Enum):
    SINGLE_IMAGE = "single_image"
    BI_TEMPORAL = "bi_temporal"
    CROSS_MODAL = "cross_modal"


class TaskName(str, Enum):
    VQA = "vqa"
    GROUNDING = "grounding"
    CHANGE_VQA = "change_vqa"
    FUSION = "fusion"


class ImageRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    modality: Modality = Modality.UNKNOWN
    crs: str | None = None
    bounds: tuple[float, float, float, float] | None = None
    acquired: datetime | None = None
    width: int
    height: int
    band_count: int = 3


class ToolRequest(BaseModel):
    query: str
    images: list[ImageRef] = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    original_query: str | None = None


class Evidence(BaseModel):
    kind: Literal["bbox", "mask", "overlay", "none"] = "none"
    overlay_png: str | None = None
    legend: dict[str, str] = Field(default_factory=dict)
    caption: str | None = None


class ToolResult(BaseModel):
    answer: str
    evidence: Evidence = Field(default_factory=Evidence)
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_method: str
    tool_name: TaskName
    model_id: str
    latency_ms: int = 0
    notes: list[str] = Field(default_factory=list)


class TraceStep(BaseModel):
    step: int
    stage: Literal["validate", "classify", "route", "execute", "combine"]
    detail: str
    model_id: str | None = None
    duration_ms: int = 0


class QueryResponse(BaseModel):
    answer: str
    evidence: Evidence
    confidence: float
    confidence_method: str
    task_family: TaskFamily
    tools_used: list[TaskName] = Field(default_factory=list)
    trace: list[TraceStep] = Field(default_factory=list)
    language: str = "eng_Latn"
    answer_original: str | None = None
