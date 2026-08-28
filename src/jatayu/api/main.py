"""HTTP API.

Kept separate from the UI so the frontend never needs a GPU, model weights, or
the geospatial stack installed — it develops against this over HTTP.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import traceback
import uuid
from pathlib import Path

import rasterio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import jatayu.tools  # noqa: F401 — importing registers every tool
from jatayu.controller.orchestrator import Orchestrator
from jatayu.schemas import (
    Evidence,
    ImageRef,
    Modality,
    QueryResponse,
    TaskFamily,
    ValidationReport,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Jatayu", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = Orchestrator()

# Uploads are transient; rendered overlays must outlive the request so the
# browser can fetch them, so they live in a served directory instead.
UPLOAD_DIR = Path(tempfile.gettempdir()) / "jatayu-uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = Path("data/samples/jatayu_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES_DIR = Path("data/samples")

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


# ---------------------------------------------------------------------------
# API envelope — language info without modifying frozen schemas.py
# ---------------------------------------------------------------------------


class LanguageInfo(BaseModel):
    detected: str = "eng_Latn"
    display_name: str = "English"
    was_translated: bool = False
    original_query: str | None = None


class AnalyzeResponse(BaseModel):
    """Wraps the frozen QueryResponse with language info and request metadata."""

    result: QueryResponse
    language: LanguageInfo | None = None
    request_id: str | None = None


# ---------------------------------------------------------------------------
# Sample data registry
# ---------------------------------------------------------------------------


class SampleScenario(BaseModel):
    id: str
    title: str
    description: str
    family: str
    suggested_query: str
    files: list[str]
    modalities: list[str]


# Built from the files actually on disk. Extend as you add more samples.
SAMPLE_SCENARIOS: list[SampleScenario] = [
    SampleScenario(
        id="kolkata_water",
        title="Kolkata — Water Body Detection",
        description="Multispectral optical tile over the Hooghly river and surrounding areas.",
        family="single_image",
        suggested_query="Highlight the water body in this image.",
        files=["kolkata_optical.tif"],
        modalities=["multispectral"],
    ),
    SampleScenario(
        id="kolkata_landcover",
        title="Kolkata — Land Cover Description",
        description="Describe what land cover types are visible in this Kolkata scene.",
        family="single_image",
        suggested_query="Describe the land cover in this image.",
        files=["kolkata_optical.tif"],
        modalities=["multispectral"],
    ),
    SampleScenario(
        id="kolkata_fusion",
        title="Kolkata — Optical + SAR Fusion",
        description="Co-registered optical and SAR pair for cross-modal analysis.",
        family="cross_modal",
        suggested_query="Use the optical and SAR images together to identify built-up and water-covered regions.",
        files=[
            "adapt_kolkata_urban_20241101_20250228_chip0000_opt.tif",
            "adapt_kolkata_urban_20241101_20250228_chip0000_sar.tif",
        ],
        modalities=["multispectral", "sar"],
    ),
    SampleScenario(
        id="kolkata_change",
        title="Kolkata — Vegetation Change",
        description="Two optical scenes from different dates for bi-temporal change analysis.",
        family="bi_temporal",
        suggested_query="What changed in vegetation between these two dates?",
        files=[
            "adapt_kolkata_urban_20241101_20250228_chip0000_opt.tif",
            "adapt_kolkata_urban_20241101_20250228_chip0001_opt.tif",
        ],
        modalities=["multispectral", "multispectral"],
    ),
    SampleScenario(
        id="kolkata_crop_stress",
        title="Kolkata — Crop Health Assessment",
        description="Assess crop stress using spectral indices on a multispectral scene.",
        family="single_image",
        suggested_query="Is the crop in this area stressed?",
        files=["kolkata_optical.tif"],
        modalities=["multispectral"],
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _infer_modality(band_names: list[str], band_count: int) -> Modality:
    if any(n in ("vv", "vh", "hh", "hv") for n in band_names):
        return Modality.SAR
    if band_count >= 4:
        return Modality.MULTISPECTRAL
    if band_count == 3:
        return Modality.OPTICAL
    if band_count in (1, 2):
        return Modality.SAR
    return Modality.UNKNOWN


def _load(path: Path, modality: Modality | None) -> ImageRef:
    with rasterio.open(path) as src:
        names = [d.lower() for d in (src.descriptions or []) if d]
        return ImageRef(
            path=str(path),
            modality=modality or _infer_modality(names, src.count),
            crs=str(src.crs) if src.crs else None,
            bounds=tuple(src.bounds) if src.crs else None,
            width=src.width,
            height=src.height,
            band_count=src.count,
            band_names=names,
        )


def _parse_modalities(raw: str | None, count: int) -> list[Modality | None]:
    if not raw:
        return [None] * count
    parts = [p.strip().lower() for p in raw.split(",")]
    if len(parts) != count:
        raise HTTPException(400, f"Got {len(parts)} modalities for {count} files.")
    try:
        return [Modality(p) for p in parts]
    except ValueError as exc:
        raise HTTPException(400, f"Unknown modality: {exc}") from exc


def _to_url(path_str: str | None) -> str | None:
    if not path_str:
        return None
    p = Path(path_str)
    if p.exists():
        # Copy to output dir if not already there
        dest = OUTPUT_DIR / p.name
        if not dest.exists():
            shutil.copy2(p, dest)
        return f"/outputs/{p.name}"
    return None


def _rewrite_evidence_urls(response: QueryResponse) -> QueryResponse:
    """Rewrite all on-disk paths in evidence to browser-fetchable URLs."""
    updates: dict = {}
    if response.evidence.overlay_png:
        updates["overlay_png"] = _to_url(response.evidence.overlay_png)
    if response.evidence.mask_path:
        updates["mask_path"] = _to_url(response.evidence.mask_path)
    if updates:
        new_evidence = response.evidence.model_copy(update=updates)
        response = response.model_copy(update={"evidence": new_evidence})
    return response


def _error_response(message: str, request_id: str | None = None) -> QueryResponse:
    """Return a valid QueryResponse for any error, never a raw 500."""
    return QueryResponse(
        answer=f"An error occurred: {message}",
        evidence=Evidence(kind="none"),
        confidence=0.0,
        confidence_method="not_attempted",
        task_family=TaskFamily.SINGLE_IMAGE,
        tools_used=[],
        trace=[],
        validation=ValidationReport(ok=False),
        request_id=request_id,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict:
    from jatayu.tools.registry import REGISTRY

    return {
        "status": "ok",
        "tools_registered": [n.value for n in REGISTRY],
        "samples_available": len(SAMPLE_SCENARIOS),
    }


@app.get("/api/tools")
def list_tools() -> dict[str, str]:
    from jatayu.tools.registry import DESCRIPTIONS

    return {n.value: d for n, d in DESCRIPTIONS.items()}


@app.get("/api/samples")
def list_samples() -> list[dict]:
    """Available demo scenarios. The frontend shows these as one-click buttons."""
    return [s.model_dump() for s in SAMPLE_SCENARIOS]


@app.post("/api/analyze")
async def analyze(
    query: str = Form(...),
    files: list[UploadFile] | None = File(None),
    sample_id: str | None = Form(None),
    language: str = Form("eng_Latn"),
    modalities: str | None = Form(None),
) -> AnalyzeResponse:
    """Main analysis endpoint.

    Either upload files OR provide a sample_id — not both.
    """
    request_id = uuid.uuid4().hex[:12]

    # --- resolve images from sample or upload --------------------------------
    if sample_id:
        scenario = next((s for s in SAMPLE_SCENARIOS if s.id == sample_id), None)
        if not scenario:
            return AnalyzeResponse(
                result=_error_response(
                    f"Unknown sample '{sample_id}'. Available: "
                    + ", ".join(s.id for s in SAMPLE_SCENARIOS),
                    request_id,
                ),
                request_id=request_id,
            )
        try:
            images = []
            for fname, mod_str in zip(scenario.files, scenario.modalities):
                path = SAMPLES_DIR / fname
                if not path.exists():
                    return AnalyzeResponse(
                        result=_error_response(
                            f"Sample file not found: {fname}", request_id
                        ),
                        request_id=request_id,
                    )
                images.append(_load(path, Modality(mod_str)))
        except Exception as exc:
            return AnalyzeResponse(
                result=_error_response(str(exc), request_id),
                request_id=request_id,
            )
    elif files and len(files) > 0 and files[0].filename:
        if not 1 <= len(files) <= 2:
            return AnalyzeResponse(
                result=_error_response(
                    "Provide one image, or a pair of two.", request_id
                ),
                request_id=request_id,
            )
        overrides = _parse_modalities(modalities, len(files))
        saved: list[Path] = []
        try:
            for i, upload in enumerate(files):
                name = Path(upload.filename or f"upload{i}").name
                dest = UPLOAD_DIR / f"{request_id}_{i}_{name}"
                with dest.open("wb") as fh:
                    shutil.copyfileobj(upload.file, fh)
                saved.append(dest)
            images = [_load(p, m) for p, m in zip(saved, overrides, strict=True)]
        except rasterio.errors.RasterioIOError as exc:
            return AnalyzeResponse(
                result=_error_response(
                    f"Could not read that file as a raster: {exc}", request_id
                ),
                request_id=request_id,
            )
        finally:
            for p in saved:
                p.unlink(missing_ok=True)
    else:
        return AnalyzeResponse(
            result=_error_response(
                "Provide either files to upload or a sample_id.", request_id
            ),
            request_id=request_id,
        )

    # --- run the orchestrator ------------------------------------------------
    try:
        response = orchestrator.run(query, images)
        response = _rewrite_evidence_urls(response)
    except Exception as exc:
        logger.error("Orchestrator failed: %s\n%s", exc, traceback.format_exc())
        response = _error_response(str(exc), request_id)

    # --- wrap with language info (never touches schemas.py) ------------------
    lang_info = LanguageInfo(
        detected=language,
        original_query=query,
    )

    return AnalyzeResponse(
        result=response,
        language=lang_info,
        request_id=request_id,
    )


# Keep the old /query endpoint for backward compatibility
@app.post("/query")
async def query_legacy(
    query: str = Form(...),
    files: list[UploadFile] = File(...),
    language: str = Form("eng_Latn"),
    modalities: str | None = Form(None),
) -> AnalyzeResponse:
    """Legacy endpoint — redirects to /api/analyze."""
    return await analyze(
        query=query, files=files, language=language, modalities=modalities,
    )
