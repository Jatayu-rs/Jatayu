"""
Jatayu FastAPI backend.

The frontend talks to this module over HTTP.
Heavy geospatial/model processing remains in the Jatayu backend.

Run from ~/jatayu:

    uvicorn jatayu.api.main:app --reload

Endpoints
---------
GET  /api/health
GET  /api/tools
GET  /api/samples
POST /api/analyze
POST /query              # backwards compatible
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import traceback
import uuid
from pathlib import Path

import rasterio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import jatayu.tools  # noqa: F401 — registers tools
from jatayu.controller.orchestrator import Orchestrator
from jatayu.schemas import (
    Evidence,
    ImageRef,
    Modality,
    QueryResponse,
    TaskFamily,
    ValidationReport,
)

logger = logging.getLogger("jatayu.api")
logging.basicConfig(level=logging.INFO)


# ============================================================================
# APP
# ============================================================================

app = FastAPI(
    title="Jatayu",
    version="0.1.0",
    description="Physics-guided geospatial analysis backend.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = Orchestrator()


# ============================================================================
# DIRECTORIES
# ============================================================================

UPLOAD_DIR = Path(tempfile.gettempdir()) / "jatayu-uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = Path("data/samples/jatayu_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES_DIR = Path("data/samples")
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# Anything generated here can be fetched by the browser.
app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="outputs",
)


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class LanguageInfo(BaseModel):
    detected: str = "eng_Latn"
    display_name: str = "English"
    was_translated: bool = False
    original_query: str | None = None


class AnalyzeResponse(BaseModel):
    result: QueryResponse
    language: LanguageInfo | None = None
    request_id: str | None = None


# ============================================================================
# SAMPLE SCENARIOS
# ============================================================================

class SampleScenario(BaseModel):
    id: str
    title: str
    description: str
    family: str
    suggested_query: str
    files: list[str]
    modalities: list[str]


SAMPLE_SCENARIOS: list[SampleScenario] = [
    SampleScenario(
        id="kolkata_water",
        title="Kolkata — Water Body Detection",
        description=(
            "Multispectral optical scene over Kolkata and the Hooghly "
            "river system."
        ),
        family="single_image",
        suggested_query="Highlight the water body in this image.",
        files=[
            "kolkata_optical.tif",
        ],
        modalities=[
            "multispectral",
        ],
    ),

    SampleScenario(
        id="kolkata_landcover",
        title="Kolkata — Land Cover",
        description=(
            "Describe the major land-cover types visible in the Kolkata scene."
        ),
        family="single_image",
        suggested_query=(
            "Describe the land cover in this image and highlight "
            "agricultural areas."
        ),
        files=[
            "kolkata_optical.tif",
        ],
        modalities=[
            "multispectral",
        ],
    ),

    SampleScenario(
        id="kolkata_fusion",
        title="Kolkata — Optical + SAR Fusion",
        description=(
            "Co-registered optical and SAR images for cross-modal "
            "surface analysis."
        ),
        family="cross_modal",
        suggested_query=(
            "Use the optical and SAR images together to identify "
            "built-up and water-covered regions."
        ),
        files=[
            "adapt_kolkata_urban_20241101_20250228_chip0000_opt.tif",
            "adapt_kolkata_urban_20241101_20250228_chip0000_sar.tif",
        ],
        modalities=[
            "multispectral",
            "sar",
        ],
    ),

    SampleScenario(
        id="kolkata_change",
        title="Kolkata — Vegetation Change",
        description=(
            "Two optical scenes from different dates for bi-temporal "
            "change analysis."
        ),
        family="bi_temporal",
        suggested_query=(
            "What changed in vegetation between these two dates?"
        ),
        files=[
            "adapt_kolkata_urban_20241101_20250228_chip0000_opt.tif",
            "adapt_kolkata_urban_20241101_20250228_chip0001_opt.tif",
        ],
        modalities=[
            "multispectral",
            "multispectral",
        ],
    ),

    SampleScenario(
        id="kolkata_crop_stress",
        title="Kolkata — Crop Health",
        description=(
            "Assess vegetation condition using spectral indices."
        ),
        family="single_image",
        suggested_query=(
            "Identify areas of vegetation stress in this image."
        ),
        files=[
            "kolkata_optical.tif",
        ],
        modalities=[
            "multispectral",
        ],
    ),
    SampleScenario(
        id="nepal_flood",
        title="Nepal — Post-Monsoon Flood Change",
        description="Pre vs post-monsoon Sentinel-2 over Kathmandu valley. "
                    "Detects increased water extent from recent flooding.",
        family="bi_temporal",
        suggested_query="What changed in water extent between pre and post monsoon?",
        files=["../demo/nepal_pre_flood.tif", "../demo/nepal_post_flood.tif"],
        modalities=["multispectral", "multispectral"],
    ),
    SampleScenario(
        id="nepal_water",
        title="Nepal — Flood Water Detection",
        description="Post-monsoon Sentinel-2 scene. Highlights standing water "
                    "and flood extent using MNDWI.",
        family="single_image",
        suggested_query="Highlight the water bodies and flood extent in this image.",
        files=["../demo/nepal_post_flood.tif"],
        modalities=["multispectral"],
    ),
    SampleScenario(
        id="punjab_crop",
        title="Punjab — Kharif Crop Health",
        description="August 2026 Sentinel-2 over Punjab wheat belt. "
                    "Multi-index crop stress assessment.",
        family="single_image",
        suggested_query="Is the crop in this area stressed?",
        files=["../demo/punjab_crop.tif"],
        modalities=["multispectral"],
    ),
]


# ============================================================================
# HELPERS
# ============================================================================

def _infer_modality(
    band_names: list[str],
    band_count: int,
) -> Modality:
    """
    Infer modality when the frontend does not explicitly provide it.
    """

    names = {n.lower() for n in band_names}

    # SAR polarisation names.
    if names.intersection({"vv", "vh", "hh", "hv"}):
        return Modality.SAR

    # Explicit SAR-style one/two-band raster.
    if band_count in (1, 2):
        return Modality.SAR

    # Standard RGB.
    if band_count == 3:
        return Modality.OPTICAL

    # Multispectral.
    if band_count >= 4:
        return Modality.MULTISPECTRAL

    return Modality.UNKNOWN


def _load(
    path: Path,
    modality: Modality | None = None,
) -> ImageRef:
    """
    Read raster metadata and create the ImageRef expected by Jatayu.
    """

    with rasterio.open(path) as src:

        descriptions = src.descriptions or ()

        band_names = [
            str(name).lower()
            for name in descriptions
            if name
        ]

        inferred = modality or _infer_modality(
            band_names,
            src.count,
        )

        bounds = (
            tuple(src.bounds)
            if src.crs is not None
            else None
        )

        return ImageRef(
            path=str(path),
            modality=inferred,
            crs=str(src.crs) if src.crs else None,
            bounds=bounds,
            width=src.width,
            height=src.height,
            band_count=src.count,
            band_names=band_names,
        )


def _parse_modalities(
    raw: str | None,
    count: int,
) -> list[Modality | None]:
    """
    Parse:

        multispectral,sar

    into:

        [Modality.MULTISPECTRAL, Modality.SAR]
    """

    if not raw:
        return [None] * count

    parts = [
        part.strip().lower()
        for part in raw.split(",")
    ]

    if len(parts) != count:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Got {len(parts)} modalities for "
                f"{count} files."
            ),
        )

    result: list[Modality | None] = []

    for part in parts:
        try:
            result.append(Modality(part))
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown modality '{part}'.",
            ) from exc

    return result


def _to_url(
    path_str: str | None,
) -> str | None:
    """
    Convert a backend filesystem path into a browser-accessible URL.
    """

    if not path_str:
        return None

    path = Path(path_str)

    if not path.exists():
        logger.warning(
            "Evidence file does not exist: %s",
            path,
        )
        return None

    try:
        destination = OUTPUT_DIR / path.name

        # Avoid copying a file onto itself.
        if path.resolve() != destination.resolve():
            shutil.copy2(path, destination)

        return f"/outputs/{destination.name}"

    except Exception:
        logger.exception(
            "Could not expose output file: %s",
            path,
        )
        return None


def _rewrite_evidence_urls(
    response: QueryResponse,
) -> QueryResponse:
    """
    Convert generated evidence paths into URLs that the frontend can fetch.
    """

    evidence = response.evidence

    updates: dict[str, str | None] = {}

    if evidence.overlay_png:
        url = _to_url(evidence.overlay_png)

        if url:
            updates["overlay_png"] = url

    if evidence.mask_path:
        url = _to_url(evidence.mask_path)

        if url:
            updates["mask_path"] = url

    if not updates:
        return response

    new_evidence = evidence.model_copy(
        update=updates,
    )

    return response.model_copy(
        update={
            "evidence": new_evidence,
        }
    )


def _error_response(
    message: str,
    request_id: str | None = None,
) -> QueryResponse:
    """
    Always return a valid QueryResponse instead of exposing a raw 500.
    """

    return QueryResponse(
        answer=f"An error occurred: {message}",
        evidence=Evidence(
            kind="none",
        ),
        confidence=0.0,
        confidence_method="not_attempted",
        task_family=TaskFamily.SINGLE_IMAGE,
        tools_used=[],
        trace=[],
        validation=ValidationReport(
            ok=False,
        ),
        request_id=request_id,
    )


# ============================================================================
# HEALTH
# ============================================================================

@app.get("/api/health")
def health() -> dict:
    """
    Backend health check.
    """

    from jatayu.tools.registry import REGISTRY

    return {
        "status": "ok",
        "service": "jatayu",
        "version": "0.1.0",
        "tools_registered": [
            name.value
            for name in REGISTRY
        ],
        "samples_available": len(
            SAMPLE_SCENARIOS
        ),
    }


# ============================================================================
# TOOLS
# ============================================================================

@app.get("/api/tools")
def list_tools() -> dict[str, str]:
    """
    List all registered specialist tools.
    """

    from jatayu.tools.registry import DESCRIPTIONS

    return {
        name.value: description
        for name, description in DESCRIPTIONS.items()
    }


# ============================================================================
# SAMPLES
# ============================================================================

@app.get("/api/samples")
def list_samples() -> list[dict]:
    """
    Return demo scenarios to the frontend.
    """

    return [
        scenario.model_dump()
        for scenario in SAMPLE_SCENARIOS
    ]


# ============================================================================
# MAIN ANALYSIS ENDPOINT
# ============================================================================

@app.post("/api/analyze")
async def analyze(
    query: str = Form(...),
    files: list[UploadFile] | None = File(None),
    sample_id: str | None = Form(None),
    language: str = Form("eng_Latn"),
    modalities: str | None = Form(None),
) -> AnalyzeResponse:

    request_id = uuid.uuid4().hex[:12]

    logger.info(
        "[%s] Analysis request: %s",
        request_id,
        query,
    )

    images: list[ImageRef] = []
    saved_paths: list[Path] = []

    # ========================================================================
    # 1. SAMPLE INPUT
    # ========================================================================

    if sample_id:

        scenario = next(
            (s for s in SAMPLE_SCENARIOS if s.id == sample_id),
            None,
        )

        if scenario is None:
            return AnalyzeResponse(
                result=_error_response(
                    f"Unknown sample '{sample_id}'.",
                    request_id,
                ),
                request_id=request_id,
            )

        try:
            for filename, modality_string in zip(
                scenario.files,
                scenario.modalities,
                strict=True,
            ):

                path = SAMPLES_DIR / filename

                if not path.exists():
                    return AnalyzeResponse(
                        result=_error_response(
                            f"Sample file not found: {filename}",
                            request_id,
                        ),
                        request_id=request_id,
                    )

                image = _load(
                    path,
                    Modality(modality_string),
                )

                images.append(image)

        except Exception as exc:

            logger.error(
                "[%s] Sample loading failed:\n%s",
                request_id,
                traceback.format_exc(),
            )

            return AnalyzeResponse(
                result=_error_response(
                    str(exc),
                    request_id,
                ),
                request_id=request_id,
            )

    # ========================================================================
    # 2. UPLOADED FILES
    # ========================================================================

    elif files:

        valid_files = [
            upload
            for upload in files
            if upload.filename
        ]

        if not valid_files:
            return AnalyzeResponse(
                result=_error_response(
                    "No valid files were uploaded.",
                    request_id,
                ),
                request_id=request_id,
            )

        if len(valid_files) > 2:
            return AnalyzeResponse(
                result=_error_response(
                    "Jatayu currently accepts at most two images.",
                    request_id,
                ),
                request_id=request_id,
            )

        try:

            overrides = _parse_modalities(
                modalities,
                len(valid_files),
            )

            # ---------------------------------------------------------------
            # SAVE UPLOADS
            # ---------------------------------------------------------------

            for index, upload in enumerate(valid_files):

                original_name = Path(
                    upload.filename
                ).name

                destination = (
                    UPLOAD_DIR
                    / f"{request_id}_{index}_{original_name}"
                )

                with destination.open("wb") as output:
                    shutil.copyfileobj(
                        upload.file,
                        output,
                    )

                saved_paths.append(destination)

                logger.info(
                    "[%s] Saved upload: %s",
                    request_id,
                    destination,
                )

            # ---------------------------------------------------------------
            # LOAD METADATA
            # ---------------------------------------------------------------

            images = [
                _load(path, modality)
                for path, modality in zip(
                    saved_paths,
                    overrides,
                    strict=True,
                )
            ]

            logger.info(
                "[%s] Loaded %d image(s)",
                request_id,
                len(images),
            )

            for image in images:

                logger.info(
                    "[%s] %s | modality=%s | "
                    "size=%sx%s | bands=%s",
                    request_id,
                    image.path,
                    image.modality.value,
                    image.width,
                    image.height,
                    image.band_count,
                )

        except rasterio.errors.RasterioIOError as exc:

            logger.error(
                "[%s] Raster loading failed: %s",
                request_id,
                exc,
            )

            # Clean up because analysis will not run.
            for path in saved_paths:
                path.unlink(missing_ok=True)

            return AnalyzeResponse(
                result=_error_response(
                    f"Could not read uploaded raster: {exc}",
                    request_id,
                ),
                request_id=request_id,
            )

        except Exception as exc:

            logger.error(
                "[%s] Upload processing failed:\n%s",
                request_id,
                traceback.format_exc(),
            )

            for path in saved_paths:
                path.unlink(missing_ok=True)

            return AnalyzeResponse(
                result=_error_response(
                    str(exc),
                    request_id,
                ),
                request_id=request_id,
            )

    # ========================================================================
    # 3. NO INPUT
    # ========================================================================

    else:

        return AnalyzeResponse(
            result=_error_response(
                "Provide either uploaded files or a sample_id.",
                request_id,
            ),
            request_id=request_id,
        )

    # ========================================================================
    # 4. RUN ORCHESTRATOR
    #
    # IMPORTANT:
    # DO NOT DELETE uploaded files before this finishes.
    # ========================================================================

    try:

        logger.info(
            "[%s] Running orchestrator...",
            request_id,
        )

        # Verify that uploaded files still exist.
        for image in images:

            image_path = Path(image.path)

            logger.info(
                "[%s] Checking input exists: %s -> %s",
                request_id,
                image_path,
                image_path.exists(),
            )

            if not image_path.exists():

                raise FileNotFoundError(
                    f"Input raster disappeared before analysis: "
                    f"{image_path}"
                )

        # ---------------------------------------------------------------
        # ACTUAL JATAYU ANALYSIS
        # ---------------------------------------------------------------

        response = orchestrator.run(
            query,
            images,
        )

        response = _rewrite_evidence_urls(
            response
        )

        # Preserve request ID.
        if response.request_id is None:

            response = response.model_copy(
                update={
                    "request_id": request_id,
                }
            )

        logger.info(
            "[%s] Analysis complete. tool=%s confidence=%.2f",
            request_id,
            (
                response.tools_used[0].value
                if response.tools_used
                else "none"
            ),
            response.confidence,
        )

    except Exception as exc:

        logger.error(
            "[%s] Orchestrator failed:\n%s",
            request_id,
            traceback.format_exc(),
        )

        response = _error_response(
            str(exc),
            request_id,
        )

    # ========================================================================
    # 5. CLEAN UP UPLOADS
    #
    # ONLY NOW is it safe to delete temporary input files.
    # ========================================================================

    finally:

        for path in saved_paths:

            try:

                if path.exists():
                    path.unlink()

                    logger.info(
                        "[%s] Deleted temporary upload: %s",
                        request_id,
                        path,
                    )

            except Exception:

                logger.warning(
                    "[%s] Could not delete temporary upload: %s",
                    request_id,
                    path,
                )

    # ========================================================================
    # 6. LANGUAGE INFO
    # ========================================================================

    language_info = LanguageInfo(
        detected=language,
        original_query=query,
    )

    return AnalyzeResponse(
        result=response,
        language=language_info,
        request_id=request_id,
    )

    # ========================================================================
    # SAMPLE INPUT
    # ========================================================================

    if sample_id:

        scenario = next(
            (
                scenario
                for scenario in SAMPLE_SCENARIOS
                if scenario.id == sample_id
            ),
            None,
        )

        if scenario is None:
            return AnalyzeResponse(
                result=_error_response(
                    (
                        f"Unknown sample '{sample_id}'. "
                        "Available samples: "
                        + ", ".join(
                            s.id
                            for s in SAMPLE_SCENARIOS
                        )
                    ),
                    request_id,
                ),
                request_id=request_id,
            )

        try:

            images: list[ImageRef] = []

            for filename, modality_string in zip(
                scenario.files,
                scenario.modalities,
                strict=True,
            ):

                path = SAMPLES_DIR / filename

                if not path.exists():
                    return AnalyzeResponse(
                        result=_error_response(
                            (
                                f"Sample file not found: "
                                f"{filename}"
                            ),
                            request_id,
                        ),
                        request_id=request_id,
                    )

                modality = Modality(
                    modality_string
                )

                image = _load(
                    path,
                    modality,
                )

                images.append(image)

        except Exception as exc:

            logger.error(
                "[%s] Sample loading failed:\n%s",
                request_id,
                traceback.format_exc(),
            )

            return AnalyzeResponse(
                result=_error_response(
                    str(exc),
                    request_id,
                ),
                request_id=request_id,
            )

    # ========================================================================
    # UPLOAD INPUT
    # ========================================================================

    elif files:

        valid_files = [
            upload
            for upload in files
            if upload.filename
        ]

        if not valid_files:
            return AnalyzeResponse(
                result=_error_response(
                    "No valid files were uploaded.",
                    request_id,
                ),
                request_id=request_id,
            )

        if len(valid_files) > 2:
            return AnalyzeResponse(
                result=_error_response(
                    "Jatayu currently accepts at most two images.",
                    request_id,
                ),
                request_id=request_id,
            )

        overrides = _parse_modalities(
            modalities,
            len(valid_files),
        )

        saved_paths: list[Path] = []

        try:

            for index, upload in enumerate(
                valid_files
            ):

                original_name = Path(
                    upload.filename
                ).name

                destination = (
                    UPLOAD_DIR
                    / (
                        f"{request_id}_"
                        f"{index}_"
                        f"{original_name}"
                    )
                )

                with destination.open("wb") as output:

                    shutil.copyfileobj(
                        upload.file,
                        output,
                    )

                saved_paths.append(
                    destination
                )

            images = [
                _load(
                    path,
                    modality,
                )
                for path, modality in zip(
                    saved_paths,
                    overrides,
                    strict=True,
                )
            ]

            logger.info(
                "[%s] Loaded %d image(s)",
                request_id,
                len(images),
            )

            for image in images:
                logger.info(
                    "[%s] %s | modality=%s | "
                    "size=%sx%s | bands=%s",
                    request_id,
                    image.path,
                    image.modality.value,
                    image.width,
                    image.height,
                    image.band_count,
                )

        except rasterio.errors.RasterioIOError as exc:

            return AnalyzeResponse(
                result=_error_response(
                    (
                        "Could not read the uploaded file "
                        "as a raster: "
                        f"{exc}"
                    ),
                    request_id,
                ),
                request_id=request_id,
            )

        except Exception as exc:

            logger.error(
                "[%s] Upload processing failed:\n%s",
                request_id,
                traceback.format_exc(),
            )

            return AnalyzeResponse(
                result=_error_response(
                    str(exc),
                    request_id,
                ),
                request_id=request_id,
            )

        finally:

            # The specialist only needs the files during this request.
            for path in saved_paths:
                try:
                    path.unlink(
                        missing_ok=True
                    )
                except Exception:
                    logger.warning(
                        "Could not delete temporary upload: %s",
                        path,
                    )

    # ========================================================================
    # NO INPUT
    # ========================================================================

    else:

        return AnalyzeResponse(
            result=_error_response(
                (
                    "Provide either uploaded files "
                    "or a sample_id."
                ),
                request_id,
            ),
            request_id=request_id,
        )

    # ========================================================================
    # RUN ORCHESTRATOR
    # ========================================================================

    try:

        logger.info(
            "[%s] Running orchestrator...",
            request_id,
        )

        response = orchestrator.run(
            query,
            images,
        )

        response = _rewrite_evidence_urls(
            response
        )

        # Make sure the request ID survives even if
        # the orchestrator didn't set it.
        if response.request_id is None:
            response = response.model_copy(
                update={
                    "request_id": request_id,
                }
            )

        logger.info(
            "[%s] Analysis complete. tool=%s confidence=%.2f",
            request_id,
            (
                response.tools_used[0].value
                if response.tools_used
                else "none"
            ),
            response.confidence,
        )

    except Exception as exc:

        logger.error(
            "[%s] Orchestrator failed:\n%s",
            request_id,
            traceback.format_exc(),
        )

        response = _error_response(
            str(exc),
            request_id,
        )

    # ========================================================================
    # LANGUAGE INFORMATION
    # ========================================================================

    language_info = LanguageInfo(
        detected=language,
        original_query=query,
    )

    return AnalyzeResponse(
        result=response,
        language=language_info,
        request_id=request_id,
    )


# ============================================================================
# LEGACY ENDPOINT
# ============================================================================

@app.post("/query")
async def query_legacy(
    query: str = Form(...),
    files: list[UploadFile] = File(...),
    language: str = Form("eng_Latn"),
    modalities: str | None = Form(None),
) -> AnalyzeResponse:
    """
    Backwards-compatible endpoint.

    Older frontend code can continue using /query.
    """

    return await analyze(
        query=query,
        files=files,
        sample_id=None,
        language=language,
        modalities=modalities,
    )
