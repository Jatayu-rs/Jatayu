"""HTTP API.

Kept separate from the UI so the frontend never needs a GPU, model weights, or
the geospatial stack installed — it develops against this over HTTP.
"""

from __future__ import annotations
from src.jatayu.io.database import JatayuDatabaseManager

import shutil
import tempfile
import uuid
from pathlib import Path

import rasterio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import jatayu.tools  # noqa: F401 — importing registers every tool
from jatayu.controller import Orchestrator
from jatayu.schemas import ImageRef, Modality, QueryResponse

app = FastAPI(title="Jatayu")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

orchestrator = Orchestrator()

# Uploads are transient; rendered overlays must outlive the request so the
# browser can fetch them, so they live in a served directory instead.
UPLOAD_DIR = Path(tempfile.gettempdir()) / "jatayu-uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


# ---------------------------------------------------------------- helpers


def _infer_modality(band_names: list[str], band_count: int) -> Modality:
    """Guess the sensor type. Returns UNKNOWN rather than guessing wrong —
    a wrong modality routes the query to the wrong tool."""
    if any(n in ("vv", "vh", "hh", "hv") for n in band_names):
        return Modality.SAR
    if band_count >= 4:
        return Modality.MULTISPECTRAL
    if band_count == 3:
        return Modality.OPTICAL
    if band_count in (1, 2):
        return Modality.SAR  # single/dual polarisation
    return Modality.UNKNOWN


def _load(path: Path, modality: Modality | None) -> ImageRef:
    """Open a raster and return its real metadata. Does not read pixels."""
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
    """Rewrite an on-disk overlay path into something the browser can fetch."""
    if not path_str:
        return None
    return f"/outputs/{Path(path_str).name}"


# ---------------------------------------------------------------- endpoints


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tools")
def list_tools() -> dict[str, str]:
    """Registered specialists and what they do. Useful for the UI and for judges."""
    from jatayu.tools.registry import DESCRIPTIONS

    return {n.value: d for n, d in DESCRIPTIONS.items()}


@app.post("/query", response_model=QueryResponse)
async def query(
    query: str = Form(...),
    files: list[UploadFile] = File(...),
    language: str = Form("eng_Latn"),
    modalities: str | None = Form(None),
) -> QueryResponse:
    """Analyse one or two images against a natural-language query.

    `modalities` is an optional comma-separated override matching the file order,
    e.g. "multispectral,sar", for when automatic inference gets it wrong.
    """
    if not 1 <= len(files) <= 2:
        raise HTTPException(400, "Provide one image, or a pair of two.")

    request_id = uuid.uuid4().hex[:12]
    overrides = _parse_modalities(modalities, len(files))
    saved: list[Path] = []

    try:
        for i, upload in enumerate(files):
            name = Path(upload.filename or f"upload{i}").name
            dest = UPLOAD_DIR / f"{request_id}_{i}_{name}"
            with dest.open("wb") as fh:
                shutil.copyfileobj(upload.file, fh)
            saved.append(dest)

        try:
            images = [_load(p, m) for p, m in zip(saved, overrides, strict=True)]
        except rasterio.errors.RasterioIOError as exc:
            raise HTTPException(
                400,
                "Could not read that file as a raster. Supported: GeoTIFF/TIFF, "
                f"or PNG/JPEG for benchmark datasets. ({exc})",
            ) from exc

        response = orchestrator.run(query, images)
        response.language = language
        response.evidence.overlay_png = _to_url(response.evidence.overlay_png)
        return response

    finally:
        # Uploads are disposable; the rendered overlay in outputs/ is what persists.
        for p in saved:
            p.unlink(missing_ok=True)


db = JatayuDatabaseManager()

def handle_workspace_access(email, password):
    result = db.authenticate_user(email, password)
    if result["status"] == "success":
        return "🔓 Access Granted. Routing telemetry feeds..."
    else:
        return f"❌ Access Denied: {result['message']}"

