from __future__ import annotations 
from fastapi import FastAPI, File, Form, HTTPException, UploadFile 
from fastapi.middleware.cors import CORSMiddleware 
import jatayu.tools
from jatayu.controller import Orchestrator 
from jatayu.schemas import ImageRef, QueryResponse



app = FastAPI(title="Jatayu")
app.add_middleware(CORSMiddleware, allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

orchestrator = Orchestrator()


@app.get("/health")
def health() -> dict[str , str]:
    return {"status":"ok"}

@app.get("/tools")
def list_tools() -> dict[str,str]:
    from jatayu.tools.registry import DESCRIPTIONS 
    return {n.value:d for n , d in DESCRIPTIONS.items()}

@app.post("/query", response_model=QueryResponse)
async def query(
    query: str = Form(...),
    files: list[UploadFile] = File(...),
    language: str = Form("eng_Latn"),
) -> QueryResponse:
    if not 1 <= len(files) <= 2:
        raise HTTPException(400, "Provide one image, or a pair of two.")
    # Real GeoTIFF loading comes in a later stage; placeholder metadata for now.
    images = [ImageRef(path=f.filename or "upload", width=512, height=512)
              for f in files]
    response = orchestrator.run(query, images)
    response.language = language
    return response
