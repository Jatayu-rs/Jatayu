"""
Thin HTTP client used by the Streamlit frontend to talk to the Jatayu FastAPI backend.
"""

import requests

# Point this at wherever your FastAPI server is actually running.
# If backend + frontend run on the same machine during dev, localhost is fine.
API_BASE_URL = "http://127.0.0.1:8000"
QUERY_ENDPOINT = f"{API_BASE_URL}/query"

DEFAULT_TIMEOUT = 60  # seconds; bump this up if the vision model is slow


def query(uploaded_files, question: str, lang: str = "en") -> dict:
    """
    Sends uploaded satellite image(s) + a question to the backend /query endpoint.

    Args:
        uploaded_files: list of Streamlit UploadedFile objects (from st.file_uploader)
        question: the user's question text
        lang: language code (e.g. "en", "hi", "bn")

    Returns:
        Parsed JSON response as a dict, matching the shape the frontend expects
        (answer / answer_translated / answer_english / confidence / issues /
        overlay_png / legend / trace).

    Raises:
        RuntimeError if the backend call fails, with a message suitable for
        surfacing directly in the Streamlit UI.
    """
    if not uploaded_files:
        raise RuntimeError("No files provided to query().")

    # Build multipart file payload. Streamlit's UploadedFile is already file-like,
    # so we can pass it straight through to requests.
    files = []
    for f in uploaded_files:
        f.seek(0)  # ensure pointer is at the start in case it was read before
        files.append(("files", (f.name, f.read(), f.type or "application/octet-stream")))

    data = {
        "query": question,
        "language": lang,
    }

    try:
        response = requests.post(
            QUERY_ENDPOINT,
            files=files,
            data=data,
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Could not reach the backend at {QUERY_ENDPOINT}. "
            f"Is `uv run uvicorn jatayu.api.main:app --reload` running? ({e})"
        ) from e
    except requests.exceptions.Timeout as e:
        raise RuntimeError(
            f"Backend took too long to respond (>{DEFAULT_TIMEOUT}s). "
            f"The vision model may still be processing a large request. ({e})"
        ) from e
    except requests.exceptions.HTTPError as e:
        # Surface backend's error detail if it sent one (FastAPI HTTPException body)
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise RuntimeError(f"Backend returned an error: {detail}") from e

    return _map_backend_response(response.json(), lang)


def _map_backend_response(raw: dict, lang: str) -> dict:
    """
    Reshapes the raw QueryResponse JSON (from jatayu.schemas.QueryResponse,
    as returned by POST /query) into the flat dict shape app.py's Streamlit
    UI expects.

    Backend QueryResponse shape (from schemas.py):
      answer: str
      evidence: { kind, overlay_png, legend, caption }
      confidence: float
      confidence_method: str
      task_family: str
      tools_used: list
      trace: list[TraceStep]
      language: str
      answer_original: str | None

    Frontend (app.py) expects (flat dict):
      answer, answer_translated, answer_english, confidence, issues,
      overlay_png, legend, trace
    """
    evidence = raw.get("evidence") or {}

    mapped = {
        "answer": raw.get("answer", ""),
        "confidence": raw.get("confidence", 0.0),
        # backend has no `issues` field yet -- stubbed empty until orchestrator populates one
        "issues": raw.get("issues", []),
        "overlay_png": evidence.get("overlay_png"),
        "legend": evidence.get("legend", {}),
        "trace": raw.get("trace", []),
    }

    # Backend translates `answer` server-side when language != eng_Latn/en,
    # and keeps the English source in `answer_original`.
    if lang not in ("en", "eng_Latn"):
        mapped["answer_translated"] = raw.get("answer", "")
        mapped["answer_english"] = raw.get("answer_original") or raw.get("answer", "")

    return mapped
