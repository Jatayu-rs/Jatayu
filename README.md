# Jatayu

**An agentic vision-language assistant for multimodal remote-sensing image analysis.**

Jatayu is our implementation of **SatQuery AI** — Smart India Hackathon problem statement
`SIH26167`, issued by ISRO / Space Applications Centre.

Ask a question in plain language about satellite imagery. Jatayu works out which task you
mean, checks your images are suitable, routes the query to the right specialist
remote-sensing model, and returns an answer grounded in visual evidence — along with a
full trace of how it got there.

> *In the Ramayana, Jatayu witnesses Sita's abduction from the air, and with his last
> breath tells Rama what happened and which way Ravana went. Observation, location,
> testimony — reported honestly, and never guessed at.*

---

## Status

🚧 **Early development.** The skeleton runs end to end on stubbed models. Real models are
being swapped in behind stable interfaces.

| Capability | State |
|---|---|
| GeoTIFF ingest, validation, preprocessing | 🔨 in progress |
| Agentic controller + execution trace | 🔨 in progress |
| Single-image VQA | ⬜ stubbed |
| Text-guided region grounding | ⬜ stubbed |
| Bi-temporal change VQA | ⬜ stubbed |
| Optical–SAR cross-modal analysis | ⬜ stubbed |
| Report generation (PDF / GeoJSON) | ⬜ stubbed |
| Remote-sensing domain adaptation | ⬜ not started |

Benchmark scores live in [`docs/evaluation.md`](docs/evaluation.md) and are updated as
runs complete. If a number isn't in that file, we don't claim it.

---

## What it does

Jatayu accepts three input configurations and answers natural-language queries about them:

| Input | Example query |
|---|---|
| **Single image** (optical, multispectral, or SAR) | *"Describe the land cover and major objects visible in this image."* |
| | *"Highlight the water body referred to in the query."* |
| **Bi-temporal pair** (same area, two dates) | *"What changed between these two dates, and where?"* |
| | *"Has the built-up area increased, decreased, or stayed the same?"* |
| **Cross-modal pair** (co-registered optical + SAR) | *"Use the optical and SAR images together to identify built-up and water-covered regions."* |

Every answer comes back with three things: the result, the visual evidence behind it
(bounding box, mask, or change map), and a confidence estimate with a stated method.

### Why not just use a general-purpose VLM?

Because it doesn't work. CLIP-family vision encoders are trained on web photographs —
oblique views, natural lighting, human-scale objects. Feed one a nadir-view 10 m/pixel
Sentinel-2 tile and the embeddings carry almost no signal distinguishing coniferous from
mixed forest, or a paddy field at tillering from one at heading. The language model
downstream then confabulates fluently from noise.

Jatayu instead adapts its vision component to remote-sensing imagery and dispatches to
purpose-built specialist models per task, rather than asking one generic model to pretend
it can do everything.

---

## Architecture

```
  Query + image(s)
         │
         ▼
  ┌─────────────────┐
  │  IO & validate  │  GeoTIFF read, CRS/bounds/band checks, co-registration
  │  jatayu.io      │  verification, band selection, percentile stretch
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │   Controller    │  Stage 1 — deterministic gate: image count, modality,
  │ jatayu.control- │            dates, sensor → narrows to a task family
  │      ler        │  Stage 2 — LLM tool selection within that family,
  └────────┬────────┘            including multi-tool chaining
           ▼
  ┌─────────────────┐
  │ Specialist tool │  VQA · Grounding · Change-VQA · Optical–SAR fusion
  │  jatayu.tools   │  Each: run(ToolRequest) -> ToolResult. Nothing else.
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │ Evidence, conf- │  Overlays, GeoJSON geometries, confidence estimate,
  │ idence, report  │  execution trace, downloadable PDF
  └─────────────────┘
```

Full detail in [`docs/architecture.md`](docs/architecture.md).

### The tool contract

Every specialist model is a single function with a fixed signature:

```python
def run(req: ToolRequest) -> ToolResult: ...
```

No tool knows about the controller, the API, the UI, or any other tool. This is what lets
six people build in parallel — the frontend, the report generator, the evaluation harness,
and the controller were all written against stubbed tools before a single model existed.

The schemas are in [`src/jatayu/schemas.py`](src/jatayu/schemas.py) and documented in
[`docs/tool-contract.md`](docs/tool-contract.md). **Changing them is a team decision, not
a quiet commit.**

---

## Quickstart

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/<org>/jatayu.git
cd jatayu
uv sync

# backend
uv run uvicorn jatayu.api.main:app --reload

# frontend, in a second terminal
uv run streamlit run app/main.py
```

Open http://localhost:8501, upload an image, ask a question.

No GPU is needed to run the app against stubbed tools — useful if you're working on the
UI, the report generator, or the controller.

### Tests

```bash
uv run pytest -q -m "not gpu"   # fast suite, no models required
uv run pytest -q                 # everything, needs a GPU and downloaded weights
uv run ruff check . --fix        # lint
```

---

## Repository layout

```
src/jatayu/
├── schemas.py       # the tool contract — read this first
├── io/              # GeoTIFF loading, validation, preprocessing
├── tools/           # specialist models, one file each
├── controller/      # deterministic gate, LLM router, execution trace
├── confidence/      # per-task confidence estimators
├── report/          # PDF and GeoJSON output
└── api/             # FastAPI

app/                 # Streamlit frontend
training/            # domain adaptation and LoRA fine-tuning scripts
eval/                # benchmark harnesses — RSVQA, VRSBench, CDVQA, routing
docs/                # architecture, tool contract, datasets, evaluation, ADRs
tests/
```

---

## Datasets

| Purpose | Dataset |
|---|---|
| Remote-sensing domain adaptation | BigEarthNet |
| Single-image VQA | RSVQA |
| Captioning / grounding | VRSBench |
| Bi-temporal change VQA | CDVQA |
| Final evaluation (private) | ISRO/SAC — co-registered Cartosat-2S optical + RISAT SAR |

Sources, licences, sizes, and download instructions are in
[`docs/datasets.md`](docs/datasets.md). Datasets and model weights are **never** committed
to this repository — see `.gitignore`.

---

## Design decisions

Recorded as ADRs in [`docs/adr/`](docs/adr/). Two worth surfacing here:

**Optical–SAR fusion is physics-based, not learned.** Water is specular to radar, so σ⁰
collapses to roughly −20 dB; built-up structures produce corner-reflector double-bounce
and spike toward 0 dB. Cross-tabulated against NDWI and NDBI from the optical bands, this
yields a four-class agreement map whose disagreement classes are themselves informative.
There is little precedent for a reliable learned optical–SAR fusion model at this scale;
we would rather ship something physically grounded and explainable than a black box we
can't justify. A learned model remains a stretch goal.

**Confidence is estimated per task, with a named method.** Every `ToolResult` carries a
`confidence_method` field. Closed-set VQA uses softmax over the answer vocabulary; fusion
uses agreement between independent optical and radar signals. We do not report a number
we can't explain the origin of.

---

## Team

| Role | Owns |
|---|---|
| Vision & adaptation engineer | domain adaptation, VQA, grounding |
| Temporal & change engineer | change-VQA, optical–SAR fusion |
| Orchestration & geospatial lead | IO, controller, contracts, API |
| Data & evaluation engineer | dataset pipeline, benchmark harnesses |
| Frontend & docs engineer | Streamlit app, documentation |
| Reporting & demo lead | PDF/GeoJSON reports, demo, pitch |

Contribution workflow, branch naming, and PR rules are in
[`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## Licence

<TBD — MIT or Apache-2.0. Check the licence terms of every base model checkpoint before
choosing, as some carry redistribution restrictions.>
