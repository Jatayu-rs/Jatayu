# UI / UX Build Spec

**Owner:** Person F (spec) · **Builds:** Apoorv (L3 controller + L4 GUI)
**Last updated:** 23 Aug 2026 · **Target:** Streamlit, ~4 weeks, one developer part-time

> **This is a build spec, not a wishlist.** Everything in §3 must exist. Everything in §4 is optional. Everything in §5 is explicitly out of scope, with reasons — use that section to push back when someone suggests a feature three days before the deadline.

---

## 1. The one design rule

**No answer appears without the evidence it came from.**

Every other decision in this document follows from that. A sentence on its own is a chatbot; a sentence with the measurement underneath it is an analysis tool. The problem statement grades orchestration transparency, and this is how the UI earns that mark.

Practically: the answer panel is never just text. It is always `answer + evidence + confidence + which model + how long`.

---

## 2. Screen layout

One page. Sidebar for input, main area for output. No routing, no multi-page navigation — it adds bugs and buys nothing in a 5-minute demo.

```
┌─ SIDEBAR ──────────┬─ MAIN ────────────────────────────────────────┐
│                    │                                                │
│  Demo mode  [ON]   │   ┌─ Query bar ──────────────────────────────┐ │
│                    │   │ Ask about this imagery...        [Run]   │ │
│  ── Input ──       │   │ [chip] [chip] [chip] [chip] [chip]       │ │
│  Image A  [upload] │   └──────────────────────────────────────────┘ │
│   modality: optical│                                                │
│   ✓ 512×512, EPSG..│   ┌─ Answer ─────────────────────────────────┐ │
│                    │   │ Vegetation decreased 12.4% in the NE...  │ │
│  Image B  [upload] │   │                                          │ │
│   modality: SAR    │   │ EVIDENCE                                 │ │
│   ✓ co-registered  │   │   veg change    −12.4%                   │ │
│                    │   │   area          3.42 km²                 │ │
│  ── Samples ──     │   │   quadrant      NE                       │ │
│  [Flood: Kerala]   │   │                                          │ │
│  [Urban: Delhi]    │   │ Confidence ████████░░ 0.81               │ │
│  [Crop: Punjab]    │   └──────────────────────────────────────────┘ │
│                    │                                                │
│  ── Registry ──    │   ┌─ Viewer ─────────────────────────────────┐ │
│  6 tools loaded    │   │  [A] [B] [Overlay] [Swipe]               │ │
│                    │   │      (image + change mask overlay)       │ │
│                    │   └──────────────────────────────────────────┘ │
│                    │                                                │
│                    │   ▸ Execution trace   change_vqa · 1.24s      │
└────────────────────┴────────────────────────────────────────────────┘
```

---

## 3. P0 — must build

These eight are the deliverable. If only these exist, the demo works and the mandatory requirements are met.

### 3.1 Input panel with modality tagging

- `st.file_uploader` accepting **`.tif`, `.tiff`, `.png`, `.jpg`** — two slots, **Image A** and **Image B** (B optional).
- Each slot gets a **modality selector**: `Optical / SAR / Auto-detect`. The controller needs this to validate; do not make it guess silently.
- On upload, immediately show: **dimensions, band count, dtype, CRS, whether georeferenced**. This is cheap (`rasterio` opens the header) and it tells the user instantly whether their file is usable.

> ⚠️ **The band-stretch trap.** A raw 16-bit GeoTIFF rendered directly displays as a black rectangle. This *will* happen during the demo with a real Cartosat file. The uploader must apply a **2–98 percentile contrast stretch** for display, and let the user pick which bands map to R/G/B for multi-band files. Budget half a day for this; it is not optional and it is not obvious.

### 3.2 Query bar with preset chips

- Free-text `st.text_input` plus a **Run** button.
- **6–12 clickable preset query chips** below it that fill the box on click.
- Chips are not a convenience feature — they are demo insurance. A judge will type something ambiguous, and chips give you a guaranteed-good path back. They also show off the range of tasks without the user having to imagine queries.

Suggested chip set, one per capability:

| Chip | Routes to |
|---|---|
| "What land cover types are in this image?" | `single_image_vqa` |
| "Describe this scene." | `caption` |
| "Where is the airport runway?" | `ground` |
| "What changed between these two images?" | `change_vqa` |
| "How much vegetation was lost?" | `change_vqa` |
| "Is there flooding hidden under the cloud cover?" | `optical_sar_fusion` |
| "Compare what optical and SAR each see here." | `optical_sar_fusion` |
| "Classify the land use in this scene." | `lulc_classify` |

### 3.3 Answer panel — answer + evidence + confidence

Three stacked blocks, always all three present:

1. **Answer** — the natural-language response.
2. **Evidence** — a key/value table rendered straight from `ToolResult.evidence`. Numbers, units, region labels. If a tool returns an empty evidence dict, that is a **bug in the tool**, not a UI case to handle gracefully.
3. **Confidence** — a bar plus the numeric value from `ToolResult.confidence`.

### 3.4 Image viewer with overlay toggle

- Tabs or radio: **A · B · Overlay**.
- **Overlay** renders `ToolResult.overlay` (change mask, grounding box, or fusion agreement map) composited on the base image at ~50% alpha.
- Overlay tab is disabled — greyed, not hidden — when the tool returned no overlay. Hiding controls makes the app feel inconsistent between queries.

### 3.5 Execution trace panel ⭐

**This is the graded feature. Build it first, not last.**

Two levels:

- **Always visible, one line:** `change_vqa · qwen-2b+cdvqa-lora · 1.24s`
- **Expandable (`st.expander`), full trace:**

```
Task classified      change_vqa           (confidence 0.94)
Input validation     ✓ 2 images
                     ✓ both georeferenced, EPSG:32644
                     ✓ co-registration check passed (offset < 1px)
Tool selected        change_vqa
Model                qwen-2b + cdvqa-lora-v2
Parameters           threshold=otsu, index=ndvi, tile=512
Intermediate         change mask area   3.42 km²
                     veg delta          −12.4%
                     built-up delta     +2.1%
Runtime              1.24s
```

Every line here comes from `ToolResult` fields — `model_id`, `params_used`, `evidence`, `runtime_ms` — plus the controller's own classification and validation output. **No new backend work is required to populate it**, which is why it should be built early rather than treated as polish.

### 3.6 Readable validation failures

Validation runs **before** any model executes, and failures render as a clear message in the answer panel — never a Python traceback.

Required cases, each with its own message:

| Situation | Message |
|---|---|
| Change query, one image | "This question compares two time periods — please upload a second image." |
| Fusion query, both optical | "Cross-modal analysis needs one optical and one SAR image. Both uploads look optical." |
| Image pair not co-registered | "These two images don't align (offset ≈ 47px). Results would be unreliable." |
| No georeferencing | "This file has no CRS. Area measurements will be in pixels, not km²." *(warn, then proceed)* |
| Unreadable file | "Couldn't read this file — is it a valid GeoTIFF?" |

Judges hand you bad inputs on purpose. A clean refusal scores better than a crash, and it demonstrates the "input validation" requirement the PS explicitly lists.

### 3.7 Preloaded sample scenarios

**3–4 buttons in the sidebar that load a complete image pair with zero uploading.**

Empty states are where demos die — a judge opens the app, sees an empty file picker, and 60 seconds evaporate. One click should produce a working result.

Suggested: `Flood (optical+SAR)` · `Urban growth (bi-temporal)` · `Crop cycle (bi-temporal)` · `Single scene`.

### 3.8 Demo mode toggle

A sidebar switch, **default ON**, that serves precomputed cached results for the sample scenarios instead of running live inference.

This is the single most important reliability feature in the app. Free Kaggle sessions die without warning. With demo mode on, the app is a static, instant, guaranteed-working demo; flipping it off proves the live pipeline is real. **Present it openly** — "this is cached so the demo doesn't depend on a free GPU staying alive; here's the same query running live" is a confident answer, not an admission.

---

## 4. P1 — build if time remains after §3 is done and tested

Ordered by value-per-hour. Do not start any of these while a P0 item is incomplete.

| Feature | Why | Effort |
|---|---|---|
| **Before/after swipe slider** | Most compelling single visual in the app for bi-temporal. `streamlit-image-comparison` is a drop-in. | ~2h |
| **Session query history** | Previous queries in the sidebar, clickable to restore. Makes it feel like a tool rather than a form. `st.session_state`. | ~2h |
| **Download result** | Button producing a PNG of the overlay + a JSON of the full trace. Judges like taking something away. | ~2h |
| **Model registry viewer** | Sidebar expander listing the 6 tools, what each does, whether loaded. Makes the orchestration visible even before a query runs. | ~1h |
| **Confidence explanation** | Hover/caption saying *why* confidence is what it is ("mask covers 3% of scene — small changes are less reliable"). | ~2h |
| **Georeferenced map overlay** | `streamlit-folium` with the result on a basemap. Genuinely impressive to ISRO judges. ⚠️ But CRS→EPSG:4326 reprojection is fiddlier than it looks. Only start this with a full spare day. | ~6h |

---

## 5. P2 — do NOT build in these four weeks

Each of these is a reasonable idea and a bad use of the remaining time. Keep the list; it becomes the **"Future work"** slide, which judges reward.

| Feature | Why not now |
|---|---|
| **User accounts / auth** | Zero demo value. Nobody logs into a hackathon demo. |
| **Voice + multilingual (Bhashini)** | Already scoped as future work — see `docs/bhashini.md`. A broken voice demo is worse than none. |
| **AOI drawing tools** | Draw-a-polygon-on-a-map is a multi-day rabbit hole and every mandatory requirement works on whole uploaded images. |
| **Batch / multi-image processing** | The PS asks for single, paired, and bi-temporal. Not batch. |
| **PDF or GeoJSON report export** | The P1 PNG+JSON download covers the same judge instinct at a tenth of the cost. |
| **Side-by-side model comparison** | Interesting research UI, irrelevant to the pitch, and doubles inference cost per query. |
| **Real-time collaboration / sharing** | No. |
| **Chat history across sessions** | Needs a database. The P1 in-session history gets 90% of the feel for none of the infrastructure. |
| **Fancy theming / custom CSS** | Streamlit's default theme is clean. Time here is time not spent on the trace panel. |

---

## 6. Backend contract — what the UI needs from each tool

The UI is a pure function of `ToolResult` (defined in the 4-week plan). Every widget maps to one field:

| UI element | `ToolResult` field | If missing |
|---|---|---|
| Answer text | `answer` | Hard failure — always required |
| Evidence table | `evidence` (dict) | Bug in the tool; do not paper over |
| Confidence bar | `confidence` (float 0–1) | Hide the bar, log a warning |
| Overlay tab | `overlay` (ndarray or None) | Disable the tab, keep it visible |
| Trace: model line | `model_id` | Hard failure — required for the audit trace |
| Trace: parameters | `params_used` (dict) | Show "not reported" |
| Trace: runtime | `runtime_ms` | Show "—" |

Plus, from the controller itself: **task classification + its confidence**, and the **validation results** list.

> **Apoorv:** build the whole UI against fake `ToolResult` objects first. It should be fully working with stub tools by **day 5** of week 1. That is the milestone that de-risks integration for the entire team — if the UI only works once real models exist, three people are blocked on each other for two weeks.

---

## 7. Build order

Each step leaves a working app. Never a half-migrated state.

1. Page skeleton + sidebar/main layout — *empty but renders*
2. `ToolResult` stub generator + answer panel — *fake answers appear*
3. **Execution trace panel** — *the graded feature exists before anything real does*
4. Query bar + preset chips wired to the stub router
5. File uploader + metadata display + **percentile stretch**
6. Image viewer + overlay tab
7. Validation messages
8. Sample scenario buttons
9. Demo mode / caching layer
10. → swap stubs for real tools as they land
11. P1 items, only if §3 is complete and rehearsed

---

## 8. Definition of done

The GUI is finished when a person who has never seen it can, unaided:

- [ ] Click one sample scenario and get an answer with evidence in under 5 seconds
- [ ] Type their own question and get a sensible route or a clear refusal
- [ ] Upload a raw multi-band GeoTIFF and **see it**, not a black square
- [ ] Point at the screen and say which model answered and why
- [ ] Give it a deliberately wrong input and get a readable explanation, not a crash
- [ ] Run the whole thing with the GPU disconnected

**Test this on a non-team member during week 4's dry run.** Not on each other — you all know where to click.
