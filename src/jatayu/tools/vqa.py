"""Single-image question answering via RemoteCLIP candidate scoring.

Not a generative model. We embed the image once, embed each candidate answer
phrased as a full caption, and pick the best match by cosine similarity. The
softmax over candidates is a genuine calibrated confidence — unlike a generative
model's token log-probs, which measure fluency, not correctness.

Honest limitations, stated here and surfaced in the demo:
- Good at closed-set (yes/no, category). Poor at open-ended captioning.
- Poor at counting. COUNT questions abstain rather than guessing.
- Sensitive to phrasing — candidates written as captions score better than bare
  labels. The candidate bank below uses the caption style deliberately.
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from jatayu.schemas import (
    Evidence, ImageRef, TaskFamily, TaskName, ToolRequest, ToolResult,
)
from jatayu.tools.registry import register

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model singleton — load once, not per request
# ---------------------------------------------------------------------------

_MODEL = None
_PREPROCESS = None
_TOKENIZER = None
_DEVICE = None
_MODEL_ID = "chendelong/RemoteCLIP (ViT-B-32)"


def _ensure_model():
    global _MODEL, _PREPROCESS, _TOKENIZER, _DEVICE

    if _MODEL is not None:
        return

    import open_clip
    from huggingface_hub import hf_hub_download

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = hf_hub_download(
        "chendelong/RemoteCLIP",
        filename="RemoteCLIP-ViT-B-32.pt",
        local_files_only=True,
    )
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=False)
    model.eval().to(_DEVICE)

    _MODEL = model
    _PREPROCESS = preprocess
    _TOKENIZER = open_clip.get_tokenizer("ViT-B-32")
    logger.info("RemoteCLIP loaded on %s", _DEVICE)


# ---------------------------------------------------------------------------
# Candidate banks — captions, not bare labels
# ---------------------------------------------------------------------------

_BOOLEAN_TEMPLATES = {
    "water": (
        "a satellite image containing a river, lake, or water body",
        "a satellite image with no visible water",
    ),
    "vegetation": (
        "a satellite image with green vegetation and healthy plant cover",
        "a satellite image with no significant vegetation",
    ),
    "built_up": (
        "a satellite image showing buildings, roads, and urban development",
        "a satellite image with no built-up structures or urban areas",
    ),
    "crop_health": (
        "a satellite image of healthy, actively growing crops",
        "a satellite image of stressed, sparse, or unhealthy crops",
    ),
    "burn_scar": (
        "a satellite image showing burned or fire-damaged land",
        "a satellite image with no burn scars or fire damage",
    ),
    "bare_soil": (
        "a satellite image of exposed bare soil or ground",
        "a satellite image with vegetation or water covering the ground",
    ),
    "flood": (
        "a satellite image showing flooding or inundated areas",
        "a satellite image with no visible flooding",
    ),
    "mangrove": (
        "a satellite image containing mangrove forest along the coast",
        "a satellite image with no mangrove vegetation",
    ),
}

_LAND_COVER_BANK = [
    "a satellite image dominated by open water such as rivers, lakes, or ocean",
    "a satellite image of dense green vegetation and forest canopy",
    "a satellite image of agricultural fields and cropland",
    "a satellite image of urban areas with buildings and roads",
    "a satellite image of bare soil, desert, or exposed ground",
    "a satellite image of wetland or marshy areas with mixed water and vegetation",
    "a satellite image of coastal areas with sand and shoreline",
    "a satellite image of mountainous or hilly terrain",
    "a satellite image of sparse grassland or scrubland",
    "a satellite image showing industrial areas or infrastructure",
]

_CATEGORY_TEMPLATES = {
    "land_cover": _LAND_COVER_BANK,
    "built_up": [
        "a satellite image of dense urban area with high-rise buildings",
        "a satellite image of suburban residential area",
        "a satellite image of industrial or commercial zone",
        "a satellite image of rural area with no significant construction",
    ],
    "vegetation": [
        "a satellite image of dense forest",
        "a satellite image of agricultural crops",
        "a satellite image of grassland or sparse vegetation",
        "a satellite image of bare or barren land with no vegetation",
    ],
}

# Minimum margin between top two candidates to report a confident answer.
# Below this, the model cannot discriminate and should abstain.
_MIN_MARGIN = 0.08


# ---------------------------------------------------------------------------
# Image loading for CLIP
# ---------------------------------------------------------------------------

def _load_image_for_clip(img: ImageRef):
    """Load a GeoTIFF as a PIL RGB image suitable for CLIP preprocessing.

    Band descriptions are used to find red, green, blue in the correct order.
    Multispectral GeoTIFFs often store bands as (blue, green, red, nir, swir1)
    and reading [1,2,3] naively gives BGR, which makes CLIP see water as
    reddish and vegetation as blueish — producing wrong answers.
    """
    from PIL import Image
    import rasterio

    with rasterio.open(img.path) as src:
        decimate = max(1, max(src.width, src.height) // 1024)
        out_h = src.height // decimate
        out_w = src.width // decimate

        if src.count >= 3:
            # Use band descriptions to find RGB in the right order
            descs = [d.lower() if d else "" for d in src.descriptions]
            try:
                r_idx = descs.index("red") + 1      # rasterio is 1-indexed
                g_idx = descs.index("green") + 1
                b_idx = descs.index("blue") + 1
            except ValueError:
                # No descriptions — assume common ordering (B, G, R, ...)
                r_idx, g_idx, b_idx = 3, 2, 1
            data = src.read(
                [r_idx, g_idx, b_idx],
                out_shape=(3, out_h, out_w),
            )
        else:
            band = src.read(1, out_shape=(1, out_h, out_w))
            data = np.stack([band[0]] * 3)

    # Percentile stretch to uint8
    rgb = np.moveaxis(data, 0, -1).astype(np.float64)
    for c in range(3):
        channel = rgb[:, :, c]
        valid = channel[np.isfinite(channel)]
        if valid.size == 0:
            continue
        lo = np.percentile(valid, 2)
        hi = np.percentile(valid, 98)
        if hi > lo:
            channel = np.clip((channel - lo) / (hi - lo) * 255, 0, 255)
        else:
            channel = np.zeros_like(channel)
        rgb[:, :, c] = channel

    # Clean NaN before casting to uint8
    rgb = np.nan_to_num(rgb, nan=0.0)
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def _score_candidates(
    image: ImageRef, candidates: list[str]
) -> tuple[list[float], list[str]]:
    """Score candidate captions against an image. Returns (probs, candidates)."""
    _ensure_model()

    pil_image = _load_image_for_clip(image)
    image_tensor = _PREPROCESS(pil_image).unsqueeze(0).to(_DEVICE)

    tokens = _TOKENIZER(candidates).to(_DEVICE)

    with torch.no_grad():
        image_features = _MODEL.encode_image(image_tensor)
        text_features = _MODEL.encode_text(tokens)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Cosine similarity, scaled to logits
        logits = (100.0 * image_features @ text_features.T).squeeze(0)
        probs = torch.softmax(logits, dim=0).cpu().tolist()

    return probs, candidates


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@register(
    TaskName.VQA,
    families={TaskFamily.SINGLE_IMAGE},
    description="Answers a factual question about one image: presence, count, "
                "extent, land-cover type. Use when the user asks what is in an image.",
)
def run(req: ToolRequest) -> ToolResult:
    t0 = perf_counter()
    notes: list[str] = []
    answer_type = req.params.get("answer_type", "description")
    target = req.params.get("target", "land_cover")

    notes.append(f"answer_type={answer_type}")
    notes.append(f"target={target}")

    # --- model load guard ----------------------------------------------------
    try:
        _ensure_model()
    except Exception as exc:
        return ToolResult(
            answer=f"RemoteCLIP is not available: {exc}",
            confidence=0.0, confidence_method="not_attempted",
            tool_name=TaskName.VQA, model_id=_MODEL_ID,
            abstained=True, notes=[f"Model load failed: {exc}"],
        )

    image = req.images[0]

    # --- COUNT: abstain honestly ---------------------------------------------
    if answer_type == "count":
        return ToolResult(
            answer=(
                "I cannot reliably count objects in satellite imagery. "
                "RemoteCLIP is a contrastive model that compares captions, "
                "not a detection model that locates individual objects."
            ),
            confidence=0.0,
            confidence_method="answer_softmax",
            tool_name=TaskName.VQA, model_id=_MODEL_ID,
            abstained=True,
            notes=["COUNT questions abstain: RemoteCLIP cannot count."],
        )

    # --- build candidates based on answer type --------------------------------
    if answer_type == "boolean":
        templates = _BOOLEAN_TEMPLATES.get(target, _BOOLEAN_TEMPLATES.get("water"))
        candidates = list(templates)
    elif answer_type == "category":
        candidates = list(_CATEGORY_TEMPLATES.get(target, _LAND_COVER_BANK))
    else:
        # description / location / land_cover — score the full land-cover bank
        candidates = list(_LAND_COVER_BANK)

    notes.append(f"candidates={len(candidates)}")

    # --- score ---------------------------------------------------------------
    try:
        probs, candidates = _score_candidates(image, candidates)
    except Exception as exc:
        return ToolResult(
            answer=f"Could not analyse this image: {exc}",
            confidence=0.0, confidence_method="not_attempted",
            tool_name=TaskName.VQA, model_id=_MODEL_ID,
            abstained=True, notes=[str(exc)],
        )

    # --- interpret results ---------------------------------------------------
    ranked = sorted(zip(probs, candidates), reverse=True)
    top_prob, top_caption = ranked[0]
    second_prob = ranked[1][0] if len(ranked) > 1 else 0.0
    margin = top_prob - second_prob

    notes.append(f"top_score={top_prob:.3f}")
    notes.append(f"margin={margin:.3f}")

    # Abstain when near chance
    if margin < _MIN_MARGIN and answer_type in ("boolean","category"):
        answer = (
            f"I cannot determine this confidently. The top two interpretations "
            f"scored {top_prob:.1%} and {second_prob:.1%} — too close to call. "
            f"A higher-resolution image or a more specific question might help."
        )
        return ToolResult(
            answer=answer,
            confidence=round(top_prob, 3),
            confidence_method="answer_softmax",
            tool_name=TaskName.VQA, model_id=_MODEL_ID,
            latency_ms=int((perf_counter() - t0) * 1000),
            abstained=True,
            notes=notes + ["Margin below threshold — abstaining."],
        )

    # --- build the answer ----------------------------------------------------
    if answer_type == "boolean":
        # First candidate is the "yes" phrasing
        is_yes = (ranked[0][1] == candidates[0])
        if is_yes:
            answer = f"Yes. The image shows evidence of {target.replace('_', ' ')} ({top_prob:.0%} confidence)."
        else:
            answer = f"No. The image does not appear to contain {target.replace('_', ' ')} ({top_prob:.0%} confidence)."
    elif answer_type == "category":
        # Strip the "a satellite image of/showing" prefix for readability
        description = top_caption
        for prefix in ("a satellite image of ", "a satellite image showing ",
                        "a satellite image dominated by ", "a satellite image with "):
            if description.lower().startswith(prefix):
                description = description[len(prefix):]
                break
        answer = f"This scene most closely matches: {description} ({top_prob:.0%} confidence)."
    else:
        # Description: assemble from top scoring captions
        above_threshold = [(p, c) for p, c in ranked if p > 0.05]
        descriptions = []
        for p, c in above_threshold[:3]:
            for prefix in ("a satellite image of ", "a satellite image showing ",
                            "a satellite image dominated by ", "a satellite image with ",
                            "a satellite image containing "):
                if c.lower().startswith(prefix):
                    c = c[len(prefix):]
                    break
            descriptions.append(f"{c} ({p:.0%})")
        answer = "This image appears to contain: " + "; ".join(descriptions) + "."

    elapsed = int((perf_counter() - t0) * 1000)

    return ToolResult(
        answer=answer,
        evidence=Evidence(kind="none", caption="Candidate scoring — no spatial evidence."),
        confidence=round(top_prob, 3),
        confidence_method="answer_softmax",
        tool_name=TaskName.VQA,
        model_id=_MODEL_ID,
        params_used={"answer_type": answer_type, "target": target,
                     "n_candidates": len(candidates)},
        latency_ms=elapsed,
        abstained=False,
        notes=notes,
    )
