"""Stage 2: pick a tool within the family.

Two implementations:
- RuleRouter: keyword heuristics, deterministic, always works offline.
- LLMRouter: uses Qwen2.5-0.5B-Instruct to select the tool from the registry
  descriptions. Falls back to RuleRouter on any failure — an expired download,
  a CUDA OOM, or a parse error must degrade the demo, never break it.

Set JATAYU_ROUTER_OFFLINE=1 to force RuleRouter regardless.
"""

from __future__ import annotations

import logging
import os

from jatayu.schemas import TaskFamily, TaskName
from jatayu.tools.registry import tools_for_family, DESCRIPTIONS

logger = logging.getLogger(__name__)

LOCATION_CUES = ("highlight", "where", "locate", "show me the", "point to", "mark",
                 "delineate", "outline")


class RuleRouter:
    """Keyword heuristics. Always available, no model, deterministic."""

    def select(self, query: str, family: TaskFamily) -> TaskName:
        candidates = tools_for_family(family)
        if not candidates:
            raise KeyError(f"No tool for family {family.value!r}")
        if len(candidates) == 1:
            return candidates[0]
        wants_location = any(cue in query.lower() for cue in LOCATION_CUES)
        preferred = TaskName.GROUNDING if wants_location else TaskName.VQA
        return preferred if preferred in candidates else candidates[0]


# ---------------------------------------------------------------------------
# LLM Router
# ---------------------------------------------------------------------------

_LLM_MODEL = None
_LLM_TOKENIZER = None


def _ensure_llm():
    global _LLM_MODEL, _LLM_TOKENIZER

    if _LLM_MODEL is not None:
        return

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    _LLM_TOKENIZER = AutoTokenizer.from_pretrained(
        model_id, local_files_only=True, trust_remote_code=True,
    )
    _LLM_MODEL = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    _LLM_MODEL.eval()
    logger.info("LLM router loaded: %s", model_id)


def _build_prompt(query: str, family: TaskFamily) -> str:
    """Build a few-shot prompt listing the available tools for this family."""
    candidates = tools_for_family(family)
    tool_lines = []
    for name in candidates:
        desc = DESCRIPTIONS.get(name, "No description available.")
        tool_lines.append(f"- {name.value}: {desc}")

    tools_text = "\n".join(tool_lines)

    return f"""You are Jatayu's tool selector. Given a user query about satellite imagery and a list of available tools, pick the single best tool.

Available tools for this {family.value} task:
{tools_text}

Rules:
- Reply with ONLY the tool name, nothing else.
- If the user wants to locate, highlight, or find a specific region → grounding
- If the user asks a question about what is in the image → vqa
- If the user asks about change over time → change_vqa
- If the user wants to combine optical and SAR → fusion

Query: "{query}"

Best tool:"""


class LLMRouter:
    """Uses Qwen2.5-0.5B-Instruct to select the tool. Falls back to RuleRouter."""

    def __init__(self):
        self._fallback = RuleRouter()

    def select(self, query: str, family: TaskFamily) -> TaskName:
        candidates = tools_for_family(family)
        if not candidates:
            raise KeyError(f"No tool for family {family.value!r}")
        if len(candidates) == 1:
            return candidates[0]

        try:
            return self._llm_select(query, family, candidates)
        except Exception as exc:
            logger.warning("LLM router failed (%s), falling back to rules.", exc)
            return self._fallback.select(query, family)

    def _llm_select(
        self, query: str, family: TaskFamily, candidates: list[TaskName]
    ) -> TaskName:
        import torch

        _ensure_llm()

        prompt = _build_prompt(query, family)

        messages = [{"role": "user", "content": prompt}]
        text = _LLM_TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        inputs = _LLM_TOKENIZER(text, return_tensors="pt").to(_LLM_MODEL.device)

        with torch.no_grad():
            output = _LLM_MODEL.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                temperature=1.0,
            )

        # Decode only the generated tokens
        generated = output[0][inputs["input_ids"].shape[1]:]
        answer = _LLM_TOKENIZER.decode(generated, skip_special_tokens=True).strip().lower()

        # Parse the answer — match against candidate names
        for candidate in candidates:
            if candidate.value in answer:
                logger.info("LLM router selected: %s (raw: %r)", candidate.value, answer)
                return candidate

        # If the LLM produced something unexpected, fall back
        logger.warning("LLM router returned unparseable: %r, falling back.", answer)
        return self._fallback.select(query, family)


def default_router():
    """Return the best available router based on environment."""
    if os.environ.get("JATAYU_ROUTER_OFFLINE", "").strip() in ("1", "true", "yes"):
        logger.info("JATAYU_ROUTER_OFFLINE set — using RuleRouter.")
        return RuleRouter()

    try:
        _ensure_llm()
        logger.info("LLM router available — using LLMRouter.")
        return LLMRouter()
    except Exception as exc:
        logger.warning("Cannot load LLM router (%s) — using RuleRouter.", exc)
        return RuleRouter()
