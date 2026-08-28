

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from jatayu.analysis.indices import IndexAnalyser
from jatayu.analysis.ontology import Target, indices_for, match_target, target_for_index
from jatayu.schemas import TaskFamily, TaskName


class AnswerType(str, Enum):
   

    BOOLEAN = "boolean"
    COUNT = "count"
    CATEGORY = "category"
    LOCATION = "location"
    DESCRIPTION = "description"


class ChangeDirection(str, Enum):
  

    INCREASE = "increase"
    DECREASE = "decrease"
    ANY = "any"


class IntentSource(str, Enum):
  

    ALIAS_ORIGINAL = "alias_original_language"
    ALIAS_ENGLISH = "alias_english"
    INDEX_ROUTER = "index_router"
    LLM = "llm"
    UNRESOLVED = "unresolved"


_YES_NO_OPENERS = frozenset(
    {"is", "are", "was", "were", "has", "have", "had", "does", "do", "did",
     "can", "could", "will", "would", "should", "any"}
)
_COUNT_MARKERS = ("how many", "how much", "number of", "count of")
_LOCATION_MARKERS = (
    "where", "highlight", "locate", "delineate", "mark", "outline", "point out",
    "show me the", "which part", "which area", "which region",
)
_INCREASE_WORDS = frozenset(
    {"increase", "increased", "grow", "grown", "growth", "expand", "expanded",
     "expansion", "gain", "gained", "rise", "risen", "more"}
)
_DECREASE_WORDS = frozenset(
    {"decrease", "decreased", "shrink", "shrunk", "decline", "declined",
     "loss", "lost", "reduce", "reduced", "reduction", "erosion", "less"}
)
_CHANGE_WORDS = frozenset(
    {"change", "changed", "difference", "differ", "between", "since", "before",
     "after", "compare", "compared", "trend"}
)

_WORD_RE = re.compile(r"[a-z]+")


class Intent(BaseModel):
  

    model_config = ConfigDict(frozen=True)

    task: TaskName | None = Field(
        default=None,
        description="Specific tool within the family. None when unresolved.",
    )
    target: Target | None = Field(
        default=None,
        description="What the user is asking about. None is a legitimate outcome "
        "and MUST be handled as 'intent not understood', never guessed past.",
    )
    answer_type: AnswerType = AnswerType.DESCRIPTION
    change_direction: ChangeDirection | None = Field(
        default=None, description="Only meaningful for bi-temporal queries"
    )
    candidate_indices: tuple[str, ...] = Field(
        default=(), description="Best first. Empty for whole-scene tasks."
    )

  
    source: IntentSource = IntentSource.UNRESOLVED
    matched_terms: tuple[str, ...] = Field(
        default=(),
        description="The words that selected the target, so the user can see WHY",
    )
    detected_language: str | None = None

    @property
    def is_understood(self) -> bool:
        """True when there is enough to route on. Check this before dispatching."""
        return self.target is not None and self.task is not None

    def trace_detail(self) -> str:
        """One line for the execution trace, in the voice the UI already uses."""
        if not self.is_understood:
            return "Could not determine what the query refers to."
        terms = ", ".join(self.matched_terms) or "phrasing"
        index = self.candidate_indices[0] if self.candidate_indices else "no index"
        return (
            f"Read as a {self.answer_type.value} question about "
            f"{self.target.value.replace('_', ' ')} (from {terms}); "
            f"planning to use {index}."
        )


def _classify_answer_type(query: str, words: set[str]) -> AnswerType:
    """Infer the expected answer shape from interrogative phrasing."""
    lowered = query.lower()

    if any(marker in lowered for marker in _COUNT_MARKERS):
        return AnswerType.COUNT

    if words & _INCREASE_WORDS and words & _DECREASE_WORDS:
        return AnswerType.CATEGORY
    if any(marker in lowered for marker in _LOCATION_MARKERS):
        return AnswerType.LOCATION
    first = next(iter(_WORD_RE.findall(lowered)), "")
    if first in _YES_NO_OPENERS:
        return AnswerType.BOOLEAN
    return AnswerType.DESCRIPTION


def _classify_direction(words: set[str]) -> ChangeDirection | None:
    """Which way the user expects the change to run, if they said."""
    up = bool(words & _INCREASE_WORDS)
    down = bool(words & _DECREASE_WORDS)
    if up and down:
        return ChangeDirection.ANY
    if up:
        return ChangeDirection.INCREASE
    if down:
        return ChangeDirection.DECREASE
    if words & _CHANGE_WORDS:
        return ChangeDirection.ANY
    return None


def _select_task(family: TaskFamily, answer_type: AnswerType, target=None) -> TaskName:
    if family is TaskFamily.CROSS_MODAL:
        return TaskName.FUSION
    if family is TaskFamily.BI_TEMPORAL:
        return TaskName.CHANGE_VQA
    if target and target.value == "crop_health":
        return TaskName.CROP_STRESS
    if answer_type is AnswerType.LOCATION:
        return TaskName.GROUNDING
    return TaskName.VQA


def parse_intent(
    query: str,
    family: TaskFamily,
    *,
    original_query: str | None = None,
    detected_language: str | None = None,
    analyser: IndexAnalyser | None = None,
) -> Intent:
 
    words = set(_WORD_RE.findall(query.lower()))
    answer_type = _classify_answer_type(query, words)
    direction = (
        _classify_direction(words) if family is TaskFamily.BI_TEMPORAL else None
    )


    target, terms = (None, ())
    source = IntentSource.UNRESOLVED
    if original_query:
        target, terms = match_target(original_query)
        if target is not None:
            source = IntentSource.ALIAS_ORIGINAL


    if target is None:
        target, terms = match_target(query)
        if target is not None:
            source = IntentSource.ALIAS_ENGLISH


    candidates: tuple[str, ...] = ()
    if target is None:
        ranked = (analyser or IndexAnalyser()).select_indices(query)
        if ranked:
            target = target_for_index(ranked[0])
            if target is not None:
                candidates = tuple(ranked[:3])
                terms = ()
                source = IntentSource.INDEX_ROUTER

    if target is None:
     
        return Intent(
            answer_type=answer_type,
            change_direction=direction,
            source=IntentSource.UNRESOLVED,
            detected_language=detected_language,
        )

    if not candidates:
        candidates = indices_for(target)

    return Intent(
        task=_select_task(family, answer_type,target),
        target=target,
        answer_type=answer_type,
        change_direction=direction,
        candidate_indices=candidates,
        source=source,
        matched_terms=terms,
        detected_language=detected_language,
    )


__all__ = [
    "AnswerType",
    "ChangeDirection",
    "Intent",
    "IntentSource",
    "parse_intent",
]
