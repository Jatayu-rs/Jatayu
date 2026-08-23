"""Stage 2: pick a tool within the family. Keyword rules; LLM comes later."""

from jatayu.schemas import TaskFamily, TaskName
from jatayu.tools.registry import tools_for_family

LOCATION_CUES = ("highlight", "where", "locate", "show me the", "point to", "mark")


class RuleRouter:
    def select(self, query: str, family: TaskFamily) -> TaskName:
        candidates = tools_for_family(family)
        if not candidates:
            raise KeyError(f"No tool for family {family.value!r}")
        if len(candidates) == 1:
            return candidates[0]
        wants_location = any(cue in query.lower() for cue in LOCATION_CUES)
        preferred = TaskName.GROUNDING if wants_location else TaskName.VQA
        return preferred if preferred in candidates else candidates[0]
