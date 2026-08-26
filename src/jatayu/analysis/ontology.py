from __future__ import annotations 

import re 
from enum import Enum 


from jatayu.analysis.indices import INDEX_REGISTRY
 
 
class Target(str, Enum):
    """A class of surface phenomenon a user can ask about.
 
    Deliberately small. Every entry must map to at least one registered index or
    to a documented whole-scene task - an unreachable target is worse than a
    missing one, because it looks supported and silently is not.
    """
 
    WATER = "water"
    TURBID_WATER = "turbid_water"
    FLOOD = "flood"
    VEGETATION = "vegetation"
    CROP_HEALTH = "crop_health"
    CANOPY_MOISTURE = "canopy_moisture"
    MANGROVE = "mangrove"
    BUILT_UP = "built_up"
    BARE_SOIL = "bare_soil"
    BURN_SCAR = "burn_scar"
    SOIL_SALINITY = "soil_salinity"
    MINERAL = "mineral"
    LAND_COVER = "land_cover"
 
 
# Ordered best-first. The head of each tuple is what a grounding query gets when
# nothing in the phrasing discriminates further.
TARGET_INDICES: dict[Target, tuple[str, ...]] = {
    Target.WATER: ("MNDWI", "NDWI"),
    Target.TURBID_WATER: ("NDTI", "MNDWI"),
    Target.FLOOD: ("MNDWI", "LSWI", "NDWI"),
    Target.VEGETATION: ("NDVI", "EVI", "SAVI"),
    Target.CROP_HEALTH: ("NDRE", "NDVI", "CHLOROPHYLL_RATIO"),
    Target.CANOPY_MOISTURE: ("NDMI", "LSWI"),
    Target.MANGROVE: ("CMR", "NDVI"),
    Target.BUILT_UP: ("NDBI", "UI"),
    Target.BARE_SOIL: ("BSI",),
    Target.BURN_SCAR: ("NBR", "NBR2", "BAI"),
    Target.SOIL_SALINITY: ("SALINITY_PROXY",),
    Target.MINERAL: ("FERROUS_RATIO", "CLAY_RATIO", "RI"),
    # Whole-scene: no single index answers "describe the land cover". Routed to
    # VQA against RemoteCLIP candidate scoring instead.
    Target.LAND_COVER: (),
}



TARGET_ALIASES: dict[Target, tuple[str, ...]] = {
    Target.WATER: (
        "water", "water body", "waterbody", "lake", "river", "pond", "tank",
        "reservoir", "lagoon", "canal", "wetland", "shoreline",
        "जल", "पानी", "तालाब", "नदी", "झील",
        "জল", "পুকুর", "নদী", "জলাশয়",
        "ଜଳ", "ପୋଖରୀ", "ନଦୀ", "ଜଳାଶୟ",
    ),
    Target.TURBID_WATER: (
        "turbid", "turbidity", "sediment", "silt", "murky", "water clarity",
    ),
    Target.FLOOD: (
        "flood", "flooding", "inundation", "waterlogged", "submerged",
        "बाढ़", "বন্যা", "ବନ୍ୟା",
    ),
    Target.VEGETATION: (
        "vegetation", "greenery", "forest", "tree", "canopy", "grassland",
        "plantation", "जंगल", "वन", "पेड़", "জঙ্গল", "বন", "গাছ",
        "ଜଙ୍ଗଲ", "ବଣ", "ଗଛ",
    ),
    Target.CROP_HEALTH: (
        "crop", "cropland", "paddy", "rice", "farmland", "field", "harvest",
        "yield", "nitrogen", "chlorophyll", "crop health", "crop stress",
        "फसल", "खेत", "धान", "ফসল", "ধান", "ଫସଲ", "ଧାନ",
    ),
    Target.CANOPY_MOISTURE: (
        "moisture", "drought", "water stress", "dry", "dryness",
    ),
    Target.MANGROVE: ("mangrove", "bhitarkanika", "sundarban", "sundarbans"),
    Target.BUILT_UP: (
        "built up", "built-up", "builtup", "urban", "city", "town", "building",
        "settlement", "construction", "impervious", "infrastructure",
        "शहर", "इमारत", "শহর", "বাড়ি", "ସହର", "ଘର",
    ),
    Target.BARE_SOIL: (
        "bare soil", "bare ground", "exposed soil", "fallow", "quarry",
        "mining", "mine", "pit", "excavation",
    ),
    Target.BURN_SCAR: (
        "burn", "burnt", "burned", "burn scar", "fire", "wildfire",
        "forest fire", "आग", "আগুন", "ନିଆଁ",
    ),
    Target.SOIL_SALINITY: ("salinity", "saline", "salt", "salt-affected"),
    Target.MINERAL: (
        "mineral", "iron ore", "iron", "ferrous", "laterite", "lateritic",
        "clay", "alteration", "geology", "geological", "ore",
    ),
    Target.LAND_COVER: (
        "land cover", "landcover", "land use", "landuse", "lulc",
        "describe the scene", "what is in this image",
    ),
}
 
 
def _validate_ontology() -> None:
    """Fail fast if a target points at an index that does not exist."""
    for target, index_names in TARGET_INDICES.items():
        unknown = [name for name in index_names if name not in INDEX_REGISTRY]
        if unknown:
            raise ValueError(
                f"Target {target.value!r} maps to unregistered index/indices "
                f"{unknown}. Available: {sorted(INDEX_REGISTRY)}"
            )
        if not index_names and target is not Target.LAND_COVER:
            raise ValueError(
                f"Target {target.value!r} has no indices and is not the "
                "documented whole-scene exception - it would be unreachable."
            )
    missing_aliases = set(TARGET_INDICES) - set(TARGET_ALIASES)
    if missing_aliases:
        raise ValueError(
            f"Targets with no aliases are unreachable from a query: "
            f"{sorted(t.value for t in missing_aliases)}"
        )
 
 
_validate_ontology()
 
 
# (?<!\w) / (?!\w) rather than \b: works identically for Latin and for Indic
# scripts, and does not misfire on multi-word aliases ending in punctuation.
_ALIAS_PATTERNS: list[tuple[Target, str, re.Pattern[str]]] = [
    (target, alias, re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", re.IGNORECASE))
    for target, aliases in TARGET_ALIASES.items()
    # Longest alias first, so "water body" is preferred over "water" and
    # "built-up area" over "urban" when both appear.
    for alias in sorted(aliases, key=len, reverse=True)
]
 
# First target claiming an index owns it. Used only as a fallback when no alias
# matched but the index router found something.
_INDEX_TO_TARGET: dict[str, Target] = {}
for _target, _names in TARGET_INDICES.items():
    for _name in _names:
        _INDEX_TO_TARGET.setdefault(_name, _target)
 
 
def match_target(text: str) -> tuple[Target | None, tuple[str, ...]]:
    """Find the target a query refers to, plus the alias terms that matched.
 
    Returns the target with the most distinct alias hits, so "turbidity in the
    water body" resolves to TURBID_WATER rather than WATER. Ties break on the
    longest matched alias - a two-word domain phrase is stronger evidence than a
    single common noun.
 
    The matched terms are returned for the execution trace: a user should be able
    to see WHICH word in their query selected the analysis.
    """
    if not text:
        return None, ()
 
    hits: dict[Target, list[str]] = {}
    for target, alias, pattern in _ALIAS_PATTERNS:
        if pattern.search(text):
            hits.setdefault(target, []).append(alias)
 
    if not hits:
        return None, ()
 
    best = max(
        hits.items(),
        key=lambda item: (len(item[1]), max(len(a) for a in item[1])),
    )
    return best[0], tuple(sorted(best[1]))
 
 
def indices_for(target: Target | None) -> tuple[str, ...]:
    """Candidate indices for a target, best first. Empty tuple is legitimate."""
    if target is None:
        return ()
    return TARGET_INDICES[target]
 
 
def target_for_index(index_name: str) -> Target | None:
    """Reverse lookup, for when the index router matched but no alias did."""
    return _INDEX_TO_TARGET.get(index_name)
 
 
__all__ = [
    "TARGET_ALIASES",
    "TARGET_INDICES",
    "Target",
    "indices_for",
    "match_target",
    "target_for_index",
]
