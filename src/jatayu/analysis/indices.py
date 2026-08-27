
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


# ============================================================================
# TYPES
# ============================================================================

FloatArray = NDArray[np.float64]
Polarity = Literal["positive", "negative"]
Computation = Callable[[dict[str, FloatArray]], FloatArray]


# ============================================================================
# CANONICAL BANDS
# ============================================================================

CANONICAL_BANDS = frozenset(
    {
        "blue",
        "green",
        "red",
        "rededge1",
        "rededge2",
        "rededge3",
        "nir",
        "swir1",
        "swir2",
    }
)

REFLECTANCE_MAX = 1.5


# ============================================================================
# QUERY TOKENISATION
# ============================================================================

_TOKEN_RE = re.compile(r"[a-z0-9]+")


_RAW_STOPWORDS = frozenset(
    {
        # English function words
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
        "what",
        "where",
        "which",
        "how",
        "can",
        "you",
        "your",
        "please",
        "any",
        "all",
        "such",
        "between",
        "their",
        "there",
        "whether",

        # Task verbs
        "highlight",
        "show",
        "identify",
        "find",
        "detect",
        "locate",
        "mark",
        "give",
        "tell",
        "measure",
        "assess",
        "monitor",
        "screen",
        "check",
        "compare",
        "estimate",
        "map",
        "support",
        "separate",
        "refine",
        "distinguish",
        "investigate",
        "inspect",
        "delineate",
        "calculate",
        "analyse",
        "analyze",
        "determine",
        "look",

        # Generic imagery words
        "region",
        "regions",
        "area",
        "areas",
        "image",
        "images",
        "imagery",
        "scene",
        "picture",
        "query",
        "output",
        "result",

        # Generic index vocabulary
        "index",
        "normalized",
        "normalised",
        "difference",
        "band",
        "ratio",
        "value",
        "used",
        "use",
        "useful",
        "likely",
        "more",
        "less",
        "other",
        "extent",
        "status",
        "condition",
    }
)


def _singular(token: str) -> str:
    """Simple deterministic plural folding."""
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"

    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]

    return token


STOPWORDS = frozenset(
    _singular(word) for word in _RAW_STOPWORDS
)


def _token_sequence(text: str) -> list[str]:
    return [
        _singular(token)
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) > 2
    ]


def _content_tokens(tokens: list[str]) -> set[str]:
    return {
        token
        for token in tokens
        if token not in STOPWORDS
    }


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:]))


# ============================================================================
# NUMERICAL HELPERS
# ============================================================================

def _normalised_difference(
    a: FloatArray,
    b: FloatArray,
) -> FloatArray:
    """
    (a - b) / (a + b)

    Zero denominator -> 0.
    NaNs remain NaN.
    """

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    out = np.zeros(
        np.broadcast_shapes(a.shape, b.shape),
        dtype=np.float64,
    )

    numerator = a - b
    denominator = a + b

    np.divide(
        numerator,
        denominator,
        out=out,
        where=denominator != 0,
    )

    invalid = np.isnan(numerator) | np.isnan(denominator)
    out[invalid] = np.nan

    return out


def _safe_ratio(
    numerator: FloatArray,
    denominator: FloatArray,
) -> FloatArray:

    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)

    out = np.zeros(
        np.broadcast_shapes(
            numerator.shape,
            denominator.shape,
        ),
        dtype=np.float64,
    )

    np.divide(
        numerator,
        denominator,
        out=out,
        where=denominator != 0,
    )

    invalid = np.isnan(numerator) | np.isnan(denominator)
    out[invalid] = np.nan

    return out


# ============================================================================
# SPECTRAL INDEX FORMULAE
# ============================================================================

def _evi(b: dict[str, FloatArray]) -> FloatArray:
    denominator = (
        b["nir"]
        + 6.0 * b["red"]
        - 7.5 * b["blue"]
        + 1.0
    )

    out = np.zeros(
        np.broadcast_shapes(
            b["nir"].shape,
            b["red"].shape,
            b["blue"].shape,
        ),
        dtype=np.float64,
    )

    np.divide(
        2.5 * (b["nir"] - b["red"]),
        denominator,
        out=out,
        where=denominator != 0,
    )

    invalid = (
        np.isnan(b["nir"])
        | np.isnan(b["red"])
        | np.isnan(b["blue"])
        | np.isnan(denominator)
    )

    out[invalid] = np.nan

    return out


def _savi(b: dict[str, FloatArray]) -> FloatArray:

    denominator = (
        b["nir"]
        + b["red"]
        + 0.5
    )

    out = np.zeros(
        np.broadcast_shapes(
            b["nir"].shape,
            b["red"].shape,
        ),
        dtype=np.float64,
    )

    np.divide(
        1.5 * (b["nir"] - b["red"]),
        denominator,
        out=out,
        where=denominator != 0,
    )

    invalid = (
        np.isnan(b["nir"])
        | np.isnan(b["red"])
        | np.isnan(denominator)
    )

    out[invalid] = np.nan

    return out


def _bai(b: dict[str, FloatArray]) -> FloatArray:

    denominator = (
        (b["red"] - 0.10) ** 2
        + (b["nir"] - 0.06) ** 2
    )

    denominator = np.maximum(
        denominator,
        1e-12,
    )

    out = 1.0 / denominator

    invalid = (
        np.isnan(b["red"])
        | np.isnan(b["nir"])
    )

    return np.where(
        invalid,
        np.nan,
        out,
    )


# ============================================================================
# INDEX DEFINITION
# ============================================================================

@dataclass(frozen=True, slots=True)
class IndexDefinition:
    required_bands: tuple[str, ...]
    compute: Computation

    positive_means: str
    negative_means: str

    physical_floor: float
    valid_range: tuple[float, float]

    use_cases: tuple[str, ...]

    polarity: Polarity = "positive"

    requires_reflectance: bool = False

    priority: int = 50


# ============================================================================
# INDEX REGISTRY
# ============================================================================

INDEX_REGISTRY: dict[str, IndexDefinition] = {

    # ------------------------------------------------------------------------
    # WATER
    # ------------------------------------------------------------------------

    "MNDWI": IndexDefinition(
        required_bands=("green", "swir1"),

        compute=lambda b: _normalised_difference(
            b["green"],
            b["swir1"],
        ),

        positive_means="open water and inundation are more likely",
        negative_means="land, vegetation and built-up surfaces are more likely",

        physical_floor=0.0,
        valid_range=(-1.0, 1.0),

        use_cases=(
            "map open water",
            "map standing water",
            "map flooded areas",
            "delineate water bodies",
            "find ponds lakes rivers lagoons",
            "detect urban flooding",
            "detect post rainfall flooding",
            "identify inundation",
            "map coastal water",
            "map waterlogging",
        ),

        priority=5,
    ),

    "NDWI": IndexDefinition(
        required_bands=("green", "nir"),

        compute=lambda b: _normalised_difference(
            b["green"],
            b["nir"],
        ),

        positive_means="open surface water is more likely",
        negative_means="vegetation and dry land are more likely",

        physical_floor=0.0,
        valid_range=(-1.0, 1.0),

        use_cases=(
            "detect open surface water",
            "delineate rivers lakes ponds wetlands",
            "estimate water extent",
        ),

        priority=80,
    ),

    # ------------------------------------------------------------------------
    # VEGETATION
    # ------------------------------------------------------------------------

    "NDVI": IndexDefinition(
        required_bands=("red", "nir"),

        compute=lambda b: _normalised_difference(
            b["nir"],
            b["red"],
        ),

        positive_means="green healthy vegetation is more likely",
        negative_means="bare soil water senescent vegetation or sparse cover",

        physical_floor=0.10,
        valid_range=(-1.0, 1.0),

        use_cases=(
            "measure vegetation greenness",
            "measure crop health",
            "check crop growth",
            "assess paddy vigour",
            "detect crop stress",
            "monitor agricultural fields",
            "monitor forest condition",
            "identify healthy vegetation",
        ),

        priority=20,
    ),

    "EVI": IndexDefinition(
        required_bands=("blue", "red", "nir"),

        compute=_evi,

        positive_means="dense healthy vegetation is more likely",
        negative_means="sparse vegetation or non vegetation",

        physical_floor=0.10,
        valid_range=(-1.0, 1.0),

        use_cases=(
            "assess dense crop vegetation",
            "measure vegetation health",
            "monitor forest canopy",
            "monitor agricultural vigour",
        ),
    ),

    "SAVI": IndexDefinition(
        required_bands=("red", "nir"),

        compute=_savi,

        positive_means="vegetation is more likely",
        negative_means="bare soil and sparse vegetation",

        physical_floor=0.10,
        valid_range=(-1.0, 1.0),

        use_cases=(
            "measure vegetation with exposed soil",
            "assess sparse crops",
            "monitor dryland agriculture",
            "detect vegetation over soil",
        ),

        priority=25,
    ),

    "NDRE": IndexDefinition(
        required_bands=("rededge1", "nir"),

        compute=lambda b: _normalised_difference(
            b["nir"],
            b["rededge1"],
        ),

        positive_means="higher chlorophyll and crop vigour",
        negative_means="lower chlorophyll and vegetation stress",

        physical_floor=0.05,
        valid_range=(-1.0, 1.0),

        use_cases=(
            "detect crop nitrogen stress",
            "detect crop chlorophyll stress",
            "check wheat stress",
            "check paddy stress",
            "monitor crop health",
            "detect subtle crop stress",
        ),

        priority=15,
    ),

    "CHLOROPHYLL_RATIO": IndexDefinition(
        required_bands=("rededge1", "nir"),

        compute=lambda b: (
            _safe_ratio(
                b["nir"],
                b["rededge1"],
            )
            - 1.0
        ),

        positive_means="higher chlorophyll vegetation is more likely",
        negative_means="lower chlorophyll or non vegetation",

        physical_floor=0.05,
        valid_range=(-1.0, float("inf")),

        use_cases=(
            "measure crop chlorophyll",
            "estimate crop nitrogen status",
            "assess crop nutrient stress",
            "screen vegetation chlorophyll",
        ),

        priority=40,
    ),

    # ------------------------------------------------------------------------
    # VEGETATION / MOISTURE
    # ------------------------------------------------------------------------

    "NDMI": IndexDefinition(
        required_bands=("nir", "swir1"),

        compute=lambda b: _normalised_difference(
            b["nir"],
            b["swir1"],
        ),

        positive_means="higher vegetation moisture",
        negative_means="vegetation water stress and drying",

        physical_floor=0.10,
        valid_range=(-1.0, 1.0),

        use_cases=(
            "measure vegetation moisture",
            "measure crop moisture",
            "detect drought stress",
            "detect vegetation water stress",
            "monitor wheat moisture",
            "monitor paddy moisture",
            "monitor canopy moisture",
        ),
    ),

    "LSWI": IndexDefinition(
        required_bands=("nir", "swir1"),

        compute=lambda b: _normalised_difference(
            b["nir"],
            b["swir1"],
        ),

        positive_means="surface and vegetation moisture",
        negative_means="dry vegetation and dry soil",

        physical_floor=0.0,
        valid_range=(-1.0, 1.0),

        use_cases=(
            "measure crop water status",
            "measure surface moisture",
            "detect flooded agriculture",
            "detect waterlogged agriculture",
            "monitor paddy water",
            "monitor wetland moisture",
        ),
    ),

    # ------------------------------------------------------------------------
    # URBAN
    # ------------------------------------------------------------------------

    "NDBI": IndexDefinition(
        required_bands=("nir", "swir1"),

        compute=lambda b: _normalised_difference(
            b["swir1"],
            b["nir"],
        ),

        positive_means="built-up urban surfaces are more likely",
        negative_means="vegetated and non built surfaces",

        physical_floor=0.0,
        valid_range=(-1.0, 1.0),

        use_cases=(
            "map built up areas",
            "map urban areas",
            "detect urban expansion",
            "find new construction",
            "detect settlements",
            "detect built up land",
        ),

        priority=10,
    ),

    "UI": IndexDefinition(
        required_bands=("nir", "swir2"),

        compute=lambda b: _normalised_difference(
            b["swir2"],
            b["nir"],
        ),

        positive_means="dense urban built-up surfaces",
        negative_means="vegetated and non urban surfaces",

        physical_floor=0.0,
        valid_range=(-1.0, 1.0),

        use_cases=(
            "detect dense construction",
            "map impervious surfaces",
            "map urban built up land",
            "separate buildings from vegetation",
        ),

        priority=60,
    ),

    # ------------------------------------------------------------------------
    # BARE SOIL / AGRICULTURE
    # ------------------------------------------------------------------------

    "BSI": IndexDefinition(
        required_bands=(
            "blue",
            "red",
            "nir",
            "swir1",
        ),

        compute=lambda b: _normalised_difference(
            b["swir1"] + b["red"],
            b["nir"] + b["blue"],
        ),

        positive_means="bare exposed soil is more likely",
        negative_means="vegetation or water is more likely",

        physical_floor=0.0,
        valid_range=(-1.0, 1.0),

        use_cases=(
            "map bare soil",
            "map exposed soil",
            "detect fallow agricultural land",
            "detect fallow fields",
            "identify fields left fallow",
            "identify agricultural land",
            "detect exposed ground",
            "separate soil from vegetation",
        ),

        priority=20,
    ),

    # ------------------------------------------------------------------------
    # SALINITY
    # ------------------------------------------------------------------------

    "SALINITY_PROXY": IndexDefinition(
        required_bands=("red", "nir"),

        compute=lambda b: _normalised_difference(
            b["red"],
            b["nir"],
        ),

        positive_means="higher saline soil spectral response",
        negative_means="lower salinity spectral response",

        physical_floor=0.0,
        valid_range=(-1.0, 1.0),

        use_cases=(
            "map soil salinity",
            "detect saline soil",
            "detect salt affected agricultural land",
            "assess coastal salinity",
            "monitor salinity after cyclone",
            "detect salt stress in paddy",
            "screen coastal agricultural salinity",
        ),

        priority=12,
    ),

    # ------------------------------------------------------------------------
    # MANGROVE
    # ------------------------------------------------------------------------

    "CMR": IndexDefinition(
        required_bands=("green", "red", "nir"),

        compute=lambda b: (
            _normalised_difference(
                b["nir"],
                b["red"],
            )
            - _normalised_difference(
                b["green"],
                b["nir"],
            )
        ),

        positive_means="mangrove vegetation is more likely",
        negative_means="non mangrove vegetation or open water",

        physical_floor=0.20,
        valid_range=(-2.0, 2.0),

        use_cases=(
            "map mangrove extent",
            "map mangrove forest",
            "assess mangrove health",
            "monitor mangrove density",
            "detect coastal mangrove",
        ),

        priority=10,
    ),

    # ------------------------------------------------------------------------
    # TURBIDITY
    # ------------------------------------------------------------------------

    "NDTI": IndexDefinition(
        required_bands=("green", "red"),

        compute=lambda b: _normalised_difference(
            b["red"],
            b["green"],
        ),

        positive_means="more turbid sediment rich water",
        negative_means="clearer water",

        physical_floor=0.0,
        valid_range=(-1.0, 1.0),

        use_cases=(
            "measure water turbidity",
            "detect turbid water",
            "detect sediment rich water",
            "measure sediment",
            "compare water clarity",
            "monitor coastal sediment",
        ),

        priority=30,
    ),

    # ------------------------------------------------------------------------
    # BURN
    # ------------------------------------------------------------------------

    "NBR": IndexDefinition(
        required_bands=("nir", "swir2"),

        compute=lambda b: _normalised_difference(
            b["nir"],
            b["swir2"],
        ),

        positive_means="healthy or unburned vegetation",
        negative_means="burned or disturbed vegetation",

        physical_floor=-0.10,
        valid_range=(-1.0, 1.0),

        use_cases=(
            "detect burn scars",
            "detect wildfire damage",
            "map burned forest",
            "detect fire disturbance",
            "compare vegetation before and after fire",
        ),

        polarity="negative",
        priority=10,
    ),

    "NBR2": IndexDefinition(
        required_bands=("swir1", "swir2"),

        compute=lambda b: _normalised_difference(
            b["swir1"],
            b["swir2"],
        ),

        positive_means="less severe burn or intact vegetation",
        negative_means="burn related SWIR change",

        physical_floor=-0.05,
        valid_range=(-1.0, 1.0),

        use_cases=(
            "assess burn severity",
            "refine fire damage",
            "distinguish fire effects",
        ),

        polarity="negative",
        priority=40,
    ),

    "BAI": IndexDefinition(
        required_bands=("red", "nir"),

        compute=_bai,

        positive_means="spectra close to burned reference",
        negative_means="spectra farther from burned reference",

        physical_floor=10.0,
        valid_range=(0.0, float("inf")),

        use_cases=(
            "detect burn scars",
            "find recently burned areas",
            "screen forest fire impacts",
        ),

        requires_reflectance=True,
        priority=60,
    ),

    # ------------------------------------------------------------------------
    # GEOLOGY
    # ------------------------------------------------------------------------

    "RI": IndexDefinition(
        required_bands=("blue", "green", "red"),

        compute=lambda b: _safe_ratio(
            b["red"] ** 2,
            b["blue"] * b["green"] ** 3,
        ),

        positive_means="reddish iron oxide rich soil",
        negative_means="less reddish soil",

        physical_floor=1.0,
        valid_range=(0.0, float("inf")),

        use_cases=(
            "map reddish soil",
            "detect iron oxide",
            "map lateritic soil",
            "investigate iron rich soil",
        ),

        requires_reflectance=True,
        priority=40,
    ),

    "CLAY_RATIO": IndexDefinition(
        required_bands=("swir1", "swir2"),

        compute=lambda b: _safe_ratio(
            b["swir1"],
            b["swir2"],
        ),

        positive_means="clay rich material",
        negative_means="less clay like spectral response",

        physical_floor=1.0,
        valid_range=(0.0, float("inf")),

        use_cases=(
            "detect clay minerals",
            "map clay rich soil",
            "detect hydrothermal alteration",
            "support geological interpretation",
        ),
    ),

    "FERROUS_RATIO": IndexDefinition(
        required_bands=("nir", "swir1"),

        compute=lambda b: _safe_ratio(
            b["swir1"],
            b["nir"],
        ),

        positive_means="ferrous mineral spectral response",
        negative_means="less ferrous response",

        physical_floor=1.0,
        valid_range=(0.0, float("inf")),

        use_cases=(
            "detect iron minerals",
            "map iron ore",
            "detect ferrous minerals",
            "investigate mining signatures",
            "map mineralised ground",
        ),

        priority=20,
    ),
}


# ============================================================================
# VALIDATION
# ============================================================================

def _validate_registry() -> None:

    for name, definition in INDEX_REGISTRY.items():

        missing = (
            set(definition.required_bands)
            - CANONICAL_BANDS
        )

        if missing:
            raise ValueError(
                f"{name} requires non canonical bands: "
                f"{sorted(missing)}"
            )

        if (
            definition.valid_range[0]
            > definition.valid_range[1]
        ):
            raise ValueError(
                f"{name} has invalid valid_range"
            )

        if not definition.use_cases:
            raise ValueError(
                f"{name} must have router use cases"
            )

        vocabulary = _index_vocabulary(
            name,
            definition,
        )

        if not vocabulary[0]:
            raise ValueError(
                f"{name} has no router vocabulary"
            )


def _index_vocabulary(
    name: str,
    definition: IndexDefinition,
) -> tuple[
    set[str],
    set[tuple[str, str]],
]:

    unigrams: set[str] = set()
    bigrams: set[tuple[str, str]] = set()

    for phrase in definition.use_cases:

        sequence = _token_sequence(
            phrase
        )

        unigrams |= _content_tokens(
            sequence
        )

        bigrams |= _bigrams(
            sequence
        )

    # Allow explicit index names:
    # NDVI, NDMI, NDBI, etc.
    unigrams.add(
        _singular(name.lower())
    )

    for part in name.lower().split("_"):
        if len(part) > 2:
            unigrams.add(
                _singular(part)
            )

    return unigrams, bigrams


_validate_registry()


_INDEX_VOCAB = {
    name: _index_vocabulary(
        name,
        definition,
    )
    for name, definition in INDEX_REGISTRY.items()
}


# ============================================================================
# INDEX ANALYSER
# ============================================================================

class IndexAnalyser:
    """
    Physics-first spectral index engine.

    Main operations:

        compute()
        classify()
        diff()
        change_map()
        select_indices()
        explain_selection()
    """

    _UNIGRAM_WEIGHT = 1.0
    _BIGRAM_WEIGHT = 3.0
    _MIN_SCORE = 1.0

    # ------------------------------------------------------------------------
    # COMPUTE
    # ------------------------------------------------------------------------
    # Query-specific primary physics.
    #
    # These are deliberately narrow. Generic vocabulary matching is useful
    # for discovery, but it should not cause unrelated indices to enter the
    # physical analysis pipeline.

    QUERY_PRIORITIES = {
        "crop_stress": (
            "NDVI",
            "NDRE",
            "NDMI",
        ),

        "soil_moisture": (
            "LSWI",
            "NDMI",
        ),

        "waterlogging": (
            "MNDWI",
            "LSWI",
        ),

        "built_up": (
            "NDBI",
            "UI",
        ),

        "crop_health": (
            "NDVI",
            "EVI",
            "NDRE",
            "NDMI",
        ),

        "salinity": (
            "SALINITY_PROXY",
        ),
    }
    def _query_physics_class(self, query: str) -> str | None:
        """
        Detect the dominant physical question.

        This is deterministic and intentionally conservative.
        """

        q = query.lower()

        # Crop stress / crop health
        if any(word in q for word in (
            "stressed",
            "stress",
            "crop stress",
            "field stressed",
            "wheat stressed",
            "paddy stressed",
        )):
            return "crop_stress"

        # Waterlogging
        if any(word in q for word in (
            "waterlogging",
            "water logged",
            "waterlogged",
            "standing water",
            "inundated",
            "flooded field",
        )):
            return "waterlogging"

        # Soil moisture
        if any(word in q for word in (
            "soil moisture",
            "surface moisture",
            "surface soil moisture",
            "moisture changed",
            "moisture change",
        )):
            return "soil_moisture"

        # Built-up
        if any(word in q for word in (
            "built-up",
            "built up",
            "urban",
            "construction",
            "settlement",
            "impervious",
        )):
            return "built_up"

        # Salinity
        if any(word in q for word in (
            "salinity",
            "saline soil",
            "salt affected",
            "salt-affected",
        )):
            return "salinity"

        # General crop vegetation
        if any(word in q for word in (
            "crop vegetation",
            "vegetation health",
            "vegetation healthiest",
            "crop health",
            "crop vigour",
            "crop vigor",
            "healthy vegetation",
        )):
            return "crop_health"

        return None
    def compute(
        self,
        bands: dict[str, np.ndarray],
        names: list[str],
    ) -> dict[str, np.ndarray]:

        if not names:
            return {}

        unknown = [
            name
            for name in names
            if name not in INDEX_REGISTRY
        ]

        if unknown:
            raise KeyError(
                f"Unknown spectral indices: {unknown}. "
                f"Available: {sorted(INDEX_REGISTRY)}"
            )

        arrays = {
            key: np.asarray(
                value,
                dtype=np.float64,
            )
            for key, value in bands.items()
        }

        shapes = {
            arr.shape
            for arr in arrays.values()
        }

        if len(shapes) > 1:
            raise ValueError(
                "Band arrays must have identical shapes; "
                f"found {sorted(shapes)}"
            )

        result: dict[str, np.ndarray] = {}

        for name in dict.fromkeys(names):

            definition = INDEX_REGISTRY[name]

            missing = [
                band
                for band in definition.required_bands
                if band not in arrays
            ]

            if missing:
                raise ValueError(
                    f"{name} requires bands "
                    f"{definition.required_bands}; "
                    f"missing {missing}"
                )

            # Reflectance-dependent indices
            if definition.requires_reflectance:

                probe = arrays[
                    definition.required_bands[0]
                ]

                finite = probe[
                    np.isfinite(probe)
                ]

                if (
                    finite.size
                    and float(np.max(finite))
                    > REFLECTANCE_MAX
                ):
                    raise ValueError(
                        f"{name} requires surface reflectance "
                        f"in approximately [0, 1]. "
                        f"Maximum input value was "
                        f"{float(np.max(finite)):.2f}."
                    )

            result[name] = definition.compute(
                arrays
            )

        return result

    # ------------------------------------------------------------------------
    # CLASSIFY
    # ------------------------------------------------------------------------

    def classify(
        self,
        name: str,
        arr: np.ndarray,
        percentile: float = 90,
    ) -> tuple[np.ndarray, dict]:

        if name not in INDEX_REGISTRY:
            raise KeyError(
                f"Unknown spectral index {name!r}"
            )

        if not 0 < percentile <= 100:
            raise ValueError(
                "percentile must be > 0 and <= 100"
            )

        values = np.asarray(
            arr,
            dtype=np.float64,
        )

        finite = np.isfinite(values)

        mask = np.zeros(
            values.shape,
            dtype=bool,
        )

        definition = INDEX_REGISTRY[name]

        floor = definition.physical_floor

        metadata = {
            "index": name,
            "percentile": percentile,
            "physical_floor": floor,
            "floor_disabled": False,
            "threshold": None,
            "fraction_selected": 0.0,
        }

        if not np.any(finite):

            metadata["floor_disabled"] = True

            metadata["note"] = (
                "No finite index values."
            )

            return mask, metadata

        if definition.polarity == "positive":

            scene_extreme = float(
                np.nanmax(values)
            )

            if scene_extreme < floor:

                metadata["floor_disabled"] = True

                metadata["note"] = (
                    f"Scene maximum {scene_extreme:.4f} "
                    f"is below physical floor {floor:.4f}."
                )

                return mask, metadata

            threshold = max(
                float(
                    np.nanpercentile(
                        values,
                        percentile,
                    )
                ),
                floor,
            )

            mask = (
                finite
                & (values >= threshold)
            )

        else:

            scene_extreme = float(
                np.nanmin(values)
            )

            if scene_extreme > floor:

                metadata["floor_disabled"] = True

                metadata["note"] = (
                    f"Scene minimum {scene_extreme:.4f} "
                    f"is above physical floor {floor:.4f}."
                )

                return mask, metadata

            threshold = min(
                float(
                    np.nanpercentile(
                        values,
                        100.0 - percentile,
                    )
                ),
                floor,
            )

            mask = (
                finite
                & (values <= threshold)
            )

        metadata["threshold"] = threshold

        metadata["fraction_selected"] = float(
            np.mean(mask)
        )

        return mask, metadata

    # ------------------------------------------------------------------------
    # DIFFERENCE
    # ------------------------------------------------------------------------

    def diff(
        self,
        before: np.ndarray,
        after: np.ndarray,
    ) -> np.ndarray:

        before_arr = np.asarray(
            before,
            dtype=np.float64,
        )

        after_arr = np.asarray(
            after,
            dtype=np.float64,
        )

        if before_arr.shape != after_arr.shape:
            raise ValueError(
                "before and after must have identical "
                f"shapes; found "
                f"{before_arr.shape} and "
                f"{after_arr.shape}"
            )

        return after_arr - before_arr

    # ------------------------------------------------------------------------
    # NORMALISED CHANGE
    # ------------------------------------------------------------------------

    def relative_change(
        self,
        before: np.ndarray,
        after: np.ndarray,
    ) -> np.ndarray:

        before_arr = np.asarray(
            before,
            dtype=np.float64,
        )

        after_arr = np.asarray(
            after,
            dtype=np.float64,
        )

        if before_arr.shape != after_arr.shape:
            raise ValueError(
                "before and after must have identical shapes"
            )

        denominator = np.abs(
            before_arr
        )

        out = np.zeros(
            before_arr.shape,
            dtype=np.float64,
        )

        np.divide(
            after_arr - before_arr,
            denominator,
            out=out,
            where=denominator > 1e-12,
        )

        invalid = (
            np.isnan(before_arr)
            | np.isnan(after_arr)
        )

        out[invalid] = np.nan

        return out

    # ------------------------------------------------------------------------
    # CHANGE CLASSIFICATION
    # ------------------------------------------------------------------------

    def change_map(
        self,
        before: np.ndarray,
        after: np.ndarray,
        threshold: float,
    ) -> dict[str, np.ndarray]:

        delta = self.diff(
            before,
            after,
        )

        increased = delta >= threshold

        decreased = delta <= -threshold

        stable = (
            ~increased
            & ~decreased
            & np.isfinite(delta)
        )

        return {
            "delta": delta,
            "increased": increased,
            "decreased": decreased,
            "stable": stable,
        }

    # ------------------------------------------------------------------------
    # ROUTER
    # ------------------------------------------------------------------------
    def select_indices(
        self,
        query: str,
    ) -> list[str]:

        # ------------------------------------------------------------
        # 1. Detect dominant physical question
        # ------------------------------------------------------------

        physics_class = self._query_physics_class(query)

        if physics_class is not None:
            return list(
                self.QUERY_PRIORITIES[physics_class]
            )

        # ------------------------------------------------------------
        # 2. Generic physics vocabulary fallback
        # ------------------------------------------------------------

        sequence = _token_sequence(query)

        q_unigrams = _content_tokens(sequence)

        if not q_unigrams:
            return []

        q_bigrams = _bigrams(sequence)

        ranked: list[
            tuple[float, int, str]
        ] = []

        for (
            name,
            definition,
        ) in INDEX_REGISTRY.items():

            unigrams, bigrams = (
                _INDEX_VOCAB[name]
            )

            unigram_matches = (
                q_unigrams & unigrams
            )

            bigram_matches = (
                q_bigrams & bigrams
            )

            score = (
                self._UNIGRAM_WEIGHT
                * len(unigram_matches)
                +
                self._BIGRAM_WEIGHT
                * len(bigram_matches)
            )

            if score >= self._MIN_SCORE:

                ranked.append(
                    (
                        score,
                        definition.priority,
                        name,
                    )
                )

        ranked.sort(
            key=lambda row: (
                -row[0],
                row[1],
                row[2],
            )
        )

        return [
            name
            for _, _, name
            in ranked
        ]
    # ------------------------------------------------------------------------
    # ROUTER EXPLANATION
    # ------------------------------------------------------------------------

    def explain_selection(
        self,
        query: str,
    ) -> list[dict]:

        sequence = _token_sequence(
            query
        )

        q_unigrams = _content_tokens(
            sequence
        )

        q_bigrams = _bigrams(
            sequence
        )

        trace: list[dict] = []

        for (
            name,
            definition,
        ) in INDEX_REGISTRY.items():

            unigrams, bigrams = (
                _INDEX_VOCAB[name]
            )

            matched_unigrams = sorted(
                q_unigrams & unigrams
            )

            matched_bigrams = sorted(
                " ".join(pair)
                for pair in (
                    q_bigrams & bigrams
                )
            )

            score = (
                self._UNIGRAM_WEIGHT
                * len(matched_unigrams)
                +
                self._BIGRAM_WEIGHT
                * len(matched_bigrams)
            )

            if score >= self._MIN_SCORE:

                trace.append(
                    {
                        "index": name,
                        "score": score,
                        "priority": definition.priority,
                        "matched_terms": matched_unigrams,
                        "matched_phrases": matched_bigrams,
                        "positive_means": definition.positive_means,
                        "negative_means": definition.negative_means,
                    }
                )

        trace.sort(
            key=lambda row: (
                -row["score"],
                row["priority"],
                row["index"],
            )
        )

        return trace
    # ------------------------------------------------------------------------
    # HIGH-LEVEL ANALYSIS
    # ------------------------------------------------------------------------

    def analyse(
        self,
        bands: dict[str, np.ndarray],
        query: str,
        percentile: float = 90,
    ) -> dict:
        """
        High-level physics-first analysis.

        1. Understand the query and select relevant indices.
        2. Compute those indices from the supplied bands.
        3. Classify the strongest physically plausible regions.
        4. Return routing + computed results + classification metadata.

        No raster I/O is performed here. `bands` must already contain
        canonical band names such as red, nir, swir1, etc.
        """

        # ------------------------------------------------------------
        # STEP 1 — Query -> relevant physics
        # ------------------------------------------------------------

        selected = self.select_indices(query)

        routing = self.explain_selection(query)

        if not selected:
            return {
                "query": query,
                "indices": [],
                "results": {},
                "routing": routing,
            }

        # ------------------------------------------------------------
        # STEP 2 — Compute selected spectral indices
        # ------------------------------------------------------------

        computed = self.compute(
            bands,
            selected,
        )

        # ------------------------------------------------------------
        # STEP 3 — Convert each index into a physical class mask
        # ------------------------------------------------------------

        results: dict[str, dict] = {}

        for name, array in computed.items():

            mask, metadata = self.classify(
                name,
                array,
                percentile=percentile,
            )

            results[name] = {
                "array": array,
                "mask": mask,
                "metadata": metadata,
            }

        # ------------------------------------------------------------
        # STEP 4 — Return complete analysis object
        # ------------------------------------------------------------

        return {
            "query": query,
            "indices": selected,
            "results": results,
            "routing": routing,
        }

# ============================================================================
# CONVENIENCE API
# ============================================================================

_DEFAULT_ANALYSER = IndexAnalyser()


def compute_index(
    name: str,
    bands: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Convenience wrapper:

        compute_index("NDVI", bands)
    """

    return _DEFAULT_ANALYSER.compute(
        bands,
        [name],
    )[name]


def select_indices(
    query: str,
) -> list[str]:
    """
    Convenience wrapper for the router.
    """

    return _DEFAULT_ANALYSER.select_indices(
        query
    )


def explain_query(
    query: str,
) -> list[dict]:
    """
    Explain why particular indices were selected.
    """

    return _DEFAULT_ANALYSER.explain_selection(
        query
    )
def analyse_query(
    bands: dict[str, np.ndarray],
    query: str,
    percentile: float = 90,
) -> dict:
    """
    Convenience wrapper for complete physics-first analysis.
    """

    return _DEFAULT_ANALYSER.analyse(
        bands,
        query,
        percentile=percentile,
    )

# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "CANONICAL_BANDS",
    "STOPWORDS",
    "INDEX_REGISTRY",
    "IndexDefinition",
    "IndexAnalyser",
    "compute_index",
    "select_indices",
    "explain_query",
    "analyse_query",
]
