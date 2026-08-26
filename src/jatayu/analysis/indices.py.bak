"""Physics-first spectral index registry used by Jatayu analysis tools.

The registry deliberately keeps formulae and their physical interpretation together.
All functions operate on canonical band names and never perform raster I/O.

References are given per index below. Operational ``physical_floor`` values are
conservative scene-plausibility guards, not universal class thresholds:
percentiles determine sensitivity, while the floor prevents a mathematically
valid percentile from inventing a class in a scene where the phenomenon is not
spectrally plausible.

Router matching (``IndexAnalyser.select_indices``) is whole-token, never
substring. A substring test lets English function words reach band maths:
"referred to in the query" scored NDVI through "moni-to-r", "grow-in-g" and
"whe-the-r", which outranked the actual water indices for a water query.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Polarity = Literal["positive", "negative"]
Computation = Callable[[dict[str, FloatArray]], FloatArray]

CANONICAL_BANDS = frozenset(
    {"blue", "green", "red", "rededge1", "nir", "swir1", "swir2"}
)

# Above this, band values are assumed to be digital numbers rather than surface
# reflectance. Sentinel-2 L2A stores reflectance scaled by 10000.
REFLECTANCE_MAX = 1.5


# ---------------------------------------------------------------------------
# Query tokenisation for the router
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z]+")

# Stopwords fall into three groups, all of which caused false matches under the
# previous substring scorer:
#   1. English function words - "the", "to", "in" carry no class information.
#   2. Task verbs - they describe what to DO, not what to look FOR. "monitor
#      the water" and "map the water" should select the same index.
#   3. Words shared by nearly every registry entry - matching them ranks the
#      entry with the most prose, not the most relevant physics.
_RAW_STOPWORDS = frozenset(
    {
        # english function words
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
        "have", "in", "into", "is", "it", "its", "of", "on", "or", "that",
        "the", "this", "to", "was", "were", "with", "what", "where", "which",
        "how", "can", "you", "your", "please", "any", "all", "such", "between",
        "their", "there", "whether",
        # task verbs and UI scaffolding
        "highlight", "show", "identify", "find", "detect", "locate", "mark",
        "give", "tell", "measure", "assess", "monitor", "screen", "check",
        "compare", "estimate", "map", "support", "separate", "refine",
        "distinguish", "investigate", "inspect", "delineate",
        "region", "area", "image", "imagery", "scene", "picture", "query",
        "referred", "refer", "output", "result",
        # words shared by nearly every registry entry
        "index", "normalized", "normalised", "difference", "band", "ratio",
        "value", "used", "use", "useful", "likely", "more", "less", "other",
        "extent", "status", "condition",
    }
)


def _singular(token: str) -> str:
    """Crude, deterministic plural folding so 'bodies' matches 'body'.

    Deliberately not a real stemmer: no dependency, no surprises, and every
    transformation here is reversible by eye when debugging a ranking.
    """
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


STOPWORDS = frozenset(_singular(word) for word in _RAW_STOPWORDS)


def _token_sequence(text: str) -> list[str]:
    """Normalised token sequence with order preserved, so bigrams stay meaningful."""
    return [_singular(t) for t in _TOKEN_RE.findall(text.lower()) if len(t) > 2]


def _content_tokens(tokens: list[str]) -> set[str]:
    """Deduplicated content words. A set, so repeating "the" cannot score twice."""
    return {t for t in tokens if t not in STOPWORDS}


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    """Adjacent token pairs, letting "water body" outrank an incidental "water"."""
    return set(zip(tokens, tokens[1:]))


# ---------------------------------------------------------------------------
# Index computations
# ---------------------------------------------------------------------------


def _normalised_difference(a: FloatArray, b: FloatArray) -> FloatArray:
    """Compute (a-b)/(a+b) without emitting divide-by-zero warnings.

    A zero denominator represents no usable contrast, so it is mapped to 0.0.
    NaN inputs stay NaN, preserving missing-data provenance.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = np.zeros(np.broadcast_shapes(a.shape, b.shape), dtype=np.float64)
    numerator = a - b
    denominator = a + b
    np.divide(numerator, denominator, out=out, where=denominator != 0)
    invalid = np.isnan(numerator) | np.isnan(denominator)
    out[invalid] = np.nan
    return out


def _safe_ratio(numerator: FloatArray, denominator: FloatArray) -> FloatArray:
    """Compute a ratio with zero denominators represented as zero."""
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    out = np.zeros(
        np.broadcast_shapes(numerator.shape, denominator.shape),
        dtype=np.float64,
    )
    np.divide(numerator, denominator, out=out, where=denominator != 0)
    invalid = np.isnan(numerator) | np.isnan(denominator)
    out[invalid] = np.nan
    return out


def _evi(b: dict[str, FloatArray]) -> FloatArray:
    """Enhanced Vegetation Index, Huete et al. (2002).

    Improves vegetation sensitivity while reducing residual atmospheric and
    soil-background effects relative to NDVI.
    """
    denominator = b["nir"] + 6.0 * b["red"] - 7.5 * b["blue"] + 1.0
    out = np.zeros(
        np.broadcast_shapes(b["nir"].shape, b["red"].shape, b["blue"].shape)
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
    """Soil Adjusted Vegetation Index, Huete (1988), with conventional L=0.5.

    L=0.5 is a practical middle-cover choice that reduces exposed-soil influence
    without requiring a separate vegetation-density estimate.
    """
    denominator = b["nir"] + b["red"] + 0.5
    out = np.zeros(np.broadcast_shapes(b["nir"].shape, b["red"].shape))
    np.divide(
        1.5 * (b["nir"] - b["red"]),
        denominator,
        out=out,
        where=denominator != 0,
    )
    invalid = np.isnan(b["nir"]) | np.isnan(b["red"]) | np.isnan(denominator)
    out[invalid] = np.nan
    return out


def _bai(b: dict[str, FloatArray]) -> FloatArray:
    """Burned Area Index, Martin (1998).

    Inverse squared spectral distance from a canonical burned-area reference
    point, so unlike normalised indices it is intentionally unbounded.

    A zero denominator is an exact match to the burned reference, which should
    be MAXIMAL BAI. Flooring the denominator rather than special-casing to zero
    keeps the polarity correct.
    """
    denominator = (b["red"] - 0.10) ** 2 + (b["nir"] - 0.06) ** 2
    denominator = np.maximum(denominator, 1e-12)
    out = 1.0 / denominator
    invalid = np.isnan(b["red"]) | np.isnan(b["nir"])
    return np.where(invalid, np.nan, out)


@dataclass(frozen=True, slots=True)
class IndexDefinition:
    """Immutable contract for one spectral index."""

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


# Primary references:
# - McFeeters (1996), Int. J. Remote Sensing, 17, 1425-1432.
# - Xu (2006), Int. J. Remote Sensing, 27, 3025-3033.
# - Gao (1996), Remote Sensing of Environment, 56, 134-140.
# - Rouse et al. (1974), Third ERTS Symposium.
# - Huete (1988), Remote Sensing of Environment, 25, 295-309.
# - Huete et al. (2002), Remote Sensing of Environment, 83, 195-213.
# - Gitelson & Merzlyak (1994), J. Photochem. Photobiol. B, 22, 247-252.
# - Gupta et al. (2018), MethodsX, 5, 1129-1139.
# - Zha et al. (2003), Int. J. Remote Sensing, 24, 583-594.
# - Rikimaru et al. (2002), Forest Canopy Density model.
# - Pouget et al. (1990), soil-colour remote sensing.
# - Martin (1998), burned-area spectral index.
# - Key & Benson (2006), Normalized Burn Ratio / fire severity.
# - Lacaux et al. (2007), turbidity indices.
# - Taghadosi et al. (2019), Eur. J. Remote Sensing, 52, 138-154.
#
# Clay and ferrous ratios are classical geological band-ratio techniques.
#
# NDSI is deliberately ABSENT. Its formula, nd(green, swir1), is byte-identical
# to MNDWI, so it fires on any water body - it would report snow on Chilika
# Lake. Reinstate only behind a latitude/elevation plausibility gate.
#
# use_cases are ROUTER INPUT, not documentation. Every phrase here is scored
# against user queries, so a class noun that never appears in any use_case is
# unreachable no matter how good the physics is. When adding an index, write
# the nouns a user would actually type.

INDEX_REGISTRY: dict[str, IndexDefinition] = {
    "MNDWI": IndexDefinition(
        ("green", "swir1"),
        lambda b: _normalised_difference(b["green"], b["swir1"]),
        "open water is more likely as MNDWI increases",
        "land, vegetation, and built-up surfaces are more likely",
        0.0,
        (-1.0, 1.0),
        (
            "map open water and flooded areas",
            "delineate water bodies and their boundaries",
            "find ponds, lakes, rivers, lagoons, and inundation",
            "detect water in built-up areas with reduced urban noise",
            "map water in coastal Odisha and Chilika lagoon",
        ),
        # Default water index: outranks CMR and NDTI on a bare "water" query,
        # both of which are specialisations (mangrove, turbidity).
        priority=5,
    ),
    "NDWI": IndexDefinition(
        ("green", "nir"),
        lambda b: _normalised_difference(b["green"], b["nir"]),
        "open water is more likely as NDWI increases",
        "vegetation and dry land are more likely",
        0.0,
        (-1.0, 1.0),
        (
            "detect open surface water",
            "delineate water bodies and their boundaries",
            "map rivers, lakes, ponds, and wetlands",
            "estimate water extent from optical imagery",
        ),
        # Retained to demonstrate its documented failure over turbid water:
        # on the Hooghly, NDWI's 95th percentile reaches 0.03 while MNDWI's
        # reaches 0.45. Prefer MNDWI for Indian rivers.
        priority=80,
    ),
    "LSWI": IndexDefinition(
        ("nir", "swir1"),
        lambda b: _normalised_difference(b["nir"], b["swir1"]),
        "higher vegetation or surface moisture is more likely",
        "drier vegetation or soil is more likely",
        0.0,
        (-1.0, 1.0),
        (
            "measure crop and vegetation moisture",
            "detect flooded or waterlogged agriculture",
            "monitor paddy water status and wetland moisture",
        ),
    ),
    "NDVI": IndexDefinition(
        ("red", "nir"),
        lambda b: _normalised_difference(b["nir"], b["red"]),
        "green vegetation and photosynthetic activity are more likely",
        "bare soil, water, senescent vegetation, or sparse cover are more likely",
        0.10,
        (-1.0, 1.0),
        (
            "measure vegetation greenness and health",
            "check whether crops are growing",
            "assess paddy vigour and vegetation cover",
            "monitor forest condition and agricultural fields",
        ),
        priority=20,
    ),
    "EVI": IndexDefinition(
        ("blue", "red", "nir"),
        _evi,
        "dense, healthy vegetation is more likely",
        "sparse vegetation or non-vegetated surfaces are more likely",
        0.10,
        (-1.0, 1.0),
        (
            "assess dense crop and forest vegetation",
            "measure vegetation health with reduced soil and atmospheric influence",
            "monitor forest canopy and agricultural vigour",
        ),
    ),
    "SAVI": IndexDefinition(
        ("red", "nir"),
        _savi,
        "vegetation is more likely",
        "bare soil and sparse vegetation are more likely",
        0.10,
        (-1.0, 1.0),
        (
            "measure vegetation where exposed soil is important",
            "assess sparse crops and dryland agriculture",
            "monitor vegetation in soil-dominated fields",
        ),
    ),
    "NDRE": IndexDefinition(
        ("rededge1", "nir"),
        lambda b: _normalised_difference(b["nir"], b["rededge1"]),
        "higher leaf chlorophyll and vegetation vigour are more likely",
        "lower chlorophyll, stress, or non-vegetation is more likely",
        0.05,
        (-1.0, 1.0),
        (
            "detect crop nitrogen or chlorophyll stress",
            "check whether paddy is stressed",
            "monitor mid-to-late season crop health",
            "assess subtle forest canopy stress",
        ),
        priority=15,
    ),
    "CMR": IndexDefinition(
        ("green", "red", "nir"),
        lambda b: (
            _normalised_difference(b["nir"], b["red"])
            - _normalised_difference(b["green"], b["nir"])
        ),
        "mangrove vegetation is more likely",
        "non-mangrove vegetation or open water is more likely",
        0.20,
        (-2.0, 2.0),
        (
            "map mangrove extent",
            "assess mangrove health and density",
            "distinguish mangroves from non-mangrove vegetation and water",
            "monitor Bhitarkanika mangroves",
            "monitor mangrove change in Indian coastal wetlands",
        ),
        priority=10,
    ),
    # NOTE: NDMI and LSWI are the identical formula, nd(nir, swir1). Both are
    # retained because the literature uses different names by context - NDMI for
    # canopy moisture, LSWI for surface and paddy water. Do not "fix" one into a
    # different formula.
    "NDMI": IndexDefinition(
        ("nir", "swir1"),
        lambda b: _normalised_difference(b["nir"], b["swir1"]),
        "higher canopy moisture is more likely",
        "vegetation water stress or drying is more likely",
        0.10,
        (-1.0, 1.0),
        (
            "measure vegetation moisture and crop moisture",
            "detect drought or canopy water stress",
            "assess mangrove moisture and health",
            "monitor forest moisture in Similipal",
        ),
    ),
    "NDBI": IndexDefinition(
        ("nir", "swir1"),
        lambda b: _normalised_difference(b["swir1"], b["nir"]),
        "built-up surfaces are more likely",
        "vegetation or non-built surfaces are more likely",
        0.0,
        (-1.0, 1.0),
        (
            "map built-up and urban areas",
            "detect urban expansion",
            "find new construction and settlements",
        ),
        priority=10,
    ),
    "UI": IndexDefinition(
        ("nir", "swir2"),
        lambda b: _normalised_difference(b["swir2"], b["nir"]),
        "built-up or urban surfaces are more likely",
        "vegetated and non-urban surfaces are more likely",
        0.0,
        (-1.0, 1.0),
        (
            "detect urban built-up land",
            "map dense construction and impervious surfaces",
            "separate built-up areas from vegetation",
        ),
        priority=60,
    ),
    "BAI": IndexDefinition(
        ("red", "nir"),
        _bai,
        "spectra close to the canonical burned target are more likely",
        "spectra farther from the canonical burned target are more likely",
        10.0,
        (0.0, float("inf")),
        (
            "detect burn scars",
            "find recently burned areas",
            "screen forest fire impacts",
            "map burned land in Similipal",
        ),
        requires_reflectance=True,  # absolute constants 0.10 / 0.06 are reflectance
        priority=60,  # NBR is the standard burn index; prefer it
    ),
    "BSI": IndexDefinition(
        ("blue", "red", "nir", "swir1"),
        lambda b: _normalised_difference(
            b["swir1"] + b["red"], b["nir"] + b["blue"]
        ),
        "bare or exposed soil is more likely",
        "vegetated or water-covered surfaces are more likely",
        0.0,
        (-1.0, 1.0),
        (
            "map bare soil",
            "detect exposed ground and agricultural fallow land",
            "separate soil from vegetation",
            "detect mining pits and quarry expansion",
        ),
        priority=20,
    ),
    "RI": IndexDefinition(
        ("blue", "green", "red"),
        lambda b: _safe_ratio(b["red"] ** 2, b["blue"] * b["green"] ** 3),
        "red or iron-oxide-rich soil is more likely",
        "less reddish soil is more likely",
        1.0,
        (0.0, float("inf")),
        (
            "map reddish soils and iron-oxide signatures",
            "screen lateritic and iron-rich ground",
            "support geological interpretation in mining areas",
            "investigate iron-rich soils around Keonjhar",
        ),
        requires_reflectance=True,  # degree -2 homogeneous: scales by 1/k^2
        priority=40,
    ),
    "CLAY_RATIO": IndexDefinition(
        ("swir1", "swir2"),
        lambda b: _safe_ratio(b["swir1"], b["swir2"]),
        "clay-bearing or OH-mineral-rich material is more likely",
        "less clay-like spectral response is more likely",
        1.0,
        (0.0, float("inf")),
        (
            "screen clay minerals and hydrothermal alteration",
            "support geological and mining interpretation",
            "inspect clay-rich exposed ground around mining areas",
        ),
    ),
    "FERROUS_RATIO": IndexDefinition(
        ("nir", "swir1"),
        lambda b: _safe_ratio(b["swir1"], b["nir"]),
        "iron-bearing or ferrous-mineral spectral response is more likely",
        "less ferrous spectral response is more likely",
        1.0,
        (0.0, float("inf")),
        (
            "screen iron-bearing minerals",
            "support iron ore exploration and illegal mining detection",
            "investigate mining signatures in Keonjhar",
            "map exposed mineralised ground",
        ),
        priority=20,
    ),
    "NBR": IndexDefinition(
        ("nir", "swir2"),
        lambda b: _normalised_difference(b["nir"], b["swir2"]),
        "healthy or unburned vegetation is more likely",
        "burned or strongly disturbed vegetation is more likely",
        -0.10,
        (-1.0, 1.0),
        (
            "detect burn scars",
            "estimate wildfire disturbance",
            "map burned forest in Similipal and Odisha",
            "compare vegetation condition before and after fire",
        ),
        polarity="negative",
        priority=10,  # standard burn index, outranks BAI
    ),
    "NBR2": IndexDefinition(
        ("swir1", "swir2"),
        lambda b: _normalised_difference(b["swir1"], b["swir2"]),
        "less severe burn or more intact vegetation is more likely",
        "burn-related SWIR spectral change is more likely",
        -0.05,
        (-1.0, 1.0),
        (
            "assess burn severity",
            "refine burned-area mapping",
            "distinguish fire effects in dry vegetation",
        ),
        polarity="negative",
        priority=40,
    ),
    "NDTI": IndexDefinition(
        ("green", "red"),
        lambda b: _normalised_difference(b["red"], b["green"]),
        "higher relative red reflectance is more turbid water",
        "clearer water is more likely",
        0.0,
        (-1.0, 1.0),
        (
            "measure water turbidity",
            "estimate how turbid a lagoon is",
            "compare turbidity and clarity between water bodies",
            "monitor sediment-rich water in Chilika lagoon and coastal Odisha",
        ),
        # Turbidity queries always carry an explicit turbid/sediment/clarity
        # token, so NDTI does not need to win a generic "water body" tie.
        priority=30,
    ),
    "CHLOROPHYLL_RATIO": IndexDefinition(
        ("rededge1", "nir"),
        lambda b: _safe_ratio(b["nir"], b["rededge1"]) - 1.0,
        "higher chlorophyll concentration is more likely",
        "lower chlorophyll or non-vegetated water is more likely",
        0.05,
        # Can go negative over water and bare soil where rededge1 exceeds nir.
        (-1.0, float("inf")),
        (
            "screen chlorophyll-rich vegetation",
            "assess crop chlorophyll and nitrogen status",
            "monitor aquatic or vegetation chlorophyll proxies",
            "screen algal bloom in lagoons",
        ),
        priority=40,
    ),
    "SALINITY_PROXY": IndexDefinition(
        ("red", "nir"),
        lambda b: _normalised_difference(b["red"], b["nir"]),
        "higher soil salinity is more likely",
        "lower soil salinity is more likely",
        0.0,
        (-1.0, 1.0),
        (
            "screen saline soil",
            "map salt-affected agricultural land",
            "assess salinity risk in irrigated fields and coastal areas",
            "screen salinity around coastal Odisha",
        ),
        priority=20,
    ),
}


def _validate_registry() -> None:
    """Fail fast during development if a definition violates the module contract."""
    for name, definition in INDEX_REGISTRY.items():
        missing = set(definition.required_bands) - CANONICAL_BANDS
        if missing:
            raise ValueError(f"{name} requires non-canonical bands: {sorted(missing)}")
        if definition.valid_range[0] > definition.valid_range[1]:
            raise ValueError(f"{name} has an invalid valid_range")
        if not definition.use_cases:
            raise ValueError(f"{name} must expose at least one router use case")
        if not _index_vocabulary(name, definition)[0]:
            raise ValueError(
                f"{name} has no content tokens in its use_cases - it is "
                "unreachable by the router. Add class nouns a user would type."
            )


def _index_vocabulary(
    name: str, definition: IndexDefinition
) -> tuple[set[str], set[tuple[str, str]]]:
    """Build the router vocabulary for one index: content unigrams plus bigrams."""
    unigrams: set[str] = set()
    bigrams: set[tuple[str, str]] = set()
    for phrase in definition.use_cases:
        sequence = _token_sequence(phrase)
        unigrams |= _content_tokens(sequence)
        bigrams |= _bigrams(sequence)
    # The index's own name is a legitimate query term ("show me NDVI").
    unigrams |= {
        _singular(part) for part in name.lower().split("_") if len(part) > 2
    }
    return unigrams, bigrams


_validate_registry()

# Precomputed once at import: select_indices stays a pure set-intersection.
_INDEX_VOCAB: dict[str, tuple[set[str], set[tuple[str, str]]]] = {
    name: _index_vocabulary(name, definition)
    for name, definition in INDEX_REGISTRY.items()
}


class IndexAnalyser:
    """Compute and classify registered indices without touching raster I/O.

    ``compute`` is O(kN) time and O(kN) output space for k requested indices and
    N pixels. ``classify`` is O(N) for the percentile plus O(N) mask space.
    ``diff`` is O(N). ``select_indices`` is O(I) set intersections over I
    registry entries against a precomputed vocabulary - negligible at 21
    entries, and deterministic.
    """

    # Router scoring. A matched adjacent pair is worth more than the two tokens
    # separately, so "water body" beats an index that merely mentions "water".
    _UNIGRAM_WEIGHT = 1.0
    _BIGRAM_WEIGHT = 3.0
    _MIN_SCORE = 1.0

    def compute(
        self, bands: dict[str, np.ndarray], names: list[str]
    ) -> dict[str, np.ndarray]:
        """Compute requested indices after validating bands, shape, and scaling."""
        if not names:
            return {}

        unknown = [name for name in names if name not in INDEX_REGISTRY]
        if unknown:
            raise KeyError(
                f"Unknown spectral index/indices: {unknown}. "
                f"Available: {sorted(INDEX_REGISTRY)}"
            )

        arrays = {
            key: np.asarray(value, dtype=np.float64) for key, value in bands.items()
        }

        shapes = {arr.shape for arr in arrays.values()}
        if len(shapes) > 1:
            raise ValueError(
                f"Band arrays must have identical shapes; found: {sorted(shapes)}"
            )

        result: dict[str, np.ndarray] = {}

        for name in dict.fromkeys(names):
            definition = INDEX_REGISTRY[name]

            missing = [
                band for band in definition.required_bands if band not in arrays
            ]
            if missing:
                raise ValueError(
                    f"{name} requires canonical bands {definition.required_bands}; "
                    f"missing {missing}. Resolve source aliases to canonical "
                    "names first."
                )

            # Scale check BEFORE computing, so we never return a silently
            # meaningless number. Normalised differences are scale-invariant and
            # skip this; power ratios and indices with absolute constants are not.
            if definition.requires_reflectance:
                probe = arrays[definition.required_bands[0]]
                finite = probe[np.isfinite(probe)]
                if finite.size and float(np.max(finite)) > REFLECTANCE_MAX:
                    raise ValueError(
                        f"{name} is not scale-invariant and requires surface "
                        f"reflectance in [0, 1]; found values up to "
                        f"{float(np.max(finite)):.1f}. Sentinel-2 L2A is stored as "
                        "DN scaled by 10000 - divide by 10000 first. "
                        "Normalised-difference indices are unaffected."
                    )

            result[name] = definition.compute(arrays)

        return result

    def classify(
        self, name: str, arr: np.ndarray, percentile: float = 90
    ) -> tuple[np.ndarray, dict]:
        """Create a scene-adaptive class mask constrained by the physical floor.

        For positive phenomena the upper percentile is used and cannot fall below
        the floor. For negative phenomena the symmetric lower percentile is used
        and cannot rise above the floor.

        If the scene never reaches the floor in the physically relevant
        direction, the class is DISABLED rather than manufacturing a class from
        the percentile distribution. This is what stops a bone-dry scene
        reporting 10% water.
        """
        if name not in INDEX_REGISTRY:
            raise KeyError(
                f"Unknown spectral index {name!r}. Available: {sorted(INDEX_REGISTRY)}"
            )

        if not 0 < percentile <= 100:
            raise ValueError("percentile must be > 0 and <= 100")

        values = np.asarray(arr, dtype=np.float64)
        finite = np.isfinite(values)
        mask = np.zeros(values.shape, dtype=bool)

        definition = INDEX_REGISTRY[name]
        floor = definition.physical_floor

        metadata: dict = {
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
                "Class disabled: scene contains no finite index values, so the "
                "physical floor cannot be evaluated."
            )
            return mask, metadata

        if definition.polarity == "positive":
            scene_extreme = float(np.nanmax(values))
            if scene_extreme < floor:
                metadata["floor_disabled"] = True
                metadata["note"] = (
                    f"Class disabled: scene maximum {scene_extreme:.6g} is below "
                    f"the physical floor {floor:.6g}."
                )
                return mask, metadata
            threshold = max(float(np.nanpercentile(values, percentile)), floor)
            mask = finite & (values >= threshold)
        else:
            scene_extreme = float(np.nanmin(values))
            if scene_extreme > floor:
                metadata["floor_disabled"] = True
                metadata["note"] = (
                    f"Class disabled: scene minimum {scene_extreme:.6g} is above "
                    f"the physical floor {floor:.6g}."
                )
                return mask, metadata
            threshold = min(float(np.nanpercentile(values, 100.0 - percentile)), floor)
            mask = finite & (values <= threshold)

        metadata["threshold"] = threshold
        metadata["fraction_selected"] = float(np.mean(mask))
        return mask, metadata

    def diff(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """Return after-before while rejecting mismatched raster shapes."""
        before_arr = np.asarray(before, dtype=np.float64)
        after_arr = np.asarray(after, dtype=np.float64)

        if before_arr.shape != after_arr.shape:
            raise ValueError(
                "before and after must have identical shapes; found "
                f"{before_arr.shape} and {after_arr.shape}."
            )

        return after_arr - before_arr

    def explain_selection(self, query: str) -> list[dict]:
        """Return the full scoring trace for a query, best first.

        Provenance for the UI and for debugging a ranking: every entry shows
        which query tokens and which adjacent pairs matched, so an index choice
        can be argued with rather than taken on trust.
        """
        sequence = _token_sequence(query)
        q_unigrams = _content_tokens(sequence)
        q_bigrams = _bigrams(sequence)

        trace: list[dict] = []
        for name, definition in INDEX_REGISTRY.items():
            unigrams, bigrams = _INDEX_VOCAB[name]
            matched_unigrams = sorted(q_unigrams & unigrams)
            matched_bigrams = sorted(" ".join(pair) for pair in q_bigrams & bigrams)
            score = (
                self._UNIGRAM_WEIGHT * len(matched_unigrams)
                + self._BIGRAM_WEIGHT * len(matched_bigrams)
            )
            if score >= self._MIN_SCORE:
                trace.append(
                    {
                        "index": name,
                        "score": score,
                        "priority": definition.priority,
                        "matched_terms": matched_unigrams,
                        "matched_phrases": matched_bigrams,
                    }
                )

        trace.sort(key=lambda row: (-row["score"], row["priority"], row["index"]))
        return trace

    def select_indices(self, query: str) -> list[str]:
        """Rank indices by whole-token overlap with router-facing use-case phrases.

        Tokens are matched as WORDS, never as substrings. The previous substring
        test let English function words reach band maths: "the" matched NDVI and
        NDRE through "whe-the-r", "to" through "moni-to-r", and "in" through
        almost everything, so "Highlight the water body referred to in the
        query." returned NDVI ahead of MNDWI. Query tokens are also deduplicated,
        so a word repeated in a sentence cannot score twice.

        Adjacent-pair matches are weighted above single tokens, so "water body"
        beats an incidental "water". A query whose every token is a stopword
        scores nothing and returns [] rather than guessing - callers MUST handle
        the empty list as "intent not understood", not index into it blindly.

        Ties break on declared ``priority`` (lower wins), then on name, so
        "burn scar" returns NBR before BAI because NBR is the standard burn
        index - and the order is stable across runs and Python versions.
        """
        sequence = _token_sequence(query)
        q_unigrams = _content_tokens(sequence)
        if not q_unigrams:
            return []
        q_bigrams = _bigrams(sequence)

        ranked: list[tuple[float, int, str]] = []
        for name, definition in INDEX_REGISTRY.items():
            unigrams, bigrams = _INDEX_VOCAB[name]
            score = (
                self._UNIGRAM_WEIGHT * len(q_unigrams & unigrams)
                + self._BIGRAM_WEIGHT * len(q_bigrams & bigrams)
            )
            if score >= self._MIN_SCORE:
                ranked.append((score, definition.priority, name))

        ranked.sort(key=lambda row: (-row[0], row[1], row[2]))
        return [name for _, _, name in ranked]


__all__ = [
    "CANONICAL_BANDS",
    "INDEX_REGISTRY",
    "STOPWORDS",
    "IndexAnalyser",
    "IndexDefinition",
]
