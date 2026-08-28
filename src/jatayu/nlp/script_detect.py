from __future__ import annotations

_SCRIPT_RANGES: tuple[tuple[int, int, str], ...] = (
    (0x0900, 0x097F, "hin_Deva"),   # Devanagari
    (0x0980, 0x09FF, "ben_Beng"),   # Bengali/Assamese
    (0x0A00, 0x0A7F, "pan_Guru"),   # Gurmukhi
    (0x0A80, 0x0AFF, "guj_Gujr"),   # Gujarati
    (0x0B00, 0x0B7F, "ory_Orya"),   # Odia
    (0x0B80, 0x0BFF, "tam_Taml"),   # Tamil
    (0x0C00, 0x0C7F, "tel_Telu"),   # Telugu
    (0x0C80, 0x0CFF, "kan_Knda"),   # Kannada
    (0x0D00, 0x0D7F, "mal_Mlym"),   # Malayalam
)

ENGLISH = "eng_Latn"


def detect_language(text: str, *, min_ratio: float = 0.15) -> str:
    """Return a FLORES-200 code for the dominant script in `text`."""
    counts: dict[str, int] = {}
    letters = 0
    for char in text:
        if not char.isalpha():
            continue
        letters += 1
        code = _script_of(ord(char))
        if code is not None:
            counts[code] = counts.get(code, 0) + 1

    if not letters or not counts:
        return ENGLISH

    code, count = max(counts.items(), key=lambda kv: kv[1])
    return code if count / letters >= min_ratio else ENGLISH


def _script_of(codepoint: int) -> str | None:
    for start, end, code in _SCRIPT_RANGES:
        if start <= codepoint <= end:
            return code
    return None
