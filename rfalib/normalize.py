"""Arabic normalization rule (plan section 6.7).

Used ONLY for comparisons (uniqueness checks, truth-vs-label agreement,
blocklist screening, duplicate-pool detection).  NEVER rewrites stored values.

Steps (exact, in order):
  1. Unicode NFC
  2. strip combining marks (unicodedata.combining)
  3. remove tatweel U+0640
  4. remove U+200B U+200C U+200E U+200F U+FEFF
  5. collapse whitespace, strip

Never touched (they distinguish real names): alef variants (ا أ إ آ),
ي vs ى, ة vs ه  ->  علي != على is test-enforced.
"""
from __future__ import annotations

import unicodedata

_STRIP_CHARS = set("\u200b\u200c\u200e\u200f\ufeff")  # ZWSP ZWNJ LRM RLM BOM
_TATWEEL = "\u0640"


def normalize(text: str) -> str:
    """Apply the section 6.7 normalization rule; returns a comparison key."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFC", str(text))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace(_TATWEEL, "")
    s = "".join(ch for ch in s if ch not in _STRIP_CHARS)
    s = " ".join(s.split())
    return s.strip()


def same(a: str, b: str) -> bool:
    """Normalized equality - the ONLY sanctioned way to compare Arabic values."""
    return normalize(a) == normalize(b)
