# -*- coding: utf-8 -*-
"""
Transliteration Engine
=======================
Offline script detection and conversion between Devanagari, IAST, and WX notation.

Uses the ``indic-transliteration`` package for Devanagari ↔ IAST.
WX conversion is handled with a built-in mapping table (needed for the
Perl-based sandhi scripts which operate in WX notation).

All operations are fully offline.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Optional

logger = logging.getLogger("varnabuddhi.transliteration")

# ---------------------------------------------------------------------------
# indic-transliteration availability
# ---------------------------------------------------------------------------
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as _transliterate

    _HAS_INDIC = True
except ImportError:
    _HAS_INDIC = False
    sanscript = None  # type: ignore
    _transliterate = None  # type: ignore

# ---------------------------------------------------------------------------
# IAST diacritical characters for heuristic script detection
# ---------------------------------------------------------------------------
IAST_DIACRITIC_CHARS = set("āīūṛṝḷḹṅñṭḍṇśṣṃṁḥĀĪŪṚṜḶḸṄÑṬḌṆŚṢṂṀḤ")

# ---------------------------------------------------------------------------
# WX ↔ IAST mapping tables
# ---------------------------------------------------------------------------
# WX is the transliteration scheme used by the sandhi Perl scripts.
_IAST_TO_WX: dict[str, str] = {
    # Vowels
    "a": "a", "ā": "A", "i": "i", "ī": "I", "u": "u", "ū": "U",
    "ṛ": "q", "ṝ": "Q", "ḷ": "L", "ḹ": "LL",
    "e": "e", "ai": "E", "o": "o", "au": "O",
    # Anusvara / Visarga / Chandrabindu
    "ṃ": "M", "ṁ": "M", "ḥ": "H",
    # Consonants — velars
    "k": "k", "kh": "K", "g": "g", "gh": "G", "ṅ": "f",
    # Consonants — palatals
    "c": "c", "ch": "C", "j": "j", "jh": "J", "ñ": "F",
    # Consonants — retroflexes
    "ṭ": "t", "ṭh": "T", "ḍ": "d", "ḍh": "D", "ṇ": "N",
    # Consonants — dentals
    "t": "w", "th": "W", "d": "x", "dh": "X", "n": "n",
    # Consonants — labials
    "p": "p", "ph": "P", "b": "b", "bh": "B", "m": "m",
    # Semi-vowels
    "y": "y", "r": "r", "l": "l", "v": "v",
    # Sibilants & aspirate
    "ś": "S", "ṣ": "R", "s": "s", "h": "h",
}

# Build the reverse map
_WX_TO_IAST: dict[str, str] = {}
for _iast_key, _wx_val in _IAST_TO_WX.items():
    # For multi-char WX values, prefer the first IAST mapping encountered
    if _wx_val not in _WX_TO_IAST:
        _WX_TO_IAST[_wx_val] = _iast_key

# Sort keys by length (longest first) so greedy matching works correctly
_IAST_TO_WX_SORTED = sorted(_IAST_TO_WX.items(), key=lambda kv: -len(kv[0]))
_WX_TO_IAST_SORTED = sorted(_WX_TO_IAST.items(), key=lambda kv: -len(kv[0]))


# ===================================================================
# Public API
# ===================================================================


def is_available() -> bool:
    """Return True if the indic-transliteration package is installed."""
    return _HAS_INDIC


def detect_script(text: str) -> str:
    """Heuristically detect whether *text* is Devanagari, IAST, or plain Latin.

    Returns one of ``"devanagari"``, ``"iast"``, or ``"latin"``.
    """
    for ch in text:
        if "\u0900" <= ch <= "\u097f":
            return "devanagari"
        if ch in IAST_DIACRITIC_CHARS:
            return "iast"
    if any(ch.isalpha() for ch in text):
        return "latin"
    return "latin"


def normalize(text: str) -> str:
    """Unicode NFC-normalize the input."""
    return unicodedata.normalize("NFC", text)


def to_iast(text: str) -> str:
    """Convert *text* to IAST.

    Devanagari is converted via ``indic-transliteration``; IAST or Latin input
    is returned as-is after normalization.
    """
    text = normalize(text)
    script = detect_script(text)
    if script == "devanagari":
        if not _HAS_INDIC:
            raise RuntimeError(
                "Devanagari → IAST conversion requires the 'indic-transliteration' "
                "package.  Install with:  pip install indic-transliteration"
            )
        return normalize(_transliterate(text, sanscript.DEVANAGARI, sanscript.IAST))
    return text


def to_devanagari(text: str) -> str:
    """Convert IAST *text* to Devanagari."""
    if not _HAS_INDIC:
        raise RuntimeError(
            "IAST → Devanagari conversion requires the 'indic-transliteration' "
            "package.  Install with:  pip install indic-transliteration"
        )
    text = normalize(text)
    return normalize(_transliterate(text, sanscript.IAST, sanscript.DEVANAGARI))


def to_wx(text: str) -> str:
    """Convert IAST *text* to WX transliteration (used by sandhi Perl scripts).

    The conversion applies a greedy left-to-right character substitution.
    """
    text = normalize(text)
    # If input is Devanagari, first convert to IAST
    if detect_script(text) == "devanagari":
        text = to_iast(text)
    result: list[str] = []
    i = 0
    while i < len(text):
        matched = False
        for iast_key, wx_val in _IAST_TO_WX_SORTED:
            if text[i: i + len(iast_key)] == iast_key:
                result.append(wx_val)
                i += len(iast_key)
                matched = True
                break
        if not matched:
            result.append(text[i])
            i += 1
    return "".join(result)


def from_wx(text: str) -> str:
    """Convert WX-transliterated *text* back to IAST."""
    result: list[str] = []
    i = 0
    while i < len(text):
        matched = False
        for wx_key, iast_val in _WX_TO_IAST_SORTED:
            if text[i: i + len(wx_key)] == wx_key:
                result.append(iast_val)
                i += len(wx_key)
                matched = True
                break
        if not matched:
            result.append(text[i])
            i += 1
    return "".join(result)


def ensure_iast(text: str) -> str:
    """Normalize *text* to IAST regardless of input script.

    Convenience wrapper: detects the input script and converts as needed.
    """
    script = detect_script(normalize(text))
    if script == "devanagari":
        return to_iast(text)
    return normalize(text)


def auto_transliterate(text: str, target: str = "auto") -> str:
    """Transliterate *text* to the *target* script.

    Parameters
    ----------
    target : str
        ``"iast"``, ``"devanagari"``, ``"wx"``, or ``"auto"``
        (auto mirrors detected input).
    """
    text = normalize(text)
    if target == "auto":
        return text
    if target == "iast":
        return to_iast(text)
    if target == "devanagari":
        return to_devanagari(text)
    if target == "wx":
        return to_wx(text)
    logger.warning("Unknown target script '%s'; returning input unchanged.", target)
    return text
