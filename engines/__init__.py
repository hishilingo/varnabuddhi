# -*- coding: utf-8 -*-
"""
Varnabuddhi Engine Package
===========================
Modular engines for Sanskrit ↔ English translation.

Engines:
    - transliteration_engine: Offline script detection and conversion
    - lexicon_engine: Local lexical database lookup
    - verification_engine: Morphological verification
    - sandhi_engine: Sandhi joining and splitting
    - llm_engine: Multi-provider LLM translation
"""

__all__ = [
    "transliteration_engine",
    "lexicon_engine",
    "verification_engine",
    "sandhi_engine",
    "llm_engine",
]
