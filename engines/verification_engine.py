# -*- coding: utf-8 -*-
"""
Verification Engine
====================
Morphological verification of Sanskrit word forms.

Parses the following reference files in the ``decls/`` directory:

* **declensions.txt** — Nominal declension paradigms (Devanagari + IAST).
* **verbs.txt** — Verb conjugation paradigms (IAST).
* **mwse72.dict.txt** — Monier-Williams Sanskrit–English dictionary headwords.

Capabilities:

* Detect whether a word form exists in the known paradigms.
* Tag its grammatical category (case/number/gender or tense/person/voice).
* Provide warnings for uncertain forms.
* Support transliteration normalization on retry.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from engines import transliteration_engine as te
from engines.cache_db import MorphCache

logger = logging.getLogger("varnabuddhi.verification")


@dataclass
class GrammarTag:
    """Grammatical annotation for a verified word form."""

    form: str  # The word form as found
    root: str = ""  # The root / lemma
    category: str = ""  # "noun", "verb", "dict_headword"
    details: str = ""  # e.g. "Nom. Sg (m)" or "3rd Person Sg (Present) Parasmaipada"
    gender: str = ""  # m / f / n  (nouns only)


@dataclass
class VerificationResult:
    """Result of verifying one or more Sanskrit words."""

    word: str
    verified: bool = False
    tags: List[GrammarTag] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing regexes
# ---------------------------------------------------------------------------
_DECL_ROOT_RE = re.compile(r"^Root:\s*of(.+)$")
_DECL_FORM_RE = re.compile(r"^((?:Nom|Voc|Acc|Ins|Dat|Abl|Gen|Loc)\.?\s*(?:Sg|Dual|Pl)):\s*(.+)$")
_VERB_ROOT_RE = re.compile(r"^Root:\s*(.+)$")
_VERB_FORM_RE = re.compile(r"^(.+?):\s*(\S+)$")
_DICT_ENTRY_RE = re.compile(r"<start><b>([^<]+)</b>\s*(.+?)</start>", re.DOTALL)


def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def _extract_iast_from_mixed(text: str) -> str:
    """Given a mixed Devanagari+IAST string like 'अकारःakāraḥ', extract the IAST part."""
    # Find where Latin/IAST characters start
    iast_chars: list[str] = []
    for ch in text:
        if ch.isascii() or ch in te.IAST_DIACRITIC_CHARS:
            iast_chars.append(ch)
    return _norm("".join(iast_chars))


def _extract_gender(root_text: str) -> str:
    """Extract gender marker from root text like 'akāra (m)'."""
    m = re.search(r"\(([mfn])\)\s*$", root_text)
    return m.group(1) if m else ""


class VerificationEngine:
    """Load declension/verb paradigms and verify Sanskrit word forms."""

    def __init__(self, decls_dir: str | Path, cache_dir: str | Path = "") -> None:
        self.decls_dir = Path(decls_dir)
        # IAST form → list of GrammarTag
        self._known_forms: Dict[str, List[GrammarTag]] = {}
        # Dictionary headwords (IAST, lowercase) → definition snippet
        self._dict_headwords: Dict[str, str] = {}
        self._loaded = False
        # SQLite cache
        self._cache: Optional[MorphCache] = None
        self._using_cache = False
        if cache_dir:
            self._cache = MorphCache(cache_dir)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self, *, load_declensions: bool = True,
             load_verbs: bool = True,
             load_dictionary: bool = True) -> None:
        """Parse reference files and build the verification index."""
        if self._loaded:
            return

        # --- Try SQLite cache first ---
        if self._cache is not None:
            source_files = [
                self.decls_dir / "declensions.txt",
                self.decls_dir / "verbs.txt",
                self.decls_dir / "mwse72.dict.txt",
            ]
            if self._cache.is_fresh(source_files):
                n_forms = self._cache.count_forms()
                n_dict = self._cache.count_dict()
                logger.info(
                    "Loading from SQLite cache (%d forms, %d headwords).",
                    n_forms, n_dict,
                )
                self._using_cache = True
                self._loaded = True
                return
            else:
                logger.info("SQLite cache is stale or missing; rebuilding...")

        # --- Parse text files (cold path) ---
        if load_declensions:
            self._load_declensions()
        if load_verbs:
            self._load_verbs()
        if load_dictionary:
            self._load_dictionary()

        # --- Store into SQLite cache for next time ---
        if self._cache is not None:
            self._store_to_cache()

        self._loaded = True
        logger.info(
            "Verification engine loaded: %d known forms, %d dict headwords.",
            len(self._known_forms), len(self._dict_headwords),
        )

    def _load_declensions(self) -> None:
        path = self.decls_dir / "declensions.txt"
        if not path.exists():
            logger.warning("declensions.txt not found in %s", self.decls_dir)
            return
        file_mb = path.stat().st_size / (1024 * 1024)
        print(f"    Parsing declensions.txt ({file_mb:.0f} MB) — this will take a few minutes on first run...", flush=True)
        logger.info("Loading declensions (%.0f MB)...", file_mb)
        current_root = ""
        current_gender = ""
        count = 0
        line_no = 0
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line_no += 1
                    if line_no % 500_000 == 0:
                        print(f"    ... {line_no:,} lines read, {count:,} forms so far", end="\r", flush=True)
                    line = line.rstrip("\n\r")
                    if not line:
                        continue
                    rm = _DECL_ROOT_RE.match(line)
                    if rm:
                        root_text = rm.group(1).strip()
                        current_root = _extract_iast_from_mixed(root_text)
                        current_gender = _extract_gender(root_text)
                        continue
                    fm = _DECL_FORM_RE.match(line)
                    if fm and current_root:
                        case_label = fm.group(1).strip()
                        form_text = fm.group(2).strip()
                        iast_form = _extract_iast_from_mixed(form_text)
                        if iast_form:
                            tag = GrammarTag(
                                form=iast_form,
                                root=current_root,
                                category="noun",
                                details=case_label,
                                gender=current_gender,
                            )
                            self._known_forms.setdefault(iast_form.lower(), []).append(tag)
                            count += 1
        except OSError as exc:
            logger.error("Cannot read declensions.txt: %s", exc)
        print(f"    Declensions: {count:,} forms from {line_no:,} lines          ", flush=True)
        logger.info("Loaded %d declension forms.", count)

    def _load_verbs(self) -> None:
        path = self.decls_dir / "verbs.txt"
        if not path.exists():
            logger.warning("verbs.txt not found in %s", self.decls_dir)
            return
        logger.info("Loading verb conjugations...")
        current_root = ""
        count = 0
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.rstrip("\n\r")
                    if not line:
                        continue
                    rm = _VERB_ROOT_RE.match(line)
                    if rm:
                        current_root = _norm(rm.group(1))
                        continue
                    fm = _VERB_FORM_RE.match(line)
                    if fm and current_root:
                        label = fm.group(1).strip()
                        form = _norm(fm.group(2))
                        if form:
                            tag = GrammarTag(
                                form=form,
                                root=current_root,
                                category="verb",
                                details=label,
                            )
                            self._known_forms.setdefault(form.lower(), []).append(tag)
                            count += 1
        except OSError as exc:
            logger.error("Cannot read verbs.txt: %s", exc)
        logger.info("Loaded %d verb forms.", count)

    def _load_dictionary(self) -> None:
        path = self.decls_dir / "mwse72.dict.txt"
        if not path.exists():
            logger.warning("mwse72.dict.txt not found in %s", self.decls_dir)
            return
        logger.info("Loading Monier-Williams dictionary headwords...")
        count = 0
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            for m in _DICT_ENTRY_RE.finditer(content):
                headword = _norm(m.group(1))
                defn_snippet = m.group(2).strip()[:200]
                self._dict_headwords[headword.lower()] = defn_snippet
                count += 1
        except OSError as exc:
            logger.error("Cannot read mwse72.dict.txt: %s", exc)
        logger.info("Loaded %d dictionary headwords.", count)

    def _store_to_cache(self) -> None:
        """Persist in-memory forms and dict headwords to SQLite."""
        if self._cache is None:
            return
        print("    Writing SQLite cache...", flush=True)
        # Forms
        rows: List[tuple] = []
        for form_lower, tags in self._known_forms.items():
            for tag in tags:
                rows.append((
                    form_lower, tag.form, tag.root,
                    tag.category, tag.details, tag.gender,
                ))
        self._cache.store_forms(rows)
        # Dictionary
        dict_rows = list(self._dict_headwords.items())
        self._cache.store_dict(dict_rows)
        self._cache.finalize()
        print(f"    SQLite cache built: {len(rows):,} forms, {len(dict_rows):,} headwords", flush=True)
        logger.info("SQLite cache built (%d forms, %d headwords).",
                    len(rows), len(dict_rows))

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    def verify(self, word: str) -> VerificationResult:
        """Verify a single Sanskrit word form.

        Checks declension/verb paradigms and the MW dictionary.
        On failure, retries with transliteration normalization.
        """
        if not self._loaded:
            self.load()

        word = _norm(word)
        result = VerificationResult(word=word)

        # Try IAST normalization
        try:
            iast_word = te.ensure_iast(word)
        except Exception:
            iast_word = word

        # Check known forms (in-memory dict or SQLite cache)
        for candidate in (iast_word.lower(), word.lower()):
            tags = self._lookup_forms(candidate)
            if tags:
                result.verified = True
                result.tags.extend(tags[:10])
                return result

        # Check dictionary headwords
        for candidate in (iast_word.lower(), word.lower()):
            defn = self._lookup_headword(candidate)
            if defn is not None:
                result.verified = True
                result.tags.append(GrammarTag(
                    form=word,
                    root=candidate,
                    category="dict_headword",
                    details=defn[:100],
                ))
                return result

        # Retry with transliteration normalization
        if te.is_available() and te.detect_script(word) == "devanagari":
            try:
                iast_retry = te.to_iast(word)
                for candidate in (iast_retry.lower(),):
                    tags = self._lookup_forms(candidate)
                    if tags:
                        result.verified = True
                        result.tags.extend(tags[:10])
                        return result
                    defn = self._lookup_headword(candidate)
                    if defn is not None:
                        result.verified = True
                        result.tags.append(GrammarTag(
                            form=word,
                            root=candidate,
                            category="dict_headword",
                            details=defn[:100],
                        ))
                        return result
            except Exception as exc:
                result.warnings.append(f"Transliteration retry failed: {exc}")

        result.warnings.append(f"Form '{word}' not found in declension/verb/dictionary data.")
        logger.debug("Unverified form: %s", word)
        return result

    def verify_text(self, text: str) -> List[VerificationResult]:
        """Verify all words in a Sanskrit text."""
        words = re.split(r"[\s\-–—,;।॥]+", _norm(text))
        results = []
        for w in words:
            w = w.strip()
            if len(w) < 2:
                continue
            results.append(self.verify(w))
        return results

    # ------------------------------------------------------------------
    # Unified lookup helpers (in-memory or SQLite)
    # ------------------------------------------------------------------
    def _lookup_forms(self, candidate: str) -> List[GrammarTag]:
        """Look up grammar tags by lowercased form, from memory or cache."""
        if self._using_cache and self._cache is not None:
            rows = self._cache.get_forms(candidate)
            return [
                GrammarTag(form=r[0], root=r[1], category=r[2],
                           details=r[3], gender=r[4])
                for r in rows
            ]
        return self._known_forms.get(candidate, [])

    def _lookup_headword(self, candidate: str) -> Optional[str]:
        """Look up a dictionary headword, from memory or cache."""
        if self._using_cache and self._cache is not None:
            return self._cache.get_headword(candidate)
        return self._dict_headwords.get(candidate)

    def lookup_headword(self, word: str) -> Optional[str]:
        """Return the MW dictionary definition for a headword, if found."""
        if not self._loaded:
            self.load()
        word = _norm(word).lower()
        return self._lookup_headword(word)
