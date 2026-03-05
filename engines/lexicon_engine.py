# -*- coding: utf-8 -*-
"""
Lexicon Engine
===============
Deterministic Sanskrit → English lookup from local data files.

Parses all ``data/*.txt`` files at startup, building dictionaries keyed by
normalized Sanskrit text (Devanagari and IAST).  Supports exact match for
words, phrases, and complete sentences.  When no full match is found,
individual word-level matches are returned so the caller can merge them with
LLM output.
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
from engines.cache_db import LexiconCache

logger = logging.getLogger("varnabuddhi.lexicon")


@dataclass
class LexiconEntry:
    """A single Sanskrit ↔ English pair from the local data files."""

    sanskrit: str  # Original Sanskrit text (as stored)
    english: str  # English translation
    source_file: str  # Filename the entry was loaded from
    entry_id: str = ""  # e.g. "[42] BhG.htm (BhG.2.3)"
    sanskrit_iast: str = ""  # IAST-normalized form (populated post-init)


@dataclass
class LookupResult:
    """Result of a lexicon query."""

    exact_match: Optional[LexiconEntry] = None
    word_matches: Dict[str, List[LexiconEntry]] = field(default_factory=dict)

    @property
    def has_exact(self) -> bool:
        return self.exact_match is not None

    @property
    def has_partial(self) -> bool:
        return len(self.word_matches) > 0


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
_ENTRY_HEADER = re.compile(r"^\[(\d+)\]\s*(.+)$")
_SANSKRIT_LINE = re.compile(r"^SANSKRIT:\s*(.+)$")
_ENGLISH_LINE = re.compile(r"^ENGLISH:\s*(.+)$")


def _normalize_key(text: str) -> str:
    """NFC-normalize and strip trailing dandas for Sanskrit key comparison."""
    text = unicodedata.normalize("NFC", text).strip()
    # Remove common trailing/leading punctuation that shouldn't affect matching
    text = re.sub(r"[।॥\u0964\u0965|]+$", "", text).strip()
    return text


def _normalize_eng_key(text: str) -> str:
    """Aggressively normalize English text for reverse-lookup matching.

    Strips punctuation, collapses whitespace, lowercases.
    """
    text = unicodedata.normalize("NFC", text).strip()
    # Remove all punctuation except apostrophes inside words
    text = re.sub(r"[^\w\s']", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


class LexiconEngine:
    """Load and query the local Sanskrit–English lexicon."""

    def __init__(self, data_dir: str | Path, cache_dir: str | Path = "") -> None:
        self.data_dir = Path(data_dir)
        # Primary lookup: normalized Sanskrit → list of entries
        self._by_sanskrit: Dict[str, List[LexiconEntry]] = {}
        # Word-level lookup: single normalized word → list of entries
        self._by_word: Dict[str, List[LexiconEntry]] = {}
        # Reverse lookup: normalized English → list of entries
        self._by_english: Dict[str, List[LexiconEntry]] = {}
        self._loaded = False
        # SQLite + FTS5 cache
        self._cache: Optional[LexiconCache] = None
        self._using_cache = False
        if cache_dir:
            self._cache = LexiconCache(cache_dir)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self) -> int:
        """Parse all ``*.txt`` files in *data_dir* and populate indexes.

        Returns the total number of entries loaded.
        """
        if self._loaded:
            return sum(len(v) for v in self._by_sanskrit.values())

        if not self.data_dir.exists():
            logger.warning("Data directory '%s' does not exist.", self.data_dir)
            return 0

        # --- Try SQLite cache first ---
        if self._cache is not None:
            print("    Checking lexicon cache...", flush=True)
            source_files = list(self.data_dir.glob("*.txt"))
            if self._cache.is_fresh(source_files):
                n = self._cache.count()
                logger.info("Lexicon loading from SQLite cache (%d entries).", n)
                self._using_cache = True
                self._loaded = True
                return n
            else:
                print("    Lexicon cache stale or missing; rebuilding...", flush=True)
                logger.info("Lexicon cache is stale or missing; rebuilding...")

        # --- Parse text files (cold path) ---
        total = 0
        txt_files = sorted(self.data_dir.glob("*.txt"))
        for i, txt_file in enumerate(txt_files, 1):
            print(f"    Parsing lexicon file [{i}/{len(txt_files)}]: {txt_file.name}", flush=True)
            count = self._parse_file(txt_file)
            logger.info("Loaded %d entries from %s", count, txt_file.name)
            total += count

        # --- Store into SQLite cache for next time ---
        if self._cache is not None:
            self._store_to_cache()

        self._loaded = True
        logger.info("Lexicon loaded: %d total entries.", total)
        return total

    def _parse_file(self, path: Path) -> int:
        """Parse a single data file and return the number of entries added."""
        count = 0
        entry_id = ""
        sanskrit_text = ""
        english_text = ""
        in_english = False

        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            logger.error("Cannot read %s: %s", path, exc)
            return 0

        def _flush() -> None:
            nonlocal count, entry_id, sanskrit_text, english_text, in_english
            if sanskrit_text and english_text:
                self._add_entry(
                    LexiconEntry(
                        sanskrit=sanskrit_text.strip(),
                        english=english_text.strip(),
                        source_file=path.name,
                        entry_id=entry_id,
                    )
                )
                count += 1
            entry_id = ""
            sanskrit_text = ""
            english_text = ""
            in_english = False

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                # Blank line may signal end of an entry block
                if sanskrit_text and english_text:
                    _flush()
                continue

            hdr = _ENTRY_HEADER.match(line_stripped)
            if hdr:
                # New entry header — flush any pending entry
                _flush()
                entry_id = line_stripped
                continue

            san = _SANSKRIT_LINE.match(line_stripped)
            if san:
                sanskrit_text = san.group(1)
                in_english = False
                continue

            eng = _ENGLISH_LINE.match(line_stripped)
            if eng:
                english_text = eng.group(1)
                in_english = True
                continue

            # Continuation line (multi-line English text)
            if in_english and english_text:
                english_text += " " + line_stripped

        # Flush trailing entry
        _flush()
        return count

    def _add_entry(self, entry: LexiconEntry) -> None:
        """Index a single entry in all lookup tables."""
        # Normalize the Sanskrit key
        san_key = _normalize_key(entry.sanskrit)

        # Try to get IAST form
        try:
            iast_form = te.ensure_iast(san_key)
            entry.sanskrit_iast = iast_form
        except Exception:
            entry.sanskrit_iast = san_key
            iast_form = san_key

        # Add to primary lookup (by full Sanskrit text)
        for key in {san_key, iast_form.lower(), san_key.lower()}:
            if key:
                self._by_sanskrit.setdefault(key, []).append(entry)

        # Add to word-level lookup
        for word in re.split(r"[\s\-–—,;।॥]+", san_key):
            word = word.strip()
            if len(word) >= 2:
                self._by_word.setdefault(word, []).append(entry)
                word_lower = word.lower()
                if word_lower != word:
                    self._by_word.setdefault(word_lower, []).append(entry)
                # Also index the IAST form of the word
                try:
                    iast_word = te.ensure_iast(word)
                    if iast_word != word:
                        self._by_word.setdefault(iast_word, []).append(entry)
                        self._by_word.setdefault(iast_word.lower(), []).append(entry)
                except Exception:
                    pass

        # Add to reverse English lookup (both raw-normalized and aggressively normalized)
        eng_key = _normalize_key(entry.english).lower()
        if eng_key:
            self._by_english.setdefault(eng_key, []).append(entry)
        eng_key_clean = _normalize_eng_key(entry.english)
        if eng_key_clean and eng_key_clean != eng_key:
            self._by_english.setdefault(eng_key_clean, []).append(entry)

    def _store_to_cache(self) -> None:
        """Persist all in-memory entries to SQLite + FTS5."""
        if self._cache is None:
            return
        seen: Set[int] = set()
        cache_entries: List[Dict[str, str]] = []
        for entries in self._by_sanskrit.values():
            for e in entries:
                eid = id(e)
                if eid in seen:
                    continue
                seen.add(eid)
                cache_entries.append({
                    "sanskrit_iast": e.sanskrit_iast or e.sanskrit,
                    "sanskrit_original": e.sanskrit,
                    "english": e.english,
                    "english_lower": _normalize_eng_key(e.english),
                    "source_file": e.source_file,
                    "entry_id": e.entry_id,
                })
        self._cache.store_entries(cache_entries)
        self._cache.finalize()
        logger.info("Lexicon SQLite cache built (%d entries).", len(cache_entries))

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------
    def lookup(self, sanskrit_input: str) -> LookupResult:
        """Look up *sanskrit_input* in the lexicon.

        First attempts an exact match (full text).  If none is found,
        splits into words and returns word-level matches.
        Falls back to ChromaDB exact match when in-memory index misses.
        """
        result = LookupResult()
        if not self._loaded:
            self.load()

        normalized = _normalize_key(sanskrit_input)
        iast = ""
        try:
            iast = te.ensure_iast(normalized)
        except Exception:
            iast = normalized

        # --- Exact match (in-memory) ---
        for key in (normalized, iast, normalized.lower(), iast.lower()):
            entries = self._by_sanskrit.get(key)
            if entries:
                result.exact_match = entries[0]
                return result

        # --- Exact match (SQLite cache fallback) ---
        if self._cache is not None:
            for key in (iast.lower(), normalized.lower()):
                hits = self._cache.exact_lookup(key)
                if hits:
                    h = hits[0]
                    result.exact_match = LexiconEntry(
                        sanskrit=h.get("sanskrit_original", key),
                        english=h.get("english", ""),
                        source_file=h.get("source_file", "cache"),
                        entry_id=h.get("entry_id", ""),
                        sanskrit_iast=h.get("sanskrit_iast", key),
                    )
                    return result

        # --- Word-level matches ---
        words = re.split(r"[\s\-–—,;।॥]+", normalized)
        for word in words:
            word = word.strip()
            if len(word) < 2:
                continue
            for variant in (word, word.lower()):
                matches = self._by_word.get(variant)
                if matches:
                    result.word_matches[word] = matches[:5]  # Limit per word
                    break
            else:
                # Try IAST variant
                try:
                    iast_word = te.ensure_iast(word)
                    for variant in (iast_word, iast_word.lower()):
                        matches = self._by_word.get(variant)
                        if matches:
                            result.word_matches[word] = matches[:5]
                            break
                except Exception:
                    pass

        return result

    def reverse_lookup(self, english_input: str) -> List[LexiconEntry]:
        """Look up an English term and return matching Sanskrit entries."""
        if not self._loaded:
            self.load()

        # Try exact match (in-memory) with both normalizations
        for key in (_normalize_key(english_input).lower(),
                    _normalize_eng_key(english_input)):
            hits = self._by_english.get(key, [])
            if hits:
                return hits

        # SQLite exact English match (works in cache-only mode)
        if self._cache is not None:
            for key in (_normalize_eng_key(english_input),
                        _normalize_key(english_input).lower()):
                try:
                    db_hits = self._cache.exact_english_lookup(key)
                except Exception:
                    db_hits = []
                if db_hits:
                    return [
                        LexiconEntry(
                            sanskrit=h.get("sanskrit_original", h.get("sanskrit_iast", "")),
                            english=h.get("english", ""),
                            source_file=h.get("source_file", "cache"),
                            entry_id=h.get("entry_id", ""),
                            sanskrit_iast=h.get("sanskrit_iast", ""),
                        )
                        for h in db_hits
                    ]

        # FTS5 fallback — fuzzy match across English and Sanskrit text
        if self._cache is not None:
            fts_hits = self._cache.fts_search(english_input, n_results=3)
            return [
                LexiconEntry(
                    sanskrit=h.get("sanskrit_original", h.get("sanskrit_iast", "")),
                    english=h.get("english", ""),
                    source_file=h.get("source_file", "cache"),
                    entry_id=h.get("entry_id", ""),
                    sanskrit_iast=h.get("sanskrit_iast", ""),
                )
                for h in fts_hits
            ]

        return []

    def reverse_lookup_words(self, english_text: str) -> Dict[str, LexiconEntry]:
        """Word-level reverse lookup: split English into n-grams and find
        Sanskrit equivalents for individual words/phrases.

        Returns a dict of {english_phrase: LexiconEntry} for all hits.
        Tries trigrams first, then bigrams, then single words (greedy).
        """
        if not self._loaded:
            self.load()

        words = english_text.split()
        if not words:
            return {}

        found: Dict[str, LexiconEntry] = {}
        consumed: set[int] = set()  # indices of words already matched

        # Try n-grams from largest to smallest
        for n in (3, 2, 1):
            for i in range(len(words) - n + 1):
                # Skip if any word in this n-gram is already consumed
                if any(j in consumed for j in range(i, i + n)):
                    continue
                phrase = " ".join(words[i:i + n])
                hits = self.reverse_lookup(phrase)
                if hits:
                    found[phrase] = hits[0]
                    for j in range(i, i + n):
                        consumed.add(j)

        return found

    def semantic_lookup(
        self, query: str, n_results: int = 5
    ) -> List[LexiconEntry]:
        """Fuzzy search via SQLite FTS5 full-text index.

        Returns a list of LexiconEntry sorted by relevance.
        """
        if self._cache is None:
            return []
        hits = self._cache.fts_search(query, n_results=n_results)
        return [
            LexiconEntry(
                sanskrit=h.get("sanskrit_original", h.get("sanskrit_iast", "")),
                english=h.get("english", ""),
                source_file=h.get("source_file", "cache"),
                entry_id=h.get("entry_id", ""),
                sanskrit_iast=h.get("sanskrit_iast", ""),
            )
            for h in hits
        ]

    @property
    def entry_count(self) -> int:
        """Total number of unique entries in the lexicon."""
        if self._using_cache and self._cache is not None:
            return self._cache.count()
        seen: Set[int] = set()
        for entries in self._by_sanskrit.values():
            for e in entries:
                seen.add(id(e))
        return len(seen)
