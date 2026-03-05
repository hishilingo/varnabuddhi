# -*- coding: utf-8 -*-
"""
Cache Database Layer
=====================
Persistent caching for Varnabuddhi's language data to eliminate slow
text-file parsing on every startup.

* **MorphCache** (SQLite) — morphological data (declensions, verbs, MW dict).
* **LexiconCache** (SQLite + FTS5) — Sanskrit ↔ English lexicon with
  full-text search for fuzzy matching.  Zero external dependencies.

Both backends detect stale caches by comparing source-file modification
times and rebuild automatically when the underlying data changes.

Cache files are stored under ``<project_root>/.cache/``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("varnabuddhi.cache")


# ===================================================================
# SQLite cache — morphological data (verification engine)
# ===================================================================


class MorphCache:
    """SQLite-backed cache for declension / verb / dictionary data.

    Schema
    ------
    forms:   (form_lower TEXT, form TEXT, root TEXT, category TEXT,
              details TEXT, gender TEXT)
    dict:    (headword TEXT PRIMARY KEY, definition TEXT)
    meta:    (key TEXT PRIMARY KEY, value TEXT)
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "morphology.db"
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Freshness check
    # ------------------------------------------------------------------
    def is_fresh(self, source_files: List[Path]) -> bool:
        """Return True if the cache exists and is newer than all *source_files*."""
        if not self.db_path.exists():
            return False
        try:
            conn = self._connect()
            row = conn.execute(
                "SELECT value FROM meta WHERE key = 'built_at'"
            ).fetchone()
            if not row:
                return False
            built_at = float(row[0])
        except Exception:
            return False
        for src in source_files:
            if src.exists() and src.stat().st_mtime > built_at:
                return False
        return True

    # ------------------------------------------------------------------
    # Schema creation
    # ------------------------------------------------------------------
    def _create_tables(self, conn: sqlite3.Connection) -> None:
        conn.executescript("""
            DROP TABLE IF EXISTS forms;
            DROP TABLE IF EXISTS dict;
            DROP TABLE IF EXISTS meta;

            CREATE TABLE forms (
                form_lower TEXT NOT NULL,
                form       TEXT NOT NULL,
                root       TEXT NOT NULL,
                category   TEXT NOT NULL,
                details    TEXT NOT NULL DEFAULT '',
                gender     TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE dict (
                headword   TEXT PRIMARY KEY,
                definition TEXT NOT NULL
            );

            CREATE TABLE meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE INDEX idx_forms_lower ON forms(form_lower);
        """)

    # ------------------------------------------------------------------
    # Bulk write
    # ------------------------------------------------------------------
    def store_forms(
        self,
        forms: List[Tuple[str, str, str, str, str, str]],
    ) -> None:
        """Store a batch of (form_lower, form, root, category, details, gender)."""
        conn = self._connect()
        self._create_tables(conn)
        conn.executemany(
            "INSERT INTO forms VALUES (?, ?, ?, ?, ?, ?)", forms
        )
        conn.commit()
        logger.info("Stored %d forms in SQLite cache.", len(forms))

    def store_dict(self, entries: List[Tuple[str, str]]) -> None:
        """Store (headword, definition) pairs."""
        conn = self._connect()
        conn.executemany(
            "INSERT OR REPLACE INTO dict VALUES (?, ?)", entries
        )
        conn.commit()
        logger.info("Stored %d dict headwords in SQLite cache.", len(entries))

    def finalize(self) -> None:
        """Write build timestamp and optimize."""
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO meta VALUES ('built_at', ?)",
            (str(time.time()),),
        )
        conn.execute("ANALYZE")
        conn.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def get_forms(self, form_lower: str) -> List[Tuple[str, str, str, str, str]]:
        """Return list of (form, root, category, details, gender) for *form_lower*."""
        conn = self._connect()
        return conn.execute(
            "SELECT form, root, category, details, gender "
            "FROM forms WHERE form_lower = ?",
            (form_lower,),
        ).fetchall()

    def get_headword(self, headword: str) -> Optional[str]:
        """Return the definition for *headword*, or None."""
        conn = self._connect()
        row = conn.execute(
            "SELECT definition FROM dict WHERE headword = ?",
            (headword,),
        ).fetchone()
        return row[0] if row else None

    def count_forms(self) -> int:
        conn = self._connect()
        row = conn.execute("SELECT COUNT(*) FROM forms").fetchone()
        return row[0] if row else 0

    def count_dict(self) -> int:
        conn = self._connect()
        row = conn.execute("SELECT COUNT(*) FROM dict").fetchone()
        return row[0] if row else 0


# ===================================================================
# SQLite + FTS5 cache — lexicon data (lexicon engine)
# ===================================================================


# Cache schema version — bump when columns/indexes change so stale caches
# are automatically rebuilt on the next startup.
_LEXICON_CACHE_VERSION = "2"


class LexiconCache:
    """SQLite-backed persistent store for Sanskrit ↔ English lexicon.

    Uses a regular table for exact lookups (indexed on ``sanskrit_lower``
    and ``english_lower``) and an FTS5 virtual table for fuzzy /
    partial-word search across both Sanskrit and English text.

    Schema
    ------
    lexicon:       (id INTEGER PK, sanskrit_lower, sanskrit_iast,
                    sanskrit_original, english, english_lower,
                    source_file, entry_id)
    lexicon_fts:   FTS5 virtual table on (sanskrit_iast, english)
    meta:          (key TEXT PK, value TEXT)
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "lexicon.db"
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Freshness check
    # ------------------------------------------------------------------
    def is_fresh(self, source_files: List[Path]) -> bool:
        """Return True if the cache exists, has the current schema version,
        and is newer than all *source_files*."""
        if not self.db_path.exists():
            return False
        try:
            conn = self._connect()
            # Check schema version
            ver_row = conn.execute(
                "SELECT value FROM meta WHERE key = 'cache_version'"
            ).fetchone()
            if not ver_row or ver_row[0] != _LEXICON_CACHE_VERSION:
                logger.info("Lexicon cache version mismatch → rebuilding.")
                return False
            row = conn.execute(
                "SELECT value FROM meta WHERE key = 'built_at'"
            ).fetchone()
            if not row:
                return False
            built_at = float(row[0])
        except Exception:
            return False
        for src in source_files:
            if src.exists() and src.stat().st_mtime > built_at:
                return False
        return True

    # ------------------------------------------------------------------
    # Schema creation
    # ------------------------------------------------------------------
    def _create_tables(self, conn: sqlite3.Connection) -> None:
        conn.executescript("""
            DROP TABLE IF EXISTS lexicon;
            DROP TABLE IF EXISTS lexicon_fts;
            DROP TABLE IF EXISTS meta;

            CREATE TABLE lexicon (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                sanskrit_lower    TEXT NOT NULL,
                sanskrit_iast     TEXT NOT NULL,
                sanskrit_original TEXT NOT NULL,
                english           TEXT NOT NULL,
                english_lower     TEXT NOT NULL DEFAULT '',
                source_file       TEXT NOT NULL DEFAULT '',
                entry_id          TEXT NOT NULL DEFAULT ''
            );

            CREATE INDEX idx_lex_san_lower ON lexicon(sanskrit_lower);
            CREATE INDEX idx_lex_eng_lower ON lexicon(english_lower);

            CREATE VIRTUAL TABLE lexicon_fts USING fts5(
                sanskrit_iast,
                english,
                content='lexicon',
                content_rowid='id'
            );

            CREATE TABLE meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)

    # ------------------------------------------------------------------
    # Bulk write
    # ------------------------------------------------------------------
    def store_entries(
        self,
        entries: List[Dict[str, str]],
    ) -> None:
        """Store lexicon entries into SQLite + FTS5.

        Each entry dict must have: sanskrit_iast, english, source_file,
        entry_id, sanskrit_original.
        """
        conn = self._connect()
        self._create_tables(conn)

        total = len(entries)
        print(f"    Writing {total:,} lexicon entries to SQLite...", flush=True)

        rows = [
            (
                e["sanskrit_iast"].lower(),
                e["sanskrit_iast"],
                e["sanskrit_original"],
                e["english"],
                e.get("english_lower", e["english"].lower()),
                e.get("source_file", ""),
                e.get("entry_id", ""),
            )
            for e in entries
        ]
        conn.executemany(
            "INSERT INTO lexicon "
            "(sanskrit_lower, sanskrit_iast, sanskrit_original, english, "
            " english_lower, source_file, entry_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        # Populate FTS index
        conn.execute(
            "INSERT INTO lexicon_fts(rowid, sanskrit_iast, english) "
            "SELECT id, sanskrit_iast, english FROM lexicon"
        )
        conn.commit()
        print(f"    Lexicon SQLite cache built: {total:,} entries", flush=True)
        logger.info("Stored %d lexicon entries in SQLite + FTS5.", total)

    def finalize(self) -> None:
        """Write build timestamp, cache version, and optimize."""
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO meta VALUES ('built_at', ?)",
            (str(time.time()),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta VALUES ('cache_version', ?)",
            (_LEXICON_CACHE_VERSION,),
        )
        conn.execute("ANALYZE")
        conn.commit()

    # ------------------------------------------------------------------
    # Query — exact
    # ------------------------------------------------------------------
    def exact_lookup(self, sanskrit_key: str) -> List[Dict[str, str]]:
        """Exact match on the lowercased Sanskrit key."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT sanskrit_iast, sanskrit_original, english, source_file, entry_id "
            "FROM lexicon WHERE sanskrit_lower = ? LIMIT 10",
            (sanskrit_key.lower(),),
        ).fetchall()
        return [
            {
                "sanskrit_iast": r[0],
                "sanskrit_original": r[1],
                "english": r[2],
                "source_file": r[3],
                "entry_id": r[4],
            }
            for r in rows
        ]

    def exact_english_lookup(self, english_key: str) -> List[Dict[str, str]]:
        """Exact match on the normalized English text."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT sanskrit_iast, sanskrit_original, english, source_file, entry_id "
            "FROM lexicon WHERE english_lower = ? LIMIT 10",
            (english_key,),
        ).fetchall()
        return [
            {
                "sanskrit_iast": r[0],
                "sanskrit_original": r[1],
                "english": r[2],
                "source_file": r[3],
                "entry_id": r[4],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Query — FTS5 fuzzy search
    # ------------------------------------------------------------------
    def fts_search(self, query: str, n_results: int = 5) -> List[Dict[str, str]]:
        """Full-text search across Sanskrit IAST and English fields.

        Supports prefix matching (e.g. ``dharma*``), phrase queries, and
        implicit OR across words.
        """
        conn = self._connect()
        # Sanitize: strip ALL non-word characters from each token to
        # prevent FTS5 syntax errors (e.g. ':' is the column-filter
        # operator, ',' / '.' can break OR-joined expressions).
        tokens = query.split()
        safe_tokens = []
        for t in tokens:
            t = re.sub(r'[^\w]', '', t)
            if t:
                safe_tokens.append(t + "*")
        if not safe_tokens:
            return []
        fts_query = " OR ".join(safe_tokens)
        try:
            rows = conn.execute(
                "SELECT l.sanskrit_iast, l.sanskrit_original, l.english, "
                "       l.source_file, l.entry_id "
                "FROM lexicon_fts f "
                "JOIN lexicon l ON f.rowid = l.id "
                "WHERE lexicon_fts MATCH ? "
                "ORDER BY rank LIMIT ?",
                (fts_query, n_results),
            ).fetchall()
        except Exception:
            return []
        return [
            {
                "sanskrit_iast": r[0],
                "sanskrit_original": r[1],
                "english": r[2],
                "source_file": r[3],
                "entry_id": r[4],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Counting
    # ------------------------------------------------------------------
    def count(self) -> int:
        try:
            conn = self._connect()
            row = conn.execute("SELECT COUNT(*) FROM lexicon").fetchone()
            return row[0] if row else 0
        except Exception:
            return 0
