# -*- coding: utf-8 -*-
"""
Baba_SandhiSplit
================

CLI utility for running the pretrained sandhi splitter (TensorFlow SavedModel
shipped with this repository) on UTF-8 text files. Handles both individual
files and directory trees, runs completely offline, and can emit IAST or
Devanagari output.

Example usage:

    # Single file -> auto-named output (same folder, *_unsandhied.txt)
    python baba_sandhi_split.py --input input/example.txt

    # Explicit output file, keep 20% token headroom for translation stages
    python baba_sandhi_split.py --input input/example.txt \
        --output output/example_unsandhied.txt --reserve 20

    # Batch process all .txt files in a folder recursively
    python baba_sandhi_split.py --input corpus/ --output results/ --recursive

    # Devanagari input, request Devanagari output
    python baba_sandhi_split.py --input devanagari.txt --output-script devanagari

The script reuses the training configuration and helper utilities under ./code.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import tempfile
import re
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple
import unicodedata

import numpy as np  # noqa  (TensorFlow expects NumPy to be present)

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover - environment specific
    raise SystemExit(
        "TensorFlow is required to run Baba_SandhiSplit. Install a TensorFlow "
        "1.x compatible build (e.g. tensorflow==1.15.*) and re-run."
    ) from exc

tf.compat.v1.disable_eager_execution()

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import configuration  # type: ignore  # noqa
import data_loader  # type: ignore  # noqa
import defines  # type: ignore  # noqa


try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
except ImportError:  # pragma: no cover - optional dependency
    sanscript = None  # type: ignore
    transliterate = None  # type: ignore


IAST_DIACRITIC_CHARS = set("āīūṛṝḷḹṅñṭḍṇśṣṃṁḥĀĪŪṚṜḶḸṄÑṬḌṆŚṢṂṀḤ")
TOKEN_REGEX = re.compile(r"\s+|\S+\s*", re.UNICODE)
WORD_CAPTURE = re.compile(r"[\u0900-\u097F\u1E00-\u1EFFA-Za-z]+", re.UNICODE)
PUNCT_STRIP = ".,;:!?\"'()[]{}“”‘’«»"


DECLENSION_CACHE_VERSION = 3
VERB_CACHE_VERSION = 2
INDECL_CACHE_VERSION = 1
ADVERB_CACHE_VERSION = 1
PRONOUN_CACHE_VERSION = 1
_CACHE_VERSION_KEY = "__version__"
_CACHE_ENTRIES_KEY = "entries"


def log(message: str) -> None:
    print(message, flush=True)


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def detect_script(sample: str) -> str:
    """Heuristically detect whether the text is Devanagari or IAST/Latin."""
    for ch in sample:
        if "\u0900" <= ch <= "\u097f":
            return "devanagari"
        if ch in IAST_DIACRITIC_CHARS:
            return "iast"
    # fall back: if we saw any ASCII letters assume IAST/Latin
    if any("A" <= ch <= "z" for ch in sample):
        return "iast"
    return "iast"


class TransliterationUnavailable(RuntimeError):
    pass


def ensure_transliteration() -> None:
    if transliterate is None:
        raise TransliterationUnavailable(
            "Devanagari <-> IAST transliteration requires the "
            "'indic_transliteration' package. Install it locally (no internet "
            "needed) and re-run."
        )


def to_iast(text: str) -> str:
    ensure_transliteration()
    return transliterate(text, sanscript.DEVANAGARI, sanscript.IAST)


def from_iast(text: str, target: str) -> str:
    if target == "iast":
        return text
    ensure_transliteration()
    return transliterate(text, sanscript.IAST, sanscript.DEVANAGARI)


def is_devanagari_token(token: str) -> bool:
    return any("\u0900" <= ch <= "\u097f" for ch in token)


def normalize_token(token: str) -> str:
    return unicodedata.normalize("NFC", token)


def _char_category(ch: str) -> str:
    if "\u0900" <= ch <= "\u097f":
        return "dev"
    if ch.isalpha() or ch in IAST_DIACRITIC_CHARS:
        return "iast"
    return "other"


def extract_iast_tokens(text: str) -> List[str]:
    """Extract IAST tokens from a label value, ignoring Devanagari segments."""
    tokens: List[str] = []
    current_chars: List[str] = []
    current_cat: Optional[str] = None
    for ch in text:
        cat = _char_category(ch)
        if cat != current_cat:
            if current_cat == "iast" and current_chars:
                token = normalize_token("".join(current_chars).strip(PUNCT_STRIP))
                if token:
                    tokens.append(token)
            current_chars = []
        if cat == "iast":
            current_chars.append(ch)
        elif cat == "other":
            current_chars = []
        current_cat = cat if cat != "other" else None
    if current_cat == "iast" and current_chars:
        token = normalize_token("".join(current_chars).strip(PUNCT_STRIP))
        if token:
            tokens.append(token)
    return tokens


def add_mapping_entry(
    mapping: Dict[str, Set[str]],
    token: str,
    label: str,
    value: Optional[str] = None,
) -> None:
    token_norm = normalize_token(token)
    if not token_norm:
        return
    if is_devanagari_token(token_norm):
        if transliterate is None:
            return
        try:
            token_norm = normalize_token(
                transliterate(token_norm, sanscript.DEVANAGARI, sanscript.IAST)
            )
        except Exception:
            return
    if not token_norm or is_devanagari_token(token_norm):
        return
    key = token_norm.casefold()
    entry_value = normalize_token(value) if value else token_norm
    entry = f"{label}: {entry_value}" if label else entry_value
    mapping[key].add(entry)


def _normalise_cache_payload(raw: Dict[str, Iterable[str]]) -> Dict[str, Set[str]]:
    """Coerce cached payload into the runtime mapping format."""
    result: Dict[str, Set[str]] = {}
    for key, values in raw.items():
        key_str = key if isinstance(key, str) else str(key)
        bucket: Set[str] = set()
        if isinstance(values, set):
            bucket.update(str(item) for item in values if item)
        elif isinstance(values, (list, tuple)):
            bucket.update(str(item) for item in values if item)
        elif values:
            bucket.add(str(values))
        if bucket:
            result[key_str] = bucket
    return result


def _read_cached_mapping(cache_path: Path, expected_version: int, kind_label: str) -> Optional[Tuple[Dict[str, Set[str]], bool]]:
    """Return cached mapping and legacy flag if available."""
    try:
        with cache_path.open("rb") as fh:
            payload = pickle.load(fh)
    except Exception as exc:
        log(f"    {kind_label} cache load failed ({exc}); rebuilding...")
        return None

    legacy = False
    data = payload
    if isinstance(payload, dict) and _CACHE_VERSION_KEY in payload:
        version = payload.get(_CACHE_VERSION_KEY)
        if version != expected_version:
            log(
                f"    {kind_label} cache version mismatch ({version} != {expected_version}); rebuilding..."
            )
            return None
        data = payload.get(_CACHE_ENTRIES_KEY, {})
    elif isinstance(payload, dict):
        legacy = True
        data = payload
    else:
        log(f"    {kind_label} cache payload invalid type: {type(payload).__name__}; rebuilding...")
        return None

    if not isinstance(data, dict):
        log(f"    {kind_label} cache entries invalid type: {type(data).__name__}; rebuilding...")
        return None

    try:
        mapping = _normalise_cache_payload(data)
    except Exception as exc:
        log(f"    {kind_label} cache normalisation failed ({exc}); rebuilding...")
        return None

    return mapping, legacy


def _write_cached_mapping(cache_path: Path, mapping: Dict[str, Set[str]], cache_version: int, kind_label: str) -> None:
    payload = {
        _CACHE_VERSION_KEY: cache_version,
        _CACHE_ENTRIES_KEY: mapping,
    }
    log(f"    writing {kind_label} cache to {cache_path} ...")
    try:
        with cache_path.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        log(f"    {kind_label} cache write complete")
    except Exception as exc:
        log(f"    {kind_label} cache write failed ({exc})")


def _build_mapping_from_file(path: Path, cache_version: int, kind_label: str) -> Dict[str, Set[str]]:
    cache_path = path.with_suffix(path.suffix + ".cache")

    # Prefer cache if present and either source is missing or cache is newer
    if cache_path.exists() and (not path.exists() or cache_path.stat().st_mtime >= path.stat().st_mtime):
        log(f"[+] loading {kind_label} cache from {cache_path}")
        cached_payload = _read_cached_mapping(cache_path, cache_version, kind_label)
        if cached_payload is not None:
            cached_map, legacy = cached_payload
            log(f"    {kind_label} cache keys: {len(cached_map)}")
            if legacy:
                _write_cached_mapping(cache_path, cached_map, cache_version, kind_label)
            if not path.exists():
                log(f"    {kind_label} list missing; using cache only")
            return cached_map
        else:
            log("    cache invalid; rebuilding...")

    if not path.exists():
        log(f"[+] {kind_label} list not found at {path}; no annotations will be added")
        return {}

    log(f"[+] loading {kind_label} list from {path}")
    total_bytes = max(path.stat().st_size, 1)
    progress_step = max(total_bytes // 20, 25_000_000)  # Report every ~5% or ~25MB
    next_progress = progress_step
    last_log = time.monotonic()
    log(f"    parsing {kind_label} list... 0%")
    mapping: DefaultDict[str, Set[str]] = defaultdict(set)
    total = 0
    with path.open("rb") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            try:
                line = raw_line.decode("utf-8", errors="ignore").strip()
            except Exception:
                continue
            if not line or line.startswith("#"):
                continue
            if kind_label == "verb":
                entries = _parse_verb_line(line)
            elif kind_label == "indeclinable":
                entries = _parse_indeclinable_line(line)
            elif kind_label == "adverb":
                entries = _parse_adverb_line(line)
            elif kind_label == "pronoun":
                entries = _parse_pronoun_line(line)
            else:
                entries = _parse_declension_line(line)
            if not entries:
                continue
            for token, entry_label, entry_value in entries:
                add_mapping_entry(mapping, token, entry_label, entry_value)
            total += len(entries)
            current_pos = fh.tell()
            now = time.monotonic()
            if current_pos >= next_progress or (now - last_log) >= 1.5:
                percent = min(99, int(current_pos * 100 / total_bytes))
                log(
                    f"    parsing {kind_label} list... {percent}% "
                    f"({line_no} lines, {current_pos / 1_000_000:.1f} MB)"
                )
                last_log = now
                while current_pos >= next_progress:
                    next_progress += progress_step
    log(f"    parsing {kind_label} list... 100% (done)")
    result = {key: set(values) for key, values in mapping.items()}
    log(f"    {kind_label} entries indexed: {len(result)} keys, {total} values")
    _write_cached_mapping(cache_path, result, cache_version, kind_label)
    return result


def _parse_declension_line(line: str) -> List[Tuple[str, str, Optional[str]]]:
    if ":" in line:
        label, value = line.split(":", 1)
        label = label.strip()
        value = value.strip()
    else:
        label, value = "Form", line
    if not value:
        return []
    tokens = extract_iast_tokens(value)
    if not tokens and transliterate is not None:
        fallback_tokens: List[str] = []
        for dev_token in WORD_CAPTURE.findall(value):
            if not is_devanagari_token(dev_token):
                continue
            try:
                fallback_tokens.append(
                    normalize_token(
                        transliterate(dev_token, sanscript.DEVANAGARI, sanscript.IAST)
                    ).strip(PUNCT_STRIP)
                )
            except Exception:
                continue
        tokens = [tok for tok in fallback_tokens if tok]
    return [(token, label, None) for token in tokens]


def _parse_verb_line(line: str) -> List[Tuple[str, str, Optional[str]]]:
    parts = line.split()
    if len(parts) < 2:
        return []
    form = parts[0]
    lemma = parts[1]
    details = " ".join(parts[2:]).strip()
    label = details if details else "verb"
    entry_value = lemma.strip()
    return [(form, label, entry_value)]


def _parse_indeclinable_line(line: str) -> List[Tuple[str, str, Optional[str]]]:
    parts = line.split()
    if not parts:
        return []
    form = parts[0]
    lemma = parts[1] if len(parts) > 1 else ""
    label = "indeclinable"
    entry_value = lemma.strip() if lemma else form
    return [(form, label, entry_value)]


def _parse_adverb_line(line: str) -> List[Tuple[str, str, Optional[str]]]:
    parts = line.split()
    if not parts:
        return []
    form = parts[0]
    lemma = parts[1] if len(parts) > 1 else ""
    details = " ".join(parts[2:]).strip()
    label = details if details else "adverb"
    entry_value = lemma.strip() if lemma else form
    return [(form, label, entry_value)]


def _parse_pronoun_line(line: str) -> List[Tuple[str, str, Optional[str]]]:
    token = line.strip()
    if not token:
        return []
    return [(token, "pronoun", None)]


def build_declension_map(path: Path) -> Dict[str, Set[str]]:
    return _build_mapping_from_file(path, DECLENSION_CACHE_VERSION, "declension")


def build_verb_map(path: Path) -> Dict[str, Set[str]]:
    return _build_mapping_from_file(path, VERB_CACHE_VERSION, "verb")


def build_indeclinable_map(path: Path) -> Dict[str, Set[str]]:
    return _build_mapping_from_file(path, INDECL_CACHE_VERSION, "indeclinable")


def build_adverb_map(path: Path) -> Dict[str, Set[str]]:
    return _build_mapping_from_file(path, ADVERB_CACHE_VERSION, "adverb")


def build_pronoun_map(path: Path) -> Dict[str, Set[str]]:
    return _build_mapping_from_file(path, PRONOUN_CACHE_VERSION, "pronoun")


def chunk_line(line: str, limit: int) -> List[str]:
    if not line:
        return []
    tokens = TOKEN_REGEX.findall(line)
    chunks: List[str] = []
    current = ""
    for token in tokens:
        token_len = len(token)
        if token_len > limit:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, token_len, limit):
                chunks.append(token[i : i + limit])
            continue
        if current and len(current) + token_len > limit:
            chunks.append(current)
            current = token
        else:
            current += token
            if len(current) >= limit:
                chunks.append(current)
                current = ""
    if current:
        chunks.append(current)

    safe: List[str] = []
    for chunk in chunks:
        if len(chunk) <= limit:
            safe.append(chunk)
        else:
            for i in range(0, len(chunk), limit):
                safe.append(chunk[i : i + limit])
    return safe


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@dataclass
class SandhiPrediction:
    path_in: Path
    path_out: Path
    input_script: str
    output_script: str


class SandhiModel:
    """Wraps TensorFlow session + DataLoader."""

    def __init__(
        self,
        project_root: Path,
        config: dict,
        declension_map: Optional[Dict[str, Set[str]]] = None,
        verb_map: Optional[Dict[str, Set[str]]] = None,
        indeclinable_map: Optional[Dict[str, Set[str]]] = None,
        adverb_map: Optional[Dict[str, Set[str]]] = None,
        pronoun_map: Optional[Dict[str, Set[str]]] = None,
    ):
        self.project_root = project_root
        self.config = dict(config)
        self.stack = ExitStack()
        self.session: Optional[tf.compat.v1.Session] = None
        self.graph = tf.Graph()
        self.data = None
        self.x_ph = None
        self.split_cnts_ph = None
        self.dropout_ph = None
        self.seqlen_ph = None
        self.predictions_ph = None
        self.declension_map: Dict[str, Set[str]] = declension_map or {}
        self.verb_map: Dict[str, Set[str]] = verb_map or {}
        self.indeclinable_map: Dict[str, Set[str]] = indeclinable_map or {}
        self.adverb_map: Dict[str, Set[str]] = adverb_map or {}
        self.pronoun_map: Dict[str, Set[str]] = pronoun_map or {}
        self.annotation_cache: Dict[Tuple[str, str], str] = {}
        self.output_script: str = "iast"  # Target script for annotations

    def __enter__(self) -> "SandhiModel":
        data_dir = (self.project_root / "data" / "input").resolve()
        model_dir = (self.project_root / "data" / "models").resolve()
        self.config["model_directory"] = str(model_dir)
        self.data = self.stack.enter_context(
            data_loader.DataLoader(
                str(data_dir),
                self.config,
                load_data_into_ram=False,
                load_data=False,
            )
        )

        log("    creating TensorFlow session...")
        self.session = tf.compat.v1.Session(graph=self.graph)
        self.stack.callback(self.session.close)

        with self.graph.as_default():
            log("    restoring saved model...")
            tf.compat.v1.saved_model.loader.load(
                self.session,
                [tf.compat.v1.saved_model.tag_constants.SERVING],
                str(model_dir),
            )
            log("    model restored")
            self.x_ph = self.graph.get_tensor_by_name("inputs:0")
            self.split_cnts_ph = self.graph.get_tensor_by_name("split_cnts:0")
            self.dropout_ph = self.graph.get_tensor_by_name("dropout_keep_prob:0")
            self.seqlen_ph = self.graph.get_tensor_by_name("seqlens:0")
            self.predictions_ph = self.graph.get_tensor_by_name("predictions:0")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stack.close()

    def annotate_lines(self, lines: List[str]) -> List[str]:
        if not any(
            [
                self.declension_map,
                self.verb_map,
                self.indeclinable_map,
                self.adverb_map,
                self.pronoun_map,
            ]
        ):
            return lines
        return [self.annotate_line(line) for line in lines]

    def annotate_line(self, line: str) -> str:
        if not line or not any(
            [
                self.declension_map,
                self.verb_map,
                self.indeclinable_map,
                self.adverb_map,
                self.pronoun_map,
            ]
        ):
            return line
        result: List[str] = []
        last = 0
        for match in re.finditer(r"\S+", line):
            result.append(line[last:match.start()])
            token = match.group()
            result.append(self.annotate_token(token))
            last = match.end()
        result.append(line[last:])
        return "".join(result)

    def format_mapping_entries(self, entries: Iterable[str]) -> List[str]:
        """Return concise summary grouped by surface form."""
        value_lookup: Dict[str, str] = {}
        label_groups: DefaultDict[str, Set[str]] = defaultdict(set)

        for entry in entries:
            if not entry:
                continue
            raw = entry.strip()
            if not raw:
                continue
            if ":" in raw:
                label, value = raw.split(":", 1)
                label = label.strip()
                value = value.strip()
            else:
                label = ""
                value = raw
            if not value:
                continue
            key = normalize_token(value).casefold()
            if key not in value_lookup:
                value_lookup[key] = value
            if label:
                label_groups[key].add(label)

        if not value_lookup:
            return []

        segments: List[str] = []
        for key in sorted(value_lookup.keys(), key=lambda k: value_lookup[k].casefold()):
            display = value_lookup[key]
            labels = sorted(label_groups.get(key, set()), key=str.casefold)
            if labels:
                segments.append(f"{display} ({', '.join(labels)})")
            else:
                segments.append(display)
        return segments

    def annotate_token(self, token: str) -> str:
        if not any(
            [
                self.declension_map,
                self.verb_map,
                self.indeclinable_map,
                self.adverb_map,
                self.pronoun_map,
            ]
        ):
            return self._convert_token_script(token)
        cache_key = (token, self.output_script)
        cached = self.annotation_cache.get(cache_key)
        if cached is not None:
            return cached
        stripped = token.strip(PUNCT_STRIP)
        if not stripped:
            result = self._convert_token_script(token)
            self.annotation_cache[cache_key] = result
            return result
        sections: List[Tuple[str, List[str]]] = []
        lookup_sources: List[Tuple[str, Callable[[str], List[str]]]] = [
            ("forms", self.lookup_forms),
            ("verbs", self.lookup_verbs),
            ("indecls", self.lookup_indeclinables),
            ("adverbs", self.lookup_adverbs),
            ("pronouns", self.lookup_pronouns),
        ]
        for name, lookup in lookup_sources:
            matches = lookup(stripped)
            if matches:
                formatted = self.format_mapping_entries(matches)
                if formatted:
                    sections.append((name, formatted))
        components = [comp for comp in re.split(r"[-=]", stripped) if comp and comp != stripped]
        seen_components: Set[str] = set()
        component_annotations: List[Tuple[str, List[str]]] = []
        for comp in components:
            comp_norm = normalize_token(comp)
            if comp_norm in seen_components:
                continue
            seen_components.add(comp_norm)
            detail_segments: List[str] = []
            for name, lookup in lookup_sources:
                matches = lookup(comp)
                if not matches:
                    continue
                formatted = self.format_mapping_entries(matches)
                if formatted:
                    detail_segments.append(f"{name}: {'; '.join(formatted)}")
            component_annotations.append((comp, detail_segments))
        if component_annotations:
            part_entries: List[str] = []
            for comp, details in component_annotations:
                filtered_details = [seg for seg in details if seg]
                if not filtered_details:
                    part_entries.append(comp)
                else:
                    part_entries.append(f"{comp} → {' | '.join(filtered_details)}")
            sections.append(("parts", part_entries))

        if not sections:
            result = self._convert_token_script(token)
            self.annotation_cache[cache_key] = result
            return result

        display_token = self._convert_token_script(token)
        annotation_lines: List[str] = [f"{display_token} ["]
        for name, entries in sections:
            annotation_lines.append(f"    {name}:")
            for entry in entries:
                if entry.startswith("  - "):
                    annotation_lines.append(f"        {entry}")
                else:
                    annotation_lines.append(f"        - {entry}")
        annotation_lines.append("]")
        result = "\n".join(annotation_lines)
        self.annotation_cache[cache_key] = result
        return result

    def _convert_token_script(self, token: str) -> str:
        """Convert a token to the target output script."""
        if self.output_script == "iast":
            return token
        if transliterate is None:
            return token
        try:
            return transliterate(token, sanscript.IAST, sanscript.DEVANAGARI)
        except Exception:
            return token

    def _lookup_in_map(self, token: str, mapping: Dict[str, Set[str]]) -> List[str]:
        if not token:
            return []
        token_norm = normalize_token(token)
        key = token_norm.casefold()
        matches: Set[str] = set()
        entries = mapping.get(key)
        if entries:
            matches.update(entries)
        if transliterate is not None:
            try:
                if is_devanagari_token(token_norm):
                    alt = normalize_token(transliterate(token_norm, sanscript.DEVANAGARI, sanscript.IAST))
                else:
                    alt = normalize_token(transliterate(token_norm, sanscript.IAST, sanscript.DEVANAGARI))
                alt_entries = mapping.get(alt.casefold())
                if alt_entries:
                    matches.update(alt_entries)
            except Exception:
                pass
        return sorted(matches)

    def lookup_forms(self, token: str) -> List[str]:
        if not self.declension_map:
            return []
        return self._lookup_in_map(token, self.declension_map)

    def lookup_verbs(self, token: str) -> List[str]:
        if not self.verb_map:
            return []
        return self._lookup_in_map(token, self.verb_map)

    def lookup_indeclinables(self, token: str) -> List[str]:
        if not self.indeclinable_map:
            return []
        return self._lookup_in_map(token, self.indeclinable_map)

    def lookup_adverbs(self, token: str) -> List[str]:
        if not self.adverb_map:
            return []
        return self._lookup_in_map(token, self.adverb_map)

    def lookup_pronouns(self, token: str) -> List[str]:
        if not self.pronoun_map:
            return []
        return self._lookup_in_map(token, self.pronoun_map)

    def predict_segments(self, path_for_model: Path) -> List[str]:
        if self.session is None or self.data is None:
            raise RuntimeError("Model session is not initialised.")

        seqs, lens, splitcnts, lines_orig = self.data.load_external_text(str(path_for_model))

        if (
            seqs is None
            or lens is None
            or splitcnts is None
            or lines_orig is None
            or len(seqs) == 0
        ):
            return []

        total = seqs.shape[0]
        batch_size = 500
        predictions: List[np.ndarray] = []
        start = 0

        log(f"    processing {total} segment{'s' if total != 1 else ''}")

        if total > 1:
            print(f"    processing {total} segments", flush=True)

        while start < total:
            end = min(start + batch_size, total)
            feed = {
                self.x_ph: seqs[start:end, :],
                self.split_cnts_ph: splitcnts[start:end, :, :],
                self.seqlen_ph: lens[start:end],
                self.dropout_ph: 1.0,
            }
            preds = self.session.run(self.predictions_ph, feed_dict=feed)
            predictions.append(preds)
            pct = (end / total) * 100.0
            log(f"    progress {pct:5.1f}% ({end}/{total})")
            start = end

        P = (
            np.concatenate(predictions, axis=0)
            if predictions
            else np.zeros((0, seqs.shape[1]), dtype=np.int32)
        )
        if total > 0:
            log("    inference complete")

        results: List[str] = []
        for row in range(P.shape[0]):
            seq_len = lens[row]
            pred_symbols = [self.data.deenc_output.get_sym(x) for x in P[row, :seq_len]]
            decoded = ""
            for pred_sym, original_sym in zip(pred_symbols[1:], lines_orig[row][1:]):
                if pred_sym == defines.SYM_IDENT:
                    decoded += original_sym
                elif pred_sym == defines.SYM_SPLIT:
                    decoded += original_sym + "-"
                else:
                    decoded += pred_sym
            decoded = (
                self.data.internal_transliteration_to_unicode(decoded)
                .replace("- ", " ")
                .replace("= ", " ")
            )
            results.append(decoded)
        return results

    def split_file(self, prediction: SandhiPrediction) -> None:
        if self.session is None or self.data is None:
            raise RuntimeError("Model session is not initialised.")

        input_path = prediction.path_in
        output_path = prediction.path_out

        raw_text = read_text(input_path)
        trailing_newline = raw_text.endswith("\n") or raw_text.endswith("\r")
        lines_original = raw_text.splitlines()
        input_script = prediction.input_script

        if input_script == "devanagari":
            lines_iast = [to_iast(line) for line in lines_original]
        else:
            lines_iast = list(lines_original)

        limit = max(1, int(self.config.get("max_sequence_length_sen", 128)) - 1)
        final_lines = list(lines_iast)
        line_has_segments = [False] * len(lines_iast)
        segments: List[str] = []
        segment_to_line: List[int] = []

        for idx, line in enumerate(lines_iast):
            if not line.strip():
                continue
            parts = chunk_line(line, limit)
            if not parts:
                continue
            line_has_segments[idx] = True
            segments.extend(parts)
            segment_to_line.extend([idx] * len(parts))

        temp_input_path: Optional[Path] = None
        segment_count = len(segments)
        log(f"    segments to process: {segment_count}")

        if segments:
            temp_input = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False)
            temp_input_path = Path(temp_input.name)
            for seg in segments:
                temp_input.write(seg.rstrip("\n"))
                temp_input.write("\n")
            temp_input.close()
            path_for_model = temp_input_path
            for idx, flag in enumerate(line_has_segments):
                if flag:
                    final_lines[idx] = ""
        else:
            path_for_model = None

        predicted_segments: List[str] = []

        if path_for_model is not None:
            log("    running model inference...")
            predicted_segments = self.predict_segments(path_for_model)
            if len(predicted_segments) != len(segments):
                raise RuntimeError(
                    f"Segment count mismatch: expected {len(segments)}, got {len(predicted_segments)}"
                )
            for seg_idx, prediction_text in enumerate(predicted_segments):
                line_idx = segment_to_line[seg_idx]
                final_lines[line_idx] += prediction_text
        else:
            # No inference needed (empty or whitespace-only file). Write output directly.
            final_lines = lines_iast
            log("    no inference required (text within model length limit)")

        # Set output script before annotating so tokens get converted
        self.output_script = prediction.output_script
        self.annotation_cache.clear()  # Clear cache when script changes
        final_lines = self.annotate_lines(final_lines)

        if temp_input_path:
            temp_input_path.unlink(missing_ok=True)

        final_text = "\n".join(final_lines)
        if trailing_newline and not final_text.endswith("\n"):
            final_text += "\n"

        write_text(output_path, final_text)


def collect_files(path: Path, extensions: List[str], recursive: bool) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path not found: {path}")
    pattern = "**/*" if recursive else "*"
    files = [
        p for p in path.glob(pattern)
        if p.is_file() and p.suffix.lower() in extensions
    ]
    return sorted(files)


def determine_output_path(
    input_path: Path,
    input_root: Path,
    output_argument: Optional[Path],
) -> Path:
    if output_argument is None:
        suffix = input_path.suffix or ".txt"
        return input_path.parent / f"{input_path.stem}_unsandhied{suffix}"

    if output_argument.is_file() or output_argument.suffix:
        if input_path.is_dir():
            raise ValueError("--output must be a directory when --input is a directory.")
        return output_argument

    output_argument.mkdir(parents=True, exist_ok=True)
    if _is_relative_to(input_path, input_root):
        relative = input_path.relative_to(input_root)
    else:
        relative = Path(input_path.name)
    target = output_argument / relative
    suffix = target.suffix or ".txt"
    return target.with_name(f"{target.stem}_unsandhied{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline sandhi splitter (TensorFlow SavedModel wrapper)."
    )
    parser.add_argument("--input", required=True, help="Input file or directory (UTF-8 .txt).")
    parser.add_argument("--output", help="Output file or directory. Defaults to *_unsandhied.txt beside source.")
    parser.add_argument("--recursive", action="store_true", help="Recurse into sub-folders when --input is a directory.")
    parser.add_argument(
        "--extensions",
        default=".txt",
        help="Comma-separated list of filename extensions to process when input is a directory. Default: .txt",
    )
    parser.add_argument(
        "--input-script",
        choices=["auto", "iast", "devanagari"],
        default="auto",
        help="Hint for input script detection. Default: auto.",
    )
    parser.add_argument(
        "--output-script",
        choices=["auto", "iast", "devanagari"],
        default="auto",
        help="Output script. Default: auto (same as input).",
    )

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_arg = Path(args.output).resolve() if args.output else None
    extensions = [ext.lower().strip() if ext.startswith(".") else f".{ext.lower().strip()}"
                  for ext in args.extensions.split(",")]

    if input_path.is_dir() and output_arg and output_arg.is_file():
        raise ValueError("When processing a directory, --output must be a directory path.")

    project_root = CODE_DIR.parent.resolve()
    declension_path = SCRIPT_DIR / "declensions.txt"
    declension_map = build_declension_map(declension_path)
    verbs_path = SCRIPT_DIR / "verbs.txt"
    verb_map = build_verb_map(verbs_path)
    indecls_path = SCRIPT_DIR / "indecls.txt"
    indeclinable_map = build_indeclinable_map(indecls_path)
    adverbs_path = SCRIPT_DIR / "adverbs.txt"
    adverb_map = build_adverb_map(adverbs_path)
    pronouns_path = SCRIPT_DIR / "pronouns.txt"
    pronoun_map = build_pronoun_map(pronouns_path)

    files = collect_files(input_path, extensions, recursive=args.recursive)
    if not files:
        print("No input files found.", file=sys.stderr)
        sys.exit(1)

    log("[+] Initialising sandhi model (first call may take a few seconds)...")
    with SandhiModel(
        project_root,
        configuration.config,
        declension_map,
        verb_map,
        indeclinable_map,
        adverb_map,
        pronoun_map,
    ) as model:
        for file_path in files:
            raw_text = read_text(file_path)
            detected_script = detect_script(raw_text) if args.input_script == "auto" else args.input_script
            if detected_script == "devanagari" and transliterate is None:
                ensure_transliteration()
            output_script = args.output_script
            if output_script == "auto":
                output_script = detected_script

            output_path = determine_output_path(
                file_path,
                input_path if input_path.is_dir() else file_path.parent,
                output_arg,
            )

            prediction = SandhiPrediction(
                path_in=file_path,
                path_out=output_path,
                input_script=detected_script,
                output_script=output_script,
            )
            log(
                f"[+] Splitting {file_path} -> {output_path} "
                f"[{prediction.input_script} -> {prediction.output_script}]"
            )
            model.split_file(prediction)

    log("Done.")


if __name__ == "__main__":
    main()
