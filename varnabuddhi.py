#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Varnabuddhi — Academic Sanskrit ⇄ English Translator
=====================================================
Cross-platform Python CLI combining LLMs, local lexical databases,
sandhi processing, morphological verification, and offline transliteration.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so ``engines`` can be imported
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from engines.transliteration_engine import (
    detect_script,
    ensure_iast,
    to_devanagari,
    is_available as translit_available,
    normalize as unicode_normalize,
)
from engines.lexicon_engine import LexiconEngine, LookupResult
from engines.verification_engine import VerificationEngine
from engines.sandhi_engine import SandhiEngine, SandhiJoinError, SandhiSplitError
from engines.llm_engine import LLMEngine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_FILE = _PROJECT_ROOT / "config.json"

BANNER = r"""
 ╔══════════════════════════════════════════════════════════════╗
 ║                                                              ║
 ║    वर्णबुद्धि                                                 ║
 ║    V A R N A B U D D H I                                     ║
 ║                                                              ║
 ║    Academic Sanskrit ⇄ English Translator                    ║
 ║                                                              ║
 ║    Author:  {author:<46s} ║
 ║    Year:    {year:<46s} ║
 ║    Version: {version:<46s} ║
 ║                                                              ║
 ╚══════════════════════════════════════════════════════════════╝
"""

DIRECTION_LABELS = {
    "san_to_eng": "Sanskrit → English",
    "eng_to_san": "English → Sanskrit",
}

HELP_TEXT = textwrap.dedent("""\
    Translation:
      <text>                  Translate text in the current direction
      --split <text>          Translate with sandhi splitting pre-process
      --iteration N <text>    Translate with N refinement iterations (default: 5)
      translate-file <file>   Translate a text file (verse-by-verse)

    Analysis:
      verify <word>           Verify a Sanskrit word form (morphological check)
      split <text>            Perform sandhi splitting on Sanskrit text
      dict <word>             Look up a word in the Monier-Williams dictionary

    Settings:
      dir                     Toggle translation direction
      model [p] [m]           Switch LLM provider [p] and/or model [m]
      profile [name]          Switch prompt profile
      script [mode]           Toggle output script: iast / devanagari / both
      status                  Show current configuration status

    General:
      help                    Show this help message
      quit / exit             Exit the application
""")


# ===================================================================
# Config helpers
# ===================================================================


def load_config(path: Path) -> Dict[str, Any]:
    """Load and validate config.json."""
    if not path.exists():
        print(f"[ERROR] Configuration file not found: {path}")
        print("Please create config.json in the project root.")
        sys.exit(1)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Invalid JSON in {path}: {exc}")
        sys.exit(1)

    # Validate required top-level sections
    for section in ("application", "llm", "translation", "paths"):
        if section not in cfg:
            print(f"[ERROR] Missing required section '{section}' in config.json")
            sys.exit(1)

    # Ensure at least one provider entry exists (model/key can be empty)
    providers = cfg.get("llm", {}).get("providers", {})
    if not providers:
        print("[WARNING] No LLM providers defined in config.json → llm → providers.")
        print("          Translation will only work via the local lexicon.")

    return cfg


def resolve_path(cfg: Dict[str, Any], key: str) -> Path:
    """Resolve a path from config relative to the project root."""
    raw = cfg.get("paths", {}).get(key, key)
    p = Path(raw)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return p


# ===================================================================
# Logging setup
# ===================================================================


def setup_logging(cfg: Dict[str, Any]) -> None:
    log_cfg = cfg.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_file = log_cfg.get("file", "")

    handlers: list[logging.Handler] = []
    if log_file:
        log_path = _PROJECT_ROOT / log_file
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=handlers or None,
    )


# ===================================================================
# Translation pipeline
# ===================================================================


# ---------------------------------------------------------------------------
# Grammar-enrichment helpers
# ---------------------------------------------------------------------------


def _build_grammar_context(
    words: list[str], verifier: VerificationEngine
) -> Dict[str, str]:
    """Run each *word* through the verification engine and return a dict
    mapping word → grammatical description string."""
    grammar: Dict[str, str] = {}
    for w in words:
        w = w.strip(".,;:!? ")
        if len(w) < 2:
            continue
        vr = verifier.verify(w)
        if vr.verified and vr.tags:
            parts: list[str] = []
            for tag in vr.tags[:3]:  # limit to top 3 parses
                if tag.category == "noun":
                    g = f" [{tag.gender}]" if tag.gender else ""
                    parts.append(f"{tag.details}{g} (root: {tag.root})")
                elif tag.category == "verb":
                    parts.append(f"{tag.details} (dhātu: {tag.root})")
                elif tag.category == "dict_headword":
                    parts.append(f"headword — {tag.details[:80]}")
            if parts:
                grammar[w] = " | ".join(parts)
    return grammar


def _compute_confidence(
    translation: str,
    word_hints: Dict[str, str],
    grammar_tags: Dict[str, str],
    split_text: Optional[str],
) -> str:
    """Return a confidence label based on how much supporting evidence exists."""
    score = 0
    # Evidence from grammar
    if grammar_tags:
        score += min(len(grammar_tags), 5)  # up to 5 pts
    # Evidence from lexicon
    if word_hints:
        score += min(len(word_hints), 5)    # up to 5 pts
    # Sandhi split succeeded
    if split_text:
        score += 2
    # Fraction of input words that have grammar tags
    if split_text:
        n_words = len(split_text.split())
        if n_words > 0:
            coverage = len(grammar_tags) / n_words
            if coverage >= 0.8:
                score += 3
            elif coverage >= 0.5:
                score += 1

    if score >= 8:
        return "HIGH"
    if score >= 4:
        return "MEDIUM"
    return "LOW"


def _apply_output_script(text: str, script_mode: str) -> str:
    """Convert Sanskrit IAST tokens in *text* to the desired output script."""
    if script_mode == "iast" or not translit_available():
        return text
    if script_mode == "devanagari":
        return to_devanagari(text)
    if script_mode == "both":
        return f"{text}\n{to_devanagari(text)}"
    return text


# ---------------------------------------------------------------------------
# Streaming helper
# ---------------------------------------------------------------------------


def _llm_stream_print(
    llm: LLMEngine,
    text: str,
    direction: str,
    word_hints: Optional[Dict[str, str]] = None,
    grammar_tags: Optional[Dict[str, str]] = None,
) -> str:
    """Call LLM with streaming, print tokens live, return full text."""
    chunks: list[str] = []
    try:
        for chunk in llm.translate_stream(
            text, direction=direction,
            word_hints=word_hints, grammar_tags=grammar_tags,
        ):
            print(chunk, end="", flush=True)
            chunks.append(chunk)
    except RuntimeError:
        # Streaming failed — fall back to non-streaming
        if not chunks:
            result = llm.translate(
                text, direction=direction,
                word_hints=word_hints, grammar_tags=grammar_tags,
            )
            print(result, flush=True)
            return result.strip()
    print(flush=True)  # final newline
    return "".join(chunks).strip()


# ---------------------------------------------------------------------------
# Padapāṭha display
# ---------------------------------------------------------------------------


def _build_padapatha_display(
    iast_text: str,
    grammar_tags: Dict[str, str],
    word_hints: Dict[str, str],
    verifier: VerificationEngine,
    lexicon: LexiconEngine,
) -> str:
    """Build a word-by-word breakdown: word → grammar → English gloss."""
    words = re.split(r"[\s]+", iast_text)
    lines: list[str] = []
    for w in words:
        w_clean = w.strip(".,;:!?|। ")
        if len(w_clean) < 2:
            continue

        # Grammar info
        grammar = grammar_tags.get(w_clean, "")
        if not grammar:
            vr = verifier.verify(w_clean)
            if vr.verified and vr.tags:
                tag = vr.tags[0]
                if tag.category == "noun":
                    g = f" [{tag.gender}]" if tag.gender else ""
                    grammar = f"{tag.details}{g}, root: {tag.root}"
                elif tag.category == "verb":
                    grammar = f"{tag.details}, dhātu: {tag.root}"
                elif tag.category == "dict_headword":
                    grammar = "headword"

        # English gloss
        gloss = word_hints.get(w_clean, "")
        if not gloss:
            lr = lexicon.lookup(w_clean)
            if lr.has_exact:
                gloss = lr.exact_match.english[:60]
            elif lr.has_partial:
                for _, entries in lr.word_matches.items():
                    if entries:
                        gloss = entries[0].english[:60]
                        break

        parts = [f"  {w_clean}"]
        if grammar:
            parts.append(f"  —  {grammar}")
        if gloss:
            parts.append(f'  —  "{gloss}"')
        lines.append("".join(parts))

    if not lines:
        return ""
    return "Padapāṭha:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Samāsa (compound) analysis
# ---------------------------------------------------------------------------


def _detect_likely_compounds(
    words: list[str], verifier: VerificationEngine
) -> list[str]:
    """Return words that are likely compounds (long + not a simple paradigm form)."""
    compounds: list[str] = []
    for w in words:
        w = w.strip(".,;:!?|। ")
        if len(w) < 8:
            continue
        vr = verifier.verify(w)
        # If verified as a simple noun/verb form, it's probably not a compound
        if vr.verified and vr.tags:
            tag = vr.tags[0]
            if tag.category in ("noun", "verb") and tag.root != w:
                continue  # Known inflected form of a shorter root
        compounds.append(w)
    return compounds


def _analyze_compounds(
    words: list[str],
    verifier: VerificationEngine,
    llm: LLMEngine,
) -> str:
    """Detect likely compounds and ask the LLM to decompose them."""
    compounds = _detect_likely_compounds(words, verifier)
    if not compounds:
        return ""

    compound_list = "\n".join(f"  - {c}" for c in compounds[:8])  # limit
    prompt = (
        "Analyze the following Sanskrit compounds (IAST). For each, provide:\n"
        "1. Samāsa type (tatpuruṣa / bahuvrīhi / dvandva / karmadhāraya / "
        "avyayībhāva / dvigu)\n"
        "2. Constituent words separated by ' + '\n"
        "3. Meaning of each constituent\n\n"
        f"{compound_list}\n\n"
        "Format each as: COMPOUND → TYPE: parts (meanings)\n"
        "Be concise. One line per compound."
    )
    try:
        result = llm.translate(prompt, direction="san_to_eng")
        if result and len(result.strip()) > 5:
            return "Samāsa analysis:\n" + textwrap.indent(result.strip(), "  ")
    except RuntimeError:
        pass
    return ""


# ---------------------------------------------------------------------------
# Multi-verse splitting
# ---------------------------------------------------------------------------


def _split_verses(text: str) -> list[str]:
    """Split Sanskrit text on double-danda boundaries (॥, ।।, ||).

    Returns a list of verse strings.  Single-danda (।) is NOT used as a
    verse boundary — it typically marks a half-verse (pāda).
    """
    # Normalize: ॥ (U+0965) and ।। (two U+0964) and || (ASCII)
    parts = re.split(r"\u0965|\u0964\u0964|\|\|", text)
    verses: list[str] = []
    for p in parts:
        p = p.strip(" \t\n\r।|")
        if p:
            verses.append(p)
    return verses


# ---------------------------------------------------------------------------
# Translation pipelines
# ---------------------------------------------------------------------------


def translate_san_to_eng(
    text: str,
    lexicon: LexiconEngine,
    llm: LLMEngine,
    sandhi: SandhiEngine,
    verifier: VerificationEngine,
    cfg: Dict[str, Any],
    *,
    use_split: bool = False,
) -> str:
    """Sanskrit → English translation pipeline."""
    # 1. Normalize to IAST
    detected = detect_script(text)
    if detected == "devanagari":
        print("  [Devanagari detected → converting to IAST]", flush=True)
    iast_text = ensure_iast(text)

    # 2. Deterministic lexicon lookup — use as reference + context, not replacement
    print("  Lexicon lookup...", end="", flush=True)
    annotations: list[str] = []
    db_translation: Optional[str] = None
    if cfg.get("translation", {}).get("use_lexicon_first", True):
        result = lexicon.lookup(text)
        if result.has_exact:
            entry = result.exact_match
            db_translation = entry.english
            source_label = entry.source_file or "lexicon"
            annotations.append(
                f"  [DB reference]: {entry.english}  ({source_label})"
            )
    print(" done.", flush=True)

    # 3. Optional sandhi splitting (only when explicitly requested)
    split_text: Optional[str] = None
    if use_split and sandhi.split_available:
        try:
            raw_split = sandhi.split(iast_text)
            first_line = raw_split.split("\n")[0].split("[")[0].strip()
            if first_line and first_line != iast_text:
                split_text = first_line.replace("-", " ")
                annotations.append(f"  [Sandhi split: {first_line}]")
        except SandhiSplitError:
            pass

    # 4. Grammatical analysis (vibhakti / lakāra tagging)
    print("  Grammar analysis...", end="", flush=True)
    analysis_words = (split_text or iast_text).split()
    grammar_tags = _build_grammar_context(analysis_words, verifier)
    print(f" {len(grammar_tags)} tags.", flush=True)

    # 5. Gather word hints for LLM context
    lookup_text = split_text or text
    word_hints: Dict[str, str] = {}
    # Seed with full-sentence DB translation as a strong hint
    if db_translation:
        word_hints["[full sentence]"] = db_translation
    if cfg.get("translation", {}).get("merge_lexicon_with_llm", True):
        result = lexicon.lookup(lookup_text)
        if result.has_partial:
            for word, entries in result.word_matches.items():
                if entries:
                    word_hints[word] = entries[0].english

    # 6. FTS5 search fallback — boost word hints with fuzzy matches
    if not word_hints and hasattr(lexicon, "semantic_lookup"):
        sem_hits = lexicon.semantic_lookup(lookup_text, n_results=3)
        for h in sem_hits:
            if h.english:
                key = h.sanskrit_iast or h.sanskrit
                word_hints[key] = h.english
        if word_hints:
            annotations.append("  [Semantic search hints provided]")

    # 7. LLM translation with enriched context (streamed)
    print("  LLM translating (streaming):", flush=True)
    llm_input = iast_text
    if split_text and split_text != iast_text:
        llm_input = f"{iast_text}\n\nPadapāṭha (unsandhied): {split_text}"

    try:
        translation = _llm_stream_print(
            llm, llm_input, "san_to_eng",
            word_hints=word_hints or None,
            grammar_tags=grammar_tags or None,
        )
    except RuntimeError as exc:
        return f"[LLM Error] {exc}"

    # 8. Confidence scoring + retry loop
    confidence = _compute_confidence(
        translation, word_hints, grammar_tags, split_text
    )

    # --- Retry pass 1: try with sandhi split if not already used ---
    if confidence == "LOW" and not use_split and sandhi.split_available:
        print("  Retrying with sandhi split...", flush=True)
        annotations.append("  [LOW confidence → retrying with sandhi split...]")
        try:
            raw_split = sandhi.split(iast_text)
            first_line = raw_split.split("\n")[0].split("[")[0].strip()
            if first_line and first_line != iast_text:
                retry_split = first_line.replace("-", " ")
                retry_words = retry_split.split()
                retry_grammar = _build_grammar_context(retry_words, verifier)
                retry_hints: Dict[str, str] = {}
                retry_result = lexicon.lookup(retry_split)
                if retry_result.has_partial:
                    for word, entries in retry_result.word_matches.items():
                        if entries:
                            retry_hints[word] = entries[0].english
                retry_input = f"{iast_text}\n\nPadapāṭha (unsandhied): {retry_split}"
                translation = llm.translate(
                    retry_input,
                    direction="san_to_eng",
                    word_hints=retry_hints or word_hints or None,
                    grammar_tags=retry_grammar or grammar_tags or None,
                )
                # Merge new evidence for scoring
                grammar_tags = {**grammar_tags, **retry_grammar}
                word_hints = {**word_hints, **retry_hints}
                split_text = retry_split
                confidence = _compute_confidence(
                    translation, word_hints, grammar_tags, split_text
                )
        except (SandhiSplitError, RuntimeError):
            pass

    # --- Retry pass 2: LLM self-critique ---
    if confidence == "LOW":
        print("  LLM self-critique...", flush=True)
        annotations.append("  [LOW confidence → requesting LLM self-critique...]")
        critique_prompt = (
            f"You previously translated this Sanskrit text:\n"
            f"  {iast_text}\n\n"
            f"Your translation was:\n"
            f"  {translation}\n\n"
            f"Please critically review your translation for accuracy. "
            f"Check each word against its grammatical form, verify "
            f"compound (samāsa) analysis, and correct any errors. "
            f"Output ONLY the improved translation, nothing else."
        )
        try:
            refined = llm.translate(
                critique_prompt,
                direction="san_to_eng",
                word_hints=word_hints or None,
                grammar_tags=grammar_tags or None,
            )
            if refined and len(refined.strip()) > 5:
                translation = refined
                annotations.append("  [Refined via self-critique]")
                confidence = _compute_confidence(
                    translation, word_hints, grammar_tags, split_text
                )
        except RuntimeError:
            pass

    annotations.append(f"  [Confidence: {confidence}]")
    if grammar_tags:
        annotations.append(f"  [Grammar analysis: {len(grammar_tags)} words tagged]")
    if word_hints:
        annotations.append("  [Lexicon hints were provided to the LLM]")

    # --- Samāsa (compound) analysis ---
    compound_section = ""
    analysis_w = (split_text or iast_text).split()
    compounds_found = _detect_likely_compounds(analysis_w, verifier)
    if compounds_found:
        print("  Analyzing compounds...", flush=True)
        compound_section = _analyze_compounds(analysis_w, verifier, llm)

    # --- Padapāṭha (word-by-word breakdown) ---
    padapatha = _build_padapatha_display(
        iast_text, grammar_tags, word_hints, verifier, lexicon,
    )

    # --- Assemble output ---
    output_parts = [translation]
    if padapatha:
        output_parts.append(padapatha)
    if compound_section:
        output_parts.append(compound_section)
    output_parts.append("\n".join(annotations))
    return "\n\n".join(output_parts)


def _verify_sanskrit_output(
    raw_sanskrit: str, verifier: VerificationEngine
) -> tuple[int, int, list[str]]:
    """Verify each word in *raw_sanskrit* and return (verified, total, unverified_list)."""
    verified = 0
    total = 0
    unverified: list[str] = []
    for w in raw_sanskrit.split():
        w = w.strip(".,;:!? ")
        if len(w) < 2:
            continue
        total += 1
        vr = verifier.verify(w)
        if vr.verified:
            verified += 1
        else:
            unverified.append(w)
    return verified, total, unverified


def _build_grammar_report(
    raw_sanskrit: str, verifier: VerificationEngine
) -> str:
    """Build a detailed Pāṇinian grammar analysis of *raw_sanskrit* output.

    Returns a human-readable report suitable for inclusion in an LLM
    self-critique prompt so the model can see exactly what each word
    was parsed as (vibhakti, liṅga, lakāra, etc.).
    """
    lines: list[str] = []
    for w in raw_sanskrit.split():
        w = w.strip(".,;:!? ")
        if len(w) < 2:
            continue
        vr = verifier.verify(w)
        if vr.verified and vr.tags:
            tag = vr.tags[0]  # primary parse
            if tag.category == "noun":
                g = f", {tag.gender}" if tag.gender else ""
                lines.append(f"  {w}: {tag.details}{g} (prātipadika: {tag.root})")
            elif tag.category == "verb":
                lines.append(f"  {w}: {tag.details} (dhātu: {tag.root})")
            elif tag.category == "dict_headword":
                lines.append(f"  {w}: dictionary headword — {tag.details[:60]}")
        else:
            lines.append(f"  {w}: not found in local paradigm tables")
    return "\n".join(lines)


def _enrich_word_hints_with_grammar(
    word_hints: Dict[str, str], verifier: VerificationEngine
) -> Dict[str, str]:
    """Augment word hints with grammar info (gender, category) for each
    Sanskrit value so the LLM knows the liṅga / prātipadika class."""
    enriched: Dict[str, str] = {}
    for eng_key, san_val in word_hints.items():
        # Skip the full-sentence pseudo-key
        if eng_key.startswith("["):
            enriched[eng_key] = san_val
            continue
        # Look up the Sanskrit form's root/gender
        for san_word in san_val.split():
            san_word = san_word.strip(".,;:!? ")
            if len(san_word) < 2:
                continue
            vr = verifier.verify(san_word)
            if vr.verified and vr.tags:
                tag = vr.tags[0]
                if tag.category == "noun" and tag.gender:
                    san_val += f" [{tag.gender}., root: {tag.root}]"
                elif tag.category == "verb":
                    san_val += f" [dhātu: {tag.root}]"
                break  # one annotation per hint is enough
        enriched[eng_key] = san_val
    return enriched


def _compute_eng_to_san_confidence(
    verified: int,
    total: int,
    word_hints_count: int,
    consensus_count: int,
) -> str:
    """Confidence for ENG→SAN factoring morphology, lexicon hits, and consensus."""
    score = 0.0
    # Morphology ratio (0-5 pts)
    if total > 0:
        score += (verified / total) * 5.0
    # Lexicon word hints found (0-3 pts)
    score += min(word_hints_count, 3)
    # Consensus bonus (0-3 pts)
    if consensus_count >= 3:
        score += 3
    elif consensus_count >= 2:
        score += 2

    if score >= 7:
        return "HIGH"
    if score >= 4:
        return "MEDIUM"
    return "LOW"


def translate_eng_to_san(
    text: str,
    lexicon: LexiconEngine,
    llm: LLMEngine,
    sandhi: SandhiEngine,
    verifier: VerificationEngine,
    cfg: Dict[str, Any],
    script_mode: str = "iast",
    *,
    iterations: int = 0,
) -> str:
    """English → Sanskrit translation pipeline with multi-iteration refinement."""
    max_iter = iterations or cfg.get("translation", {}).get("iteration_count", 5)
    output_lines: list[str] = []

    # ── Step 0: Full-sentence reverse lookup (reference, not replacement) ──
    db_reference: Optional[str] = None  # IAST form of DB hit, shown alongside LLM
    if cfg.get("translation", {}).get("use_lexicon_first", True):
        print("  [Lexicon] Searching for exact match...", flush=True)
        entries = lexicon.reverse_lookup(text)
        if entries:
            entry = entries[0]
            iast_san = entry.sanskrit_iast or ensure_iast(entry.sanskrit)
            db_reference = iast_san
            source = entry.source_file or "lexicon"
            print(f"  [Lexicon] ✓ Reference found in {source}", flush=True)
            output_lines.append(
                f"  [DB reference]: {_apply_output_script(iast_san, script_mode)}"
                f"  ({source})"
            )
        else:
            print("  [Lexicon] No exact sentence match.", flush=True)

    # ── Step 1: Word-level reverse lookup → build hints ──────────────
    word_hints: Dict[str, str] = {}
    # Seed with full-sentence DB reference as a strong hint
    if db_reference:
        word_hints["[full sentence]"] = db_reference
    print("  [Lexicon] Word-level reverse lookup...", flush=True)
    word_hits = lexicon.reverse_lookup_words(text)
    found_words: list[str] = []
    missed_words: list[str] = []
    for w in text.split():
        w_clean = w.strip(".,;:!?")
        if len(w_clean) < 2:
            continue
        # Check if this word was part of any matched n-gram
        matched = False
        for phrase, entry in word_hits.items():
            if w_clean.lower() in phrase.lower():
                matched = True
                break
        if matched:
            found_words.append(w_clean)
        else:
            missed_words.append(w_clean)

    for phrase, entry in word_hits.items():
        san = entry.sanskrit_iast or entry.sanskrit
        word_hints[phrase] = san
        output_lines.append(f"  [Lexicon] \"{phrase}\" → {san}")

    if found_words:
        print(f"  [Lexicon] ✓ Found: {', '.join(found_words)}", flush=True)
    if missed_words:
        print(f"  [Lexicon] ✗ Not found: {', '.join(missed_words)}", flush=True)
    if not word_hits:
        # FTS5 broad fallback for any partial context
        sem_hits = lexicon.semantic_lookup(text, n_results=3)
        for h in sem_hits:
            if h.english:
                key = h.sanskrit_iast or h.sanskrit
                word_hints[key] = h.english
        if word_hints:
            output_lines.append("  [Semantic search hints provided]")
            print("  [Lexicon] Semantic search provided context hints.", flush=True)

    # Enrich word hints with grammar info (gender, dhātu, etc.)
    word_hints = _enrich_word_hints_with_grammar(word_hints, verifier)

    # ── Step 2: Multi-iteration LLM loop ─────────────────────────────
    candidates: list[dict] = []  # {"text": str, "conf": str, "verified": int, "total": int, "unverified": list}
    best_idx = 0
    best_score = -1
    CONF_SCORE = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "UNKNOWN": 0}

    for iteration in range(1, max_iter + 1):
        print(f"  [Iteration {iteration}/{max_iter}] Translating...", flush=True)

        # Build prompt: on pass 2+ include grammar-enriched self-critique
        if iteration == 1:
            try:
                raw = llm.translate(
                    text,
                    direction="eng_to_san",
                    word_hints=word_hints or None,
                )
            except RuntimeError as exc:
                return f"[LLM Error] {exc}"
        else:
            prev = candidates[best_idx]["text"]
            unverified_list = candidates[best_idx].get("unverified", [])

            # Only include detailed grammar analysis if there are actual
            # problems (unverified forms).  For clean output, keep the
            # critique light so the LLM doesn't over-correct good work.
            if unverified_list:
                print(f"  [Iteration {iteration}/{max_iter}] Analyzing grammar...", flush=True)
                grammar_report = _build_grammar_report(prev, verifier)
                critique_prompt = (
                    f"You previously translated this English text into Sanskrit:\n"
                    f"  {text}\n\n"
                    f"Your Sanskrit output was:\n"
                    f"  {prev}\n\n"
                    f"Morphological analysis (from Pāṇinian tables):\n"
                    f"{grammar_report}\n\n"
                    f"The following forms were NOT found in local paradigm "
                    f"tables: {', '.join(unverified_list)}\n\n"
                    f"Please fix only the problematic forms. Ensure vibhakti, "
                    f"liṅga, and vacana agreement. Output ONLY the corrected "
                    f"IAST Sanskrit, nothing else."
                )
            else:
                critique_prompt = (
                    f"You previously translated this English text into Sanskrit:\n"
                    f"  {text}\n\n"
                    f"Your Sanskrit output was:\n"
                    f"  {prev}\n\n"
                    f"All forms verified in Pāṇinian tables. If the translation "
                    f"is accurate, output it again unchanged. Otherwise make "
                    f"only minimal corrections and output ONLY IAST Sanskrit."
                )
            try:
                raw = llm.translate(
                    critique_prompt,
                    direction="eng_to_san",
                    word_hints=word_hints or None,
                )
            except RuntimeError:
                print(f"  [Iteration {iteration}/{max_iter}] LLM error, skipping.", flush=True)
                continue

        if not raw or len(raw.strip()) < 3:
            print(f"  [Iteration {iteration}/{max_iter}] Empty response, skipping.", flush=True)
            continue

        raw = raw.strip()

        # Verify morphology
        verified, total, unverified = _verify_sanskrit_output(raw, verifier)

        # Count consensus
        consensus = sum(1 for c in candidates if c["text"].strip() == raw)

        conf = _compute_eng_to_san_confidence(
            verified, total, len(word_hints), consensus + 1
        )
        score = CONF_SCORE.get(conf, 0)
        # Tie-break: prefer higher verification ratio
        if total > 0:
            score += (verified / total) * 0.5

        candidates.append({
            "text": raw,
            "conf": conf,
            "verified": verified,
            "total": total,
            "unverified": unverified,
        })

        if score > best_score:
            best_score = score
            best_idx = len(candidates) - 1

        v_label = f"{verified}/{total} verified" if total > 0 else "no morphology data"
        print(
            f"  [Iteration {iteration}/{max_iter}] {conf} confidence "
            f"({v_label})",
            flush=True,
        )

        # Early exit: MEDIUM or HIGH → no need for more iterations
        if conf in ("HIGH", "MEDIUM"):
            print(f"  [Iteration {iteration}/{max_iter}] {conf} → stopping early.", flush=True)
            break

    if not candidates:
        return "[Error] All LLM iterations failed."

    # ── Step 3: Consensus detection ──────────────────────────────────
    text_counts = Counter(c["text"].strip() for c in candidates)
    most_common_text, most_common_count = text_counts.most_common(1)[0]
    if most_common_count >= 2:
        # Find the candidate with the consensus text and best score
        for i, c in enumerate(candidates):
            if c["text"].strip() == most_common_text:
                best_idx = i
                break
        output_lines.append(
            f"  [Consensus: {most_common_count}/{len(candidates)} iterations agree]"
        )
        # Recalculate with consensus boost
        bc = candidates[best_idx]
        bc["conf"] = _compute_eng_to_san_confidence(
            bc["verified"], bc["total"], len(word_hints), most_common_count
        )

    # ── Step 4: Assemble final output ────────────────────────────────
    best = candidates[best_idx]
    raw_sanskrit = best["text"]
    final_lines = [_apply_output_script(raw_sanskrit, script_mode)]

    # Pairwise sandhi joining (per word-boundary, not whole sentence)
    if cfg.get("translation", {}).get("apply_sandhi_on_output", True) and sandhi.join_available:
        san_words = raw_sanskrit.split()
        if len(san_words) > 1:
            try:
                sandhied, sandhi_details = sandhi.join_pairwise(list(san_words))
                if sandhied != raw_sanskrit:
                    final_lines.append(
                        f"  [Sandhied]: {_apply_output_script(sandhied, script_mode)}"
                    )
                if sandhi_details:
                    for left, right, joined in sandhi_details:
                        final_lines.append(
                            f"    {left} + {right} → {joined}"
                        )
            except SandhiJoinError as exc:
                final_lines.append(f"  [Sandhi unavailable: {exc}]")

    if best["unverified"]:
        final_lines.append(f"  [Unverified forms: {', '.join(best['unverified'])}]")

    # Annotations from lexicon lookup
    final_lines.extend(output_lines)

    final_lines.append(f"  [Confidence: {best['conf']}]")
    if len(candidates) > 1:
        final_lines.append(f"  [Iterations: {len(candidates)} passes]")
    if word_hints:
        final_lines.append(
            f"  [Lexicon hints: {len(word_hints)} word(s) provided to LLM]"
        )

    return "\n".join(final_lines)


# ===================================================================
# Command handlers
# ===================================================================


def cmd_verify(args: str, verifier: VerificationEngine) -> str:
    word = args.strip()
    if not word:
        return "Usage: verify <word>"
    vr = verifier.verify(word)
    lines = [f"Word: {vr.word}"]
    if vr.verified:
        lines.append("Status: VERIFIED")
        for tag in vr.tags[:5]:
            detail = f"  {tag.category}: root={tag.root}"
            if tag.details:
                detail += f"  ({tag.details})"
            if tag.gender:
                detail += f"  [{tag.gender}]"
            lines.append(detail)
    else:
        lines.append("Status: NOT FOUND")
    for w in vr.warnings:
        lines.append(f"  Warning: {w}")
    return "\n".join(lines)


def cmd_split(args: str, sandhi: SandhiEngine) -> str:
    text = args.strip()
    if not text:
        return "Usage: split <text>"
    try:
        result = sandhi.split(text)
        return f"Unsandhied: {result}"
    except SandhiSplitError as exc:
        return f"[Sandhi Split Error] {exc}"


def cmd_dict(args: str, verifier: VerificationEngine) -> str:
    word = args.strip()
    if not word:
        return "Usage: dict <word>"
    defn = verifier.lookup_headword(word)
    if defn:
        return f"{word}: {defn}"
    return f"'{word}' not found in Monier-Williams dictionary."


def cmd_profile(args: str, llm: LLMEngine) -> str:
    name = args.strip()
    if not name:
        profiles = llm.available_profiles
        lines = [f"Active profile: {llm.active_profile} — {llm.active_profile_description}"]
        if profiles:
            lines.append("Available profiles:")
            for p in profiles:
                marker = " *" if p == llm.active_profile else ""
                desc = llm.config.get("profiles", {}).get(p, {}).get("description", "")
                lines.append(f"  {p}{marker}  — {desc}" if desc else f"  {p}{marker}")
        return "\n".join(lines)
    try:
        llm.set_profile(name)
        return f"Profile: {llm.active_profile} — {llm.active_profile_description}"
    except ValueError as exc:
        return f"[Error] {exc}"


def cmd_script(args: str, cfg: Dict[str, Any]) -> tuple[str, str]:
    """Toggle or set the output script mode. Returns (message, new_mode)."""
    current = cfg.get("transliteration", {}).get("output_script", "iast")
    mode = args.strip().lower()
    if not mode:
        cycle = {"iast": "devanagari", "devanagari": "both", "both": "iast"}
        mode = cycle.get(current, "iast")
    if mode not in ("iast", "devanagari", "both"):
        return f"Unknown script mode '{mode}'. Use: iast, devanagari, both", current
    cfg.setdefault("transliteration", {})["output_script"] = mode
    labels = {"iast": "IAST", "devanagari": "Devanāgarī", "both": "IAST + Devanāgarī"}
    return f"Output script: {labels.get(mode, mode)}", mode


def cmd_translate(
    args: str,
    direction: str,
    lexicon: LexiconEngine,
    llm: LLMEngine,
    sandhi: SandhiEngine,
    verifier: VerificationEngine,
    cfg: Dict[str, Any],
    script_mode: str,
) -> str:
    """Translate a text file verse-by-verse."""
    file_path = args.strip()
    if not file_path:
        return "Usage: translate-file <file_path>"
    src = Path(file_path)
    if not src.is_absolute():
        src = _PROJECT_ROOT / src
    if not src.exists():
        return f"File not found: {src}"

    try:
        raw = src.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"Cannot read file: {exc}"

    # Split into verses/paragraphs (double-newline or ॥ delimiter)
    import re as _re
    blocks = _re.split(r"\n\s*\n|\u0965", raw)
    blocks = [b.strip() for b in blocks if b.strip()]

    if not blocks:
        return "File is empty or has no translatable content."

    # Determine output path
    out_path = src.with_name(src.stem + "_translated" + src.suffix)

    results: list[str] = []
    total = len(blocks)
    for i, block in enumerate(blocks, 1):
        print(f"  Translating {i}/{total}...", end="\r")
        block_oneline = " ".join(block.splitlines())
        if direction == "san_to_eng":
            tr = translate_san_to_eng(
                block_oneline, lexicon, llm, sandhi, verifier, cfg
            )
        else:
            tr = translate_eng_to_san(
                block_oneline, lexicon, llm, sandhi, verifier, cfg, script_mode
            )
        results.append(f"--- [{i}/{total}] ---\nSource: {block_oneline}\n\n{tr}")

    output = "\n\n".join(results)
    try:
        out_path.write_text(output, encoding="utf-8")
    except OSError as exc:
        return f"Translated {total} blocks but failed to write output: {exc}\n\n{output}"

    print()  # clear the \r progress line
    return f"Translated {total} blocks → {out_path}"


def cmd_status(
    direction: str,
    script_mode: str,
    llm: LLMEngine,
    lexicon: LexiconEngine,
    sandhi: SandhiEngine,
    verifier: VerificationEngine,
) -> str:
    script_labels = {"iast": "IAST", "devanagari": "Devanāgarī", "both": "IAST + Devanāgarī"}
    lines = [
        f"Direction:       {DIRECTION_LABELS.get(direction, direction)}",
        f"Output script:   {script_labels.get(script_mode, script_mode)}",
        f"LLM Provider:    {llm.active_provider}",
        f"LLM Model:       {llm.active_model}",
        f"Profile:         {llm.active_profile} — {llm.active_profile_description}",
        f"Lexicon entries: {lexicon.entry_count}",
        f"Sandhi join:     {'available' if sandhi.join_available else 'unavailable'}",
        f"Sandhi split:    {'available' if sandhi.split_available else 'unavailable'}",
        f"Transliteration: {'available' if translit_available() else 'unavailable (install indic-transliteration)'}",
    ]
    return "\n".join(lines)


# ===================================================================
# REPL
# ===================================================================


def main() -> None:
    # --- Ensure UTF-8 console I/O (Windows PowerShell fix) ---
    if sys.platform == "win32":
        try:
            sys.stdin.reconfigure(encoding="utf-8", errors="replace")
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass  # Fallback silently if reconfigure isn't available

    # Load and validate config
    cfg = load_config(CONFIG_FILE)
    setup_logging(cfg)
    logger = logging.getLogger("varnabuddhi")

    app = cfg["application"]
    print(BANNER.format(
        author=app.get("author", ""),
        year=str(app.get("year", "")),
        version=app.get("version", ""),
    ))

    # Initialize engines
    data_dir = resolve_path(cfg, "data_dir")
    decls_dir = resolve_path(cfg, "decls_dir")
    sandhi_dir = resolve_path(cfg, "sandhi_dir")
    sandhi_split_dir = resolve_path(cfg, "sandhi_split_dir")
    cache_dir = resolve_path(cfg, "cache_dir")

    print("Initializing engines...", flush=True)

    lexicon = LexiconEngine(data_dir, cache_dir=cache_dir)
    try:
        count = lexicon.load()
        src = "SQLite cache" if lexicon._using_cache else "text files"
        print(f"  Lexicon:        {count} entries loaded ({src})", flush=True)
    except Exception as exc:
        print(f"  Lexicon:        failed ({exc})", flush=True)

    verifier = VerificationEngine(decls_dir, cache_dir=cache_dir)
    try:
        verifier.load(load_declensions=True, load_verbs=True, load_dictionary=True)
        if verifier._using_cache:
            print(f"  Verification:   ready (SQLite cache)", flush=True)
        else:
            print(f"  Verification:   ready ({len(verifier._known_forms)} forms, "
                  f"{len(verifier._dict_headwords)} headwords)", flush=True)
    except Exception as exc:
        print(f"  Verification:   failed ({exc})", flush=True)

    sandhi = SandhiEngine(sandhi_dir, sandhi_split_dir)
    print(f"  Sandhi join:    {'ready' if sandhi.join_available else 'unavailable (Perl not found or scripts missing)'}")
    print(f"  Sandhi split:   {'ready' if sandhi.split_available else 'unavailable'}")
    print(f"  Transliteration: {'ready' if translit_available() else 'unavailable (pip install indic-transliteration)'}")

    llm = LLMEngine(cfg["llm"])
    usable = llm.usable_providers
    if usable:
        model_label = llm.active_model or "(default)"
        print(f"  LLM:            {llm.active_provider} / {model_label}")
        if len(usable) > 1:
            others = [p for p in usable if p != llm.active_provider]
            print(f"                  fallbacks: {', '.join(others)}")
    else:
        print("  LLM:            [WARNING] no usable provider configured")
        print("                  Set an API key in config.json or start Ollama.")
    print(f"  Profile:        {llm.active_profile} — {llm.active_profile_description}")

    direction = cfg.get("translation", {}).get("default_direction", "san_to_eng")
    script_mode = cfg.get("transliteration", {}).get("output_script", "iast")
    if script_mode == "auto":
        script_mode = "iast"  # resolve 'auto' to a concrete default
    print(f"\nDirection: {DIRECTION_LABELS.get(direction, direction)}")
    print("Type 'help' for commands, 'quit' to exit.\n")

    # REPL loop
    _GREEN = "\033[32m"
    _RESET = "\033[0m"
    while True:
        try:
            user_input = input(f"{_GREEN}input»{_RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nNamasté.")
            break

        if not user_input:
            continue

        # --- Parse command vs. translation input ---
        parts = user_input.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        # --- Command dispatch ---
        if cmd in ("quit", "exit", "q"):
            print("Namasté.")
            break
        elif cmd == "help":
            print(HELP_TEXT)
        elif cmd == "dir":
            direction = (
                "eng_to_san" if direction == "san_to_eng" else "san_to_eng"
            )
            print(f"Direction: {DIRECTION_LABELS[direction]}")
        elif cmd == "model":
            model_parts = args.split(maxsplit=1)
            provider = model_parts[0] if model_parts else ""
            model = model_parts[1] if len(model_parts) > 1 else ""
            if not provider:
                print(f"Current: {llm.active_provider} / {llm.active_model}")
                print(f"Available: {', '.join(llm.available_providers)}")
            else:
                try:
                    llm.set_provider(provider, model)
                    print(f"Switched to {llm.active_provider} / {llm.active_model}")
                except ValueError as exc:
                    print(f"[Error] {exc}")
        elif cmd == "profile":
            print(cmd_profile(args, llm))
        elif cmd == "script":
            msg, script_mode = cmd_script(args, cfg)
            print(msg)
        elif cmd == "verify":
            print(cmd_verify(args, verifier))
        elif cmd == "split":
            print(cmd_split(args, sandhi))
        elif cmd == "dict":
            print(cmd_dict(args, verifier))
        elif cmd == "translate-file":
            print(cmd_translate(
                args, direction, lexicon, llm, sandhi, verifier, cfg,
                script_mode,
            ))
        elif cmd == "status":
            print(cmd_status(
                direction, script_mode, llm, lexicon, sandhi, verifier,
            ))
        else:
            # --- Translation ---
            use_split = False
            iter_override = 0
            translate_text = user_input
            if cmd == "--split":
                use_split = True
                translate_text = args
            elif cmd == "--iteration":
                # Parse: --iteration N <text>
                iter_parts = args.split(maxsplit=1)
                try:
                    iter_override = int(iter_parts[0]) if iter_parts else 5
                except ValueError:
                    iter_override = 5
                translate_text = iter_parts[1] if len(iter_parts) > 1 else ""
            if not translate_text:
                continue

            # --- Multi-verse splitting ---
            verses = _split_verses(translate_text) if direction == "san_to_eng" else [translate_text]
            if len(verses) > 1:
                print(f"  [{len(verses)} verses detected]\n", flush=True)

            for v_idx, verse in enumerate(verses, 1):
                if len(verses) > 1:
                    print(f"{'='*50}", flush=True)
                    print(f"  [Verse {v_idx}/{len(verses)}]: {verse[:80]}{'...' if len(verse) > 80 else ''}", flush=True)
                    print(f"{'='*50}", flush=True)

                if direction == "san_to_eng":
                    result = translate_san_to_eng(
                        verse, lexicon, llm, sandhi, verifier, cfg,
                        use_split=use_split,
                    )
                else:
                    result = translate_eng_to_san(
                        translate_text, lexicon, llm, sandhi, verifier, cfg,
                        script_mode, iterations=iter_override,
                    )

                print(f"\n{result}\n")


if __name__ == "__main__":
    main()
