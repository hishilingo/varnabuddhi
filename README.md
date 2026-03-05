# वर्णबुद्धि — Varnabuddhi

**Academic Sanskrit ⇄ English Translator**

A cross-platform Python CLI that combines LLM intelligence with local lexical databases, Pāṇinian morphological verification, sandhi processing, and offline transliteration to produce philologically rigorous Sanskrit translations.

## Features

- **Bidirectional translation** — Sanskrit → English and English → Sanskrit
- **Streaming LLM output** — tokens appear in real time as they're generated
- **Multi-provider LLM support** — Google Gemini (default), OpenAI, Anthropic, Openrouter.ai, Ollama (local)
- **5 prompt profiles** — default, literal, academic, philosophical, poetic
- **Morphological verification** — declension/conjugation tables + Monier-Williams dictionary
- **Sandhi processing** — Perl-based joining (per word-boundary) and neural splitting
- **Script support** — Devanagari input auto-detected, IAST used internally, output in IAST / Devanagari / both
- **Padapāṭha display** — word-by-word breakdown with grammar tags and English glosses
- **Samāsa analysis** — automatic compound detection and decomposition (tatpuruṣa, bahuvrīhi, etc.)
- **Multi-verse splitting** — paste an entire stotra and each verse is translated separately
- **Multi-iteration refinement** — ENG→SAN pipeline runs up to N passes with grammar-enriched self-critique
- **SQLite+FTS5 caching** — lexicon and morphology data cached for fast startup after first run
- **Confidence scoring** — each translation rated HIGH / MEDIUM / LOW based on evidence

## Requirements

- **Python 3.10+**
- **Strawberry Perl** (optional, for sandhi joining) — [strawberryperl.com](https://strawberryperl.com)
- At least one LLM provider API key (Gemini recommended)

### Python Dependencies

```
pip install google-genai indic-transliteration
```

Optional (for other LLM providers):
```
pip install openai anthropic
```

## Quick Start

1. **Clone or download** the project.

2. **Configure** — edit `config.json` and set your API key:
   ```json
   "gemini": {
       "api_key": "YOUR_GEMINI_API_KEY",
       "model": "gemini-2.5-flash"
   }
   ```

3. **Run**:
   ```
   python varnabuddhi.py
   ```

4. **Translate** — type or paste Sanskrit text at the `input»` prompt:
   ```
   input» dharmasya tattvaṃ nihitaṃ guhāyām
   ```

First run will take a few minutes while the lexicon and morphology caches are built. Subsequent starts are fast.

## Project Structure

```
varnabuddhi/
├── varnabuddhi.py              # Main application + REPL
├── config.json                 # Configuration (API keys, profiles, paths)
├── engines/
│   ├── __init__.py
│   ├── llm_engine.py           # Multi-provider LLM (streaming + non-streaming)
│   ├── lexicon_engine.py       # Local Sanskrit–English lexicon + reverse lookup
│   ├── verification_engine.py  # Morphological verification (declensions, verbs, MW dict)
│   ├── transliteration_engine.py  # Devanagari ↔ IAST ↔ WX conversion
│   ├── sandhi_engine.py        # Sandhi joining (Perl) and splitting (neural)
│   └── cache_db.py             # SQLite + FTS5 caching layer
├── data/                       # Lexicon source files (Sanskrit–English pairs)
├── decls/                      # Declension tables, verb paradigms, MW dictionary
├── sandhi/                     # Perl sandhi joining scripts
├── sandhi_split/               # Neural sandhi splitting model
└── .cache/                     # Auto-generated SQLite caches
```

## Translation Pipeline

### Sanskrit → English
1. Script detection (Devanagari auto-converted to IAST)
2. Lexicon lookup (exact + word-level, shown as DB reference)
3. Sandhi splitting (optional, with `--split`)
4. Grammatical analysis (vibhakti / lakāra tagging from Pāṇinian tables)
5. Word hint gathering (lexicon + FTS5 fallback)
6. **Streaming LLM translation** with grammar context and word hints
7. Confidence scoring → automatic retry with sandhi split or self-critique if LOW
8. Samāsa (compound) analysis for long unverified forms
9. Padapāṭha (word-by-word breakdown) display

### English → Sanskrit
1. Full-sentence reverse lexicon lookup (shown as reference)
2. Word-level reverse lookup (n-gram based: trigrams → bigrams → words)
3. Grammar-enriched word hints (gender, dhātu annotations)
4. Multi-iteration LLM loop (default 5 passes) with self-critique
5. Morphological verification each iteration
6. Consensus detection across iterations
7. Pairwise sandhi joining on final output

## Configuration

See `config.json` for all options. Key sections:

- **`llm.providers`** — API keys and models for each provider
- **`llm.profiles`** — prompt profiles (system prompts per direction)
- **`llm.generation`** — temperature, top_p, top_k, max_output_tokens
- **`translation`** — iteration count, lexicon behavior, sandhi toggles
- **`transliteration.output_script`** — `iast`, `devanagari`, or `both`

## Credits

Idea/Development: Hishiryo 2026

## License

वर्णबुद्धि — Varnabuddhi is licensed under the Apache License, Version 2.0


