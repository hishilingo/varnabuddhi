# Varnabuddhi — Usage Guide

## Starting the Application

```
python varnabuddhi.py
```

On first run, the lexicon and morphology caches are built from the text files in `data/` and `decls/`. This takes a few minutes. Subsequent starts load from the SQLite cache in seconds.

The REPL displays a green `input»` prompt. Type text to translate, or a command.

## Translation

### Basic Translation

Simply type or paste text at the prompt. The current direction (shown at startup) determines the translation:

```
input» satyaṃ jñānaṃ anantaṃ brahma
```

Output includes the translation, padapāṭha (word-by-word breakdown), samāsa analysis (if compounds are detected), and metadata annotations.

### Devanagari Input

Devanagari script is auto-detected and converted to IAST internally:

```
input» ओंकारमूर्तिशिवसर्वसुखावहाय । व्योमस्वरूपनिधनोद्भवशास्वताय।।
```

### Multi-Verse Input

Paste multiple verses separated by double dandas (`॥`, `।।`, or `||`). Each verse is translated separately with numbered headers:

```
input» धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः।। मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय।।
```

Output:
```
  [2 verses detected]

==================================================
  [Verse 1/2]: dharmakṣetre kurukṣetre samavetā yuyutsavaḥ
==================================================

  <translation of verse 1>

==================================================
  [Verse 2/2]: māmakāḥ pāṇḍavāścaiva kimakurvata sañjaya
==================================================

  <translation of verse 2>
```

### Sandhi Splitting

Pre-process the input with sandhi splitting before translation:

```
input» --split tattvamasi
```

### Iteration Control (English → Sanskrit)

Override the number of refinement iterations for ENG→SAN:

```
input» --iteration 3 The self is eternal and unchanging
```

### File Translation

Translate an entire text file (one verse per block, separated by blank lines):

```
input» translate-file path/to/verses.txt
```

Output is written to `path/to/verses_translated.txt`.

## Output Sections

Each SAN→ENG translation displays:

1. **Translation** — the streamed LLM output (appears token-by-token)
2. **Padapāṭha** — word-by-word breakdown:
   ```
   Padapāṭha:
     satyam  —  Nom. Sg [n], root: satya  —  "truth, reality"
     jñānam  —  Nom. Sg [n], root: jñāna  —  "knowledge"
     anantam —  Nom. Sg [n], root: ananta  —  "infinite"
     brahma  —  headword  —  "the Absolute, Brahman"
   ```
3. **Samāsa analysis** — compound decomposition (only if compounds detected):
   ```
   Samāsa analysis:
     oṃkāramūrti → Tatpuruṣa: oṃkāra + mūrti (oṃ-form + embodiment)
     sarvasukhāvahāya → Bahuvrīhi: sarva + sukha + āvaha (all + happiness + bringing)
   ```
4. **Annotations** — DB references, confidence score, grammar tag count, etc.

Each ENG→SAN translation displays:
1. **Sanskrit output** (IAST, with optional Devanagari)
2. **Sandhied form** with per-boundary details
3. **Lexicon references** and word-level hits
4. **Confidence score** and iteration count

## Commands

### Translation Direction

```
input» dir
```

Toggles between Sanskrit → English and English → Sanskrit.

### LLM Provider / Model

```
input» model                          # Show current provider + model
input» model gemini gemini-2.5-flash  # Switch provider and model
input» model ollama llama3            # Switch to local Ollama
```

### Prompt Profile

```
input» profile                  # List available profiles
input» profile literal          # Switch to literal word-for-word
input» profile academic         # Heavy scholarly apparatus
input» profile philosophical    # Darśana-aware translation
input» profile poetic           # Verse-aware literary style
input» profile default          # Balanced scholarly (default)
```

**Profile descriptions:**

- **default** — Balanced scholarly translation. Preserves nuance, uses IAST for untranslatable terms.
- **literal** — Strict word-for-word. Each Sanskrit term shown in parentheses. No paraphrase.
- **academic** — Journal-grade with grammatical analysis, textual variants, Pāṇini sūtra references.
- **philosophical** — Identifies darśana school, preserves technical terms (brahman ≠ god, ātman ≠ soul).
- **poetic** — Identifies meter (chandas), renders alaṅkāra (figures of speech), preserves rasa (mood).

### Output Script

```
input» script iast          # IAST only (default)
input» script devanagari    # Devanagari only
input» script both          # Show both IAST and Devanagari
```

### Analysis Tools

```
input» verify dharmasya     # Check morphology of a word
input» split tattvamasi     # Sandhi-split a compound/phrase
input» dict ātman           # Look up in Monier-Williams dictionary
```

### Status

```
input» status
```

Shows current direction, output script, LLM provider/model, profile, lexicon size, sandhi availability, and transliteration status.

### Exit

```
input» quit
input» exit
input» q
```

## Configuration Reference

All settings are in `config.json`.

### LLM Providers

```json
"providers": {
    "gemini": {
        "api_key": "YOUR_KEY",
        "model": "gemini-2.5-flash"
    },
    "openai": {
        "api_key": "YOUR_KEY",
        "model": "gpt-4o",
        "base_url": "https://api.openai.com/v1"
    },
    "anthropic": {
        "api_key": "YOUR_KEY",
        "model": "claude-sonnet-4-20250514"
    },
    "ollama": {
        "host": "localhost",
        "port": 11434,
        "model": "llama3"
    }
}
```

Only one provider needs to be configured. The `fallback_order` array controls automatic fallback if the primary fails.

### Generation Parameters

```json
"generation": {
    "temperature": 0.1,
    "max_output_tokens": 4096,
    "top_p": 0.9,
    "top_k": 40,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}
```

These can be overridden per-profile or per-provider.

### Translation Settings

```json
"translation": {
    "default_direction": "san_to_eng",
    "use_lexicon_first": true,
    "merge_lexicon_with_llm": true,
    "apply_sandhi_on_output": true,
    "split_sandhi_on_input": false,
    "verify_morphology": true,
    "iteration_count": 5
}
```

- **`iteration_count`** — default refinement passes for ENG→SAN (overridable with `--iteration N`)
- **`use_lexicon_first`** — show DB reference when an exact match exists
- **`merge_lexicon_with_llm`** — feed word-level lexicon hits as hints to the LLM
- **`apply_sandhi_on_output`** — apply pairwise sandhi joining to ENG→SAN output

### Output Script

```json
"transliteration": {
    "output_script": "iast"
}
```

Values: `iast`, `devanagari`, `both`, `auto` (resolves to `iast`).

### Logging

```json
"logging": {
    "level": "INFO",
    "file": "varnabuddhi.log"
}
```

Set `level` to `DEBUG` for detailed engine diagnostics.

## Tips

- **Long compounds** like `oṃkāramūrtiśivasarvasukhāvahāya` are automatically detected and decomposed via samāsa analysis.
- **Low-confidence translations** trigger automatic retries — first with sandhi splitting, then with LLM self-critique.
- The **academic** and **philosophical** profiles produce longer output with grammatical apparatus — pair them with `max_output_tokens: 8192` for best results.
- Use `--split` for heavily sandhied passages where word boundaries are ambiguous.
- The **lexicon is a reference, not a replacement** — DB matches are shown alongside the LLM translation, not instead of it.
- **Streaming** is used for the primary SAN→ENG translation call. Retry passes and ENG→SAN iterations use non-streaming to keep output clean.
