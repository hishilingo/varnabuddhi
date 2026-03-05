## Baba_SandhiSplit

Offline wrapper around the pretrained Neural Sandhi splitter shipped in this
repository (`data/models/saved_model.pb`). It exposes a single CLI command,
running entirely locally (no network calls), capable of processing either a
single UTF-8 text file or complete folder trees.

### Requirements

- Python 3.8+
- TensorFlow 1.x runtime (tested with 1.15). Install locally:  
  `pip install tensorflow==1.15.5` (CPU build).
- Optional (for Devanagari <-> IAST conversion):  
  `pip install indic-transliteration`

Both packages can be installed offline from pre-downloaded wheels.

### Usage

Run from the project root (same level as `code/` and `data/`):

```bash
python baba_sandhi_split.py --input path/to/input.txt
```

By default the script writes `<name>_unsandhied.txt` next to the source file.

#### Key options

| Option | Description |
| --- | --- |
| `--input` | Input file **or** directory. Required. |
| `--output` | Output file/folder. If omitted, defaults to `<file>_unsandhied.txt` beside the input file, or mirrored structure inside the provided directory. |
| `--recursive` | Recurse into sub-directories when `--input` is a folder. |
| `--extensions` | Comma-separated extension filter for directory mode (default: `.txt`). |
| `--input-script` | Force script detection (`auto`, `iast`, `devanagari`). Default: `auto`. |
| `--output-script` | Choose output script (`auto`, `iast`, `devanagari`). `auto` mirrors detected input. Requires `indic_transliteration` when Devanagari is involved. |

Examples:

```bash
# Single file -> same directory
python baba_sandhi_split.py --input samples/verse.txt

# Explicit output file
python baba_sandhi_split.py --input verse.txt --output out/verse_unsandhied.txt

# Batch process directory (recursive) into a clean output tree
python baba_sandhi_split.py --input corpus/ --output unsandhied/ --recursive

# Devanagari input, Devanagari output (needs indic_transliteration package)
python baba_sandhi_split.py --input devanagari.txt --output-script devanagari
```

### Implementation notes

- Runs the SavedModel with TensorFlow 1.x compat API; long runs show an inline
  progress percentage so you can estimate remaining time.
- Input text is optionally transliterated to IAST (for Devanagari sources); the
  model itself operates on IAST.
- Output stays in IAST unless `--output-script devanagari` is requested.
- Folder processing reproduces the input directory structure under the output
  tree.
- Entirely offline - no external APIs.
- If `declensions.txt` is present, each recognised form (and hyphenated
  component) is annotated inline with matching declension entries.
- If `verbs.txt`, `indecls.txt`, `adverbs.txt`, or `pronouns.txt` are present,
  their matches are shown in dedicated sections (`verbs`, `indecls`, `adverbs`,
  `pronouns`) so you can spot the full lexical analysis for every split token.
- Large reference lists are cached on first load (`declensions.txt.cache`,
  `verbs.txt.cache`, `indecls.txt.cache`, `adverbs.txt.cache`,
  `pronouns.txt.cache`); delete the respective cache file after editing the
  source list to force a rebuild.

### Troubleshooting

- `ModuleNotFoundError: No module named 'tensorflow'`  
  Install TensorFlow 1.x: `pip install tensorflow==1.15.5`
- `TransliterationUnavailable` (Devanagari workflows)  
  Install `indic_transliteration`:  
  `pip install indic-transliteration`

The splitter keeps the original newline structure, one line in → one line out.
