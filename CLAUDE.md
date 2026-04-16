# CLAUDE.md

## Project Overview

Clinical structured extraction pipeline using NuExtract-1.5-MLX-8bit (`mlx-community/numind-NuExtract-1.5-MLX-8bit`). Extracts structured fields (diagnoses, medications, ICD-10 codes, dosages) from clinical notes and dictation transcripts. Supports long-document chunking with field-aware merge and ICD-10-CM code validation. Streamlit web UI with a single Text input tab.

## Commands

```bash
uv sync                                    # Setup environment
uv run streamlit run streamlit_app.py      # Run the app
uv run ruff check .                        # Lint
uv run ruff format .                       # Format
uv run ty check                            # Type check
uv run pytest                              # Run tests
uv run pytest tests/test_file.py::test_name  # Run a single test
```

CI via GitHub Actions (`.github/workflows/ci.yml`): runs lint, format check, type check, and `pytest` on every push and PR to `main`. Uses `macos-14` (Apple Silicon) runners for MLX compatibility. E2E tests are excluded from CI — run manually with `uv run pytest -m e2e`.

## Architecture

Main app in `streamlit_app.py`, utilities in `utils.py`, presets in `presets.json`, chunking in `chunking.py`, merging in `merging.py`, validation in `validation.py`:

### `streamlit_app.py`

- **Constants** — `MODEL_ID`, `MAX_INPUT_TOKENS`, `DEFAULT_TEMPLATE`
- **`load_presets(path)`** — Loads extraction presets from JSON file; cached via `@st.cache_data`; falls back to SOAP Note preset if file missing or invalid
- **`load_model()`** — Loads model and tokenizer via `mlx_lm.load()`; cached via `@st.cache_resource`
- **`validate_template(template_str)`** — Validates JSON string is a non-empty dict; returns `(parsed, error)`
- **`build_prompt(template_str, text)`** — Builds NuExtract-1.5 prompt with `<|input|>/<|output|>` format
- **`extract(text, model, tokenizer, template, max_new_tokens)`** — Runs extraction via `mlx_lm.generate()`; enforces `MAX_INPUT_TOKENS` limit (raises `ValueError`); strips `<|end-output|>` marker; returns `(result, was_truncated)`
- **`_has_config_errors(template_error, template_parsed)`** — Shows first config error via `st.error` and returns `True`, or returns `False` if none
- **`_get_effective_template(json_str, source_format, template_str)`** — Returns JSON template, converting from YAML/Pydantic if needed and updating session state
- **`ConfigState`** — `NamedTuple` returned by `_render_config()`: `template_str`, `json_str`, `source_format`, `template_error`, `template_parsed`, `max_new_tokens`
- **`_render_config()`** — Renders inline config controls (preset selector, max tokens slider, template editor, format expander); returns `ConfigState`; called once above the text input
- **`_load_icd10_codes()`** — `@st.cache_data` wrapper around `load_icd10_codes()` from `validation.py`; keeps validation logic Streamlit-free
- **`_run_extraction(text, model, tokenizer, template_str, template_parsed, max_new_tokens)`** — Two-path extraction: single-chunk fast path when input fits within token budget, or multi-chunk path with progress bar, `merge_results()`, and ICD-10 validation via `annotate_icd10()`; displays results and any validation warnings
- **Streamlit UI** — Config rendered above the text input via `_render_config()` (preset selector + max tokens slider side-by-side, then template editor); single Text input with Extract button

### `utils.py`

- **Constants** — `DEFAULT_MAX_NEW_TOKENS = 2048`
- **`detect_and_convert_template(template_str)`** — Detects template format (JSON → Pydantic → YAML) and converts to JSON; returns `(json_str, source_format, error)` where source_format is `"json"`, `"yaml"`, `"pydantic"`, `"pydantic_with_unknown"`, or `None`; returns error for unrecognized formats
- **`_parse_pydantic_model(text)`** — Regex-based parser for flat Pydantic BaseModel classes; maps all types to empty string placeholders; lists map to `[]`
- **`_map_pydantic_type(type_str)`** — Maps individual Pydantic type annotations to NuExtract-1.5 placeholders

### `chunking.py`

- **Constants** — `DEFAULT_CHUNK_TOKENS`, `DEFAULT_OVERLAP_TOKENS`, `HEADER_ATTACH_ZONE`, `SECTION_HEADERS`
- **`chunk_text(text, tokenizer, max_tokens, overlap) -> list[str]`** — Splits text into overlapping token-window chunks; header-attach rule pushes standalone clinical section headers (e.g. `"ASSESSMENT:\n"`) found in the last `HEADER_ATTACH_ZONE` tokens of a chunk to the start of the next chunk
- **`_split_long_line(line, tokenizer, max_tokens) -> list[str]`** — Word-boundary fallback for single lines exceeding `max_tokens` (e.g. run-on dictation without line breaks); preserves trailing newline on last sub-line, uses space separator when input has no newline

### `merging.py`

- **`merge_results(results, template) -> dict | None`** — Field-aware merge of multiple per-chunk extraction dicts; scalar fields take the first non-empty value, list fields are concatenated and deduplicated; returns `None` if all inputs are empty

### `validation.py`

- **`load_icd10_codes(path) -> set[str]`** — Loads bundled ICD-10-CM code set from JSON file; returns empty set if file missing
- **`annotate_icd10(result, codes)`** — Walks the extraction result tree and returns an annotated copy with an `icd10_code_valid` sibling (`true`/`false`) next to each `icd10_code` field; normalizes extracted codes (strip, uppercase, remove dots) before lookup; original dict untouched
- **`count_invalid_codes(result) -> int`** — Returns the count of ICD-10 codes in the result that failed validation

### `presets.json`

5 clinical extraction presets (SOAP Note, Discharge Summary, H&P, Medication Reconciliation, Problem List) with templates and sample clinical text. Templates use empty string placeholders (`""` for fields, `[]` for arrays). Loaded by `load_presets()` at app startup.

### `data/icd10_cm_2025.json`

Bundled ICD-10-CM code set (dev subset in repo). Filename matches the CMS source year. CMS dotless uppercase format (e.g. `"J18.9"` stored as `"J189"`). Used by `validation.py` for code lookup.

### `scripts/generate_icd10_data.py`

Converts the CMS ICD-10-CM source file to the bundled JSON format consumed by `validation.py`.

Shared fixtures in `tests/conftest.py` (`sample_icd10_codes`, `sample_3_chunk_text`, `sample_per_chunk_results`). Tests in `tests/test_chunking.py` (22 tests), `tests/test_merging.py` (13 tests), `tests/test_validation.py` (16 tests), `tests/test_streamlit_app.py` (51 tests), and `tests/test_utils.py` (16 tests). Total: 118 tests.

## Key Details

- Default 2048 new tokens per extraction (`DEFAULT_MAX_NEW_TOKENS`); configurable via inline slider (64–4096)
- Max 4,096 input tokens (`MAX_INPUT_TOKENS`); raises `ValueError` if exceeded per chunk
- NuExtract-1.5 prompt format: `<|input|>\n### Template:\n{template}\n### Text:\n{text}\n\n<|output|>\n`
- Template field accepts JSON, YAML, or Pydantic models; YAML/Pydantic auto-detected and converted on Extract
- Detection order: JSON → Pydantic → YAML (Pydantic before YAML because class syntax is valid YAML)
- NuExtract-1.5 templates use empty strings (`""`) as field placeholders and empty arrays (`[]`) for list fields
- Text-only extraction (no image/vision support)
- No ICL examples (NuExtract-1.5 does not officially support in-context learning)
- MLX-based inference optimized for Apple Silicon; no GPU device selection needed
- Supports 6 languages: English, French, Spanish, German, Portuguese, Italian
- Warning suppression: `transformers.modeling_rope_utils` logger set to ERROR to suppress rope config warning from Phi-3.5 base model
- Long-document chunking activates when input exceeds `MAX_INPUT_TOKENS - template_overhead`. Default 3500-token chunks with 200-token overlap. Hybrid token-window + header-attach rule (clinical section headers pushed to next chunk if in last 100 tokens).
- ICD-10-CM validation: bundled code set annotates extracted codes with `icd10_code_valid` sibling. CMS dotless uppercase format. Missing data file → `st.warning('validation skipped')`.
- Single Text input (no CSV Batch tab); pandas dependency removed
- Dependencies in `pyproject.toml`; `mlx-lm>=0.20.0` required; `chunking.py`, `merging.py`, `validation.py` are pure-Python stdlib + tokenizer modules (no additional runtime deps)
