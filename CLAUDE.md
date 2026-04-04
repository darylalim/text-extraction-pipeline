# CLAUDE.md

## Project Overview

Structured extraction pipeline using NuExtract-1.5-MLX-8bit (`mlx-community/numind-NuExtract-1.5-MLX-8bit`). Accepts text with a user-defined JSON/YAML/Pydantic template. Streamlit web UI with two tabs: Text and CSV Batch.

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

No CI/CD configured.

## Architecture

Main app in `streamlit_app.py`, utilities in `utils.py`, presets in `presets.json`:

### `streamlit_app.py`

- **Constants** — `MODEL_ID`, `MAX_INPUT_TOKENS`, `DEFAULT_TEMPLATE`
- **`load_presets(path)`** — Loads extraction presets from JSON file; cached via `@st.cache_data`; falls back to Person preset if file missing or invalid
- **`load_model()`** — Loads model and tokenizer via `mlx_lm.load()`; cached via `@st.cache_resource`
- **`validate_template(template_str)`** — Validates JSON string is a non-empty dict; returns `(parsed, error)`
- **`build_prompt(template_str, text)`** — Builds NuExtract-1.5 prompt with `<|input|>/<|output|>` format
- **`extract(text, model, tokenizer, template, max_new_tokens)`** — Runs extraction via `mlx_lm.generate()`; enforces `MAX_INPUT_TOKENS` limit (raises `ValueError`); strips `<|end-output|>` marker; returns `(result, was_truncated)`
- **`_has_config_errors(template_error, template_parsed)`** — Shows first config error via `st.error` and returns `True`, or returns `False` if none
- **`_get_effective_template(json_str, source_format, template_str)`** — Returns JSON template, converting from YAML/Pydantic if needed and updating session state
- **`_run_single_extraction(text, model, tokenizer, template_str, max_new_tokens)`** — Runs single extraction and displays results; handles truncation warning, JSON parse failure, ValueError, and RuntimeError
- **`_display_csv_results(df, results, truncated_rows, template_parsed, selected_column, filename)`** — Displays CSV extraction results: truncated warnings, preview dataframe, metrics (Total/Extracted/Failed), download button
- **Streamlit UI** — Sidebar with preset selector, generation slider (64–4096 tokens), template config; two tabs: Text, CSV Batch
- **Text tab** — Single text input with Extract button
- **CSV Batch tab** — CSV upload with text column selector; sequential extraction with progress bar

### `utils.py`

- **Constants** — `DEFAULT_MAX_NEW_TOKENS = 2048`
- **`detect_and_convert_template(template_str)`** — Detects template format (JSON → Pydantic → YAML) and converts to JSON; returns `(json_str, source_format, error)` where source_format is `"json"`, `"yaml"`, `"pydantic"`, `"pydantic_with_unknown"`, or `None`; returns error for unrecognized formats
- **`_parse_pydantic_model(text)`** — Regex-based parser for flat Pydantic BaseModel classes; maps all types to empty string placeholders; lists map to `[]`
- **`_map_pydantic_type(type_str)`** — Maps individual Pydantic type annotations to NuExtract-1.5 placeholders

### `presets.json`

5 extraction presets (Person, Job Posting, Invoice, Product, Scientific Paper) with templates and sample text. Templates use empty string placeholders (`""` for fields, `[]` for arrays). Loaded by `load_presets()` at app startup.

Shared test helpers in `tests/conftest.py`. Tests in `tests/test_streamlit_app.py` and `tests/test_utils.py`.

## Key Details

- Default 2048 new tokens per extraction (`DEFAULT_MAX_NEW_TOKENS`); configurable via sidebar slider (64–4096)
- Max 4,096 input tokens (`MAX_INPUT_TOKENS`); raises `ValueError` if exceeded
- NuExtract-1.5 prompt format: `<|input|>\n### Template:\n{template}\n### Text:\n{text}\n\n<|output|>\n`
- Template field accepts JSON, YAML, or Pydantic models; YAML/Pydantic auto-detected and converted on Extract
- Detection order: JSON → Pydantic → YAML (Pydantic before YAML because class syntax is valid YAML)
- NuExtract-1.5 templates use empty strings (`""`) as field placeholders and empty arrays (`[]`) for list fields
- Text-only extraction (no image/vision support)
- No ICL examples (NuExtract-1.5 does not officially support in-context learning)
- MLX-based inference optimized for Apple Silicon; no GPU device selection needed
- Supports 6 languages: English, French, Spanish, German, Portuguese, Italian
- Dependencies in `pyproject.toml`; `mlx-lm>=0.20.0` required
- Sample test data: `tests/data/csv/sample_persons.csv` (30 rows)
