# CLAUDE.md

## Project Overview

Structured extraction pipeline using NuExtract-2.0-4B (`numind/NuExtract-2.0-4B`). Accepts text or images with a user-defined JSON/YAML/Pydantic template and optional ICL examples. Streamlit web UI with text, image, image batch, and CSV batch processing tabs.

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

- **Constants** — `MODEL_ID`, `MAX_INPUT_TOKENS`, `DEFAULT_TEMPLATE`, `DEFAULT_EXAMPLES`
- **`load_presets(path)`** — Loads extraction presets from JSON file; cached via `@st.cache_data`; falls back to Person preset if file missing or invalid
- **`get_device()`** — Auto-detects compute: MPS → CUDA → CPU
- **`load_model(device)`** — Loads model and processor in BF16; cached via `@st.cache_resource`
- **`validate_template(template_str)`** — Validates JSON string is a non-empty dict; returns `(parsed, error)`
- **`parse_examples(examples_str)`** — Validates JSON array of `{"input", "output"}` objects; input can be a string (text) or dict with `{"type": "image", "image": url}` for image examples; returns `(list, error)`
- **`extract(..., image=None, max_new_tokens=DEFAULT_MAX_NEW_TOKENS)`** — Thin wrapper around `extract_batch()`; maps `input_content` to `text` (text-only) or `context` (with image); enforces `MAX_INPUT_TOKENS` limit (raises `ValueError`); returns `(result, was_truncated)`
- **`extract_batch(inputs, ..., chunk_size=4, progress_callback=None)`** — Core batch inference function; accepts list of `{"text", "image", "context"}` dicts; processes in chunks via `_process_chunk`; OOM fallback retries items sequentially with `_clear_device_cache`; token limit violations skip items in batch mode, raise `ValueError` in single-item mode; returns `list[(dict|None, bool)]`
- **`_build_message(item)`** — Builds chat message from batch input dict (text-only, image-only, or image+context)
- **`_process_chunk(...)`** — Processes a single chunk as a batched forward pass; handles OOM and ValueError fallback to sequential
- **`_run_batch_inference(...)`** — Runs batched model inference: collects images, tokenizes, checks token limits post-processor, generates, trims by `padded_input_len`, decodes, parses JSON per item
- **`_clear_device_cache(device)`** — Clears GPU memory cache (CUDA/MPS) after OOM errors
- **`_load_csv_image(value)`** — Loads image from URL or file path for CSV batch; handles NaN, empty, invalid inputs; returns PIL Image or None
- **`_has_config_errors(template_error, examples_error, template_parsed)`** — Shows first config error via `st.error` and returns `True`, or returns `False` if none; gates extraction when no template is available
- **`_convert_template_if_needed(json_str, source_format)`** — Converts YAML/Pydantic template to JSON and updates session state on Extract
- **Streamlit UI** — Sidebar with preset selector, generation slider (64–4096 tokens), template/examples config, and "Generate Template" button (appears for natural language input); template field accepts JSON, YAML, Pydantic, or natural language; four tabs: Text, Image, Image Batch, CSV Batch; all tabs catch `ValueError` and `RuntimeError` with truncation and OOM warnings
- **Image Batch tab** — Multi-image upload with `accept_multiple_files=True`; shared context with optional per-image overrides; configurable batch size (1–8, default 4); results as individual expandable cards + summary dataframe with metrics and CSV download
- **CSV Batch tab** — Extended with optional image column selector (excludes text column); when image column selected, uses `extract_batch()` with configurable batch size and `_load_csv_image` for URL/file path loading; text-only mode preserves original sequential `extract()` loop
- **Warning suppression** — Filters known MPS padding warnings

### `utils.py`

- **Constants** — `DEFAULT_MAX_NEW_TOKENS = 2048`
- **`detect_and_convert_template(template_str)`** — Detects template format (JSON → Pydantic → YAML → natural language) and converts to JSON; returns `(json_str, source_format, error)` where source_format is `"json"`, `"yaml"`, `"pydantic"`, `"pydantic_with_unknown"`, or `None`
- **`_parse_pydantic_model(text)`** — Regex-based parser for flat Pydantic BaseModel classes; maps `str`/`int`/`float`/`bool`/`datetime`/`list[X]`/`Optional[X]` to NuExtract types; nested models fall back to `"string"`
- **`_map_pydantic_type(type_str)`** — Maps individual Pydantic type annotations to NuExtract types
- **`generate_template(description, model, processor, device)`** — Generates a JSON extraction template from a natural language description using NuExtract's native `template=None` mode; uses hardcoded 256 max tokens; returns `(dict, None)` or `(None, error)`
- **`process_all_vision_info(messages, examples=None)`** — Extracts images from both ICL examples and user messages; supports single input (`messages` is list of dicts) or batch input (`messages` is list of lists); normalizes examples (None, flat list broadcast, or list-of-lists); raises `ValueError` on batch length mismatch; returns flat list in per-item order or `None`

### `presets.json`

5 extraction presets (Person, Job Posting, Invoice, Product, Scientific Paper) with templates, ICL examples, and sample text. Loaded by `load_presets()` at app startup.

Tests in `tests/test_streamlit_app.py` (89 tests) and `tests/test_utils.py` (35 tests).

## Key Details

- Default 2048 new tokens per extraction (`DEFAULT_MAX_NEW_TOKENS`); configurable via sidebar slider (64–4096)
- Max 10,000 input tokens (`MAX_INPUT_TOKENS`) to prevent OOM; raises `ValueError` if exceeded
- `DEFAULT_TEMPLATE` and `DEFAULT_EXAMPLES` provide a person-extraction starting point
- Template field accepts JSON, YAML, Pydantic models, or natural language; "Generate Template" button converts descriptions to JSON; YAML/Pydantic auto-detected and converted on Extract
- Detection order: JSON → Pydantic → YAML → natural language (Pydantic before YAML because class syntax is valid YAML)
- Image examples use URL references (`http://` or `https://` only) and are processed on the Image and Image Batch tabs
- Image Batch tab supports multi-image upload with shared/per-image context and configurable batch size (1–8)
- CSV image column supports URLs and local file paths; NaN values silently skipped; invalid paths fall back to text-only with warning
- Image support requires `qwen-vl-utils` and `torchvision`
- `HF_TOKEN` env var enables authenticated Hub access (optional; model is public)
- Sample test data: `tests/data/csv/sample_persons.csv` (30 rows)
- Dependencies in `pyproject.toml` pinned to specific versions
