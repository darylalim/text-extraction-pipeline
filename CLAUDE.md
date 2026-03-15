# CLAUDE.md

## Project Overview

Structured extraction pipeline using NuExtract-2.0-4B (`numind/NuExtract-2.0-4B`). Accepts text or images with a user-defined JSON template and optional ICL examples. Streamlit web UI with text, image, and CSV batch processing tabs.

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
- **`get_device()`** — Auto-detects compute: MPS → CUDA → CPU
- **`load_model(device)`** — Loads model and processor in BF16; cached via `@st.cache_resource`
- **`load_presets(path)`** — Loads extraction presets from JSON file; cached via `@st.cache_data`; falls back to Person preset if file missing or invalid
- **`validate_template(template_str)`** — Validates JSON string is a non-empty dict; returns `(parsed, error)`
- **`parse_examples(examples_str)`** — Validates JSON array of `{"input", "output"}` objects; input can be a string (text) or dict with `{"type": "image", "image": url}` for image examples; returns `(list, error)`
- **`extract(..., image=None, max_new_tokens=DEFAULT_MAX_NEW_TOKENS)`** — Runs inference under `torch.inference_mode()` with template and ICL examples; enforces `MAX_INPUT_TOKENS` limit (raises `ValueError`); uses `process_all_vision_info` for image inputs; returns `(result, was_truncated)` where result is parsed JSON dict or `None`
- **`_has_config_errors(template_error, examples_error, template_parsed)`** — Shows first config error via `st.error` and returns `True`, or returns `False` if none; gates extraction when no template is available
- **`_convert_template_if_needed(json_str, source_format)`** — Converts YAML/Pydantic template to JSON and updates session state on Extract
- **Streamlit UI** — Sidebar with preset selector, generation slider, template/examples config, and "Generate Template" button (appears for natural language input); template field accepts JSON, YAML, Pydantic, or natural language; three tabs: Text, Image, CSV Batch; all tabs catch `ValueError` and `RuntimeError`
- **Warning suppression** — Filters known MPS padding warnings

### `utils.py`

- **Constants** — `DEFAULT_MAX_NEW_TOKENS = 2048`
- **`detect_and_convert_template(template_str)`** — Detects template format (JSON, Pydantic, YAML, or natural language) and converts to JSON; returns `(json_str, source_format, error)` where source_format is `"json"`, `"yaml"`, `"pydantic"`, `"pydantic_with_unknown"`, or `None`
- **`_parse_pydantic_model(text)`** — Regex-based parser for flat Pydantic BaseModel classes; maps str/int/float/bool/datetime/list[X]/Optional[X] to NuExtract types; nested models fall back to "string"
- **`_map_pydantic_type(type_str)`** — Maps individual Pydantic type annotations to NuExtract types
- **`generate_template(description, model, processor, device)`** — Generates a JSON extraction template from a natural language description using NuExtract's native `template=None` mode; uses hardcoded 256 max tokens; returns `(dict, None)` or `(None, error)`
- **`process_all_vision_info(messages, examples=None)`** — Extracts images from both ICL examples and user messages; returns flat list in correct order (example images first) or `None`

### `presets.json`

5 extraction presets (Person, Job Posting, Invoice, Product, Scientific Paper) with templates, ICL examples, and sample text. Loaded by `load_presets()` at app startup.

Tests in `tests/test_streamlit_app.py` and `tests/test_utils.py`.

## Key Details

- Default 2048 new tokens per extraction (`DEFAULT_MAX_NEW_TOKENS`); configurable via sidebar slider (64–4096)
- Max 10,000 input tokens (`MAX_INPUT_TOKENS`) to prevent OOM; raises `ValueError` if exceeded
- `DEFAULT_TEMPLATE` and `DEFAULT_EXAMPLES` provide a person-extraction starting point
- Template field accepts JSON, YAML, Pydantic models, or natural language; "Generate Template" button converts descriptions to JSON; YAML/Pydantic auto-detected and converted on Extract
- Image examples use URL references (`http://` or `https://` only) and are processed on the Image tab
- Image support requires `qwen-vl-utils` and `torchvision`
- `HF_TOKEN` env var enables authenticated Hub access (optional; model is public)
- Dependencies: `pyyaml` for YAML template detection; all deps in `pyproject.toml` pinned to specific versions
- Sample test data: `tests/data/csv/sample_persons.csv` (30 rows)
