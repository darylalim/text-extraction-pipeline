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

Main app in `streamlit_app.py` (~380 lines), utilities in `utils.py` (~50 lines):

### `streamlit_app.py`

- **Constants** — `MODEL_ID`, `MAX_NEW_TOKENS`, `MAX_INPUT_TOKENS`, `DEFAULT_TEMPLATE`, `DEFAULT_EXAMPLES`
- **`get_device()`** — Auto-detects compute: MPS → CUDA → CPU
- **`load_model(device)`** — Loads model and processor in BF16; cached via `@st.cache_resource`
- **`validate_template(template_str)`** — Validates JSON string is a non-empty dict; returns `(parsed, error)`
- **`parse_examples(examples_str)`** — Validates JSON array of `{"input", "output"}` objects; input can be a string (text) or dict with `{"type": "image", "image": url}` for image examples; returns `(list, error)`
- **`extract(..., image=None)`** — Runs inference under `torch.inference_mode()` with template and ICL examples; enforces `MAX_INPUT_TOKENS` limit (raises `ValueError`); uses `process_all_vision_info` for image inputs; returns parsed JSON dict or `None`
- **`_has_config_errors(template_error, examples_error)`** — Shows first config error via `st.error` and returns `True`, or returns `False` if none; used by all three tabs
- **Streamlit UI** — Sidebar with template/examples config and "Generate Template" button (appears when template is not valid JSON); three tabs: Text, Image, CSV Batch; all tabs catch `ValueError` from token limit
- **Warning suppression** — Filters known MPS padding warnings

### `utils.py`

- **`generate_template(description, model, processor, device)`** — Generates a JSON extraction template from a natural language description using NuExtract's native `template=None` mode; returns `(dict, None)` or `(None, error)`
- **`process_all_vision_info(messages, examples=None)`** — Extracts images from both ICL examples and user messages; returns flat list in correct order (example images first) or `None`

Tests in `tests/test_streamlit_app.py` and `tests/test_utils.py`.

## Key Details

- Max 256 new tokens per extraction (`MAX_NEW_TOKENS`) since output is JSON
- Max 10,000 input tokens (`MAX_INPUT_TOKENS`) to prevent OOM; raises `ValueError` if exceeded
- `DEFAULT_TEMPLATE` and `DEFAULT_EXAMPLES` provide a person-extraction starting point
- Template field accepts JSON or natural language; "Generate Template" button converts descriptions to JSON
- Image examples use URL references (`http://` or `https://` only) and are processed on the Image tab
- Image support requires `qwen-vl-utils` and `torchvision`
- `HF_TOKEN` env var enables authenticated Hub access (optional; model is public)
- Sample test data: `tests/data/csv/sample_persons.csv` (30 rows)
- Dependencies in `pyproject.toml` pinned to specific versions
