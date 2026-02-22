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

Single-file app in `streamlit_app.py` (~346 lines):

- **Constants** — `MODEL_ID`, `MAX_NEW_TOKENS`, `DEFAULT_TEMPLATE`, `DEFAULT_EXAMPLES`
- **`get_device()`** — Auto-detects compute: MPS → CUDA → CPU
- **`load_model(device)`** — Loads model and processor in BF16; cached via `@st.cache_resource`
- **`validate_template(template_str)`** — Validates JSON string is a non-empty dict; returns `(parsed, error)`
- **`parse_examples(examples_str)`** — Validates JSON array of `{"input", "output"}` objects; returns `(list, error)`
- **`extract(..., image=None)`** — Runs inference under `torch.inference_mode()` with template and ICL examples; supports text-only and image+text; returns parsed JSON dict or `None`
- **`_has_config_errors(template_error, examples_error)`** — Shows first config error via `st.error` and returns `True`, or returns `False` if none; used by all three tabs
- **Streamlit UI** — Sidebar for template/examples config; three tabs: Text, Image, CSV Batch
- **Warning suppression** — Filters known MPS padding warnings

Tests in `tests/test_streamlit_app.py` (32 tests): constants, template/examples validation, config error helper, extraction (text, image, zero-shot, inference_mode, token decoding, error propagation), device detection, model loading.

## Key Details

- Max 256 new tokens per extraction (`MAX_NEW_TOKENS`) since output is JSON
- `DEFAULT_TEMPLATE` and `DEFAULT_EXAMPLES` provide a person-extraction starting point
- Image support requires `qwen-vl-utils` and `torchvision`
- `HF_TOKEN` env var enables authenticated Hub access (optional; model is public)
- Sample test data: `tests/data/csv/sample_persons.csv` (30 rows)
- Dependencies in `pyproject.toml` pinned to specific versions
