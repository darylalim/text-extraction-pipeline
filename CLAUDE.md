# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

General-purpose structured extraction pipeline using NuExtract-2.0-4B (`numind/NuExtract-2.0-4B`). Accepts arbitrary text or images, a user-defined JSON template, and optional in-context learning examples to extract structured data. Provides a Streamlit web UI with text, image, and CSV batch processing tabs.

## Commands

```bash
# Setup environment
uv sync

# Run the app
uv run streamlit run streamlit_app.py

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run ty check

# Run tests
uv run pytest

# Run a single test
uv run pytest tests/test_file.py::test_name
```

There is no CI/CD configured.

## Architecture

The entire application lives in `streamlit_app.py` (~296 lines):

- **Constants** — `MODEL_ID`, `MAX_NEW_TOKENS`, `DEFAULT_TEMPLATE`, `DEFAULT_EXAMPLES` at top level; `DEFAULT_TEMPLATE` and `DEFAULT_EXAMPLES` are JSON strings that pre-populate the sidebar UI
- **`get_device()`** — Auto-detects compute device: MPS (Apple Silicon) → CUDA → CPU
- **`load_model(device)`** — Loads the NuExtract model (`AutoModelForImageTextToText`) and processor (`AutoProcessor`) from Hugging Face Hub in BF16 with `trust_remote_code=True`; cached via `@st.cache_resource` so it loads once per session. Reads `HF_TOKEN` from environment for authenticated Hub access.
- **`validate_template(template_str)`** — Parses a JSON string, checks it's a non-empty dict. Returns `(parsed_dict, error_msg)`.
- **`parse_examples(examples_str)`** — Parses a JSON array of `{"input":..., "output":...}` objects. Returns `(list, error_msg)`. Empty/whitespace input returns `([], None)`.
- **`extract(input_content, model, processor, device, template, examples, image=None)`** — Runs inference with a user-provided JSON template and ICL examples. Supports text-only and image+text inputs (using `process_vision_info` from `qwen_vl_utils` for image preprocessing). Returns the full parsed JSON dict, or `None` on parse failure.
- **Streamlit UI** — Sidebar for template/examples configuration; three tabs: **Text** (free-form input → JSON result), **Image** (image upload + optional context → JSON result), **CSV Batch** (CSV upload → column selection → batch extraction with progress bar → multi-column results → CSV download)
- **Warning suppression** — Filters MPS padding warnings at module level (known platform limitation on Apple Silicon)

Tests are in `tests/test_streamlit_app.py` (27 tests covering constants, template/examples validation, extraction with text/image/zero-shot, token decoding, device detection, and model loading).

## Key Details

- Model generates up to 256 new tokens per extraction (`MAX_NEW_TOKENS`) since output is JSON
- JSON template and ICL examples are user-configurable via the sidebar; `DEFAULT_TEMPLATE` and `DEFAULT_EXAMPLES` provide a credit facility extraction starting point
- Image support requires `qwen-vl-utils` and `torchvision` dependencies
- Set `HF_TOKEN` env var for authenticated Hugging Face Hub access (optional; model is public)
- Sample test data in `tests/data/csv/sample_10k_sentences.csv` (30 rows of synthetic 10-K sentences)
- Dependencies in `pyproject.toml` are pinned to specific versions
