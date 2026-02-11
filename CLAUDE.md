# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Text extraction pipeline that uses IBM Granite 4.0 tiny LLM (`ibm-granite/granite-4.0-h-tiny`) to extract "Line of Credit Facility Maximum Borrowing Capacity" values from 10-K financial sentences. Provides a Streamlit web UI for batch CSV processing.

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

The entire application lives in `streamlit_app.py` (~135 lines):

- **Constants** — `MODEL_ID`, `FIELD`, `MAX_NEW_TOKENS` at top level eliminate repeated magic strings
- **`get_device()`** — Auto-detects compute device: MPS (Apple Silicon) → CUDA → CPU
- **`load_model(device)`** — Loads the Granite model and tokenizer from Hugging Face Hub in FP16; cached via `@st.cache_resource` so it loads once per session. Reads `HF_TOKEN` from environment for authenticated Hub access.
- **`extract_text(text, model, tokenizer, device)`** — Runs inference using a few-shot prompt (2 examples of credit facility extraction). Decodes only new tokens (skips re-decoding the prompt). Post-processes output with regex to strip chat template tags, takes first line, and validates length (≤50 chars). Returns "N/A" on failure.
- **Streamlit UI** — CSV upload → column selection → batch extraction with progress bar → results preview with metrics → CSV download
- **Warning suppression** — Filters Mamba fast-path and MPS padding warnings at module level (known platform limitations on Apple Silicon)

Tests are in `tests/test_streamlit_app.py` (22 tests covering constants, device detection, extraction post-processing, token decoding, and model loading).

## Key Details

- Model generates up to 50 new tokens per extraction (`MAX_NEW_TOKENS`)
- Few-shot prompt examples are hardcoded in `extract_text()` — modify these to change extraction behavior
- Set `HF_TOKEN` env var for authenticated Hugging Face Hub access (optional; model is public)
- Sample test data in `tests/data/csv/sample_10k_sentences.csv` (30 rows of synthetic 10-K sentences)
- Dependencies in `pyproject.toml` are pinned to specific versions
