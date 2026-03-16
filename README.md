# NuExtract 2.0 Pipeline

Structured extraction pipeline using [NuExtract-2.0-4B](https://huggingface.co/numind/NuExtract-2.0-4B). Accepts text or images with a user-defined template and optional in-context learning (ICL) examples to extract structured data. Streamlit web UI with text, image, image batch, and CSV batch processing tabs.

## Features

- **Text extraction** — paste text, define a template, get structured output
- **Image extraction** — upload an image with optional context text
- **Image batch extraction** — upload multiple images with shared or per-image context; configurable batch size (1–8)
- **CSV batch processing** — extract from every row in a CSV file; optional image column for mixed text+image extraction
- **Multi-format templates** — accepts JSON, YAML, Pydantic models, or natural language descriptions
- **Auto template generation** — describe what to extract in plain English and generate a JSON template
- **Extraction presets** — 5 built-in presets (Person, Job Posting, Invoice, Product, Scientific Paper)
- **ICL examples** — provide input/output examples (text or image URLs) to guide extraction
- **Configurable output length** — sidebar slider for max new tokens (64–4096, default 2048)
- **Token limit** — enforces a 10,000 input token limit to prevent OOM errors
- **OOM recovery** — automatic fallback to sequential processing when batched inference runs out of memory

## Installation

```bash
uv sync
```

Optionally set your Hugging Face token for faster downloads:

```bash
export HF_TOKEN=hf_...
```

## Usage

```bash
uv run streamlit run streamlit_app.py
```

## Testing

```bash
uv run pytest
```

Sample test data is in `tests/data/csv/sample_persons.csv` (30 rows of synthetic person descriptions).

## Project Structure

```
streamlit_app.py          # Main app — UI, validation, extraction
utils.py                  # Utilities — template detection, generation, vision processing
presets.json              # 5 built-in extraction presets
tests/
  conftest.py             # Shared test fixtures and mock helpers
  test_streamlit_app.py   # Tests for main app (103 tests)
  test_utils.py           # Tests for utilities (35 tests)
  data/csv/               # Sample test data
```
