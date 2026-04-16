# NuExtract Pipeline

Clinical structured extraction pipeline using [NuExtract-1.5-MLX-8bit](https://huggingface.co/mlx-community/numind-NuExtract-1.5-MLX-8bit). Extracts structured fields (diagnoses, medications, ICD-10 codes, dosages) from clinical notes and dictation transcripts. Streamlit web UI optimized for Apple Silicon via MLX.

## Features

- **Clinical extraction** — paste a note, pick a preset, get structured JSON output
- **Long-document chunking** — automatic splitting of notes exceeding the input limit, with overlap and clinical section header-attach; per-chunk results merged field-aware (scalars: first non-empty; lists: union + de-dupe)
- **ICD-10-CM validation** — extracted codes are annotated `valid`/`invalid` against a bundled CMS code set
- **Clinical presets** — 5 built-in: SOAP Note, Discharge Summary, H&P, Medication Reconciliation, Problem List
- **Multi-format templates** — accepts JSON, YAML, or Pydantic model definitions
- **Configurable output length** — inline slider for max new tokens (64–4096, default 2048)
- **Multi-language** — supports English, French, Spanish, German, Portuguese, Italian

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.12+

## Installation

```bash
uv sync
```

## Usage

```bash
uv run streamlit run streamlit_app.py
```

The model (~4 GB) is downloaded automatically on first run.

## ICD-10 Data

The repo ships with a small dev subset of ICD-10-CM codes (`data/icd10_cm_2025.json`) sufficient for the sample presets. For production use, generate the full ~74k-code set from the CMS source:

```bash
python scripts/generate_icd10_data.py path/to/icd10cm_order_2025.txt data/icd10_cm_2025.json
```

Source: [CMS 2025 ICD-10-CM](https://www.cms.gov/medicare/coding-billing/icd-10-codes/2025-icd-10-cm).

## Testing

```bash
uv run pytest              # Fast unit + integration tests (~2s, 123 tests)
uv run pytest -m e2e       # End-to-end with real model (slow, opt-in)
```

## Clinical Use Guidance

This pipeline is a research / prototype tool. Before using it on real clinical data, review the following:

- **PHI in browser state.** Streamlit keeps the text input and template in the browser session. Content persists until the tab is closed. Don't leave an unattended session open with real clinical text on screen.
- **Local-only inference.** All extraction runs locally via MLX — no data is sent to external services. Verify this in your deployment (no network calls during extraction).
- **Code validation is syntactic, not semantic.** `icd10_code_valid: true` means the code exists in the CMS set. It does NOT mean the code is the correct one for the clinical content. Wrong-but-real codes (e.g. bronchitis code assigned to pneumonia text) pass validation silently.
- **Dose / frequency / route are not validated.** Medication names aren't normalized and dosages aren't range-checked. Extraction output is a starting point for clinical review, not a substitute.
- **Long-document chunking can miss cross-chunk context.** Overlap reduces but doesn't eliminate this. Critical information at a chunk boundary may be duplicated or underspecified in merged output.
- **Always review output against the source note** before using extracted data for clinical decisions, billing, or patient records.

## Project Structure

```
streamlit_app.py          # UI, model loading, extraction orchestration
utils.py                  # Template format detection (JSON/YAML/Pydantic)
chunking.py               # Text splitting with overlap and header-attach
merging.py                # Field-aware per-chunk result merging
validation.py             # ICD-10-CM code annotation
presets.json              # 5 clinical extraction presets
data/
  icd10_cm_2025.json      # Bundled ICD-10-CM code set (dev subset)
scripts/
  generate_icd10_data.py  # CMS source → JSON converter
tests/
  conftest.py             # Shared fixtures
  test_streamlit_app.py   # App integration tests (51)
  test_chunking.py        # Chunking tests (22)
  test_merging.py         # Merging tests (13)
  test_validation.py      # ICD-10 validation tests (16)
  test_utils.py           # Template detection tests (16)
```
