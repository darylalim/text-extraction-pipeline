# Text Extraction Pipeline
General-purpose structured extraction pipeline using NuExtract-2.0-4B. Accepts arbitrary text or images, a user-defined JSON template, and optional in-context learning examples to extract structured data.

## Installation
Run the following commands in the terminal.

- Install dependencies: `uv sync`
- (Optional) Set your Hugging Face token for faster downloads: `export HF_TOKEN=hf_...`
- Run the application in a web browser: `uv run streamlit run streamlit_app.py`

## Testing

```bash
uv run pytest
```

Sample test data is in `tests/data/csv/sample_persons.csv` (30 rows of synthetic person descriptions).
