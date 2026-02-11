# Text Extraction Pipeline
Extract the Line of Credit Facility Maximum Borrowing Capacity from 10K sentences with IBM Granite 4.0 language models.

## Installation
Run the following commands in the terminal.

- Install dependencies: `uv sync`
- (Optional) Set your Hugging Face token for faster downloads: `export HF_TOKEN=hf_...`
- Run the application in a web browser: `uv run streamlit run streamlit_app.py`

## Testing

```bash
uv run pytest
```

Sample test data is in `tests/data/csv/sample_10k_sentences.csv` (30 rows of synthetic 10-K sentences).

## Note
- With the test dataset `sample_10k_sentences.csv`, if the Granite 4.0 model does not extract text, the result is either N/A or $2.525B 😂
