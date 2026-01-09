# Text Extraction Pipeline
Extract the Line of Credit Facility Maximum Borrowing Capacity from 10K sentences with IBM Granite 4.0 language models.

## Installation
Run the following commands in the terminal.

- Set up a Python virtual environment: `python3.12 -m venv streamlit_env`
- Activate the virtual environment: `source streamlit_env/bin/activate`
- Install the required Python packages: `pip install -r requirements.txt`
- Run the application in a web browser: `streamlit run streamlit_app.py`

## Note
- With the test dataset `sample_10k_sentences.csv`, if the Granite 4.0 model does not extract text, the result is either N/A or $2.525B 😂
