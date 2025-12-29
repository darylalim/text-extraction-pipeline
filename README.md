# Text Extraction Pipeline

A Streamlit application that extracts Line of Credit Facility Maximum Borrowing Capacity from 10K sentences using an IBM Granite 4.0 language model.

## Installation

1. Clone the repository and navigate to the project folder.

2. Create and activate a virtual environment:
   ```bash
   python3.12 -m venv streamlit_env
   source streamlit_env/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the app:
   ```bash
   streamlit run streamlit_app.py
   ```

2. Upload a CSV file containing 10K sentences.

3. Select the column with the text to extract from.

4. Click **Extract** to process.

5. Review the results and download the extracted CSV.

## Test Dataset

Use `sample_10k_sentences.csv` to test the app.

## Note

Using the test dataset, if the Granite 4.0 model does not extract text, the result is either N/A or $2.525B 😂
