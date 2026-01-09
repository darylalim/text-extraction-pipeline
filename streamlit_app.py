import os

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_device():
    """Automatically detect the best available device in order of priority: MPS, CUDA, CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

@st.cache_resource
def load_model(device):
    """Load and cache the model and tokenizer."""
    model_path = "ibm-granite/granite-4.0-h-tiny"
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)    
    return model, tokenizer

def extract_text(text, model, tokenizer, device):
    """Extract Line of Credit Facility Maximum Borrowing Capacity from text."""
    prompt = f"""Extract the Line Of Credit Facility Maximum Borrowing Capacity from the 10K sentences.
Your response should only include the answer. Do not provide any further explanation.

Here are some examples, complete the last one:
10K Sentence:
The credit agreement also provides that up to $500 million in commitments may be used for letters of credit.
Line Of Credit Facility Maximum Borrowing Capacity:
$500M

10K Sentence:
In March 2020, we upsized the Credit Agreement by $100 million, which matures July 2023, to $2.525 billion.
Line Of Credit Facility Maximum Borrowing Capacity:
$2.525B

10K Sentence:
{text}
Line Of Credit Facility Maximum Borrowing Capacity:"""

    chat = [{"role": "user", "content": prompt}]
    chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
    input_tokens = tokenizer(chat, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            **input_tokens,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id
        )
    
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract the answer after the last "Line Of Credit Facility Maximum Borrowing Capacity:"
    try:
        answer = decoded.split("Line Of Credit Facility Maximum Borrowing Capacity:")[-1].strip()
        # Get the first line of the answer
        answer = answer.split("\n")[0].strip()

        # Remove chat template artifacts
        for tag in ['assistant', 'user', 'system']:
            answer = answer.replace(tag, '').strip()
        
        # Basic validation
        if not answer or len(answer) > 50:  
            return "N/A"
        return answer
    
    except Exception:
        return "N/A"

st.title("Text Extraction Pipeline")
st.write("Extract the Line of Credit Facility Maximum Borrowing Capacity from 10K sentences.")

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"],
    help="Upload a CSV file with 10K sentences"
)

device = get_device()

with st.spinner(f"Loading model on {device.upper()}..."):
    model, tokenizer = load_model(device)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"File uploaded. ({len(df)} rows)")
        
        columns = df.columns.tolist()
        selected_column = st.selectbox(
            "Select column for 10K sentences",
            options=columns
        )
        
        if st.button("Extract", type="primary"):
            # Process each row
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Extracting..."):
                for idx, row in df.iterrows():
                    text = str(row[selected_column])
                    result = extract_text(text, model, tokenizer, device)
                    results.append(result)
                    
                    # Update progress
                    progress = (idx + 1) / len(df)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing row {idx + 1} of {len(df)}")
            
            status_text.text("Done.")
            
            # Add new column
            df["Line of Credit Facility Maximum Borrowing Capacity"] = results
            
            # Store in session state for display and download
            st.session_state["processed_df"] = df
            st.session_state["selected_column"] = selected_column
            st.session_state["uploaded_file"] = uploaded_file.name
        
        if "processed_df" in st.session_state:            
            display_df = st.session_state["processed_df"][
                [st.session_state["selected_column"], "Line of Credit Facility Maximum Borrowing Capacity"]
            ]

            st.write("Preview")
            st.dataframe(display_df.head(), width='stretch')
            
            # Show extraction metrics
            total = len(st.session_state["processed_df"])
            na_count = (st.session_state["processed_df"]["Line of Credit Facility Maximum Borrowing Capacity"] == "N/A").sum()
            extracted_count = total - na_count
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Rows", total)
            col2.metric("Extracted", extracted_count)
            col3.metric("N/A", na_count)
            
            csv_data = st.session_state["processed_df"].to_csv(index=False)
            base_name = os.path.splitext(st.session_state['uploaded_file'])[0]
            output_filename = f"{base_name}_extract.csv"
            
            st.download_button(
                label="Download",
                data=csv_data,
                file_name=output_filename,
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")

else:
    st.info("Upload a CSV file.")
