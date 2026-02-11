import logging
import os
import re
import warnings

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.getLogger("transformers.models.mamba.modeling_mamba").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*MPS: The constant padding.*")

MODEL_ID = "ibm-granite/granite-4.0-h-tiny"
FIELD = "Line of Credit Facility Maximum Borrowing Capacity"
MAX_NEW_TOKENS = 50


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


@st.cache_resource
def load_model(device):
    token = os.environ.get("HF_TOKEN") or False
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map=device, dtype=torch.float16, token=token
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    return model, tokenizer


def extract_text(text, model, tokenizer, device):
    prompt = f"""Extract the {FIELD} from the 10K sentences.
Your response should only include the answer. Do not provide any further explanation.

Here are some examples, complete the last one:
10K Sentence:
The credit agreement also provides that up to $500 million in commitments may be used for letters of credit.
{FIELD}:
$500M

10K Sentence:
In March 2020, we upsized the Credit Agreement by $100 million, which matures July 2023, to $2.525 billion.
{FIELD}:
$2.525B

10K Sentence:
{text}
{FIELD}:"""

    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_tokens = tokenizer(chat, return_tensors="pt").to(device)
    input_len = input_tokens["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **input_tokens,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
        )

    answer = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
    answer = re.sub(r"\b(assistant|user|system)\b", "", answer.split("\n")[0]).strip()
    return answer if answer and len(answer) <= MAX_NEW_TOKENS else "N/A"


st.title("Text Extraction Pipeline")
st.write(f"Extract the {FIELD} from 10K sentences.")

uploaded_file = st.file_uploader(
    "Upload CSV file", type=["csv"], help="Upload a CSV file with 10K sentences"
)

device = get_device()

with st.spinner(f"Loading model on {device.upper()}..."):
    model, tokenizer = load_model(device)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"File uploaded. ({len(df)} rows)")

        selected_column = st.selectbox(
            "Select column for 10K sentences", options=df.columns.tolist()
        )

        if st.button("Extract", type="primary"):
            results = []
            progress_bar = st.progress(0, text="Starting...")

            with st.spinner("Extracting..."):
                for i, text in enumerate(df[selected_column].astype(str)):
                    results.append(extract_text(text, model, tokenizer, device))
                    progress_bar.progress(
                        (i + 1) / len(df), text=f"Processing row {i + 1} of {len(df)}"
                    )

            progress_bar.progress(1.0, text="Done.")

            df[FIELD] = results

            st.write("Preview")
            st.dataframe(df[[selected_column, FIELD]].head(), use_container_width=True)

            total = len(df)
            na_count = (df[FIELD] == "N/A").sum()

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Rows", total)
            col2.metric("Extracted", total - na_count)
            col3.metric("N/A", na_count)

            base_name = uploaded_file.name.rsplit(".", 1)[0]
            st.download_button(
                label="Download",
                data=df.to_csv(index=False),
                file_name=f"{base_name}_extract.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Upload a CSV file.")
