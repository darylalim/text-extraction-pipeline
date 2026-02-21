import json
import os
import warnings

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

warnings.filterwarnings("ignore", message=".*MPS: The constant padding.*")

MODEL_ID = "numind/NuExtract-2.0-4B"
FIELD = "Line of Credit Facility Maximum Borrowing Capacity"
MAX_NEW_TOKENS = 256
MAX_ANSWER_LENGTH = 50
TEMPLATE = json.dumps({FIELD: "verbatim-string"})
EXAMPLES = [
    {
        "input": "The credit agreement also provides that up to $500 million in commitments may be used for letters of credit.",
        "output": json.dumps({FIELD: "$500M"}),
    },
    {
        "input": "In March 2020, we upsized the Credit Agreement by $100 million, which matures July 2023, to $2.525 billion.",
        "output": json.dumps({FIELD: "$2.525B"}),
    },
]


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


@st.cache_resource
def load_model(device):
    token = os.environ.get("HF_TOKEN") or False
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
        token=token,
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        padding_side="left",
        use_fast=True,
        token=token,
    )
    return model, processor


def extract_text(text, model, processor, device):
    messages = [{"role": "user", "content": text}]
    formatted = processor.tokenizer.apply_chat_template(
        messages,
        template=TEMPLATE,
        examples=EXAMPLES,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[formatted], images=None, padding=True, return_tensors="pt")
    inputs = inputs.to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    trimmed = output[:, input_len:]
    decoded = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    try:
        parsed = json.loads(decoded[0])
        answer = parsed.get(FIELD, "").strip()
        return answer if answer and len(answer) <= MAX_ANSWER_LENGTH else "N/A"
    except (json.JSONDecodeError, IndexError, AttributeError):
        return "N/A"


st.title("Text Extraction Pipeline")
st.write(f"Extract the {FIELD} from 10K sentences.")

uploaded_file = st.file_uploader(
    "Upload CSV file", type=["csv"], help="Upload a CSV file with 10K sentences"
)

device = get_device()

with st.spinner(f"Loading model on {device.upper()}..."):
    model, processor = load_model(device)

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
                    results.append(extract_text(text, model, processor, device))
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
