import json
import os
import warnings

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

warnings.filterwarnings("ignore", message=".*MPS: The constant padding.*")
warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")

MODEL_ID = "numind/NuExtract-2.0-4B"
MAX_NEW_TOKENS = 256
DEFAULT_TEMPLATE = json.dumps(
    {
        "first_name": "verbatim-string",
        "last_name": "verbatim-string",
        "description": "string",
        "age": "integer",
        "gpa": "number",
        "birth_date": "date-time",
        "nationality": ["France", "England", "Japan", "USA", "China"],
        "languages_spoken": [["English", "French", "Japanese", "Mandarin", "Spanish"]],
    },
    indent=2,
)
DEFAULT_EXAMPLES = json.dumps(
    [
        {
            "input": "Yuki Tanaka is a 22-year-old Japanese student with a 3.8 GPA, born on March 15, 2003. She speaks Japanese, English, and French.",
            "output": json.dumps(
                {
                    "first_name": "Yuki",
                    "last_name": "Tanaka",
                    "description": "Japanese student who speaks three languages",
                    "age": 22,
                    "gpa": 3.8,
                    "birth_date": "2003-03-15",
                    "nationality": "Japan",
                    "languages_spoken": ["Japanese", "English", "French"],
                }
            ),
        },
        {
            "input": "Born July 4, 1998, James Smith is a 27-year-old American with a GPA of 3.2. He is fluent in English and Spanish.",
            "output": json.dumps(
                {
                    "first_name": "James",
                    "last_name": "Smith",
                    "description": "American who is fluent in English and Spanish",
                    "age": 27,
                    "gpa": 3.2,
                    "birth_date": "1998-07-04",
                    "nationality": "USA",
                    "languages_spoken": ["English", "Spanish"],
                }
            ),
        },
    ],
    indent=2,
)


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
        dtype=torch.bfloat16,
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


def validate_template(template_str):
    try:
        parsed = json.loads(template_str)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"
    if not isinstance(parsed, dict):
        return None, "Template must be a JSON object."
    if not parsed:
        return None, "Template must not be empty."
    return parsed, None


def parse_examples(examples_str):
    if not examples_str or not examples_str.strip():
        return [], None
    try:
        parsed = json.loads(examples_str)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"
    if not isinstance(parsed, list):
        return None, "Examples must be a JSON array."
    for i, ex in enumerate(parsed):
        if not isinstance(ex, dict) or "input" not in ex or "output" not in ex:
            return None, f'Example {i + 1} must have "input" and "output" keys.'
    return parsed, None


def extract(input_content, model, processor, device, template, examples, image=None):
    if image is not None:
        content = [{"type": "image", "image": image}]
        if input_content:
            content.append({"type": "text", "text": input_content})
        messages = [{"role": "user", "content": content}]
        image_inputs, _ = process_vision_info(messages)
    else:
        messages = [{"role": "user", "content": input_content}]
        image_inputs = None

    formatted = processor.tokenizer.apply_chat_template(
        messages,
        template=template,
        examples=examples,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[formatted], images=image_inputs, padding=True, return_tensors="pt"
    ).to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
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
        return json.loads(decoded[0])
    except (json.JSONDecodeError, IndexError):
        return None


def _has_config_errors(template_error, examples_error):
    if template_error:
        st.error(f"Fix template: {template_error}")
        return True
    if examples_error:
        st.error(f"Fix examples: {examples_error}")
        return True
    return False


# --- Streamlit UI ---

st.title("Text Extraction Pipeline")

with st.sidebar:
    st.header("Model")
    st.code(MODEL_ID, language=None)
    st.subheader("Template")
    template_str = st.text_area("JSON template", value=DEFAULT_TEMPLATE, height=100)
    template_parsed, template_error = validate_template(template_str)
    if template_error:
        st.error(template_error)
    with st.expander("Supported types"):
        st.markdown(
            "- **verbatim-string** — extract text present verbatim in the input\n"
            "- **string** — generic string, can incorporate paraphrasing/abstraction\n"
            "- **integer** — a whole number\n"
            "- **number** — a whole or decimal number\n"
            "- **date-time** — ISO formatted date\n"
            '- **array** — array of any type above, e.g. `["string"]`\n'
            '- **enum** — choice from a set of options, e.g. `["yes", "no", "maybe"]`\n'
            "- **multi-label** — enum with multiple answers, "
            'e.g. `[["A", "B", "C"]]`\n\n'
            "If no relevant information is found, the model returns `null` "
            "or `[]` for arrays/multi-labels."
        )
    st.subheader("Examples")
    examples_str = st.text_area(
        "ICL examples (JSON array)", value=DEFAULT_EXAMPLES, height=200
    )
    examples_parsed, examples_error = parse_examples(examples_str)
    if examples_error:
        st.error(examples_error)

device = get_device()

with st.spinner(f"Loading {MODEL_ID} on {device.upper()}..."):
    model, processor = load_model(device)

text_tab, image_tab, csv_tab = st.tabs(["Text", "Image", "CSV Batch"])

with text_tab:
    input_text = st.text_area(
        "Enter text to extract from", height=150, key="text_input"
    )
    if st.button("Extract", type="primary", key="text_extract"):
        if not _has_config_errors(template_error, examples_error):
            if not input_text.strip():
                st.warning("Enter some text.")
            else:
                with st.spinner("Extracting..."):
                    result = extract(
                        input_text,
                        model,
                        processor,
                        device,
                        template_str,
                        examples_parsed,
                    )
                if result is not None:
                    st.json(result)
                else:
                    st.error(
                        "Extraction failed — could not parse model output as JSON."
                    )

with image_tab:
    uploaded_image = st.file_uploader(
        "Upload image", type=["png", "jpg", "jpeg", "webp"], key="image_upload"
    )
    image_context = st.text_area(
        "Additional context (optional)", height=100, key="image_context"
    )
    if uploaded_image is not None:
        pil_image = Image.open(uploaded_image)
        st.image(pil_image, width="stretch")
    if st.button("Extract", type="primary", key="image_extract"):
        if not _has_config_errors(template_error, examples_error):
            if uploaded_image is None:
                st.warning("Upload an image.")
            else:
                with st.spinner("Extracting..."):
                    result = extract(
                        image_context or None,
                        model,
                        processor,
                        device,
                        template_str,
                        examples_parsed,
                        image=pil_image,
                    )
                if result is not None:
                    st.json(result)
                else:
                    st.error(
                        "Extraction failed — could not parse model output as JSON."
                    )

with csv_tab:
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        key="csv_upload",
        help="Upload a CSV file with text to extract from",
    )
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded. ({len(df)} rows)")

            selected_column = st.selectbox(
                "Select text column", options=df.columns.tolist()
            )

            if st.button("Extract", type="primary", key="csv_extract"):
                if not _has_config_errors(template_error, examples_error):
                    results = []
                    progress_bar = st.progress(0, text="Starting...")

                    with st.spinner("Extracting..."):
                        for i, text in enumerate(df[selected_column].astype(str)):
                            result = extract(
                                text,
                                model,
                                processor,
                                device,
                                template_str,
                                examples_parsed,
                            )
                            results.append(result)
                            progress_bar.progress(
                                (i + 1) / len(df),
                                text=f"Processing row {i + 1} of {len(df)}",
                            )

                    progress_bar.progress(1.0, text="Done.")

                    fields = list(template_parsed.keys())
                    for field in fields:
                        df[field] = [
                            r.get(field, "") if r is not None else "" for r in results
                        ]

                    st.write("Preview")
                    st.dataframe(
                        df[[selected_column] + fields].head(),
                        width="stretch",
                    )

                    total = len(df)
                    failed = sum(1 for r in results if r is None)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Rows", total)
                    col2.metric("Extracted", total - failed)
                    col3.metric("Failed", failed)

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
