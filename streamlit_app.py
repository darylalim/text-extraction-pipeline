import json
import logging
import os
import warnings

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from utils import (
    DEFAULT_MAX_NEW_TOKENS,
    detect_and_convert_template,
    generate_template,
    process_all_vision_info,
)

warnings.filterwarnings("ignore", message=".*MPS: The constant padding.*")
warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")

logger = logging.getLogger(__name__)

MODEL_ID = "numind/NuExtract-2.0-4B"
MAX_INPUT_TOKENS = 10_000
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


@st.cache_data
def load_presets(path="presets.json"):
    """Load extraction presets from JSON file.

    Returns list of valid presets. Falls back to a single Person preset
    built from DEFAULT_TEMPLATE/DEFAULT_EXAMPLES if file is missing or invalid.
    """
    fallback = [
        {
            "name": "Person",
            "template": json.loads(DEFAULT_TEMPLATE),
            "examples": json.loads(DEFAULT_EXAMPLES),
            "sample_text": "",
        }
    ]
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning("Failed to load presets from %s: %s", path, e)
        return fallback

    if not isinstance(data, list):
        logger.warning("presets.json root is not a list, using fallback")
        return fallback

    valid = []
    for i, entry in enumerate(data):
        if (
            isinstance(entry, dict)
            and isinstance(entry.get("name"), str)
            and isinstance(entry.get("template"), dict)
            and isinstance(entry.get("examples"), list)
            and isinstance(entry.get("sample_text"), str)
        ):
            valid.append(entry)
        else:
            logger.warning("Skipping invalid preset at index %d", i)

    return valid if valid else fallback


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
        inp = ex["input"]
        if isinstance(inp, dict):
            if inp.get("type") != "image":
                return None, f'Example {i + 1} input dict must have "type": "image".'
            url = inp.get("image")
            if not isinstance(url, str) or not url:
                return None, f'Example {i + 1} must have a non-empty "image" URL.'
            if not url.startswith(("http://", "https://")):
                return None, f"Example {i + 1} image URL must use http:// or https://."
    return parsed, None


def extract(
    input_content,
    model,
    processor,
    device,
    template,
    examples,
    image=None,
    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
):
    if image is not None:
        content = [{"type": "image", "image": image}]
        if input_content:
            content.append({"type": "text", "text": input_content})
        messages = [{"role": "user", "content": content}]
        image_inputs = process_all_vision_info(messages, examples)
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
    if input_len > MAX_INPUT_TOKENS:
        raise ValueError(
            f"Input too long: {input_len} tokens (limit: {MAX_INPUT_TOKENS})."
        )

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
        )

    trimmed = output[:, input_len:]
    was_truncated = trimmed.shape[1] == max_new_tokens
    decoded = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    try:
        return json.loads(decoded[0]), was_truncated
    except (json.JSONDecodeError, IndexError):
        return None, was_truncated


def _has_config_errors(template_error, examples_error, template_parsed):
    if template_error:
        st.error(f"Fix template: {template_error}")
        return True
    if template_parsed is None:
        st.error("Generate a JSON template first.")
        return True
    if examples_error:
        st.error(f"Fix examples: {examples_error}")
        return True
    return False


def _convert_template_if_needed(json_str, source_format):
    """Convert non-JSON template to JSON and update the template field."""
    if source_format in ("yaml", "pydantic", "pydantic_with_unknown"):
        st.session_state["template_input"] = json_str
        return json_str
    return None


def _clear_device_cache(device):
    """Clear device memory cache after OOM errors."""
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


# --- Streamlit UI ---

st.title("Text Extraction Pipeline")

device = get_device()

with st.spinner(f"Loading {MODEL_ID} on {device.upper()}..."):
    model, processor = load_model(device)

with st.sidebar:
    st.header("Model")
    st.code(MODEL_ID, language=None)
    st.subheader("Generation")
    max_new_tokens = st.slider(
        "Max new tokens",
        min_value=64,
        max_value=4096,
        value=DEFAULT_MAX_NEW_TOKENS,
        step=64,
        help="Maximum tokens to generate. Increase for complex templates.",
    )
    presets = load_presets()
    preset_names = [p["name"] for p in presets] + ["Custom"]

    if "prev_preset" not in st.session_state:
        st.session_state["prev_preset"] = "Custom"

    selected_preset = st.selectbox(
        "Load preset", preset_names, index=len(preset_names) - 1, key="preset_selector"
    )

    if selected_preset != st.session_state["prev_preset"]:
        st.session_state["prev_preset"] = selected_preset
        if selected_preset != "Custom":
            preset = next(p for p in presets if p["name"] == selected_preset)
            st.session_state["template_input"] = json.dumps(
                preset["template"], indent=2
            )
            st.session_state["examples_input"] = json.dumps(
                preset["examples"], indent=2
            )
            st.session_state["text_input"] = preset["sample_text"]
            st.rerun()

    st.subheader("Template")
    if "template_input" not in st.session_state:
        st.session_state["template_input"] = DEFAULT_TEMPLATE
    template_str = st.text_area(
        "JSON template or description",
        height=100,
        key="template_input",
    )
    json_str, source_format, template_error = detect_and_convert_template(template_str)
    if source_format == "json":
        template_parsed, _ = validate_template(template_str)
    elif source_format in ("yaml", "pydantic", "pydantic_with_unknown"):
        fmt_label = "pydantic" if "pydantic" in source_format else source_format
        st.info(f"Detected {fmt_label} template — will convert to JSON on extract.")
        if source_format == "pydantic_with_unknown":
            st.warning(
                "Nested models simplified to string; edit the JSON template to add structure."
            )
        template_parsed, _ = validate_template(json_str)
    elif template_error:
        st.error(template_error)
        template_parsed = None
    else:
        st.info("Template will be used as a natural language description.")
        template_parsed = None
        if st.button("Generate Template", key="generate_template"):
            with st.spinner("Generating template..."):
                generated, gen_error = generate_template(
                    template_str, model, processor, device
                )
            if generated is not None:
                st.session_state["template_input"] = json.dumps(generated, indent=2)
                st.rerun()
            else:
                st.error(f"Generation failed: {gen_error}")
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
    if "examples_input" not in st.session_state:
        st.session_state["examples_input"] = DEFAULT_EXAMPLES
    examples_str = st.text_area(
        "ICL examples (JSON array)", height=200, key="examples_input"
    )
    examples_parsed, examples_error = parse_examples(examples_str)
    if examples_error:
        st.error(examples_error)

text_tab, image_tab, csv_tab = st.tabs(["Text", "Image", "CSV Batch"])

with text_tab:
    input_text = st.text_area(
        "Enter text to extract from", height=150, key="text_input"
    )
    if st.button("Extract", type="primary", key="text_extract"):
        if not _has_config_errors(template_error, examples_error, template_parsed):
            if not input_text.strip():
                st.warning("Enter some text.")
            else:
                with st.spinner("Extracting..."):
                    try:
                        converted = _convert_template_if_needed(json_str, source_format)
                        if converted:
                            template_str = converted
                        result, was_truncated = extract(
                            input_text,
                            model,
                            processor,
                            device,
                            template_str,
                            examples_parsed,
                            max_new_tokens=max_new_tokens,
                        )
                        if was_truncated:
                            st.warning(
                                "Output may be truncated — consider increasing max tokens."
                            )
                        if result is not None:
                            st.json(result)
                        else:
                            st.error(
                                "Extraction failed — could not parse model output as JSON."
                            )
                    except ValueError as e:
                        st.error(str(e))
                    except RuntimeError as e:
                        st.error(f"Runtime error: {e}. Try reducing max tokens.")

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
        if not _has_config_errors(template_error, examples_error, template_parsed):
            if uploaded_image is None:
                st.warning("Upload an image.")
            else:
                with st.spinner("Extracting..."):
                    try:
                        converted = _convert_template_if_needed(json_str, source_format)
                        if converted:
                            template_str = converted
                        result, was_truncated = extract(
                            image_context or None,
                            model,
                            processor,
                            device,
                            template_str,
                            examples_parsed,
                            image=pil_image,
                            max_new_tokens=max_new_tokens,
                        )
                        if was_truncated:
                            st.warning(
                                "Output may be truncated — consider increasing max tokens."
                            )
                        if result is not None:
                            st.json(result)
                        else:
                            st.error(
                                "Extraction failed — could not parse model output as JSON."
                            )
                    except ValueError as e:
                        st.error(str(e))
                    except RuntimeError as e:
                        st.error(f"Runtime error: {e}. Try reducing max tokens.")

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
                if not _has_config_errors(
                    template_error, examples_error, template_parsed
                ):
                    results = []
                    progress_bar = st.progress(0, text="Starting...")

                    with st.spinner("Extracting..."):
                        converted = _convert_template_if_needed(json_str, source_format)
                        if converted:
                            template_str = converted
                        skipped_rows = []
                        truncated_rows = []
                        for i, text in enumerate(df[selected_column].astype(str)):
                            try:
                                result, was_truncated = extract(
                                    text,
                                    model,
                                    processor,
                                    device,
                                    template_str,
                                    examples_parsed,
                                    max_new_tokens=max_new_tokens,
                                )
                                if was_truncated:
                                    truncated_rows.append(i + 1)
                            except (ValueError, RuntimeError):
                                result = None
                                skipped_rows.append(i + 1)
                            results.append(result)
                            progress_bar.progress(
                                (i + 1) / len(df),
                                text=f"Processing row {i + 1} of {len(df)}",
                            )

                    progress_bar.progress(1.0, text="Done.")
                    if skipped_rows:
                        st.warning(f"Rows skipped (input too long): {skipped_rows}")
                    if truncated_rows:
                        st.warning(f"Rows possibly truncated: {truncated_rows}")

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

        except (pd.errors.ParserError, KeyError, UnicodeDecodeError) as e:
            st.error(f"Error: {e}")
    else:
        st.info("Upload a CSV file.")
