import json
import logging
from typing import NamedTuple

logging.getLogger("transformers.modeling_rope_utils").setLevel(logging.ERROR)

import streamlit as st  # noqa: E402
from mlx_lm import generate as mlx_generate  # noqa: E402
from mlx_lm import load as mlx_load  # noqa: E402

from utils import DEFAULT_MAX_NEW_TOKENS, detect_and_convert_template  # noqa: E402
from chunking import chunk_text  # noqa: E402
from merging import merge_results  # noqa: E402
from validation import annotate_icd10, count_invalid_codes, load_icd10_codes  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_ID = "mlx-community/numind-NuExtract-1.5-MLX-8bit"
MAX_INPUT_TOKENS = 4_096
DEFAULT_TEMPLATE = json.dumps(
    {
        "chief_complaint": "",
        "hpi": "",
        "review_of_systems": "",
        "vitals": {"bp": "", "hr": "", "temp": "", "rr": "", "spo2": ""},
        "exam_findings": "",
        "assessment": [{"diagnosis": "", "icd10_code": ""}],
        "plan": [],
    },
    indent=2,
)


@st.cache_data
def load_presets(path="presets.json"):
    """Load extraction presets from JSON file.

    Returns list of valid presets. Falls back to a single Person preset
    built from DEFAULT_TEMPLATE if file is missing or invalid.
    """
    fallback = [
        {
            "name": "SOAP Note",
            "template": json.loads(DEFAULT_TEMPLATE),
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
            and isinstance(entry.get("sample_text"), str)
        ):
            valid.append(entry)
        else:
            logger.warning("Skipping invalid preset at index %d", i)

    return valid if valid else fallback


@st.cache_resource
def load_model():
    """Load NuExtract-1.5-MLX model and tokenizer."""
    model, tokenizer = mlx_load(MODEL_ID)
    return model, tokenizer


@st.cache_data
def _load_icd10_codes():
    return load_icd10_codes()


def validate_template(template_str):
    """Validate that a string is valid JSON containing a non-empty dict."""
    try:
        parsed = json.loads(template_str)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"
    if not isinstance(parsed, dict):
        return None, "Template must be a JSON object."
    if not parsed:
        return None, "Template must not be empty."
    return parsed, None


def build_prompt(template_str, text):
    """Build a NuExtract-1.5 extraction prompt."""
    template_json = json.dumps(json.loads(template_str), indent=4)
    return (
        f"<|input|>\n### Template:\n{template_json}\n### Text:\n{text}\n\n<|output|>\n"
    )


def extract(text, model, tokenizer, template, max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
    """Extract structured data from text.

    Returns (dict|None, was_truncated). Raises ValueError if input exceeds
    MAX_INPUT_TOKENS.
    """
    prompt = build_prompt(template, text)

    input_tokens = tokenizer.encode(prompt)
    if len(input_tokens) > MAX_INPUT_TOKENS:
        raise ValueError(
            f"Input too long: {len(input_tokens)} tokens (limit: {MAX_INPUT_TOKENS})."
        )

    response = mlx_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_new_tokens, verbose=False
    )

    response = response.split("<|end-output|>")[0].strip()

    response_tokens = tokenizer.encode(response)
    was_truncated = len(response_tokens) >= max_new_tokens

    try:
        return json.loads(response), was_truncated
    except json.JSONDecodeError:
        return None, was_truncated


def _has_config_errors(template_error, template_parsed):
    """Show first config error via st.error and return True, or return False."""
    if template_error:
        st.error(f"Fix template: {template_error}")
        return True
    if template_parsed is None:
        st.error("Provide a valid JSON, YAML, or Pydantic template.")
        return True
    return False


def _get_effective_template(json_str, source_format, template_str):
    """Get the effective JSON template, converting from YAML/Pydantic if needed."""
    if source_format in ("yaml", "pydantic", "pydantic_with_unknown"):
        st.session_state["template_input"] = json_str
        return json_str
    return template_str


class ConfigState(NamedTuple):
    """State returned by _render_config()."""

    template_str: str
    json_str: str | None
    source_format: str | None
    template_error: str | None
    template_parsed: dict | None
    max_new_tokens: int


def _render_config():
    """Render config controls (preset, template, max tokens) and return state."""
    presets = load_presets()
    preset_names = [p["name"] for p in presets] + ["Custom"]

    if "prev_preset" not in st.session_state:
        st.session_state["prev_preset"] = "Custom"

    col1, col2 = st.columns(2)

    with col1:
        selected_preset = st.selectbox(
            "Load preset",
            preset_names,
            index=len(preset_names) - 1,
            key="preset_selector",
        )

    if selected_preset != st.session_state["prev_preset"]:
        st.session_state["prev_preset"] = selected_preset
        if selected_preset != "Custom":
            preset = next((p for p in presets if p["name"] == selected_preset), None)
            if preset is None:
                return ConfigState(
                    template_str="",
                    json_str=None,
                    source_format=None,
                    template_error="Preset not found.",
                    template_parsed=None,
                    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                )
            st.session_state["template_input"] = json.dumps(
                preset["template"], indent=2
            )
            st.session_state["text_input"] = preset["sample_text"]
            st.rerun()

    with col2:
        max_new_tokens = st.slider(
            "Max new tokens",
            min_value=64,
            max_value=4096,
            value=DEFAULT_MAX_NEW_TOKENS,
            step=64,
            help="Maximum tokens to generate. Increase for complex templates.",
        )

    if "template_input" not in st.session_state:
        st.session_state["template_input"] = DEFAULT_TEMPLATE
    template_str = st.text_area(
        "JSON template",
        height=300,
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
                "Nested models simplified to empty string; edit the JSON template to add structure."
            )
        template_parsed, _ = validate_template(json_str)
    else:
        st.error(template_error)
        template_parsed = None
    with st.expander("Template format"):
        st.markdown(
            '- Use `""` (empty string) as placeholder for text fields\n'
            "- Use `[]` (empty array) for list fields\n"
            "- Use nested objects for structured fields\n\n"
            "The model extracts text verbatim from the input to fill placeholders."
        )

    return ConfigState(
        template_str=template_str,
        json_str=json_str,
        source_format=source_format,
        template_error=template_error,
        template_parsed=template_parsed,
        max_new_tokens=max_new_tokens,
    )


def _extract_single_chunk(text, model, tokenizer, template_str, max_new_tokens):
    """Run extraction on a single chunk. Returns result dict or None; displays
    any error/warning via st.* before returning."""
    try:
        result, was_truncated = extract(
            text, model, tokenizer, template_str, max_new_tokens
        )
    except ValueError as e:
        st.error(str(e))
        return None
    except RuntimeError as e:
        st.error(f"Runtime error: {e}")
        return None

    if was_truncated:
        st.warning("Output may be truncated — consider increasing max tokens.")
    if result is None:
        st.error("Extraction failed — could not parse model output as JSON.")
        return None
    return result


def _extract_multi_chunk(
    chunks, model, tokenizer, template_str, template_parsed, max_new_tokens
):
    """Extract from each chunk, display progress, merge. Returns merged result
    dict or None; displays warnings for partial failures and truncation."""
    results = []
    truncated_chunks = []
    progress = st.progress(0, text="Starting...")

    for i, chunk in enumerate(chunks):
        try:
            r, was_truncated = extract(
                chunk, model, tokenizer, template_str, max_new_tokens
            )
        except (ValueError, RuntimeError):
            r, was_truncated = None, False
        results.append(r)
        if was_truncated:
            truncated_chunks.append(i + 1)
        progress.progress(
            (i + 1) / len(chunks), text=f"Chunk {i + 1} of {len(chunks)}..."
        )

    progress.progress(1.0, text="Done.")

    succeeded = sum(1 for r in results if r is not None)
    if succeeded == 0:
        st.error("Extraction failed — all chunks returned invalid output.")
        return None
    if succeeded < len(chunks):
        st.warning(
            f"Extraction succeeded for {succeeded} of {len(chunks)} chunks. Results may be incomplete."
        )
    if truncated_chunks:
        st.warning(
            f"Output may be truncated in chunk(s) {truncated_chunks} — consider increasing max tokens."
        )

    merged = merge_results(results, template_parsed)
    if merged is None:
        st.error("Extraction failed — could not merge results.")
    return merged


def _validate_and_display(result):
    """Annotate ICD-10 codes, display warnings, render JSON."""
    codes = _load_icd10_codes()
    if codes:
        result = annotate_icd10(result, codes)
        invalid = count_invalid_codes(result)
        if invalid:
            st.warning(f"{invalid} extracted ICD-10 code(s) not found in CMS 2025 set.")
    elif not st.session_state.get("_icd10_warning_shown"):
        st.warning("ICD-10 code list not loaded — validation skipped.")
        st.session_state["_icd10_warning_shown"] = True

    st.json(result)


def _run_extraction(
    text,
    model,
    tokenizer,
    template_str,
    template_parsed,
    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
):
    template_overhead = len(tokenizer.encode(build_prompt(template_str, "")))
    text_token_count = len(tokenizer.encode(text))
    max_text_tokens = MAX_INPUT_TOKENS - template_overhead

    if text_token_count <= max_text_tokens:
        result = _extract_single_chunk(
            text, model, tokenizer, template_str, max_new_tokens
        )
    else:
        chunks = chunk_text(text, tokenizer, max_tokens=max_text_tokens)
        st.info(
            f"Input is {text_token_count} tokens — splitting into {len(chunks)} chunks."
        )
        result = _extract_multi_chunk(
            chunks, model, tokenizer, template_str, template_parsed, max_new_tokens
        )

    if result is None:
        return
    _validate_and_display(result)


# --- Streamlit UI ---

st.title("NuExtract Pipeline")

with st.spinner(f"Loading {MODEL_ID}..."):
    model, tokenizer = load_model()

(
    template_str,
    json_str,
    source_format,
    template_error,
    template_parsed,
    max_new_tokens,
) = _render_config()

input_text = st.text_area("Enter text to extract from", height=150, key="text_input")
if st.button("Extract", type="primary", key="text_extract"):
    if not _has_config_errors(template_error, template_parsed):
        if not input_text.strip():
            st.warning("Enter some text.")
        else:
            effective = _get_effective_template(json_str, source_format, template_str)
            with st.spinner("Extracting..."):
                _run_extraction(
                    input_text,
                    model,
                    tokenizer,
                    effective,
                    template_parsed,
                    max_new_tokens=max_new_tokens,
                )
