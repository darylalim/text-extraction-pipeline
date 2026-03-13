# Design: Token Limit, Auto Template Generation, ICL Example Images

## Overview

Three features added to the text-extraction-pipeline Streamlit app:

1. **Token limit enforcement** — Prevent OOM/hangs on large inputs
2. **Auto template generation** — Generate JSON templates from natural language descriptions
3. **ICL example images** — Support images in few-shot examples on the Image tab

## 1. Token Limit Enforcement

**Constant:** `MAX_INPUT_TOKENS = 10_000`

**Behavior:** In `extract()`, after tokenizing the input, check `inputs["input_ids"].shape[1]` against `MAX_INPUT_TOKENS`. If exceeded, raise a `ValueError` with a message including the actual token count and the limit. This check occurs after preprocessing (chat template formatting, processor tokenization, device transfer) — this is intentional because the accurate token count is only available after full tokenization, and the preprocessing cost is acceptable relative to the OOM risk of running `model.generate()`.

**UI handling:**
- Text/Image tabs: wrap `extract()` call in try/except for `ValueError`. Display the error via `st.error` with the token count info.
- CSV tab: wrap each per-row `extract()` call in try/except for `ValueError`. Over-limit rows produce `result = None` and their 1-based row numbers are collected. After all rows process, a `st.warning` lists the skipped row numbers. These rows are counted in the existing "Failed" metric. The progress bar continues normally through skipped rows.

## 2. Auto Template Generation (Dual-Mode Template Field)

**Mechanism (validated from the HF demo):** When the template is not valid JSON, pass `template=None` to `processor.tokenizer.apply_chat_template()` and set the user message content to the description text. The model generates a JSON template. This is the same approach used in the official NuExtract HF Space — the model's chat template natively supports template generation when `template=None`.

**New function in `utils.py`:**

```python
def generate_template(description: str, model, processor, device: str) -> tuple[dict | None, str | None]:
```

- Builds messages: `[{"role": "user", "content": description}]`
- Calls `processor.tokenizer.apply_chat_template(messages, template=None, tokenize=False, add_generation_prompt=True)`
- Tokenizes and runs `model.generate()` with `max_new_tokens=MAX_NEW_TOKENS` (256, same as extraction — templates are small JSON objects)
- Parses the decoded output as JSON
- Returns `(template_dict, None)` on success, `(None, error_message)` on failure

**Streamlit UI and session state:**

- The template text area uses `st.session_state` with a key (e.g., `"template_input"`) so its value can be programmatically updated.
- When `validate_template()` fails, the existing error is shown AND a "Generate Template" button appears in the sidebar below the error.
- Clicking the button calls `generate_template()`, and on success, sets `st.session_state["template_input"]` to the generated JSON string. Streamlit's automatic rerun on state change causes the text area to display the new value.
- On generation failure, `st.error` shows the error message. The original description text remains in the field.
- The button is disabled with a spinner while generation is in progress.

**Flow:**
1. User types natural language in the template field (e.g., "extract person name, age, and job")
2. `validate_template()` fails — error shown, "Generate Template" button appears
3. User clicks button — `generate_template()` is called, button shows spinner
4. Generated JSON replaces the template field content via session state
5. Page reruns, `validate_template()` now passes on the generated JSON
6. User reviews/edits the template, then clicks Extract

## 3. ICL Example Images (Image Tab, URL References)

**Extended examples format:** The `input` field in examples can be either a string (text, existing behavior) or a dict for images:

```json
[
  {
    "input": {"type": "image", "image": "https://example.com/photo.png"},
    "output": "{\"name\": \"John\"}"
  }
]
```

Mixed examples (some text, some image) are allowed in a single array.

**`parse_examples()` updates:** Accepts both string and dict `input` formats. For dict format, validates that:
- `type` is `"image"`
- `image` key is present and is a non-empty string (URL)
- Only `http://` and `https://` URLs are accepted (no `file://` or data URIs, for security)

**Chat template integration for image examples:**

Image example inputs need to produce `<image>` placeholder tokens in the formatted prompt so the processor can align them with actual image data. The approach:

1. When `extract()` receives examples with image-type inputs and `image` is not None (Image tab), it converts image example inputs to the Qwen2-VL message format: `[{"type": "image", "image": url}]`. This is passed to `apply_chat_template` which renders the appropriate vision tokens.
2. `process_all_vision_info()` extracts the actual image objects from both examples and the user message, returning them in the correct order (example images first, then input image) so they align with the placeholder positions in the tokenized prompt.
3. Text-only example inputs (strings) are passed through unchanged.

**New function in `utils.py`:**

```python
def process_all_vision_info(messages: list, examples: list | None = None) -> list | None:
```

Adapted from the HF demo's `utils.py`. Uses `qwen_vl_utils.process_vision_info` and `qwen_vl_utils.fetch_image` internally. Extracts images from both ICL examples and the user message, returns them as a flat list in correct order (example images first, then input images). Returns `None` if no images found.

**`extract()` updates:** When `image` is provided, call `process_all_vision_info(messages, examples)` instead of the current direct `process_vision_info(messages)` call to gather all images (examples + input).

**Scope:** Only the Image tab uses image examples. Text and CSV tabs call `extract()` without `image=`, so `process_all_vision_info` is not invoked — image-format examples in the JSON are simply ignored (the `input` dict is passed through to `apply_chat_template` as-is, and without corresponding image data the model treats it as a text reference). This avoids the need for tab-specific validation in `parse_examples()`.

**Implementation risk:** The interaction between image example inputs, `apply_chat_template`, and the processor's image alignment has not been fully validated against the NuExtract chat template's Jinja code. During implementation, this should be tested first with a simple integration test before building out the full feature. If the chat template does not natively handle image inputs in examples, a fallback approach is to strip image examples and only pass text examples to the model.

## 4. New `utils.py` Module

Contains two functions:
- `generate_template()` — template generation from natural language
- `process_all_vision_info()` — image extraction from examples + messages

Imported by `streamlit_app.py`. No other structural changes.

## 5. Testing

Tests for new `utils.py` functions go in a new `tests/test_utils.py` file. Tests for modified `streamlit_app.py` functions remain in `tests/test_streamlit_app.py`.

**New tests in `tests/test_utils.py`:**

- `generate_template`: valid description produces a template dict; invalid model output returns error tuple; model error propagates
- `process_all_vision_info`: examples with images returns correct order; no images returns `None`; mixed text/image examples; examples=None fallback

**New tests in `tests/test_streamlit_app.py`:**

- Token limit: input under limit passes; input at exactly the limit passes; input over limit raises `ValueError` with token count in message
- `parse_examples`: image input format accepted; missing `image` key rejected; non-http URL rejected; mixed text/image examples accepted
- `extract` with image examples: `process_all_vision_info` called with correct args when image provided

**Existing tests:** All 32 existing tests remain unchanged and passing.

## 6. CLAUDE.md Updates

After implementation, update CLAUDE.md to reflect:
- New `utils.py` module and its functions
- New `MAX_INPUT_TOKENS` constant
- Updated `parse_examples` behavior (image input format)
- Updated `extract` behavior (token limit, image examples)
- New test file `tests/test_utils.py`
