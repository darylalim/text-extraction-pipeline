# Approach A Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add configurable max_new_tokens, YAML/Pydantic template input, and example presets to the extraction pipeline.

**Architecture:** Three independent features layered onto the existing `streamlit_app.py` + `utils.py` codebase. Feature 1 modifies `extract()` signature and adds a sidebar slider. Feature 2 adds `detect_and_convert_template()` to `utils.py` and replaces `validate_template()` usage in the sidebar. Feature 3 adds `presets.json` and a preset selector in the sidebar.

**Tech Stack:** Python, Streamlit, PyYAML, pytest

**Spec:** `docs/superpowers/specs/2026-03-14-approach-a-design.md`

---

## Chunk 1: Configurable max_new_tokens

### Task 1: Rename constant and update extract() signature

**Files:**
- Modify: `utils.py:6` (rename constant)
- Modify: `streamlit_app.py:15` (update import)
- Modify: `streamlit_app.py:136-179` (update `extract()` signature and body)
- Test: `tests/test_streamlit_app.py`
- Test: `tests/test_utils.py`

- [ ] **Step 1: Write failing test for new extract() signature**

In `tests/test_streamlit_app.py`, add after line 463 (after `test_extract_at_token_limit_succeeds`):

```python
def test_extract_uses_custom_max_new_tokens(app):
    output = json.dumps({"company": "Acme"})
    model, processor = make_mocks(output)
    app.extract(
        "some text", model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES,
        max_new_tokens=512,
    )
    gen_call = model.generate.call_args
    assert gen_call[1]["max_new_tokens"] == 512


def test_extract_default_max_new_tokens_is_2048(app):
    output = json.dumps({"company": "Acme"})
    model, processor = make_mocks(output)
    app.extract("some text", model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES)
    gen_call = model.generate.call_args
    assert gen_call[1]["max_new_tokens"] == 2048
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py::test_extract_uses_custom_max_new_tokens tests/test_streamlit_app.py::test_extract_default_max_new_tokens_is_2048 -v`
Expected: FAIL — `max_new_tokens` is currently 256, not 2048; custom value not accepted.

- [ ] **Step 3: Rename constant in utils.py**

In `utils.py`, change line 6:
```python
# Old:
MAX_NEW_TOKENS = 256
# New:
DEFAULT_MAX_NEW_TOKENS = 2048
```

- [ ] **Step 4: Update import in streamlit_app.py**

In `streamlit_app.py`, change line 15:
```python
# Old:
from utils import MAX_NEW_TOKENS, generate_template, process_all_vision_info
# New:
from utils import DEFAULT_MAX_NEW_TOKENS, generate_template, process_all_vision_info
```

- [ ] **Step 5: Update extract() signature and body**

In `streamlit_app.py`, replace the `extract` function (lines 136-179):
```python
def extract(input_content, model, processor, device, template, examples, image=None, max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
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
```

- [ ] **Step 6: Update generate_template() to use hardcoded 256**

In `utils.py`, change line 34 inside `generate_template()`:
```python
# Old:
            max_new_tokens=MAX_NEW_TOKENS,
# New:
            max_new_tokens=256,
```

- [ ] **Step 7: Run new tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py::test_extract_uses_custom_max_new_tokens tests/test_streamlit_app.py::test_extract_default_max_new_tokens_is_2048 -v`
Expected: PASS

- [ ] **Step 8: Fix all existing extract() call sites to unpack tuple**

Every call to `extract()` now returns `(result, was_truncated)`. Update all three tabs in `streamlit_app.py`:

**Text tab** (around line 260): change `result = extract(...)` to `result, was_truncated = extract(...)`
**Image tab** (around line 294): change `result = extract(...)` to `result, was_truncated = extract(...)`
**CSV tab** (around line 337): change `result = extract(...)` to `result, was_truncated = extract(...)` (ignore `was_truncated` for individual rows in batch)

- [ ] **Step 9: Fix all existing tests that call extract()**

Every test that calls `app.extract(...)` and checks the return value needs to unpack the tuple. Update these tests in `tests/test_streamlit_app.py`:

- `test_extract_returns_parsed_dict`: `result, _ = app.extract(...)`
- `test_extract_json_failure_returns_none`: `result, _ = app.extract(...)`
- `test_extract_empty_values_returns_dict`: `result, _ = app.extract(...)`
- `test_extract_at_token_limit_succeeds`: `result, _ = app.extract(...)`

Tests that don't check the return value (e.g., `test_extract_decodes_only_new_tokens`) need no change.

- [ ] **Step 10: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests pass.

- [ ] **Step 11: Write test that generate_template() still uses 256**

In `tests/test_utils.py`, add:

```python
def test_generate_template_uses_256_max_new_tokens():
    from utils import generate_template

    output = json.dumps({"name": "string"})
    model, processor = make_mocks(output)
    generate_template("extract name", model, processor, "cpu")
    gen_call = model.generate.call_args
    assert gen_call[1]["max_new_tokens"] == 256
```

- [ ] **Step 12: Run the new test**

Run: `uv run pytest tests/test_utils.py::test_generate_template_uses_256_max_new_tokens -v`
Expected: PASS

- [ ] **Step 13: Commit**

```bash
git add utils.py streamlit_app.py tests/test_streamlit_app.py tests/test_utils.py
git commit -m "feat: make max_new_tokens configurable with 2048 default

extract() now accepts max_new_tokens kwarg and returns (result, was_truncated).
Rename MAX_NEW_TOKENS to DEFAULT_MAX_NEW_TOKENS. generate_template() keeps 256."
```

### Task 2: Add truncation detection test

**Files:**
- Test: `tests/test_streamlit_app.py`

- [ ] **Step 1: Write test for truncation detection**

In `tests/test_streamlit_app.py`, add:

```python
def test_extract_detects_truncation(app):
    output = json.dumps({"company": "Acme"})
    model, processor = make_mocks(output)
    # make_mocks generates output tensor of shape (1, 8), input_ids shape (1, 5)
    # trimmed = 3 tokens. Set max_new_tokens=3 to trigger truncation.
    _, was_truncated = app.extract(
        "some text", model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES,
        max_new_tokens=3,
    )
    assert was_truncated is True


def test_extract_no_truncation_when_output_shorter(app):
    output = json.dumps({"company": "Acme"})
    model, processor = make_mocks(output)
    # trimmed = 3 tokens, max_new_tokens=100 → no truncation
    _, was_truncated = app.extract(
        "some text", model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES,
        max_new_tokens=100,
    )
    assert was_truncated is False
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_streamlit_app.py::test_extract_detects_truncation tests/test_streamlit_app.py::test_extract_no_truncation_when_output_shorter -v`
Expected: PASS (implementation already done in Task 1).

- [ ] **Step 3: Commit**

```bash
git add tests/test_streamlit_app.py
git commit -m "test: add truncation detection tests for extract()"
```

### Task 3: Add sidebar slider and wire up all tabs

**Files:**
- Modify: `streamlit_app.py:201-246` (sidebar section)
- Modify: `streamlit_app.py:249-276` (text tab)
- Modify: `streamlit_app.py:277-310` (image tab)
- Modify: `streamlit_app.py:312-389` (CSV tab)

- [ ] **Step 1: Add slider to sidebar**

In `streamlit_app.py`, after the `st.code(MODEL_ID, ...)` line (line 203), add:

```python
    st.subheader("Generation")
    max_new_tokens = st.slider(
        "Max new tokens",
        min_value=64,
        max_value=4096,
        value=DEFAULT_MAX_NEW_TOKENS,
        step=64,
        help="Maximum tokens to generate. Increase for complex templates.",
    )
```

- [ ] **Step 2: Pass max_new_tokens to extract() in text tab**

Update the text tab `extract()` call to pass `max_new_tokens=max_new_tokens`. Add truncation and RuntimeError handling:

```python
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
                    try:
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
                        st.error(
                            f"Runtime error: {e}. Try reducing max tokens."
                        )
```

- [ ] **Step 3: Pass max_new_tokens to extract() in image tab**

Same pattern as text tab — add `max_new_tokens=max_new_tokens`, truncation warning, and RuntimeError catch.

- [ ] **Step 4: Pass max_new_tokens to extract() in CSV tab**

Update CSV tab. Extend the existing `except ValueError` to also catch `RuntimeError`:

```python
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
```

Add `truncated_rows = []` before the loop and show a warning after:
```python
                    if truncated_rows:
                        st.warning(f"Rows possibly truncated: {truncated_rows}")
```

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests pass.

- [ ] **Step 6: Run lint and format**

Run: `uv run ruff check . && uv run ruff format .`

- [ ] **Step 7: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add max_new_tokens sidebar slider with truncation warnings

Slider range 64-4096 (default 2048). All tabs pass value to extract()
and display truncation/OOM warnings."
```

---

## Chunk 2: YAML and Pydantic template input

### Task 4: Add pyyaml dependency

**Files:**
- Modify: `pyproject.toml:6-15`

- [ ] **Step 1: Add pyyaml to pyproject.toml**

In `pyproject.toml`, add `"pyyaml==6.0.2"` to the `dependencies` list (after `"pandas==2.3.3"`).

- [ ] **Step 2: Sync dependencies**

Run: `uv sync`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add pyyaml==6.0.2 dependency"
```

### Task 5: Implement detect_and_convert_template() — JSON and YAML paths

**Files:**
- Modify: `utils.py` (add new function)
- Test: `tests/test_utils.py`

- [ ] **Step 1: Write failing tests for JSON detection**

In `tests/test_utils.py`, add:

```python
# --- detect_and_convert_template ---


def test_detect_json_valid():
    from utils import detect_and_convert_template

    json_str, fmt, error = detect_and_convert_template('{"name": "string"}')
    assert fmt == "json"
    assert json.loads(json_str) == {"name": "string"}
    assert error is None


def test_detect_json_empty_object():
    from utils import detect_and_convert_template

    json_str, fmt, error = detect_and_convert_template("{}")
    assert json_str is None
    assert fmt is None
    assert error is not None
    assert "empty" in error.lower()


def test_detect_json_non_dict():
    from utils import detect_and_convert_template

    json_str, fmt, error = detect_and_convert_template("[1, 2, 3]")
    assert json_str is None
    assert fmt is None
    assert error is not None
    assert "object" in error.lower()


def test_detect_empty_string():
    from utils import detect_and_convert_template

    json_str, fmt, error = detect_and_convert_template("")
    assert json_str is None
    assert fmt is None
    assert error is not None
    assert "empty" in error.lower()


def test_detect_whitespace_only():
    from utils import detect_and_convert_template

    json_str, fmt, error = detect_and_convert_template("   ")
    assert json_str is None
    assert fmt is None
    assert error is not None
    assert "empty" in error.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_utils.py::test_detect_json_valid tests/test_utils.py::test_detect_json_empty_object tests/test_utils.py::test_detect_json_non_dict -v`
Expected: FAIL — function does not exist.

- [ ] **Step 3: Implement JSON path of detect_and_convert_template()**

In `utils.py`, add after the imports:

```python
import yaml


def detect_and_convert_template(template_str):
    """Detect template format and convert to JSON.

    Returns (json_str, source_format, error) where source_format is
    "json", "yaml", "pydantic", or None.
    """
    if not template_str or not template_str.strip():
        return None, None, "Template must not be empty."

    # 1. Try JSON
    try:
        parsed = json.loads(template_str)
        if isinstance(parsed, dict):
            if not parsed:
                return None, None, "Template must not be empty."
            return template_str.strip(), "json", None
        else:
            return None, None, "Template must be a JSON object."
    except json.JSONDecodeError:
        pass

    # 2. Try YAML (implemented in next step)
    # 3. Try Pydantic (implemented in next task)

    # No format matched — treat as natural language
    return None, None, None
```

- [ ] **Step 4: Run JSON tests**

Run: `uv run pytest tests/test_utils.py::test_detect_json_valid tests/test_utils.py::test_detect_json_empty_object tests/test_utils.py::test_detect_json_non_dict -v`
Expected: PASS

- [ ] **Step 5: Write failing tests for YAML detection**

In `tests/test_utils.py`, add:

```python
def test_detect_yaml_valid():
    from utils import detect_and_convert_template

    yaml_input = "name: string\nage: integer\n"
    json_str, fmt, error = detect_and_convert_template(yaml_input)
    assert fmt == "yaml"
    assert json.loads(json_str) == {"name": "string", "age": "integer"}
    assert error is None


def test_detect_yaml_nested():
    from utils import detect_and_convert_template

    yaml_input = "person:\n  name: verbatim-string\n  age: integer\n"
    json_str, fmt, error = detect_and_convert_template(yaml_input)
    assert fmt == "yaml"
    parsed = json.loads(json_str)
    assert parsed == {"person": {"name": "verbatim-string", "age": "integer"}}


def test_detect_yaml_not_dict():
    from utils import detect_and_convert_template

    yaml_input = "- item1\n- item2\n"
    json_str, fmt, error = detect_and_convert_template(yaml_input)
    # Falls through to natural language
    assert json_str is None
    assert fmt is None
    assert error is None


def test_detect_yaml_invalid():
    from utils import detect_and_convert_template

    yaml_input = "{{{\ninvalid: yaml: content: [["
    json_str, fmt, error = detect_and_convert_template(yaml_input)
    # Falls through to natural language (YAML parse fails silently)
    assert json_str is None
    assert fmt is None
    assert error is None
```

- [ ] **Step 6: Implement YAML path**

In `utils.py`, in `detect_and_convert_template()`, replace the `# 2. Try YAML` comment:

```python
    # 2. Try YAML
    try:
        parsed = yaml.safe_load(template_str)
        if isinstance(parsed, dict) and parsed:
            return json.dumps(parsed, indent=2), "yaml", None
    except yaml.YAMLError:
        pass
```

- [ ] **Step 7: Run all detect tests**

Run: `uv run pytest tests/test_utils.py -k "detect" -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add utils.py tests/test_utils.py
git commit -m "feat: add detect_and_convert_template() with JSON and YAML support"
```

### Task 6: Implement Pydantic regex parser

**Files:**
- Modify: `utils.py` (extend `detect_and_convert_template()`)
- Test: `tests/test_utils.py`

- [ ] **Step 1: Write failing tests for Pydantic detection**

In `tests/test_utils.py`, add:

```python
def test_detect_pydantic_flat_model():
    from utils import detect_and_convert_template

    pydantic_input = (
        "class Person(BaseModel):\n"
        "    name: str\n"
        "    age: int\n"
        "    salary: float\n"
        "    active: bool\n"
    )
    json_str, fmt, error = detect_and_convert_template(pydantic_input)
    assert fmt == "pydantic"
    parsed = json.loads(json_str)
    assert parsed == {
        "name": "string",
        "age": "integer",
        "salary": "number",
        "active": "boolean",
    }
    assert error is None


def test_detect_pydantic_list_type():
    from utils import detect_and_convert_template

    pydantic_input = (
        "class Job(BaseModel):\n"
        "    title: str\n"
        "    skills: list[str]\n"
    )
    json_str, fmt, error = detect_and_convert_template(pydantic_input)
    assert fmt == "pydantic"
    parsed = json.loads(json_str)
    assert parsed == {"title": "string", "skills": ["string"]}


def test_detect_pydantic_optional_type():
    from utils import detect_and_convert_template

    pydantic_input = (
        "class Item(BaseModel):\n"
        "    name: str\n"
        "    description: Optional[str]\n"
    )
    json_str, fmt, error = detect_and_convert_template(pydantic_input)
    assert fmt == "pydantic"
    parsed = json.loads(json_str)
    assert parsed == {"name": "string", "description": "string"}


def test_detect_pydantic_datetime_type():
    from utils import detect_and_convert_template

    pydantic_input = (
        "class Event(BaseModel):\n"
        "    name: str\n"
        "    date: datetime\n"
    )
    json_str, fmt, error = detect_and_convert_template(pydantic_input)
    assert fmt == "pydantic"
    parsed = json.loads(json_str)
    assert parsed == {"name": "string", "date": "date-time"}


def test_detect_pydantic_nested_model_falls_back_to_string():
    from utils import detect_and_convert_template

    pydantic_input = (
        "class Address(BaseModel):\n"
        "    street: str\n"
        "\n"
        "class Person(BaseModel):\n"
        "    name: str\n"
        "    address: Address\n"
    )
    json_str, fmt, error = detect_and_convert_template(pydantic_input)
    assert fmt == "pydantic"
    parsed = json.loads(json_str)
    # Last class is used; Address is unknown type → "string"
    assert parsed == {"name": "string", "address": "string"}


def test_detect_pydantic_nested_list_falls_back():
    from utils import detect_and_convert_template

    pydantic_input = (
        "class Item(BaseModel):\n"
        "    tags: list[list[str]]\n"
    )
    json_str, fmt, error = detect_and_convert_template(pydantic_input)
    assert fmt == "pydantic"
    parsed = json.loads(json_str)
    assert parsed == {"tags": ["string"]}


def test_detect_natural_language_fallthrough():
    from utils import detect_and_convert_template

    text = "Extract the person's name and age from the text"
    json_str, fmt, error = detect_and_convert_template(text)
    assert json_str is None
    assert fmt is None
    assert error is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_utils.py -k "pydantic or natural_language" -v`
Expected: FAIL — Pydantic path not implemented.

- [ ] **Step 3: Implement Pydantic regex parser**

In `utils.py`, add a helper function before `detect_and_convert_template()`:

```python
import re


_PYDANTIC_TYPE_MAP = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "datetime": "date-time",
}


def _parse_pydantic_model(text):
    """Parse the last BaseModel class in text into a JSON-compatible dict.

    Returns (dict, has_unknown_types) or (None, False) if no model found.
    """
    # Find all BaseModel classes
    class_pattern = re.compile(
        r"class\s+\w+\s*\(\s*BaseModel\s*\)\s*:(.*?)(?=\nclass\s|\Z)",
        re.DOTALL,
    )
    matches = list(class_pattern.finditer(text))
    if not matches:
        return None, False

    # Use the last class
    body = matches[-1].group(1)
    field_pattern = re.compile(r"^\s+(\w+)\s*:\s*(.+)$", re.MULTILINE)
    fields = field_pattern.findall(body)
    if not fields:
        return None, False

    result = {}
    has_unknown = False
    for name, type_str in fields:
        type_str = type_str.strip()
        mapped = _map_pydantic_type(type_str)
        if mapped is None:
            mapped = "string"
            has_unknown = True
        result[name] = mapped

    return result, has_unknown


def _map_pydantic_type(type_str):
    """Map a Pydantic type annotation to a NuExtract type."""
    type_str = type_str.strip()

    # Direct mapping
    if type_str in _PYDANTIC_TYPE_MAP:
        return _PYDANTIC_TYPE_MAP[type_str]

    # Optional[X]
    optional_match = re.match(r"Optional\[(.+)\]$", type_str)
    if optional_match:
        inner = optional_match.group(1).strip()
        return _map_pydantic_type(inner)

    # list[X] — one level only
    list_match = re.match(r"list\[(.+)\]$", type_str, re.IGNORECASE)
    if list_match:
        inner = list_match.group(1).strip()
        inner_mapped = _map_pydantic_type(inner)
        if inner_mapped is None or isinstance(inner_mapped, list):
            # Nested list or unknown inner type
            return ["string"]
        return [inner_mapped]

    # Unknown type
    return None
```

Then in `detect_and_convert_template()`, replace the `# 3. Try Pydantic` comment:

```python
    # 3. Try Pydantic
    parsed, has_unknown = _parse_pydantic_model(template_str)
    if parsed:
        json_out = json.dumps(parsed, indent=2)
        if has_unknown:
            return json_out, "pydantic_with_unknown", None
        return json_out, "pydantic", None
```

- [ ] **Step 4: Run all Pydantic tests**

Run: `uv run pytest tests/test_utils.py -k "pydantic or natural_language" -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add utils.py tests/test_utils.py
git commit -m "feat: add Pydantic regex parser to detect_and_convert_template()

Supports str, int, float, bool, datetime, list[X], Optional[X].
Nested models and list[list[X]] fall back to string."
```

### Task 7: Wire detect_and_convert_template() into sidebar

**Files:**
- Modify: `streamlit_app.py:15` (update import)
- Modify: `streamlit_app.py:100-109` (remove or keep `validate_template` for parse_examples gating)
- Modify: `streamlit_app.py:212-224` (sidebar template handling)

- [ ] **Step 1: Update import**

In `streamlit_app.py`, change the import line:

```python
# Old:
from utils import DEFAULT_MAX_NEW_TOKENS, generate_template, process_all_vision_info
# New:
from utils import DEFAULT_MAX_NEW_TOKENS, detect_and_convert_template, generate_template, process_all_vision_info
```

- [ ] **Step 2: Replace validate_template() call in sidebar**

Replace the sidebar template validation and "Generate Template" button logic. The key changes:
- Call `detect_and_convert_template(template_str)` instead of `validate_template(template_str)`
- For `"json"` format: use as-is, `template_error = None`
- For `"yaml"` or `"pydantic"`: show `st.info`, set `template_error = None`, store `json_str` for later conversion
- For `None` format with `None` error: show info about natural language, show Generate Template button
- For error: show `st.error`

Replace the sidebar template section (the `validate_template` call through the Generate Template button) with:

```python
    json_str, source_format, template_error = detect_and_convert_template(template_str)
    if source_format == "json":
        template_parsed, _ = validate_template(template_str)
    elif source_format in ("yaml", "pydantic", "pydantic_with_unknown"):
        fmt_label = "pydantic" if "pydantic" in source_format else source_format
        st.info(f"Detected {fmt_label} template — will convert to JSON on extract.")
        if source_format == "pydantic_with_unknown":
            st.warning("Nested models simplified to string; edit the JSON template to add structure.")
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
```

- [ ] **Step 3: Add conversion on Extract button press**

In each tab, before calling `extract()`, if `source_format` is `"yaml"` or `"pydantic"`, overwrite `template_str` with `json_str` and update the session state:

Add a helper function before the UI section:

```python
def _convert_template_if_needed(json_str, source_format):
    """Convert non-JSON template to JSON and update the template field."""
    if source_format in ("yaml", "pydantic", "pydantic_with_unknown"):
        st.session_state["template_input"] = json_str
        return json_str
    return None
```

In each tab's extract block, add before the `extract()` call:
```python
                        converted = _convert_template_if_needed(json_str, source_format)
                        if converted:
                            template_str = converted
```

- [ ] **Step 4: Update _has_config_errors to handle new flow**

The `template_error` from `detect_and_convert_template` is `None` for valid templates (json/yaml/pydantic) and for natural language. It's only non-None for actual errors (empty template). The existing `_has_config_errors` checks `if template_error:` which still works. But we also need to gate extraction when no template is available (natural language without generation). Update `_has_config_errors`:

```python
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
```

Update all three tab calls to pass `template_parsed`:
```python
        if not _has_config_errors(template_error, examples_error, template_parsed):
```

- [ ] **Step 5: Update _has_config_errors tests**

In `tests/test_streamlit_app.py`, update the four `_has_config_errors` tests to pass the new `template_parsed` parameter:

```python
def test_has_config_errors_template_error(app):
    with patch("streamlit_app.st") as mock_st:
        assert app._has_config_errors("bad json", None, None) is True
        mock_st.error.assert_called_once_with("Fix template: bad json")


def test_has_config_errors_examples_error(app):
    with patch("streamlit_app.st") as mock_st:
        assert app._has_config_errors(None, "missing keys", {"a": "b"}) is True
        mock_st.error.assert_called_once_with("Fix examples: missing keys")


def test_has_config_errors_no_errors(app):
    with patch("streamlit_app.st") as mock_st:
        assert app._has_config_errors(None, None, {"a": "b"}) is False
        mock_st.error.assert_not_called()


def test_has_config_errors_template_takes_priority(app):
    with patch("streamlit_app.st") as mock_st:
        assert app._has_config_errors("bad template", "bad examples", None) is True
        mock_st.error.assert_called_once_with("Fix template: bad template")


def test_has_config_errors_no_template_parsed(app):
    with patch("streamlit_app.st") as mock_st:
        assert app._has_config_errors(None, None, None) is True
        mock_st.error.assert_called_once_with("Generate a JSON template first.")
```

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests pass.

- [ ] **Step 7: Run lint and format**

Run: `uv run ruff check . && uv run ruff format .`

- [ ] **Step 8: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: wire YAML/Pydantic detection into sidebar

Replace validate_template() with detect_and_convert_template() in sidebar.
Convert to JSON on Extract button press. Add template_parsed gate."
```

---

## Chunk 3: Example presets

### Task 8: Create presets.json

**Files:**
- Create: `presets.json`

- [ ] **Step 1: Create presets.json with 5 presets**

Create `presets.json` at project root. Each preset has `name`, `template`, `examples`, `sample_text`. The Person preset matches the existing `DEFAULT_TEMPLATE` and `DEFAULT_EXAMPLES`.

```json
[
  {
    "name": "Person",
    "template": {
      "first_name": "verbatim-string",
      "last_name": "verbatim-string",
      "description": "string",
      "age": "integer",
      "gpa": "number",
      "birth_date": "date-time",
      "nationality": ["France", "England", "Japan", "USA", "China"],
      "languages_spoken": [["English", "French", "Japanese", "Mandarin", "Spanish"]]
    },
    "examples": [
      {
        "input": "Yuki Tanaka is a 22-year-old Japanese student with a 3.8 GPA, born on March 15, 2003. She speaks Japanese, English, and French.",
        "output": "{\"first_name\": \"Yuki\", \"last_name\": \"Tanaka\", \"description\": \"Japanese student who speaks three languages\", \"age\": 22, \"gpa\": 3.8, \"birth_date\": \"2003-03-15\", \"nationality\": \"Japan\", \"languages_spoken\": [\"Japanese\", \"English\", \"French\"]}"
      },
      {
        "input": "Born July 4, 1998, James Smith is a 27-year-old American with a GPA of 3.2. He is fluent in English and Spanish.",
        "output": "{\"first_name\": \"James\", \"last_name\": \"Smith\", \"description\": \"American who is fluent in English and Spanish\", \"age\": 27, \"gpa\": 3.2, \"birth_date\": \"1998-07-04\", \"nationality\": \"USA\", \"languages_spoken\": [\"English\", \"Spanish\"]}"
      }
    ],
    "sample_text": "Maria Garcia, born December 1, 1995, is a 30-year-old student from France. She maintains a 3.5 GPA and speaks French, English, and Spanish fluently."
  },
  {
    "name": "Job Posting",
    "template": {
      "company": "verbatim-string",
      "position": "string",
      "location": "string",
      "remote": ["yes", "no", "hybrid"],
      "years_of_experience": "string",
      "required_skills": ["string"],
      "salary": "string"
    },
    "examples": [],
    "sample_text": "TechCorp is hiring a Senior Backend Engineer in San Francisco (hybrid). Requires 5+ years of experience with Python, PostgreSQL, and AWS. Salary range: $150K-$200K."
  },
  {
    "name": "Invoice",
    "template": {
      "vendor": "verbatim-string",
      "invoice_number": "verbatim-string",
      "date": "date-time",
      "line_items": [
        {
          "description": "string",
          "quantity": "integer",
          "unit_price": "number",
          "total": "number"
        }
      ],
      "subtotal": "number",
      "tax": "number",
      "total_due": "number",
      "payment_terms": "string"
    },
    "examples": [],
    "sample_text": "Invoice #INV-2026-0042 from Acme Supplies, dated March 10, 2026. Items: 50x Widget A at $12.00 each ($600.00), 20x Widget B at $25.50 each ($510.00). Subtotal: $1,110.00. Tax (8%): $88.80. Total due: $1,198.80. Payment terms: Net 30."
  },
  {
    "name": "Product",
    "template": {
      "name": "verbatim-string",
      "brand": "verbatim-string",
      "price": "number",
      "currency": "verbatim-string",
      "category": "string",
      "specifications": [
        {
          "attribute": "string",
          "value": "string"
        }
      ]
    },
    "examples": [],
    "sample_text": "The Samsung Galaxy S26 Ultra is priced at $1,299.99 USD. This flagship smartphone features a 6.9-inch AMOLED display, 256GB storage, 12GB RAM, and a 5,500mAh battery with 65W fast charging."
  },
  {
    "name": "Scientific Paper",
    "template": {
      "title": "verbatim-string",
      "authors": ["verbatim-string"],
      "journal": "verbatim-string",
      "year": "integer",
      "abstract_summary": "string",
      "methods": ["string"],
      "key_findings": ["string"]
    },
    "examples": [],
    "sample_text": "\"Attention Is All You Need\" by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin, published in NeurIPS 2017. The paper introduces the Transformer architecture, which relies entirely on self-attention mechanisms. Key methods include multi-head attention, positional encoding, and layer normalization. The model achieved state-of-the-art results on English-to-German and English-to-French translation benchmarks."
  }
]
```

- [ ] **Step 2: Commit**

```bash
git add presets.json
git commit -m "feat: add presets.json with 5 extraction presets"
```

### Task 9: Implement preset loading and sidebar selector

**Files:**
- Modify: `streamlit_app.py` (add loader function, sidebar selector, session state wiring)
- Test: `tests/test_streamlit_app.py`

- [ ] **Step 1: Write failing tests for preset loading**

In `tests/test_streamlit_app.py`, add:

```python
# --- load_presets ---


def test_load_presets_valid_file(app, tmp_path):
    presets_data = [
        {
            "name": "Test",
            "template": {"field": "string"},
            "examples": [],
            "sample_text": "test input",
        }
    ]
    f = tmp_path / "presets.json"
    f.write_text(json.dumps(presets_data))
    result = app.load_presets(str(f))
    assert len(result) == 1
    assert result[0]["name"] == "Test"


def test_load_presets_missing_file(app, tmp_path):
    result = app.load_presets(str(tmp_path / "nonexistent.json"))
    assert len(result) == 1
    assert result[0]["name"] == "Person"


def test_load_presets_invalid_json(app, tmp_path):
    f = tmp_path / "presets.json"
    f.write_text("not json {{{")
    result = app.load_presets(str(f))
    assert len(result) == 1
    assert result[0]["name"] == "Person"


def test_load_presets_skips_invalid_entries(app, tmp_path):
    presets_data = [
        {"name": "Good", "template": {"f": "string"}, "examples": [], "sample_text": "x"},
        {"name": "Bad"},  # missing keys
        {"name": "Also Good", "template": {"g": "integer"}, "examples": [], "sample_text": "y"},
    ]
    f = tmp_path / "presets.json"
    f.write_text(json.dumps(presets_data))
    result = app.load_presets(str(f))
    assert len(result) == 2
    assert result[0]["name"] == "Good"
    assert result[1]["name"] == "Also Good"


def test_load_presets_non_list_root(app, tmp_path):
    f = tmp_path / "presets.json"
    f.write_text(json.dumps({"name": "Person"}))
    result = app.load_presets(str(f))
    assert len(result) == 1
    assert result[0]["name"] == "Person"


def test_load_presets_all_entries_invalid(app, tmp_path):
    f = tmp_path / "presets.json"
    f.write_text(json.dumps([{"name": "Bad"}, {"invalid": True}]))
    result = app.load_presets(str(f))
    assert len(result) == 1
    assert result[0]["name"] == "Person"


def test_load_presets_empty_list(app, tmp_path):
    f = tmp_path / "presets.json"
    f.write_text("[]")
    result = app.load_presets(str(f))
    assert len(result) == 1
    assert result[0]["name"] == "Person"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py -k "load_presets" -v`
Expected: FAIL — function does not exist.

- [ ] **Step 3: Implement load_presets()**

In `streamlit_app.py`, add after the constants (after `DEFAULT_EXAMPLES`):

First add `import logging` to the imports at the top of `streamlit_app.py` and add `logger = logging.getLogger(__name__)` after the imports.

Also update the `app` fixture in `tests/test_streamlit_app.py` to mock `st.cache_data` alongside the existing `st.cache_resource` mock:
```python
        patch.object(st, "cache_resource", side_effect=lambda f: f),
        patch.object(st, "cache_data", side_effect=lambda f: f),
```

```python
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
```

- [ ] **Step 4: Run preset loading tests**

Run: `uv run pytest tests/test_streamlit_app.py -k "load_presets" -v`
Expected: PASS

- [ ] **Step 5: Add preset selector to sidebar and wire session state**

In `streamlit_app.py`, in the sidebar section, add the preset selector before the template text area. Also convert the examples field to use session state `key`:

```python
with st.sidebar:
    st.header("Model")
    st.code(MODEL_ID, language=None)

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
            st.session_state["template_input"] = json.dumps(preset["template"], indent=2)
            st.session_state["examples_input"] = json.dumps(preset["examples"], indent=2)
            st.session_state["text_input"] = preset["sample_text"]
            st.rerun()
```

Update the template and examples session state initialization:

```python
    st.subheader("Template")
    if "template_input" not in st.session_state:
        st.session_state["template_input"] = DEFAULT_TEMPLATE
    # ... template text_area with key="template_input" (already exists)

    st.subheader("Examples")
    if "examples_input" not in st.session_state:
        st.session_state["examples_input"] = DEFAULT_EXAMPLES
    examples_str = st.text_area(
        "ICL examples (JSON array)", height=200, key="examples_input"
    )
```

Note: Remove the old `value=DEFAULT_EXAMPLES` from the examples text_area since we're using session state now.

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests pass.

- [ ] **Step 7: Run lint and format**

Run: `uv run ruff check . && uv run ruff format .`

- [ ] **Step 8: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add preset selector with 5 extraction presets

Sidebar selectbox loads presets from presets.json. Falls back to
Person preset if file missing. Convert examples to session state."
```

---

## Chunk 4: Documentation and cleanup

### Task 10: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md**

Update the following sections:
- Add `pyyaml` to dependencies mention
- Update `extract()` description: now accepts `max_new_tokens` kwarg, returns `(result, was_truncated)` tuple
- Add `detect_and_convert_template()` to `utils.py` section
- Add `presets.json` to architecture
- Update constants: `MAX_NEW_TOKENS` → `DEFAULT_MAX_NEW_TOKENS = 2048`
- Update test counts
- Add `presets.json` to Key Details

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for approach A features"
```

### Task 11: Final verification

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests pass.

- [ ] **Step 2: Run lint**

Run: `uv run ruff check .`
Expected: No errors.

- [ ] **Step 3: Run format check**

Run: `uv run ruff format --check .`
Expected: No files need formatting.

- [ ] **Step 4: Run type check**

Run: `uv run ty check`
Expected: No new errors introduced.
