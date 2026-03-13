# Token Limit, Auto Template Generation, ICL Example Images — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add token limit enforcement, natural-language-to-JSON template generation, and image support in ICL examples to the extraction pipeline.

**Architecture:** A new `utils.py` module holds two functions (`generate_template`, `process_all_vision_info`) imported by `streamlit_app.py`. The main app gains a `MAX_INPUT_TOKENS` constant, token limit check in `extract()`, session-state-driven "Generate Template" button in the sidebar, and updated `extract()` to pass example images via `process_all_vision_info`. Tests split across `tests/test_utils.py` (new) and `tests/test_streamlit_app.py` (extended).

**Tech Stack:** Python 3.12, Streamlit, PyTorch, Hugging Face Transformers, qwen-vl-utils, pytest

**Spec:** `docs/superpowers/specs/2026-03-13-three-features-design.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `utils.py` | `generate_template()`, `process_all_vision_info()` |
| Create | `tests/test_utils.py` | Tests for both utils functions |
| Modify | `streamlit_app.py` | Token limit, sidebar Generate button, extract() image-example support |
| Modify | `tests/test_streamlit_app.py` | Tests for token limit, parse_examples image format, extract with image examples |
| Modify | `CLAUDE.md` | Document new module, constant, and behaviors |

---

## Chunk 1: Token Limit Enforcement

### Task 1: Add token limit constant and check in `extract()`

**Files:**
- Modify: `streamlit_app.py:19-20` (add constant)
- Modify: `streamlit_app.py:147-149` (add check between tokenization and generation)
- Test: `tests/test_streamlit_app.py`

- [ ] **Step 1: Write failing tests for token limit**

Add these tests after the existing `test_extract_text_only_passes_images_none` test (line 295) in `tests/test_streamlit_app.py`:

```python
# --- Token limit ---


def test_extract_under_token_limit_succeeds(app):
    output = json.dumps({"company": "Acme"})
    model, processor = _make_mocks(output)
    result = app.extract(
        "some text", model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES
    )
    assert result == {"company": "Acme"}


def test_extract_over_token_limit_raises(app):
    output = json.dumps({"company": "Acme"})
    model, processor = _make_mocks(output)
    # Override input_ids to exceed MAX_INPUT_TOKENS
    big_input_ids = torch.ones(1, 10_001, dtype=torch.long)
    proc_result = MagicMock()
    proc_result.to.return_value = {
        "input_ids": big_input_ids,
        "attention_mask": torch.ones_like(big_input_ids),
    }
    processor.return_value = proc_result
    with pytest.raises(ValueError, match="10001.*10000"):
        app.extract(
            "some text", model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES
        )


def test_extract_at_token_limit_succeeds(app):
    output = json.dumps({"company": "Acme"})
    model, processor = _make_mocks(output)
    # Override input_ids to exactly MAX_INPUT_TOKENS
    exact_input_ids = torch.ones(1, 10_000, dtype=torch.long)
    proc_result = MagicMock()
    proc_result.to.return_value = {
        "input_ids": exact_input_ids,
        "attention_mask": torch.ones_like(exact_input_ids),
    }
    processor.return_value = proc_result
    model.generate.return_value = torch.cat(
        [exact_input_ids, torch.tensor([[10, 20, 30]])], dim=1
    )
    result = app.extract(
        "some text", model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES
    )
    assert result == {"company": "Acme"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py::test_extract_over_token_limit_raises -v`
Expected: FAIL — no `ValueError` raised (extract succeeds without limit check)

- [ ] **Step 3: Add `MAX_INPUT_TOKENS` constant and token limit check**

In `streamlit_app.py`, add the constant after `MAX_NEW_TOKENS` (line 20):

```python
MAX_INPUT_TOKENS = 10_000
```

In `streamlit_app.py`, add the token limit check in `extract()` between line 147 (`input_len = inputs["input_ids"].shape[1]`) and line 149 (`with torch.inference_mode():`):

```python
    if input_len > MAX_INPUT_TOKENS:
        raise ValueError(
            f"Input too long: {input_len} tokens (limit: {MAX_INPUT_TOKENS})."
        )
```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All 35 tests PASS (32 existing + 3 new)

- [ ] **Step 5: Lint and format**

Run: `uv run ruff check . && uv run ruff format .`
Expected: Clean

- [ ] **Step 6: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add token limit enforcement in extract()"
```

### Task 2: Update UI tabs to handle `ValueError` from token limit

**Files:**
- Modify: `streamlit_app.py:219-242` (text tab)
- Modify: `streamlit_app.py:244-274` (image tab)
- Modify: `streamlit_app.py:276-346` (csv tab)

- [ ] **Step 1: Update text tab to catch `ValueError`**

In `streamlit_app.py`, replace the text tab extraction block (the `with st.spinner("Extracting..."):` block and the result handling below it, lines ~228-242) with:

```python
                with st.spinner("Extracting..."):
                    try:
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
                    except ValueError as e:
                        st.error(str(e))
```

- [ ] **Step 2: Update image tab to catch `ValueError`**

Replace the image tab extraction block (lines ~259-274) with the same pattern:

```python
                with st.spinner("Extracting..."):
                    try:
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
                    except ValueError as e:
                        st.error(str(e))
```

- [ ] **Step 3: Update CSV tab to catch `ValueError` per row**

Replace the CSV tab extraction loop (lines ~298-311) with:

```python
                    with st.spinner("Extracting..."):
                        skipped_rows = []
                        for i, text in enumerate(df[selected_column].astype(str)):
                            try:
                                result = extract(
                                    text,
                                    model,
                                    processor,
                                    device,
                                    template_str,
                                    examples_parsed,
                                )
                            except ValueError:
                                result = None
                                skipped_rows.append(i + 1)
                            results.append(result)
                            progress_bar.progress(
                                (i + 1) / len(df),
                                text=f"Processing row {i + 1} of {len(df)}",
                            )
```

After `progress_bar.progress(1.0, text="Done.")` (line ~313), add:

```python
                    if skipped_rows:
                        st.warning(
                            f"Rows skipped (input too long): {skipped_rows}"
                        )
```

- [ ] **Step 4: Run all tests to verify nothing broke**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All 35 tests PASS

- [ ] **Step 5: Lint and format**

Run: `uv run ruff check . && uv run ruff format .`
Expected: Clean

- [ ] **Step 6: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: handle token limit errors in all UI tabs"
```

---

## Chunk 2: Auto Template Generation

### Task 3: Create `utils.py` with `generate_template()`

**Files:**
- Create: `utils.py`
- Create: `tests/test_utils.py`

- [ ] **Step 1: Write failing tests for `generate_template`**

Create `tests/test_utils.py`:

```python
import json
from unittest.mock import MagicMock

import torch


def _make_mocks(decode_output):
    """Create mock model and processor that produce the given decode output."""
    processor = MagicMock()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    inputs_dict = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }
    proc_result = MagicMock()
    proc_result.to.return_value = inputs_dict
    processor.return_value = proc_result
    processor.tokenizer.apply_chat_template.return_value = "formatted prompt"
    processor.batch_decode.return_value = [decode_output]

    model = MagicMock()
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 10, 20, 30]])

    return model, processor


# --- generate_template ---


def test_generate_template_returns_dict_on_valid_output():
    from utils import generate_template

    output = json.dumps({"name": "string", "age": "integer"})
    model, processor = _make_mocks(output)
    result, error = generate_template("extract name and age", model, processor, "cpu")
    assert result == {"name": "string", "age": "integer"}
    assert error is None


def test_generate_template_passes_template_none():
    from utils import generate_template

    output = json.dumps({"name": "string"})
    model, processor = _make_mocks(output)
    generate_template("extract name", model, processor, "cpu")

    call_args = processor.tokenizer.apply_chat_template.call_args
    assert call_args[1]["template"] is None


def test_generate_template_passes_description_as_message():
    from utils import generate_template

    output = json.dumps({"name": "string"})
    model, processor = _make_mocks(output)
    generate_template("extract the person's name", model, processor, "cpu")

    call_args = processor.tokenizer.apply_chat_template.call_args
    messages = call_args[0][0]
    assert messages == [{"role": "user", "content": "extract the person's name"}]


def test_generate_template_invalid_output_returns_error():
    from utils import generate_template

    model, processor = _make_mocks("not valid json {{{")
    result, error = generate_template("extract stuff", model, processor, "cpu")
    assert result is None
    assert error is not None


def test_generate_template_model_error_propagates():
    import pytest
    from utils import generate_template

    model, processor = _make_mocks(json.dumps({"name": "string"}))
    model.generate.side_effect = RuntimeError("out of memory")
    with pytest.raises(RuntimeError, match="out of memory"):
        generate_template("extract name", model, processor, "cpu")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_utils.py::test_generate_template_returns_dict_on_valid_output -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'utils'`

- [ ] **Step 3: Create `utils.py` with `generate_template`**

Create `utils.py` in the project root:

```python
import json

import torch


def generate_template(description, model, processor, device):
    """Generate a JSON extraction template from a natural language description.

    Uses the NuExtract model's native template generation mode by passing
    template=None to apply_chat_template.

    Returns (dict, None) on success, (None, error_message) on failure.
    """
    messages = [{"role": "user", "content": description}]
    formatted = processor.tokenizer.apply_chat_template(
        messages,
        template=None,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[formatted], images=None, padding=True, return_tensors="pt"
    ).to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=256,
        )

    trimmed = output[:, input_len:]
    decoded = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    try:
        parsed = json.loads(decoded[0])
        return parsed, None
    except (json.JSONDecodeError, IndexError) as e:
        return None, f"Could not parse generated template: {e}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_utils.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Lint and format**

Run: `uv run ruff check . && uv run ruff format .`
Expected: Clean

- [ ] **Step 6: Commit**

```bash
git add utils.py tests/test_utils.py
git commit -m "feat: add generate_template() in utils.py"
```

### Task 4: Update sidebar UI for dual-mode template field

**Files:**
- Modify: `streamlit_app.py:14` (add import)
- Modify: `streamlit_app.py:182-210` (sidebar section)

- [ ] **Step 1: Add import of `generate_template`**

In `streamlit_app.py`, add after the existing imports (after line 14):

```python
from utils import generate_template
```

- [ ] **Step 2: Update sidebar template section to use session state and Generate button**

Replace the sidebar template section (lines ~185-189) with:

```python
    st.subheader("Template")
    if "template_input" not in st.session_state:
        st.session_state["template_input"] = DEFAULT_TEMPLATE
    template_str = st.text_area(
        "JSON template or description",
        height=100,
        key="template_input",
    )
    template_parsed, template_error = validate_template(template_str)
    if template_error:
        st.error(template_error)
        if st.button("Generate Template", key="generate_template"):
            with st.spinner("Generating template..."):
                generated, gen_error = generate_template(
                    template_str, model, processor, device
                )
            if generated is not None:
                st.session_state["template_input"] = json.dumps(
                    generated, indent=2
                )
                st.rerun()
            else:
                st.error(f"Generation failed: {gen_error}")
```

Note: The `model`, `processor`, and `device` variables are used before they are defined in the current code flow (they're defined at lines 212-215). The sidebar code runs at import time during Streamlit's execution. We need to move the model loading above the sidebar, or restructure.

Looking at the current code: `device = get_device()` is at line 212, and `model, processor = load_model(device)` is at line 214-215. The sidebar block starts at line 182. We need to move model loading before the sidebar.

**Restructure:** Move lines 212-215 (device detection and model loading) to before the sidebar block (before line 182). The model loading spinner will appear in the main area before the sidebar renders.

- [ ] **Step 3: Move model loading before sidebar**

Move these lines from their current position (~212-215) to just before the sidebar block (~before line 182), right after `_has_config_errors`:

```python
device = get_device()

with st.spinner(f"Loading {MODEL_ID} on {device.upper()}..."):
    model, processor = load_model(device)
```

Then the sidebar section follows, and the `st.tabs(...)` line follows after the sidebar.

Remove the original device/model lines from their old position.

- [ ] **Step 4: Run all tests to verify nothing broke**

Run: `uv run pytest -v`
Expected: All tests PASS (the `app` fixture mocks model loading, so the reorder doesn't affect tests)

- [ ] **Step 5: Lint and format**

Run: `uv run ruff check . && uv run ruff format .`
Expected: Clean

- [ ] **Step 6: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add Generate Template button for natural language descriptions"
```

---

## Chunk 3: ICL Example Images

### Task 5: Add `process_all_vision_info()` to `utils.py`

**Files:**
- Modify: `utils.py` (add function)
- Modify: `tests/test_utils.py` (add tests)

- [ ] **Step 1: Write failing tests for `process_all_vision_info`**

Append to `tests/test_utils.py`:

```python
from unittest.mock import patch


# --- process_all_vision_info ---


def test_process_all_vision_info_example_images_and_message_image():
    from utils import process_all_vision_info

    fake_example_img = MagicMock(name="example_img")
    fake_message_img = MagicMock(name="message_img")

    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": "https://example.com/input.png"}],
        }
    ]
    examples = [
        {
            "input": {"type": "image", "image": "https://example.com/ex.png"},
            "output": '{"name": "John"}',
        }
    ]

    with (
        patch(
            "utils.process_vision_info",
            return_value=([fake_message_img], None),
        ),
        patch("utils.fetch_image", return_value=fake_example_img),
    ):
        result = process_all_vision_info(messages, examples)

    assert result == [fake_example_img, fake_message_img]


def test_process_all_vision_info_no_images_returns_none():
    from utils import process_all_vision_info

    messages = [{"role": "user", "content": "just text"}]

    with patch(
        "utils.process_vision_info",
        return_value=(None, None),
    ):
        result = process_all_vision_info(messages, None)

    assert result is None


def test_process_all_vision_info_text_examples_ignored():
    from utils import process_all_vision_info

    fake_message_img = MagicMock(name="message_img")
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": "https://example.com/input.png"}],
        }
    ]
    examples = [
        {"input": "just text", "output": '{"name": "John"}'},
    ]

    with patch(
        "utils.process_vision_info",
        return_value=([fake_message_img], None),
    ):
        result = process_all_vision_info(messages, examples)

    assert result == [fake_message_img]


def test_process_all_vision_info_mixed_examples():
    from utils import process_all_vision_info

    fake_example_img = MagicMock(name="example_img")
    fake_message_img = MagicMock(name="message_img")

    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": "https://example.com/input.png"}],
        }
    ]
    examples = [
        {"input": "just text", "output": '{"name": "Alice"}'},
        {
            "input": {"type": "image", "image": "https://example.com/ex.png"},
            "output": '{"name": "Bob"}',
        },
    ]

    with (
        patch(
            "utils.process_vision_info",
            return_value=([fake_message_img], None),
        ),
        patch("utils.fetch_image", return_value=fake_example_img),
    ):
        result = process_all_vision_info(messages, examples)

    assert result == [fake_example_img, fake_message_img]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_utils.py::test_process_all_vision_info_example_images_and_message_image -v`
Expected: FAIL — `ImportError: cannot import name 'process_all_vision_info'`

- [ ] **Step 3: Add `process_all_vision_info` to `utils.py`**

Add to `utils.py` after the existing `generate_template` function:

```python
from qwen_vl_utils import fetch_image, process_vision_info


def process_all_vision_info(messages, examples=None):
    """Extract images from both ICL examples and user messages.

    Returns a flat list of images in correct order (example images first,
    then message images), or None if no images found.
    """
    all_images = []

    if examples:
        for ex in examples:
            inp = ex.get("input")
            if isinstance(inp, dict) and inp.get("type") == "image":
                all_images.append(fetch_image(inp))

    message_images = process_vision_info(messages)[0] or []
    all_images.extend(message_images)

    return all_images if all_images else None
```

Move the `from qwen_vl_utils` import to the top of the file (after the existing imports).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_utils.py -v`
Expected: All 9 tests PASS (5 generate_template + 4 process_all_vision_info)

- [ ] **Step 5: Lint and format**

Run: `uv run ruff check . && uv run ruff format .`
Expected: Clean

- [ ] **Step 6: Commit**

```bash
git add utils.py tests/test_utils.py
git commit -m "feat: add process_all_vision_info() in utils.py"
```

### Task 6: Update `parse_examples()` for image input format

**Files:**
- Modify: `streamlit_app.py:111-123` (`parse_examples` function)
- Modify: `tests/test_streamlit_app.py`

- [ ] **Step 1: Write failing tests for image input format**

Add after the existing `test_parse_examples_missing_keys` test in `tests/test_streamlit_app.py`:

```python
def test_parse_examples_image_input_accepted(app):
    examples_str = json.dumps([
        {
            "input": {"type": "image", "image": "https://example.com/img.png"},
            "output": '{"name": "John"}',
        }
    ])
    parsed, error = app.parse_examples(examples_str)
    assert parsed is not None
    assert error is None


def test_parse_examples_image_input_missing_image_key(app):
    examples_str = json.dumps([
        {
            "input": {"type": "image"},
            "output": '{"name": "John"}',
        }
    ])
    parsed, error = app.parse_examples(examples_str)
    assert parsed is None
    assert error is not None


def test_parse_examples_image_input_non_http_url_rejected(app):
    examples_str = json.dumps([
        {
            "input": {"type": "image", "image": "file:///etc/passwd"},
            "output": '{"name": "John"}',
        }
    ])
    parsed, error = app.parse_examples(examples_str)
    assert parsed is None
    assert error is not None


def test_parse_examples_mixed_text_and_image_accepted(app):
    examples_str = json.dumps([
        {"input": "text example", "output": '{"name": "Alice"}'},
        {
            "input": {"type": "image", "image": "https://example.com/img.png"},
            "output": '{"name": "Bob"}',
        },
    ])
    parsed, error = app.parse_examples(examples_str)
    assert parsed is not None
    assert len(parsed) == 2
    assert error is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py::test_parse_examples_image_input_missing_image_key -v`
Expected: FAIL — current `parse_examples` only checks for "input" and "output" keys, doesn't validate image dict format

- [ ] **Step 3: Update `parse_examples` to validate image input format**

Replace the `parse_examples` function in `streamlit_app.py` (lines 111-123):

```python
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
```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All 39 tests PASS (35 previous + 4 new)

- [ ] **Step 5: Lint and format**

Run: `uv run ruff check . && uv run ruff format .`
Expected: Clean

- [ ] **Step 6: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: support image input format in parse_examples()"
```

### Task 7: Update `extract()` to use `process_all_vision_info` for image examples

**Files:**
- Modify: `streamlit_app.py:14` (add import)
- Modify: `streamlit_app.py:126-135` (extract function, image branch)
- Modify: `tests/test_streamlit_app.py`

- [ ] **Step 1: Write failing test for extract with image examples**

Add after the token limit tests in `tests/test_streamlit_app.py`:

```python
# --- extract with image examples ---


def test_extract_image_with_image_examples_calls_process_all(app):
    output = json.dumps({"company": "Acme"})
    model, processor = _make_mocks(output)
    fake_image = MagicMock()
    fake_all_images = [MagicMock(), MagicMock()]
    image_examples = [
        {
            "input": {"type": "image", "image": "https://example.com/ex.png"},
            "output": '{"company": "Test"}',
        }
    ]

    with patch(
        "streamlit_app.process_all_vision_info",
        return_value=fake_all_images,
    ) as mock_pavi:
        app.extract(
            "context",
            model,
            processor,
            "cpu",
            TEST_TEMPLATE,
            image_examples,
            image=fake_image,
        )

        mock_pavi.assert_called_once()
        call_args = mock_pavi.call_args
        assert call_args[0][1] == image_examples

        proc_call = processor.call_args
        assert proc_call[1]["images"] is fake_all_images
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_streamlit_app.py::test_extract_image_with_image_examples_calls_process_all -v`
Expected: FAIL — `process_all_vision_info` not imported/called in `streamlit_app`

- [ ] **Step 3: Update `extract()` to use `process_all_vision_info`**

In `streamlit_app.py`, add to the imports (alongside the existing `generate_template` import):

```python
from utils import generate_template, process_all_vision_info
```

Replace the image branch in `extract()` (lines ~127-132):

```python
    if image is not None:
        content = [{"type": "image", "image": image}]
        if input_content:
            content.append({"type": "text", "text": input_content})
        messages = [{"role": "user", "content": content}]
        image_inputs = process_all_vision_info(messages, examples)
    else:
```

This replaces the old `image_inputs, _ = process_vision_info(messages)` call. The `process_all_vision_info` function handles extracting images from both examples and the user message.

- [ ] **Step 4: Run all tests to verify they pass**

Run: `uv run pytest -v`
Expected: All tests PASS

Note: The existing `test_extract_image_builds_vision_message` test patches `streamlit_app.process_vision_info` which is no longer called directly. This test needs updating — it should now patch `streamlit_app.process_all_vision_info` instead. Update the test:

Replace the patch target in `test_extract_image_builds_vision_message`:

```python
def test_extract_image_builds_vision_message(app):
    output = json.dumps({"company": "Acme"})
    model, processor = _make_mocks(output)
    fake_image = MagicMock()
    fake_image_inputs = [MagicMock()]

    with patch(
        "streamlit_app.process_all_vision_info",
        return_value=fake_image_inputs,
    ) as mock_pavi:
        app.extract(
            "context",
            model,
            processor,
            "cpu",
            TEST_TEMPLATE,
            TEST_EXAMPLES,
            image=fake_image,
        )

        mock_pavi.assert_called_once()
        messages = mock_pavi.call_args[0][0]
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert content[0]["type"] == "image"
        assert content[0]["image"] is fake_image
        assert content[1] == {"type": "text", "text": "context"}

        proc_call = processor.call_args
        assert proc_call[1]["images"] is fake_image_inputs
```

- [ ] **Step 5: Run all tests again**

Run: `uv run pytest -v`
Expected: All tests PASS

- [ ] **Step 6: Lint and format**

Run: `uv run ruff check . && uv run ruff format .`
Expected: Clean

- [ ] **Step 7: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: support image examples in extract() via process_all_vision_info"
```

---

## Chunk 4: Documentation and Cleanup

### Task 8: Remove unused `process_vision_info` import

**Files:**
- Modify: `streamlit_app.py:13`

- [ ] **Step 1: Remove the now-unused direct import**

In `streamlit_app.py`, remove:

```python
from qwen_vl_utils import process_vision_info
```

This import is no longer used directly — `process_all_vision_info` in `utils.py` handles it internally.

- [ ] **Step 2: Run all tests**

Run: `uv run pytest -v`
Expected: All tests PASS

- [ ] **Step 3: Lint and format**

Run: `uv run ruff check . && uv run ruff format .`
Expected: Clean

- [ ] **Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "refactor: remove unused process_vision_info import"
```

### Task 9: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md**

Replace the Architecture section with:

```markdown
## Architecture

Main app in `streamlit_app.py` (~380 lines), utilities in `utils.py` (~50 lines):

### `streamlit_app.py`

- **Constants** — `MODEL_ID`, `MAX_NEW_TOKENS`, `MAX_INPUT_TOKENS`, `DEFAULT_TEMPLATE`, `DEFAULT_EXAMPLES`
- **`get_device()`** — Auto-detects compute: MPS → CUDA → CPU
- **`load_model(device)`** — Loads model and processor in BF16; cached via `@st.cache_resource`
- **`validate_template(template_str)`** — Validates JSON string is a non-empty dict; returns `(parsed, error)`
- **`parse_examples(examples_str)`** — Validates JSON array of `{"input", "output"}` objects; input can be a string (text) or dict with `{"type": "image", "image": url}` for image examples; returns `(list, error)`
- **`extract(..., image=None)`** — Runs inference under `torch.inference_mode()` with template and ICL examples; enforces `MAX_INPUT_TOKENS` limit (raises `ValueError`); uses `process_all_vision_info` for image inputs; returns parsed JSON dict or `None`
- **`_has_config_errors(template_error, examples_error)`** — Shows first config error via `st.error` and returns `True`, or returns `False` if none; used by all three tabs
- **Streamlit UI** — Sidebar with template/examples config and "Generate Template" button (appears when template is not valid JSON); three tabs: Text, Image, CSV Batch; all tabs catch `ValueError` from token limit
- **Warning suppression** — Filters known MPS padding warnings

### `utils.py`

- **`generate_template(description, model, processor, device)`** — Generates a JSON extraction template from a natural language description using NuExtract's native `template=None` mode; returns `(dict, None)` or `(None, error)`
- **`process_all_vision_info(messages, examples=None)`** — Extracts images from both ICL examples and user messages; returns flat list in correct order (example images first) or `None`

Tests in `tests/test_streamlit_app.py` and `tests/test_utils.py`.
```

Replace the Key Details section with:

```markdown
## Key Details

- Max 256 new tokens per extraction (`MAX_NEW_TOKENS`) since output is JSON
- Max 10,000 input tokens (`MAX_INPUT_TOKENS`) to prevent OOM; raises `ValueError` if exceeded
- `DEFAULT_TEMPLATE` and `DEFAULT_EXAMPLES` provide a person-extraction starting point
- Template field accepts JSON or natural language; "Generate Template" button converts descriptions to JSON
- Image examples use URL references (`http://` or `https://` only) and are processed on the Image tab
- Image support requires `qwen-vl-utils` and `torchvision`
- `HF_TOKEN` env var enables authenticated Hub access (optional; model is public)
- Sample test data: `tests/data/csv/sample_persons.csv` (30 rows)
- Dependencies in `pyproject.toml` pinned to specific versions
```

- [ ] **Step 2: Lint and format**

Run: `uv run ruff check . && uv run ruff format .`
Expected: Clean

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for token limit, template generation, and image examples"
```
