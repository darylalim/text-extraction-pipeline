# Batch Vision Support Design

## Overview

Add batch vision processing to the NuExtract pipeline, enabling multi-image extraction through a new Image Batch tab and extending the CSV tab with optional image column support. Inspired by the NuMind demo's batch `process_all_vision_info` pattern.

## Requirements

- Multi-image upload with per-image extraction using the same template
- CSV rows with image URLs or file paths alongside text
- True batched inference (multiple inputs per `model.generate()` call)
- Shared context with optional per-image overrides
- Results displayed as individual cards + summary table with CSV download

## Architecture

### Approach: New tab + CSV extension, unified `extract_batch()`

- New "Image Batch" tab for standalone multi-image upload
- Existing CSV tab extended with optional image column selector
- `extract_batch()` becomes the single inference function
- `extract()` becomes a thin wrapper calling `extract_batch()` with a single item
- `process_all_vision_info` gains batch support (NuMind pattern)

## Component Design

### 1. `process_all_vision_info` batch support (`utils.py`)

**Input normalization:**
- Detects single vs. batch input by checking if `messages[0]` is a list
- Same detection for `examples` parameter
- Single inputs normalized to batch-of-one internally

**Batch processing:**
- Iterates over each item in the batch
- Collects images in per-item order: all images for item 1 (examples then message), then all images for item 2, etc. This matches how the processor resolves image placeholder tokens per sequence when given `text=[formatted1, formatted2, ...]`
- Raises `ValueError` if batched `examples` length doesn't match batched `messages` length

**Backward compatibility:**
- Signature unchanged: `process_all_vision_info(messages, examples=None) -> list | None`
- Single inputs produce identical output to current behavior

### 2. `extract_batch()` function (`streamlit_app.py`)

**Signature:**
```python
extract_batch(
    inputs: list[dict],   # [{"text": str|None, "image": PIL|None, "context": str|None}, ...]
    model, processor, device,
    template, examples,
    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    chunk_size=4,
) -> list[tuple[dict|None, bool]]
```

**Processing:**
- Splits `inputs` into chunks of `chunk_size`. Each chunk is processed as a true batched forward pass
- For each chunk:
  - Builds messages per item (text-only, image-only, or image+context). The `context` field is the text portion of an image message — if an item has both `image` and `context`, the message content is `[{"type": "image", "image": image}, {"type": "text", "text": context}]`. If only `text` is set (no image), the message is `{"role": "user", "content": text}`. This matches the existing `extract()` behavior
  - Calls `apply_chat_template` per item (tokenizer requirement)
  - Passes all formatted texts to processor at once: `text=[formatted1, formatted2, ...]`
  - Calls batched `process_all_vision_info` for all images in per-item order
  - Checks each item's token count against `MAX_INPUT_TOKENS`; items exceeding the limit are skipped (result set to `(None, False)`) and their indices collected for a warning
  - Single `model.generate()` call with left-padded batch. **Prerequisite:** processor must be configured with `padding_side="left"` (already set in `load_model()`)
  - Trims, decodes, and parses JSON per item independently
  - On OOM (`RuntimeError` with "out of memory" in the message): retries that chunk's items one at a time sequentially; if a single item also OOMs, it is skipped with `(None, False)`
- Returns aggregated list of `(dict|None, bool)` tuples across all chunks

**`extract()` wrapper:**
```python
def extract(input_content, model, processor, device, template, examples, image=None, max_new_tokens=...):
    batch_input = {"text": input_content, "image": image, "context": None}
    results = extract_batch([batch_input], model, processor, device, template, examples, max_new_tokens, chunk_size=1)
    return results[0]
```

**Backward compatibility:** The wrapper preserves the existing `extract()` signature and behavior exactly. All 64 existing `extract()` tests must continue to pass with no changes. Using `chunk_size=1` ensures a batch-of-one behaves identically to the current single-item code path (no left-padding differences).

### 3. Image Batch tab (new 4th tab)

**UI layout:**
- Tab order: Text | Image | Image Batch | CSV Batch (image tabs grouped together)
- Multi-image uploader: `st.file_uploader` with `accept_multiple_files=True` (png, jpg, jpeg, webp)
- Shared context: `st.text_area` for context applying to all images
- Per-image overrides: `st.expander("Per-image context")` with text input per uploaded image (keyed by filename), defaults to empty (falls back to shared context)
- Batch size slider: 1-8, default 4
- Extract button

**Results display:**
- Individual cards: image thumbnail + extracted JSON in `st.expander` (first 3 expanded, rest collapsed)
- Summary table: dataframe with columns `filename` + template fields
- Metrics row: Total / Extracted / Failed
- Download button: CSV export

**Template conversion:** Calls `_convert_template_if_needed()` before extraction, consistent with all other tabs.

**Processing:**
1. Build input dicts: `{"text": None, "image": pil_image, "context": per_image_context or shared_context}`
2. Call `extract_batch()` with `chunk_size` from slider (chunking is internal to `extract_batch()`)
3. Progress bar updates per chunk (e.g., 20 images with chunk_size=4 gives 5 progress updates)

### 4. CSV tab image column extension

**UI changes:**
- New optional selectbox after text column selector: `"Select image column (optional)"` with options `["None"] + df.columns.tolist()`
- Batch size slider appears when image column is selected

**Image loading:**
- URLs (starts with `http://` or `https://`): fetched via `fetch_image` from `qwen_vl_utils`
- File paths: loaded via `PIL.Image.open`. This is safe because the app runs locally and the user controls the CSV content; no sandboxing needed
- Invalid paths/URLs: row treated as text-only with warning

**Template conversion:** Calls `_convert_template_if_needed()` before extraction, same as current behavior.

**Processing:**
- With image column: `{"text": text_value, "image": loaded_image, "context": None}` — text becomes context
- Without image column: `{"text": text_value, "image": None, "context": None}` — identical to current behavior
- Batched via `extract_batch()` with `chunk_size` from slider (chunking internal to `extract_batch()`). Progress updates per chunk

### 5. Batching strategy and memory management

**Chunked batching:**
- `extract_batch()` accepts a `chunk_size` parameter (default 4) and handles chunking internally
- Each chunk is a true batched forward pass via `model.generate()`
- `extract_batch()` accepts an optional `progress_callback(completed, total)` for UI progress updates per chunk

**Chunk size control:**
- Image Batch tab: slider (1-8, default 4)
- CSV tab: slider appears when image column is selected; text-only stays sequential per-row (chunk_size=1)

**Prerequisites:**
- Processor must be configured with `padding_side="left"` for correct batched generation (already set in `load_model()`)

**Error handling:**
- OOM: detected by catching `RuntimeError` and checking for "out of memory" in the error message. On OOM for a chunk, fall back to sequential processing for that chunk's items. If a single item also OOMs, it is skipped with `(None, False)`
- Token limit (`ValueError`) items: skipped, marked as `(None, False)`, indices collected for a warning

## Testing Strategy

### `tests/test_utils.py` — batched `process_all_vision_info`
- Batch of 2 message lists with examples — verifies interleaved image order
- Batch messages with single (non-batched) examples — normalized correctly
- Mismatched batch lengths — raises `ValueError`
- Single input backward compatibility

### `tests/test_streamlit_app.py` — `extract_batch()`
- Batch of 2 text-only inputs — returns list of 2 results
- Batch with mixed text and image inputs — correct vision processing
- Batch with context per item — context passed correctly
- Token limit exceeded on one item — that item skipped, others succeed
- Truncation detection per item — independent `was_truncated` flags
- Chunk fallback on OOM — items in failed chunk processed sequentially
- Single-item batch — same result as old `extract()`

### `tests/test_streamlit_app.py` — `extract()` wrapper
- Verify `extract()` delegates to `extract_batch()` and returns same result format

### `tests/test_streamlit_app.py` — CSV image loading
- URL detected and fetched via `fetch_image`
- File path detected and loaded via `PIL.Image.open`
- Invalid path/URL falls back to text-only with warning

### No UI tests
- Consistent with existing approach (current tests don't test UI rendering)

## Files Modified

- `utils.py` — `process_all_vision_info` batch support
- `streamlit_app.py` — `extract_batch()`, `extract()` wrapper, Image Batch tab, CSV tab extension
- `tests/test_utils.py` — new batch vision tests
- `tests/test_streamlit_app.py` — new `extract_batch()` and wrapper tests
- `presets.json` — no changes
- `CLAUDE.md` — update architecture section after implementation
