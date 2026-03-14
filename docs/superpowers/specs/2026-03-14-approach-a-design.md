# Approach A: Three Targeted Features

**Date:** 2026-03-14
**Goal:** Improve day-to-day reliability and usability for structured extraction workflows

## Feature 1: Configurable max_new_tokens

**Problem:** `MAX_NEW_TOKENS = 256` silently truncates nested/list-heavy extractions. The model output gets cut off mid-JSON, parse fails, user gets `None` with no explanation.

**Design:**
- Raise default from 256 to **2048** (safe for MPS, covers most real extractions)
- Add a sidebar slider: range 64–4096, step 64, placed under existing template/examples config
- Add `max_new_tokens` keyword argument to `extract()`:
  ```python
  def extract(input_content, model, processor, device, template, examples, image=None, max_new_tokens=2048):
  ```
- All three tabs (Text, Image, CSV) read the slider value from session state and pass it explicitly to `extract()`
- `generate_template()` keeps its own fixed cap of 256 (template schemas are small)
- The `MAX_NEW_TOKENS` constant in `utils.py` is renamed to `DEFAULT_MAX_NEW_TOKENS = 2048` and used as the slider default
- Truncation detection: `extract()` returns `(result, was_truncated)` where `was_truncated` is `True` when `trimmed_output.shape[1] == max_new_tokens`. Each calling tab checks `was_truncated` and shows `st.warning("Output may be truncated — consider increasing max tokens")`

**OOM handling:** If a high token count causes OOM on MPS/CUDA, the existing `RuntimeError` propagates. Each tab's try/except is extended to catch `RuntimeError` and show a user-friendly message suggesting a lower max token value.

**Why 2048 default:** HF Space uses 4000 on A10G (24GB VRAM). MPS has less headroom. 2048 is a practical middle ground; the slider lets users push higher when needed.

## Feature 2: YAML and Pydantic template input

**Problem:** Template field only accepts JSON. Users who work with YAML configs or Pydantic models must manually convert before each extraction.

**Design:**

### New function: `detect_and_convert_template(template_str)`

Located in `utils.py`. Returns `(json_str, source_format, error)` where:
- `json_str`: the converted JSON string, or `None` on error
- `source_format`: one of `"json"`, `"yaml"`, `"pydantic"`, or `None` on error
- `error`: error message string, or `None` on success

Detection order:
1. Try `json.loads()` — if it produces a non-empty dict, return `(json_str, "json", None)`
2. Try `yaml.safe_load()` — if it produces a non-empty dict (and input is not valid JSON), return `(json.dumps(result), "yaml", None)`. Only loads single documents; multi-document YAML (`---` separators) uses only the first document.
3. Try Pydantic regex extraction — if `class ...(BaseModel):` is found, parse fields and return `(json_str, "pydantic", None)`
4. If none match, return `(None, None, None)` — caller treats as natural language

### Control flow in the sidebar

Replace `validate_template()` with `detect_and_convert_template()` in the sidebar:
- If `source_format` is `"json"`: current behavior, no message. Template is valid.
- If `source_format` is `"yaml"` or `"pydantic"`: show `st.info("Detected {format} template — will convert to JSON on extract")`. Template is considered valid. Do **not** overwrite the field yet.
- If `source_format` is `None` and `error` is `None`: show neutral message "Template will be used as a natural language description." Show "Generate Template" button (existing behavior).
- If `error` is not `None`: show `st.error(error)`.

Conversion (overwriting the template field with JSON) happens only on button press (Extract or Generate Template), not on every rerun. This lets users keep YAML/Pydantic in the field while editing.

### Pydantic regex parser

Handles **flat models only**. Nested `BaseModel` references are explicitly excluded — if a field's type annotation is another class name, it maps to `"string"` and the `st.info` message notes "Nested models simplified to string; edit the JSON template to add structure."

**Pydantic type mapping:**

| Pydantic type | NuExtract type |
|---|---|
| `str` | `"string"` |
| `int` | `"integer"` |
| `float` | `"number"` |
| `bool` | `"boolean"` |
| `datetime` | `"date-time"` |
| `list[X]` | `["X"]` (one level only; `list[list[str]]` falls back to `["string"]`) |
| `Optional[X]` | same as `X` |
| Other class names | `"string"` (with warning) |

**Limitations:** Regex parser won't handle validators, computed fields, `Literal` unions, or nested models. Goal is quick conversion of typical schemas, not full Pydantic replication.

**Dependency:** Adds `pyyaml==6.0.2` to `pyproject.toml` (pinned, consistent with existing convention). No `pydantic` dependency.

## Feature 3: Example presets

**Problem:** Only one hardcoded starting point (person extraction). Users copy-paste and modify templates for common patterns every time.

**Design:**
- New `presets.json` file at project root with 5 presets:
  1. **Person** — name, age, description, nationality, languages (current default)
  2. **Job Posting** — company, position, skills, responsibilities, salary
  3. **Invoice** — vendor, date, line items, totals, payment terms
  4. **Product** — name, brand, price, category, specifications
  5. **Scientific Paper** — title, authors, abstract, methods, findings
- Each preset contains: `name`, `template` (JSON object), `examples` (ICL array), `sample_text` (demo input)
- Sidebar gets a **selectbox** above the template field: "Load preset..." with the 5 options plus "Custom" (default after any manual edit)
- Selecting a preset populates template, examples, and text tab input via session state
- Selecting "Custom" preserves current fields
- `DEFAULT_TEMPLATE` and `DEFAULT_EXAMPLES` constants remain as fallbacks if `presets.json` fails to load; the Person preset duplicates them

### Session state management

- The examples field is converted to use the `key`-based session state pattern (matching how template already works with `key="template_input"`)
- Text tab input uses `key="text_input"` with session state initialization from preset's `sample_text`
- Re-selecting the same preset resets fields to preset values (user edits are overwritten on explicit preset selection — this is intentional)

### File loading and error handling

- `presets.json` is loaded once at app startup via `@st.cache_data`
- If the file is missing or malformed: log a warning, fall back to a single "Person" preset built from `DEFAULT_TEMPLATE` / `DEFAULT_EXAMPLES`
- Each preset is validated at load time: must have `name` (str), `template` (dict), `examples` (list), `sample_text` (str). Invalid presets are skipped with a logged warning.

**Why a JSON file:** Keeps `streamlit_app.py` clean. Easy to add/edit presets without touching app logic.

## Priority order

1. **Configurable max_new_tokens** — fixes silent data loss (highest impact, lowest effort)
2. **YAML/Pydantic templates** — removes daily friction (high impact for this user)
3. **Example presets** — quality-of-life improvement (medium impact)

## Test plan

- **Feature 1:** Tests for `extract()` with explicit `max_new_tokens` parameter; test default value is 2048; test `generate_template` still uses 256; test truncation detection (output length == max_new_tokens)
- **Feature 2:** Tests for `detect_and_convert_template()` with JSON, YAML, Pydantic, natural language, empty input, invalid YAML, ambiguous formats; test source_format values; test Pydantic type mapping for each supported type; test nested model fallback to "string"
- **Feature 3:** Tests for preset loading from valid file; malformed file fallback; preset selection populating session state; preset validation (missing keys skipped)

## Documentation

CLAUDE.md will be updated to reflect the new `extract()` signature, new `detect_and_convert_template()` function, `presets.json` file, `pyyaml` dependency, and updated `DEFAULT_MAX_NEW_TOKENS` constant.
