# Approach A: Three Targeted Features

**Date:** 2026-03-14
**Goal:** Improve day-to-day reliability and usability for structured extraction workflows

## Feature 1: Configurable max_new_tokens

**Problem:** `MAX_NEW_TOKENS = 256` silently truncates nested/list-heavy extractions. The model output gets cut off mid-JSON, parse fails, user gets `None` with no explanation.

**Design:**
- Raise default to **2048** (safe for MPS, covers most real extractions)
- Add a sidebar slider: range 64–4096, step 64, placed under existing template/examples config
- Pass value as parameter to `extract()` instead of reading constant directly
- `generate_template()` keeps its own fixed cap of 256 (template schemas are small)

**Why 2048 default:** HF Space uses 4000 on A10G (24GB VRAM). MPS has less headroom. 2048 is a practical middle ground; the slider lets users push higher when needed.

## Feature 2: YAML and Pydantic template input

**Problem:** Template field only accepts JSON. Users who work with YAML configs or Pydantic models must manually convert before each extraction.

**Design:**
- Auto-detect format on Extract (or Generate Template):
  1. Try JSON parse (current behavior, fast path)
  2. Try YAML parse (`yaml.safe_load`) — if result is a non-empty dict, convert to JSON
  3. Try Pydantic extraction — regex-based: find `class ...(BaseModel):`, extract field names + type annotations, map to NuExtract types
  4. If none match, treat as natural language (existing Generate Template behavior)
- On successful conversion, update the template field with resulting JSON and show `st.info` message (e.g., "Converted from YAML to JSON template")
- New function in `utils.py`: `detect_and_convert_template(template_str)` → returns `(json_str, source_format, error)`
- Adds `pyyaml` as a dependency. No `pydantic` dependency.

**Pydantic type mapping:**

| Pydantic type | NuExtract type |
|---|---|
| `str` | `"string"` |
| `int` | `"integer"` |
| `float` | `"number"` |
| `bool` | `"boolean"` |
| `datetime` | `"date-time"` |
| `list[X]` | `["X"]` |
| `Optional[X]` | same as `X` |
| Nested model | Recursive object |

**Limitations:** Regex parser won't handle validators, computed fields, or `Literal` unions. Goal is quick conversion of typical schemas, not full Pydantic replication.

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
- Sidebar gets a **selectbox** above the template field: "Load preset..." with 5 options plus "Custom" (default after manual edit)
- Selecting a preset populates template, examples, and text tab input via session state
- Selecting "Custom" preserves current fields

**Why a JSON file:** Keeps `streamlit_app.py` clean. Easy to add/edit presets without touching app logic.

## Priority order

1. **Configurable max_new_tokens** — fixes silent data loss (highest impact, lowest effort)
2. **YAML/Pydantic templates** — removes daily friction (high impact for this user)
3. **Example presets** — quality-of-life improvement (medium impact)
