"""End-to-end smoke tests that load the real NuExtract-1.5-MLX model.

These tests verify integration between our extraction code and the actual
model/tokenizer — bugs in prompt format, special token handling, or template
serialization that mocked tests can't catch.

Slow: each test downloads (first run) and loads the ~4 GB model.

Run with:
    uv run pytest -m e2e
"""

import json

import pytest

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def model_and_tokenizer():
    from mlx_lm import load as mlx_load

    return mlx_load("mlx-community/numind-NuExtract-1.5-MLX-8bit")


def test_extract_simple_soap_note(model_and_tokenizer):
    """Real model extracts a simple clinical note into the SOAP template."""
    from streamlit_app import extract

    model, tokenizer = model_and_tokenizer
    template = json.dumps(
        {
            "chief_complaint": "",
            "assessment": [{"diagnosis": "", "icd10_code": ""}],
        }
    )
    text = (
        "S: Patient presents with chest pain.\nA: Acute myocardial infarction (I21.9)."
    )

    result, was_truncated = extract(
        text, model, tokenizer, template, max_new_tokens=256
    )

    assert result is not None, "Real model should return parseable JSON"
    assert not was_truncated
    assert "chief_complaint" in result
    assert "assessment" in result
    assert "chest pain" in result["chief_complaint"].lower()


def test_chunking_roundtrip_with_real_tokenizer(model_and_tokenizer):
    """Real tokenizer splits a long note correctly and preserves content."""
    from chunking import chunk_text

    _, tokenizer = model_and_tokenizer
    # Long clinical-style text that should split into multiple chunks
    text = "\n".join(
        [
            f"Line {i}: clinical observation about patient status and vitals."
            for i in range(200)
        ]
    )

    chunks = chunk_text(text, tokenizer, max_tokens=500, overlap=50)

    assert len(chunks) >= 2, "Long text should split into multiple chunks"
    # Every chunk fits within the token budget
    for chunk in chunks:
        assert len(tokenizer.encode(chunk)) <= 500
    # All line numbers from the original appear somewhere in the chunks
    combined = "\n".join(chunks)
    for i in range(200):
        assert f"Line {i}:" in combined


def test_icd10_validation_on_real_extraction(model_and_tokenizer):
    """Full flow: extract with real model, validate ICD-10 codes from result."""
    from streamlit_app import extract
    from validation import annotate_icd10, load_icd10_codes

    model, tokenizer = model_and_tokenizer
    template = json.dumps({"problems": [{"diagnosis": "", "icd10_code": ""}]})
    text = "Active problems: Essential hypertension (I10), Type 2 diabetes mellitus (E11.9)."

    result, _ = extract(text, model, tokenizer, template, max_new_tokens=256)
    assert result is not None

    codes = load_icd10_codes()
    if not codes:
        pytest.skip("ICD-10 code set not loaded; skip annotation check")

    annotated = annotate_icd10(result, codes)
    # Real codes I10 and E11.9 should validate against the bundled dev subset
    flags = []

    def _collect(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "icd10_code_valid":
                    flags.append(v)
                elif isinstance(v, (dict, list)):
                    _collect(v)
        elif isinstance(obj, list):
            for item in obj:
                _collect(item)

    _collect(annotated)
    assert flags, "annotate_icd10 should have added at least one validity flag"
    assert any(f is True for f in flags), (
        "At least one real ICD-10 code should validate"
    )
