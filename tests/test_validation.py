import json

import pytest

SAMPLE_CODES = {"E119", "I10", "J189", "J441", "E785", "F411", "J209"}


@pytest.fixture
def codes():
    return SAMPLE_CODES


# --- load_icd10_codes ---


def test_load_icd10_codes_valid_file(tmp_path, codes):
    from validation import load_icd10_codes

    f = tmp_path / "codes.json"
    f.write_text(json.dumps(sorted(codes)))
    result = load_icd10_codes(str(f))
    assert result == codes


def test_load_icd10_codes_missing_file():
    from validation import load_icd10_codes

    result = load_icd10_codes("/nonexistent/path.json")
    assert result == set()


def test_load_icd10_codes_invalid_json(tmp_path):
    from validation import load_icd10_codes

    f = tmp_path / "bad.json"
    f.write_text("not json {{{")
    result = load_icd10_codes(str(f))
    assert result == set()


def test_load_icd10_codes_not_a_list(tmp_path):
    from validation import load_icd10_codes

    f = tmp_path / "bad.json"
    f.write_text('{"key": "value"}')
    result = load_icd10_codes(str(f))
    assert result == set()


# --- annotate_icd10 ---


def test_valid_code_annotated_true(codes):
    from validation import annotate_icd10

    result = {"diagnosis": "HTN", "icd10_code": "I10"}
    annotated = annotate_icd10(result, codes)
    assert annotated["icd10_code_valid"] is True


def test_invalid_code_annotated_false(codes):
    from validation import annotate_icd10

    result = {"diagnosis": "Unknown", "icd10_code": "Z99.99"}
    annotated = annotate_icd10(result, codes)
    assert annotated["icd10_code_valid"] is False


def test_dotted_code_normalized(codes):
    from validation import annotate_icd10

    # J18.9 with dot should match J189 in code set
    result = {"diagnosis": "Pneumonia", "icd10_code": "J18.9"}
    annotated = annotate_icd10(result, codes)
    assert annotated["icd10_code_valid"] is True


def test_case_insensitive_match(codes):
    from validation import annotate_icd10

    result = {"icd10_code": "j18.9"}
    annotated = annotate_icd10(result, codes)
    assert annotated["icd10_code_valid"] is True


def test_empty_code_annotated_false(codes):
    from validation import annotate_icd10

    result = {"icd10_code": ""}
    annotated = annotate_icd10(result, codes)
    assert annotated["icd10_code_valid"] is False


def test_no_icd10_field_unchanged(codes):
    from validation import annotate_icd10

    result = {"name": "Alice", "age": "30"}
    annotated = annotate_icd10(result, codes)
    assert annotated == {"name": "Alice", "age": "30"}


def test_nested_icd10_annotated(codes):
    from validation import annotate_icd10

    result = {
        "assessment": [
            {"diagnosis": "COPD", "icd10_code": "J44.1"},
            {"diagnosis": "DM2", "icd10_code": "E11.9"},
        ]
    }
    annotated = annotate_icd10(result, codes)
    assert annotated["assessment"][0]["icd10_code_valid"] is True
    assert annotated["assessment"][1]["icd10_code_valid"] is True


def test_mixed_validity(codes):
    from validation import annotate_icd10

    result = {
        "problems": [
            {"icd10_code": "I10"},
            {"icd10_code": "FAKE.1"},
        ]
    }
    annotated = annotate_icd10(result, codes)
    assert annotated["problems"][0]["icd10_code_valid"] is True
    assert annotated["problems"][1]["icd10_code_valid"] is False


def test_original_not_mutated(codes):
    from validation import annotate_icd10

    result = {"icd10_code": "I10"}
    annotated = annotate_icd10(result, codes)
    assert "icd10_code_valid" not in result
    assert "icd10_code_valid" in annotated


# --- count_invalid_codes ---


def test_count_invalid_codes_mixed():
    from validation import count_invalid_codes

    result = {
        "problems": [
            {"icd10_code": "I10", "icd10_code_valid": True},
            {"icd10_code": "FAKE", "icd10_code_valid": False},
            {"icd10_code": "BAD", "icd10_code_valid": False},
        ]
    }
    assert count_invalid_codes(result) == 2


def test_count_invalid_codes_all_valid():
    from validation import count_invalid_codes

    result = {"icd10_code": "I10", "icd10_code_valid": True}
    assert count_invalid_codes(result) == 0


def test_count_invalid_codes_no_codes():
    from validation import count_invalid_codes

    result = {"name": "Alice"}
    assert count_invalid_codes(result) == 0
