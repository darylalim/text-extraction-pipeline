# --- Single result ---


def test_single_result_returned_as_is():
    from merging import merge_results

    result = {"name": "Alice", "age": "30"}
    template = {"name": "", "age": ""}
    assert merge_results([result], template) == result


# --- All None ---


def test_all_none_returns_none():
    from merging import merge_results

    template = {"name": "", "age": ""}
    assert merge_results([None, None, None], template) is None


def test_empty_list_returns_none():
    from merging import merge_results

    assert merge_results([], {"name": ""}) is None


# --- Scalar merge ---


def test_scalar_first_non_empty_wins():
    from merging import merge_results

    results = [
        {"name": "", "age": "30"},
        {"name": "Alice", "age": "31"},
    ]
    template = {"name": "", "age": ""}
    merged = merge_results(results, template)
    assert merged["name"] == "Alice"
    assert merged["age"] == "30"


def test_scalar_all_empty_stays_empty():
    from merging import merge_results

    results = [{"name": ""}, {"name": ""}]
    template = {"name": ""}
    merged = merge_results(results, template)
    assert merged["name"] == ""


def test_scalar_skips_none_results():
    from merging import merge_results

    results = [None, {"name": "Alice"}]
    template = {"name": ""}
    merged = merge_results(results, template)
    assert merged["name"] == "Alice"


# --- List merge ---


def test_list_union_from_chunks():
    from merging import merge_results

    results = [
        {"meds": [{"name": "aspirin"}]},
        {"meds": [{"name": "ibuprofen"}]},
    ]
    template = {"meds": []}
    merged = merge_results(results, template)
    assert len(merged["meds"]) == 2
    names = {m["name"] for m in merged["meds"]}
    assert names == {"aspirin", "ibuprofen"}


def test_list_dedupes_identical_dicts():
    from merging import merge_results

    item = {"diagnosis": "HTN", "icd10_code": "I10"}
    results = [{"problems": [item]}, {"problems": [item]}]
    template = {"problems": []}
    merged = merge_results(results, template)
    assert len(merged["problems"]) == 1


def test_list_dedupes_scalar_items():
    from merging import merge_results

    results = [{"tags": ["a", "b"]}, {"tags": ["b", "c"]}]
    template = {"tags": []}
    merged = merge_results(results, template)
    assert sorted(merged["tags"]) == ["a", "b", "c"]


def test_list_preserves_order():
    from merging import merge_results

    results = [
        {"items": [{"n": "first"}, {"n": "second"}]},
        {"items": [{"n": "third"}]},
    ]
    template = {"items": []}
    merged = merge_results(results, template)
    assert [i["n"] for i in merged["items"]] == ["first", "second", "third"]


# --- Nested dict merge ---


def test_nested_dict_merges_recursively():
    from merging import merge_results

    results = [
        {"vitals": {"bp": "120/80", "hr": ""}},
        {"vitals": {"bp": "", "hr": "72"}},
    ]
    template = {"vitals": {"bp": "", "hr": ""}}
    merged = merge_results(results, template)
    assert merged["vitals"]["bp"] == "120/80"
    assert merged["vitals"]["hr"] == "72"


# --- Partial failures ---


def test_some_none_partial_merge():
    from merging import merge_results

    results = [{"name": "Alice"}, None, {"age": "30"}]
    template = {"name": "", "age": ""}
    merged = merge_results(results, template)
    assert merged["name"] == "Alice"
    assert merged["age"] == "30"


# --- Template field never populated ---


def test_unpopulated_field_keeps_placeholder():
    from merging import merge_results

    results = [{"name": "Alice"}, {"name": "Bob"}]
    template = {"name": "", "age": ""}
    merged = merge_results(results, template)
    assert merged["age"] == ""


# --- Nested values in list items ---


def test_list_dedupes_items_with_nested_dicts():
    """Items containing nested dicts must be hashable for de-dupe (uses JSON)."""
    from merging import merge_results

    item = {"name": "alpha", "metadata": {"notes": "first"}}
    results = [{"items": [item]}, {"items": [item]}]
    template = {"items": []}
    merged = merge_results(results, template)
    assert len(merged["items"]) == 1
    assert merged["items"][0] == item


def test_list_dedupes_items_with_nested_lists():
    from merging import merge_results

    item = {"name": "beta", "tags": ["a", "b"]}
    results = [{"items": [item]}, {"items": [item]}]
    template = {"items": []}
    merged = merge_results(results, template)
    assert len(merged["items"]) == 1


# --- Clinical shape via shared fixture ---


def test_merge_clinical_chunk_results(sample_per_chunk_results):
    """Exercises real clinical template shape using shared conftest fixture."""
    from merging import merge_results

    template = {
        "chief_complaint": "",
        "hpi": "",
        "medications": [{"name": "", "dose": ""}],
        "assessment": [{"diagnosis": "", "icd10_code": ""}],
    }
    merged = merge_results(sample_per_chunk_results, template)
    assert merged["chief_complaint"] == "chest pain"
    assert merged["hpi"] == "acute onset chest pain radiating to left arm"
    med_names = {m["name"] for m in merged["medications"]}
    assert med_names == {"lisinopril", "atorvastatin"}
    assert len(merged["assessment"]) == 1
    assert merged["assessment"][0]["icd10_code"] == "I24.9"
