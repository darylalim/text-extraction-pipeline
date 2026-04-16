import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TEST_TEMPLATE = json.dumps({"company": "", "revenue": ""})


@pytest.fixture(scope="module")
def app():
    """Import streamlit_app with streamlit UI and model loading mocked."""
    import streamlit as st

    with (
        patch.object(st, "title"),
        patch.object(st, "text_area", return_value=""),
        patch.object(st, "button", return_value=False),
        patch.object(st, "spinner"),
        patch.object(st, "info"),
        patch.object(st, "error"),
        patch.object(st, "columns", return_value=[MagicMock(), MagicMock()]),
        patch.object(st, "selectbox", return_value="Custom"),
        patch.object(st, "slider", return_value=2048),
        patch.object(st, "expander"),
        patch.object(st, "cache_resource", side_effect=lambda f: f),
        patch.object(st, "cache_data", side_effect=lambda f: f),
        patch("streamlit_app.mlx_load", return_value=(MagicMock(), MagicMock())),
    ):
        sys.modules.pop("streamlit_app", None)
        import streamlit_app

        yield streamlit_app
        sys.modules.pop("streamlit_app", None)


# --- Constants ---


def test_default_template_is_valid_json(app):
    parsed = json.loads(app.DEFAULT_TEMPLATE)
    assert isinstance(parsed, dict)
    assert len(parsed) > 0


def test_max_input_tokens_constant(app):
    assert app.MAX_INPUT_TOKENS == 4_096


def test_default_max_new_tokens_constant(app):
    assert app.DEFAULT_MAX_NEW_TOKENS == 2048


# --- validate_template ---


def test_validate_template_valid(app):
    parsed, error = app.validate_template('{"name": ""}')
    assert parsed == {"name": ""}
    assert error is None


def test_validate_template_invalid_json(app):
    parsed, error = app.validate_template("not json {{{")
    assert parsed is None
    assert error is not None


def test_validate_template_not_object(app):
    parsed, error = app.validate_template("[1, 2, 3]")
    assert parsed is None
    assert "object" in error.lower()


def test_validate_template_empty_object(app):
    parsed, error = app.validate_template("{}")
    assert parsed is None
    assert "empty" in error.lower()


# --- build_prompt ---


def test_build_prompt_format(app):
    prompt = app.build_prompt('{"name": ""}', "John is 30 years old.")
    assert "<|input|>" in prompt
    assert "### Template:" in prompt
    assert "### Text:" in prompt
    assert "<|output|>" in prompt
    assert "John is 30 years old." in prompt
    assert '"name": ""' in prompt


def test_build_prompt_pretty_prints_template(app):
    prompt = app.build_prompt('{"a":"","b":""}', "text")
    # Template should be indented with 4 spaces (json.dumps indent=4)
    assert '    "a": ""' in prompt


# --- _has_config_errors ---


def test_has_config_errors_template_error(app):
    with patch("streamlit_app.st") as mock_st:
        assert app._has_config_errors("bad json", None) is True
        mock_st.error.assert_called_once_with("Fix template: bad json")


def test_has_config_errors_no_errors(app):
    with patch("streamlit_app.st") as mock_st:
        assert app._has_config_errors(None, {"a": ""}) is False
        mock_st.error.assert_not_called()


def test_has_config_errors_no_template_parsed(app):
    with patch("streamlit_app.st") as mock_st:
        assert app._has_config_errors(None, None) is True
        mock_st.error.assert_called_once()


# --- _get_effective_template ---


@pytest.mark.parametrize("fmt", ["yaml", "pydantic", "pydantic_with_unknown"])
def test_get_effective_template_converts_non_json(app, fmt):
    with patch("streamlit_app.st") as mock_st:
        mock_st.session_state = {}
        result = app._get_effective_template('{"name": ""}', fmt, "original")
        assert result == '{"name": ""}'
        assert mock_st.session_state["template_input"] == '{"name": ""}'


@pytest.mark.parametrize("fmt", ["json", None])
def test_get_effective_template_returns_original(app, fmt):
    result = app._get_effective_template('{"name": ""}', fmt, "original")
    assert result == "original"


# --- _render_config ---


def test_render_config_returns_all_fields(app):
    """_render_config returns a tuple of (template_str, json_str, source_format, template_error, template_parsed, max_new_tokens)."""
    with patch("streamlit_app.st") as mock_st:
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.session_state = {
            "template_input": '{"name": ""}',
            "prev_preset": "Custom",
        }
        mock_st.text_area.return_value = '{"name": ""}'
        mock_st.slider.return_value = 2048
        mock_st.selectbox.return_value = "Custom"
        mock_st.expander.return_value.__enter__ = MagicMock()
        mock_st.expander.return_value.__exit__ = MagicMock()

        result = app._render_config()

    assert len(result) == 6
    (
        template_str,
        json_str,
        source_format,
        template_error,
        template_parsed,
        max_new_tokens,
    ) = result
    assert template_str == '{"name": ""}'
    assert source_format == "json"
    assert template_error is None
    assert template_parsed == {"name": ""}
    assert max_new_tokens == 2048


def test_render_config_detects_yaml(app):
    """_render_config detects YAML templates and returns converted JSON."""
    with patch("streamlit_app.st") as mock_st:
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.session_state = {
            "template_input": 'name: ""\nage: ""',
            "prev_preset": "Custom",
        }
        mock_st.text_area.return_value = 'name: ""\nage: ""'
        mock_st.slider.return_value = 2048
        mock_st.selectbox.return_value = "Custom"
        mock_st.expander.return_value.__enter__ = MagicMock()
        mock_st.expander.return_value.__exit__ = MagicMock()
        mock_st.info = MagicMock()

        result = app._render_config()

    _, json_str, source_format, template_error, template_parsed, _ = result
    assert source_format == "yaml"
    assert template_error is None
    assert template_parsed == {"name": "", "age": ""}


def test_render_config_invalid_template(app):
    """_render_config returns error for invalid template."""
    with patch("streamlit_app.st") as mock_st:
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.session_state = {"template_input": "not valid", "prev_preset": "Custom"}
        mock_st.text_area.return_value = "not valid"
        mock_st.slider.return_value = 2048
        mock_st.selectbox.return_value = "Custom"
        mock_st.expander.return_value.__enter__ = MagicMock()
        mock_st.expander.return_value.__exit__ = MagicMock()
        mock_st.error = MagicMock()

        result = app._render_config()

    _, _, source_format, template_error, template_parsed, _ = result
    assert source_format is None
    assert template_error is not None
    assert template_parsed is None


def test_render_config_preset_change_updates_session(app):
    """When preset changes, _render_config updates session state and reruns."""
    presets = [
        {"name": "Person", "template": {"first_name": ""}, "sample_text": "Maria"}
    ]

    with patch("streamlit_app.st") as mock_st:
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.session_state = {
            "template_input": '{"old": ""}',
            "prev_preset": "Custom",
        }
        mock_st.selectbox.return_value = "Person"
        mock_st.slider.return_value = 2048

        with patch.object(app, "load_presets", return_value=presets):
            try:
                app._render_config()
            except Exception:
                pass  # st.rerun() raises in test context

        assert mock_st.session_state["prev_preset"] == "Person"
        assert '"first_name"' in mock_st.session_state["template_input"]
        assert mock_st.session_state["text_input"] == "Maria"


# --- _run_extraction ---


def test_run_extraction_short_input(app):
    """Short input takes single-chunk path, displays result."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = list(range(50))

    with (
        patch.object(app, "extract", return_value=({"name": "Alice"}, False)),
        patch.object(app, "_load_icd10_codes", return_value=set()),
        patch("streamlit_app.st") as mock_st,
    ):
        app._run_extraction(
            "short text",
            mock_model,
            mock_tokenizer,
            TEST_TEMPLATE,
            {"company": "", "revenue": ""},
        )
    mock_st.json.assert_called_once()
    mock_st.info.assert_not_called()


def test_run_extraction_long_input_chunks_and_merges(app):
    """Long input triggers chunking, merging, and displays merged result."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    # First encode call (build_prompt overhead): 20 tokens
    # Second encode call (text tokens): 5000 tokens (exceeds 4096 - 20)
    mock_tokenizer.encode.side_effect = [list(range(20)), list(range(5000))]

    with (
        patch(
            "streamlit_app.chunk_text", return_value=["chunk1", "chunk2"]
        ) as mock_chunk,
        patch.object(
            app,
            "extract",
            side_effect=[
                ({"company": "Acme"}, False),
                ({"revenue": "1M"}, False),
            ],
        ),
        patch(
            "streamlit_app.merge_results",
            return_value={"company": "Acme", "revenue": "1M"},
        ) as mock_merge,
        patch.object(app, "_load_icd10_codes", return_value=set()),
        patch("streamlit_app.st") as mock_st,
    ):
        mock_st.progress.return_value = MagicMock()
        template_parsed = {"company": "", "revenue": ""}
        app._run_extraction(
            "very long text...",
            mock_model,
            mock_tokenizer,
            TEST_TEMPLATE,
            template_parsed,
        )
    mock_chunk.assert_called_once()
    mock_merge.assert_called_once()
    mock_st.json.assert_called_once()


def test_run_extraction_partial_chunk_failure(app):
    """Some chunks fail: warning shown, partial result displayed."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = [list(range(20)), list(range(5000))]

    with (
        patch("streamlit_app.chunk_text", return_value=["c1", "c2", "c3"]),
        patch.object(
            app,
            "extract",
            side_effect=[
                ({"company": "Acme"}, False),
                (None, False),
                ({"revenue": "1M"}, False),
            ],
        ),
        patch(
            "streamlit_app.merge_results",
            return_value={"company": "Acme", "revenue": "1M"},
        ),
        patch.object(app, "_load_icd10_codes", return_value=set()),
        patch("streamlit_app.st") as mock_st,
    ):
        mock_st.progress.return_value = MagicMock()
        app._run_extraction(
            "long text",
            mock_model,
            mock_tokenizer,
            TEST_TEMPLATE,
            {"company": "", "revenue": ""},
        )
    warnings = [call[0][0] for call in mock_st.warning.call_args_list]
    assert any("2 of 3" in w for w in warnings)


def test_run_extraction_all_chunks_fail(app):
    """All chunks fail: error shown, no JSON displayed."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = [list(range(20)), list(range(5000))]

    with (
        patch("streamlit_app.chunk_text", return_value=["c1", "c2"]),
        patch.object(app, "extract", return_value=(None, False)),
        patch.object(app, "_load_icd10_codes", return_value=set()),
        patch("streamlit_app.st") as mock_st,
    ):
        mock_st.progress.return_value = MagicMock()
        app._run_extraction(
            "long text",
            mock_model,
            mock_tokenizer,
            TEST_TEMPLATE,
            {"company": "", "revenue": ""},
        )
    mock_st.error.assert_called()
    mock_st.json.assert_not_called()


def test_run_extraction_icd10_validation(app):
    """ICD-10 codes in result get validated and warning shown for invalid."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = list(range(50))

    result_with_code = {"diagnosis": "HTN", "icd10_code": "FAKE.99"}

    with (
        patch.object(app, "extract", return_value=(result_with_code, False)),
        patch.object(app, "_load_icd10_codes", return_value={"I10", "E119"}),
        patch("streamlit_app.st") as mock_st,
    ):
        app._run_extraction(
            "text",
            mock_model,
            mock_tokenizer,
            TEST_TEMPLATE,
            {"diagnosis": "", "icd10_code": ""},
        )
    warnings = [call[0][0] for call in mock_st.warning.call_args_list]
    assert any("ICD-10" in w for w in warnings)


def test_run_extraction_icd10_missing_data(app):
    """Missing ICD-10 data: validation skipped warning shown once per session."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = list(range(50))

    with (
        patch.object(app, "extract", return_value=({"name": "Alice"}, False)),
        patch.object(app, "_load_icd10_codes", return_value=set()),
        patch("streamlit_app.st") as mock_st,
    ):
        mock_st.session_state = {}
        app._run_extraction(
            "text", mock_model, mock_tokenizer, TEST_TEMPLATE, {"name": ""}
        )
    warnings = [call[0][0] for call in mock_st.warning.call_args_list]
    assert any("validation skipped" in w.lower() for w in warnings)


def test_run_extraction_icd10_missing_warning_suppressed_after_first(app):
    """Missing ICD-10 data: warning shown only once, suppressed on subsequent calls."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = list(range(50))

    session_state = {}
    with (
        patch.object(app, "extract", return_value=({"name": "Alice"}, False)),
        patch.object(app, "_load_icd10_codes", return_value=set()),
        patch("streamlit_app.st") as mock_st,
    ):
        mock_st.session_state = session_state
        # First call — warning should fire and set the flag
        app._run_extraction(
            "text", mock_model, mock_tokenizer, TEST_TEMPLATE, {"name": ""}
        )
        # Second call — warning should NOT fire again
        app._run_extraction(
            "text", mock_model, mock_tokenizer, TEST_TEMPLATE, {"name": ""}
        )
    # Count of "validation skipped" warnings across both calls should be exactly 1
    skipped_count = sum(
        1
        for call in mock_st.warning.call_args_list
        if "validation skipped" in call[0][0].lower()
    )
    assert skipped_count == 1
    assert session_state.get("_icd10_warning_shown") is True


def test_run_extraction_valueerror_single_chunk(app):
    """ValueError in single-chunk path displays error."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = list(range(50))

    with (
        patch.object(app, "extract", side_effect=ValueError("Input too long")),
        patch("streamlit_app.st") as mock_st,
    ):
        app._run_extraction(
            "text", mock_model, mock_tokenizer, TEST_TEMPLATE, {"company": ""}
        )
    mock_st.error.assert_called_once_with("Input too long")


def test_run_extraction_truncated_chunks_warning(app):
    """Truncated chunks show warning with chunk numbers."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = [list(range(20)), list(range(5000))]

    with (
        patch("streamlit_app.chunk_text", return_value=["c1", "c2"]),
        patch.object(
            app,
            "extract",
            side_effect=[
                ({"company": "Acme"}, True),
                ({"revenue": "1M"}, False),
            ],
        ),
        patch(
            "streamlit_app.merge_results",
            return_value={"company": "Acme", "revenue": "1M"},
        ),
        patch.object(app, "_load_icd10_codes", return_value=set()),
        patch("streamlit_app.st") as mock_st,
    ):
        mock_st.progress.return_value = MagicMock()
        app._run_extraction(
            "long text",
            mock_model,
            mock_tokenizer,
            TEST_TEMPLATE,
            {"company": "", "revenue": ""},
        )
    warnings = [call[0][0] for call in mock_st.warning.call_args_list]
    assert any("truncated" in w.lower() for w in warnings)


def test_run_extraction_runtimeerror_single_chunk(app):
    """RuntimeError in single-chunk path displays error with 'Runtime error:' prefix."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = list(range(50))

    with (
        patch.object(app, "extract", side_effect=RuntimeError("memory issue")),
        patch("streamlit_app.st") as mock_st,
    ):
        app._run_extraction(
            "text", mock_model, mock_tokenizer, TEST_TEMPLATE, {"company": ""}
        )
    mock_st.error.assert_called_once()
    assert "runtime error" in mock_st.error.call_args[0][0].lower()


def test_run_extraction_json_parse_failure_single_chunk(app):
    """Single-chunk extraction returning None (JSON parse failure) shows error."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = list(range(50))

    with (
        patch.object(app, "extract", return_value=(None, False)),
        patch.object(app, "_load_icd10_codes", return_value=set()),
        patch("streamlit_app.st") as mock_st,
    ):
        app._run_extraction(
            "text", mock_model, mock_tokenizer, TEST_TEMPLATE, {"company": ""}
        )
    mock_st.error.assert_called()
    error_msgs = [call[0][0].lower() for call in mock_st.error.call_args_list]
    assert any("json" in msg for msg in error_msgs)
    mock_st.json.assert_not_called()


def test_run_extraction_truncated_single_chunk(app):
    """Truncated single-chunk result shows warning."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = list(range(50))

    with (
        patch.object(app, "extract", return_value=({"company": "Acme"}, True)),
        patch.object(app, "_load_icd10_codes", return_value=set()),
        patch("streamlit_app.st") as mock_st,
    ):
        app._run_extraction(
            "text", mock_model, mock_tokenizer, TEST_TEMPLATE, {"company": ""}
        )
    warnings = [call[0][0].lower() for call in mock_st.warning.call_args_list]
    assert any("truncated" in w for w in warnings)


# --- extract ---


def test_extract_returns_parsed_dict(app):
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = list(range(50))

    with patch("streamlit_app.mlx_generate", return_value='{"company": "Acme"}'):
        result, _ = app.extract("some text", mock_model, mock_tokenizer, TEST_TEMPLATE)
    assert result == {"company": "Acme"}


def test_extract_json_failure_returns_none(app):
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = list(range(50))

    with patch("streamlit_app.mlx_generate", return_value="not valid json {{{"):
        result, _ = app.extract("some text", mock_model, mock_tokenizer, TEST_TEMPLATE)
    assert result is None


def test_extract_strips_end_output_marker(app):
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = list(range(50))

    with patch(
        "streamlit_app.mlx_generate",
        return_value='{"company": "Acme"}<|end-output|>',
    ):
        result, _ = app.extract("some text", mock_model, mock_tokenizer, TEST_TEMPLATE)
    assert result == {"company": "Acme"}


def test_extract_over_token_limit_raises(app):
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = list(range(5_000))

    with pytest.raises(ValueError, match="5000.*4096"):
        app.extract("some text", mock_model, mock_tokenizer, TEST_TEMPLATE)


def test_extract_at_token_limit_succeeds(app):
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = list(range(4_096))

    with patch("streamlit_app.mlx_generate", return_value='{"company": "Acme"}'):
        result, _ = app.extract("some text", mock_model, mock_tokenizer, TEST_TEMPLATE)
    assert result == {"company": "Acme"}


def test_extract_detects_truncation(app):
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    # First call: input token count (under limit)
    # Second call: response token count (>= max_new_tokens)
    mock_tokenizer.encode.side_effect = [list(range(50)), list(range(100))]

    with patch("streamlit_app.mlx_generate", return_value='{"company": "Acme"}'):
        _, was_truncated = app.extract(
            "text", mock_model, mock_tokenizer, TEST_TEMPLATE, max_new_tokens=100
        )
    assert was_truncated is True


def test_extract_no_truncation(app):
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    # First call: input token count; Second call: response token count (< max)
    mock_tokenizer.encode.side_effect = [list(range(50)), list(range(10))]

    with patch("streamlit_app.mlx_generate", return_value='{"company": "Acme"}'):
        _, was_truncated = app.extract(
            "text", mock_model, mock_tokenizer, TEST_TEMPLATE, max_new_tokens=100
        )
    assert was_truncated is False


def test_extract_passes_correct_params_to_generate(app):
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = list(range(50))

    with patch(
        "streamlit_app.mlx_generate", return_value='{"company": "Acme"}'
    ) as mock_gen:
        app.extract(
            "text", mock_model, mock_tokenizer, TEST_TEMPLATE, max_new_tokens=512
        )
    mock_gen.assert_called_once()
    call_kwargs = mock_gen.call_args
    assert call_kwargs[1]["max_tokens"] == 512
    assert call_kwargs[1]["verbose"] is False


def test_extract_uses_custom_max_new_tokens(app):
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = list(range(50))

    with patch(
        "streamlit_app.mlx_generate", return_value='{"company": "Acme"}'
    ) as mock_gen:
        app.extract(
            "text", mock_model, mock_tokenizer, TEST_TEMPLATE, max_new_tokens=512
        )
    assert mock_gen.call_args[1]["max_tokens"] == 512


def test_extract_builds_correct_prompt(app):
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = list(range(50))

    with patch(
        "streamlit_app.mlx_generate", return_value='{"company": "Acme"}'
    ) as mock_gen:
        app.extract("test input", mock_model, mock_tokenizer, TEST_TEMPLATE)
    prompt = mock_gen.call_args[1]["prompt"]
    assert "<|input|>" in prompt
    assert "### Template:" in prompt
    assert "### Text:" in prompt
    assert "test input" in prompt
    assert "<|output|>" in prompt


# --- load_model ---


def test_load_model_uses_model_id(app):
    with patch(
        "streamlit_app.mlx_load", return_value=(MagicMock(), MagicMock())
    ) as mock_load:
        app.load_model()
    mock_load.assert_called_once_with(app.MODEL_ID)


# --- load_presets ---


def test_load_presets_valid_file(app, tmp_path):
    presets_data = [
        {
            "name": "Test",
            "template": {"field": ""},
            "sample_text": "test input",
        }
    ]
    f = tmp_path / "presets.json"
    f.write_text(json.dumps(presets_data))
    result = app.load_presets(str(f))
    assert len(result) == 1
    assert result[0]["name"] == "Test"


@pytest.mark.parametrize(
    "content",
    [
        None,
        "not json {{{",
        '{"name": "Person"}',
        '[{"name": "Bad"}, {"invalid": true}]',
        "[]",
    ],
)
def test_load_presets_fallback(app, tmp_path, content):
    if content is None:
        path = str(tmp_path / "nonexistent.json")
    else:
        f = tmp_path / "presets.json"
        f.write_text(content)
        path = str(f)
    result = app.load_presets(path)
    assert len(result) == 1
    assert result[0]["name"] == "SOAP Note"


def test_load_presets_skips_invalid_entries(app, tmp_path):
    presets_data = [
        {"name": "Good", "template": {"f": ""}, "sample_text": "x"},
        {"name": "Bad"},
        {"name": "Also Good", "template": {"g": ""}, "sample_text": "y"},
    ]
    f = tmp_path / "presets.json"
    f.write_text(json.dumps(presets_data))
    result = app.load_presets(str(f))
    assert len(result) == 2
    assert result[0]["name"] == "Good"
    assert result[1]["name"] == "Also Good"


def test_load_presets_actual_file(app):
    presets_path = str(Path(__file__).resolve().parent.parent / "presets.json")
    result = app.load_presets(presets_path)
    assert len(result) == 5
    names = {p["name"] for p in result}
    assert names == {
        "SOAP Note",
        "Discharge Summary",
        "H&P",
        "Medication Reconciliation",
        "Problem List",
    }
    for p in result:
        assert isinstance(p["template"], dict) and p["template"]
        assert isinstance(p["sample_text"], str) and p["sample_text"]
