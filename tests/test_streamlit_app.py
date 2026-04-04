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

    tab_mocks = [MagicMock(), MagicMock()]

    with (
        patch.object(st, "title"),
        patch.object(st, "text_area", return_value=""),
        patch.object(st, "button", return_value=False),
        patch.object(st, "file_uploader", return_value=None),
        patch.object(st, "tabs", return_value=tab_mocks),
        patch.object(st, "spinner"),
        patch.object(st, "info"),
        patch.object(st, "error"),
        patch.object(st, "sidebar", MagicMock()),
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


# --- _run_single_extraction ---


def test_run_single_extraction_success(app):
    with (
        patch.object(app, "extract", return_value=({"company": "Acme"}, False)),
        patch("streamlit_app.st") as mock_st,
    ):
        app._run_single_extraction("text", MagicMock(), MagicMock(), TEST_TEMPLATE)
    mock_st.json.assert_called_once_with({"company": "Acme"})
    mock_st.warning.assert_not_called()
    mock_st.error.assert_not_called()


def test_run_single_extraction_truncated(app):
    with (
        patch.object(app, "extract", return_value=({"company": "Acme"}, True)),
        patch("streamlit_app.st") as mock_st,
    ):
        app._run_single_extraction("text", MagicMock(), MagicMock(), TEST_TEMPLATE)
    mock_st.json.assert_called_once_with({"company": "Acme"})
    mock_st.warning.assert_called_once()
    assert "truncated" in mock_st.warning.call_args[0][0].lower()


def test_run_single_extraction_parse_failure(app):
    with (
        patch.object(app, "extract", return_value=(None, False)),
        patch("streamlit_app.st") as mock_st,
    ):
        app._run_single_extraction("text", MagicMock(), MagicMock(), TEST_TEMPLATE)
    mock_st.error.assert_called_once()
    assert "json" in mock_st.error.call_args[0][0].lower()


def test_run_single_extraction_valueerror(app):
    with (
        patch.object(app, "extract", side_effect=ValueError("Input too long")),
        patch("streamlit_app.st") as mock_st,
    ):
        app._run_single_extraction("text", MagicMock(), MagicMock(), TEST_TEMPLATE)
    mock_st.error.assert_called_once_with("Input too long")


def test_run_single_extraction_runtimeerror(app):
    with (
        patch.object(app, "extract", side_effect=RuntimeError("memory issue")),
        patch("streamlit_app.st") as mock_st,
    ):
        app._run_single_extraction("text", MagicMock(), MagicMock(), TEST_TEMPLATE)
    mock_st.error.assert_called_once()
    assert "runtime error" in mock_st.error.call_args[0][0].lower()


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
    assert result[0]["name"] == "Person"


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
    assert names == {"Person", "Job Posting", "Invoice", "Product", "Scientific Paper"}
    for p in result:
        assert isinstance(p["template"], dict) and p["template"]
        assert isinstance(p["sample_text"], str)


# --- _display_csv_results ---


def test_display_csv_results_normal(app):
    import pandas as pd

    df = pd.DataFrame({"text": ["a", "b"]})
    results = [{"company": "Acme"}, {"company": "Beta"}]

    with patch("streamlit_app.st") as mock_st:
        col1, col2, col3 = MagicMock(), MagicMock(), MagicMock()
        mock_st.columns.return_value = [col1, col2, col3]
        app._display_csv_results(df, results, [], {"company": ""}, "text", "test.csv")
    mock_st.write.assert_called_once_with("Preview")
    mock_st.dataframe.assert_called_once()
    mock_st.download_button.assert_called_once()
    assert mock_st.download_button.call_args[1]["file_name"] == "test_extract.csv"
    mock_st.warning.assert_not_called()
    col1.metric.assert_called_once_with("Total Rows", 2)
    col2.metric.assert_called_once_with("Extracted", 2)
    col3.metric.assert_called_once_with("Failed", 0)


def test_display_csv_results_truncated_rows(app):
    import pandas as pd

    df = pd.DataFrame({"text": ["a"]})
    results = [{"company": "Acme"}]

    with patch("streamlit_app.st") as mock_st:
        col1, col2, col3 = MagicMock(), MagicMock(), MagicMock()
        mock_st.columns.return_value = [col1, col2, col3]
        app._display_csv_results(
            df, results, [2, 4], {"company": ""}, "text", "test.csv"
        )
    warnings = [call[0][0] for call in mock_st.warning.call_args_list]
    assert any("truncated" in w.lower() for w in warnings)


def test_display_csv_results_all_none(app):
    import pandas as pd

    df = pd.DataFrame({"text": ["a", "b", "c"]})
    results = [None, None, None]

    with patch("streamlit_app.st") as mock_st:
        col1, col2, col3 = MagicMock(), MagicMock(), MagicMock()
        mock_st.columns.return_value = [col1, col2, col3]
        app._display_csv_results(df, results, [], {"company": ""}, "text", "test.csv")
    col1.metric.assert_called_once_with("Total Rows", 3)
    col2.metric.assert_called_once_with("Extracted", 0)
    col3.metric.assert_called_once_with("Failed", 3)
