import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="module")
def app():
    """Import streamlit_app with streamlit UI and model loading mocked."""
    import streamlit as st

    with (
        patch.object(st, "title"),
        patch.object(st, "write"),
        patch.object(st, "file_uploader", return_value=None),
        patch.object(st, "spinner"),
        patch.object(st, "info"),
        patch.object(st, "cache_resource", side_effect=lambda f: f),
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=MagicMock(),
        ),
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=MagicMock(),
        ),
    ):
        sys.modules.pop("streamlit_app", None)
        import streamlit_app

        yield streamlit_app
        sys.modules.pop("streamlit_app", None)


def _make_mocks(decode_output):
    """Create mock model and tokenizer that produce the given decode output."""
    tokenizer = MagicMock()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    tokens_result = MagicMock()
    tokens_result.to.return_value = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }
    tokenizer.return_value = tokens_result
    tokenizer.eos_token_id = 0
    tokenizer.decode.return_value = decode_output
    tokenizer.apply_chat_template.return_value = "formatted prompt"

    model = MagicMock()
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 10, 20, 30]])

    return model, tokenizer


# --- Constants ---


def test_model_id(app):
    assert app.MODEL_ID == "ibm-granite/granite-4.0-h-tiny"


def test_field_name(app):
    assert app.FIELD == "Line of Credit Facility Maximum Borrowing Capacity"


def test_max_new_tokens(app):
    assert app.MAX_NEW_TOKENS == 50


# --- get_device ---


def test_get_device_prefers_mps(app):
    with patch("torch.backends.mps.is_available", return_value=True):
        assert app.get_device() == "mps"


def test_get_device_falls_back_to_cuda(app):
    with (
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.cuda.is_available", return_value=True),
    ):
        assert app.get_device() == "cuda"


def test_get_device_falls_back_to_cpu(app):
    with (
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.cuda.is_available", return_value=False),
    ):
        assert app.get_device() == "cpu"


# --- extract_text ---


def test_extract_clean_answer(app):
    model, tokenizer = _make_mocks("$500M")
    assert app.extract_text("some text", model, tokenizer, "cpu") == "$500M"


def test_extract_strips_assistant_tag(app):
    model, tokenizer = _make_mocks("$500M assistant")
    assert app.extract_text("some text", model, tokenizer, "cpu") == "$500M"


def test_extract_strips_user_tag(app):
    model, tokenizer = _make_mocks("user $500M")
    assert app.extract_text("some text", model, tokenizer, "cpu") == "$500M"


def test_extract_strips_system_tag(app):
    model, tokenizer = _make_mocks("system $500M")
    assert app.extract_text("some text", model, tokenizer, "cpu") == "$500M"


def test_extract_strips_multiple_tags(app):
    model, tokenizer = _make_mocks("assistant $500M system")
    assert app.extract_text("some text", model, tokenizer, "cpu") == "$500M"


def test_extract_only_tags_returns_na(app):
    model, tokenizer = _make_mocks("assistant")
    assert app.extract_text("some text", model, tokenizer, "cpu") == "N/A"


def test_extract_takes_first_line(app):
    model, tokenizer = _make_mocks("$500M\nextra text on second line")
    assert app.extract_text("some text", model, tokenizer, "cpu") == "$500M"


def test_extract_empty_returns_na(app):
    model, tokenizer = _make_mocks("")
    assert app.extract_text("some text", model, tokenizer, "cpu") == "N/A"


def test_extract_whitespace_returns_na(app):
    model, tokenizer = _make_mocks("   \n  ")
    assert app.extract_text("some text", model, tokenizer, "cpu") == "N/A"


def test_extract_too_long_returns_na(app):
    model, tokenizer = _make_mocks("x" * 51)
    assert app.extract_text("some text", model, tokenizer, "cpu") == "N/A"


def test_extract_max_length_accepted(app):
    answer = "x" * 50
    model, tokenizer = _make_mocks(answer)
    assert app.extract_text("some text", model, tokenizer, "cpu") == answer


def test_extract_decodes_only_new_tokens(app):
    model, tokenizer = _make_mocks("$500M")
    app.extract_text("some text", model, tokenizer, "cpu")

    decode_call = tokenizer.decode.call_args
    decoded_tensor = decode_call[0][0]
    output_tensor = model.generate.return_value[0]
    input_len = 5  # _make_mocks creates input_ids with shape (1, 5)
    assert torch.equal(decoded_tensor, output_tensor[input_len:])
    assert decode_call[1]["skip_special_tokens"] is True


def test_extract_generate_error_propagates(app):
    model, tokenizer = _make_mocks("$500M")
    model.generate.side_effect = RuntimeError("out of memory")
    with pytest.raises(RuntimeError, match="out of memory"):
        app.extract_text("some text", model, tokenizer, "cpu")


def test_extract_builds_prompt_with_field_and_input(app):
    model, tokenizer = _make_mocks("$500M")
    app.extract_text("test sentence here", model, tokenizer, "cpu")

    prompt = tokenizer.apply_chat_template.call_args[0][0][0]["content"]
    assert app.FIELD in prompt
    assert "test sentence here" in prompt


# --- load_model ---


def test_load_model_uses_model_id(app):
    with (
        patch.object(app, "AutoModelForCausalLM") as mock_cls,
        patch.object(app, "AutoTokenizer") as mock_tok,
    ):
        mock_cls.from_pretrained.return_value = MagicMock()
        mock_tok.from_pretrained.return_value = MagicMock()
        app.load_model("cpu")
        assert mock_cls.from_pretrained.call_args[0][0] == app.MODEL_ID
        assert mock_tok.from_pretrained.call_args[0][0] == app.MODEL_ID


def test_load_model_passes_hf_token(app):
    with (
        patch.dict(os.environ, {"HF_TOKEN": "hf_test123"}),
        patch.object(app, "AutoModelForCausalLM") as mock_cls,
        patch.object(app, "AutoTokenizer") as mock_tok,
    ):
        mock_cls.from_pretrained.return_value = MagicMock()
        mock_tok.from_pretrained.return_value = MagicMock()
        app.load_model("cpu")
        assert mock_cls.from_pretrained.call_args[1]["token"] == "hf_test123"
        assert mock_tok.from_pretrained.call_args[1]["token"] == "hf_test123"


def test_load_model_no_token_passes_false(app):
    env = os.environ.copy()
    env.pop("HF_TOKEN", None)
    with (
        patch.dict(os.environ, env, clear=True),
        patch.object(app, "AutoModelForCausalLM") as mock_cls,
        patch.object(app, "AutoTokenizer") as mock_tok,
    ):
        mock_cls.from_pretrained.return_value = MagicMock()
        mock_tok.from_pretrained.return_value = MagicMock()
        app.load_model("cpu")
        assert mock_cls.from_pretrained.call_args[1]["token"] is False
        assert mock_tok.from_pretrained.call_args[1]["token"] is False
