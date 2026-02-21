import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIELD = "Line of Credit Facility Maximum Borrowing Capacity"


@pytest.fixture(scope="module")
def app():
    """Import streamlit_app with streamlit UI and model loading mocked."""
    import streamlit as st
    from transformers import AutoModelForImageTextToText, AutoProcessor

    with (
        patch.object(st, "title"),
        patch.object(st, "write"),
        patch.object(st, "file_uploader", return_value=None),
        patch.object(st, "spinner"),
        patch.object(st, "info"),
        patch.object(st, "cache_resource", side_effect=lambda f: f),
        patch.object(
            AutoModelForImageTextToText, "from_pretrained", return_value=MagicMock()
        ),
        patch.object(AutoProcessor, "from_pretrained", return_value=MagicMock()),
    ):
        sys.modules.pop("streamlit_app", None)
        import streamlit_app

        yield streamlit_app
        sys.modules.pop("streamlit_app", None)


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


# --- Constants ---


def test_model_id(app):
    assert app.MODEL_ID == "numind/NuExtract-2.0-4B"


def test_field_name(app):
    assert app.FIELD == FIELD


def test_max_new_tokens(app):
    assert app.MAX_NEW_TOKENS == 256


def test_max_answer_length(app):
    assert app.MAX_ANSWER_LENGTH == 50


def test_template(app):
    parsed = json.loads(app.TEMPLATE)
    assert FIELD in parsed


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
    model, processor = _make_mocks(json.dumps({FIELD: "$500M"}))
    assert app.extract_text("some text", model, processor, "cpu") == "$500M"


def test_extract_json_parse_failure_returns_na(app):
    model, processor = _make_mocks("not valid json {{{")
    assert app.extract_text("some text", model, processor, "cpu") == "N/A"


def test_extract_empty_returns_na(app):
    model, processor = _make_mocks(json.dumps({FIELD: ""}))
    assert app.extract_text("some text", model, processor, "cpu") == "N/A"


def test_extract_whitespace_returns_na(app):
    model, processor = _make_mocks(json.dumps({FIELD: "   "}))
    assert app.extract_text("some text", model, processor, "cpu") == "N/A"


def test_extract_too_long_returns_na(app):
    model, processor = _make_mocks(json.dumps({FIELD: "x" * 51}))
    assert app.extract_text("some text", model, processor, "cpu") == "N/A"


def test_extract_max_length_accepted(app):
    answer = "x" * 50
    model, processor = _make_mocks(json.dumps({FIELD: answer}))
    assert app.extract_text("some text", model, processor, "cpu") == answer


def test_extract_decodes_only_new_tokens(app):
    model, processor = _make_mocks(json.dumps({FIELD: "$500M"}))
    app.extract_text("some text", model, processor, "cpu")

    decode_call = processor.batch_decode.call_args
    decoded_tensor = decode_call[0][0]
    output_tensor = model.generate.return_value
    input_len = 5  # _make_mocks creates input_ids with shape (1, 5)
    expected_trimmed = output_tensor[:, input_len:]
    assert torch.equal(decoded_tensor, expected_trimmed)
    assert decode_call[1]["skip_special_tokens"] is True


def test_extract_generate_error_propagates(app):
    model, processor = _make_mocks(json.dumps({FIELD: "$500M"}))
    model.generate.side_effect = RuntimeError("out of memory")
    with pytest.raises(RuntimeError, match="out of memory"):
        app.extract_text("some text", model, processor, "cpu")


def test_extract_builds_prompt_with_template_and_examples(app):
    model, processor = _make_mocks(json.dumps({FIELD: "$500M"}))
    app.extract_text("test sentence here", model, processor, "cpu")

    call_args = processor.tokenizer.apply_chat_template.call_args
    messages = call_args[0][0]
    assert messages[0]["content"] == "test sentence here"
    assert call_args[1]["template"] == app.TEMPLATE
    assert call_args[1]["examples"] == app.EXAMPLES


# --- load_model ---


def test_load_model_uses_model_id(app):
    with (
        patch.object(app, "AutoModelForImageTextToText") as mock_cls,
        patch.object(app, "AutoProcessor") as mock_proc,
    ):
        mock_cls.from_pretrained.return_value = MagicMock()
        mock_proc.from_pretrained.return_value = MagicMock()
        app.load_model("cpu")
        assert mock_cls.from_pretrained.call_args[0][0] == app.MODEL_ID
        assert mock_proc.from_pretrained.call_args[0][0] == app.MODEL_ID


def test_load_model_passes_hf_token(app):
    with (
        patch.dict(os.environ, {"HF_TOKEN": "hf_test123"}),
        patch.object(app, "AutoModelForImageTextToText") as mock_cls,
        patch.object(app, "AutoProcessor") as mock_proc,
    ):
        mock_cls.from_pretrained.return_value = MagicMock()
        mock_proc.from_pretrained.return_value = MagicMock()
        app.load_model("cpu")
        assert mock_cls.from_pretrained.call_args[1]["token"] == "hf_test123"
        assert mock_proc.from_pretrained.call_args[1]["token"] == "hf_test123"


def test_load_model_no_token_passes_false(app):
    env = os.environ.copy()
    env.pop("HF_TOKEN", None)
    with (
        patch.dict(os.environ, env, clear=True),
        patch.object(app, "AutoModelForImageTextToText") as mock_cls,
        patch.object(app, "AutoProcessor") as mock_proc,
    ):
        mock_cls.from_pretrained.return_value = MagicMock()
        mock_proc.from_pretrained.return_value = MagicMock()
        app.load_model("cpu")
        assert mock_cls.from_pretrained.call_args[1]["token"] is False
        assert mock_proc.from_pretrained.call_args[1]["token"] is False


def test_load_model_uses_trust_remote_code(app):
    with (
        patch.object(app, "AutoModelForImageTextToText") as mock_cls,
        patch.object(app, "AutoProcessor") as mock_proc,
    ):
        mock_cls.from_pretrained.return_value = MagicMock()
        mock_proc.from_pretrained.return_value = MagicMock()
        app.load_model("cpu")
        assert mock_cls.from_pretrained.call_args[1]["trust_remote_code"] is True
        assert mock_proc.from_pretrained.call_args[1]["trust_remote_code"] is True
