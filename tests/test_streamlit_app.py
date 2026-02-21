import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TEST_TEMPLATE = json.dumps({"company": "string", "revenue": "string"})
TEST_EXAMPLES = [
    {
        "input": "Acme Corp reported $1B in revenue.",
        "output": json.dumps({"company": "Acme Corp", "revenue": "$1B"}),
    }
]


@pytest.fixture(scope="module")
def app():
    """Import streamlit_app with streamlit UI and model loading mocked."""
    import streamlit as st
    from transformers import AutoModelForImageTextToText, AutoProcessor

    tab_mocks = [MagicMock(), MagicMock(), MagicMock()]

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


def test_default_template_is_valid_json(app):
    parsed = json.loads(app.DEFAULT_TEMPLATE)
    assert isinstance(parsed, dict)
    assert len(parsed) > 0


def test_default_examples_have_required_keys(app):
    parsed = json.loads(app.DEFAULT_EXAMPLES)
    assert isinstance(parsed, list)
    assert len(parsed) > 0
    for ex in parsed:
        assert "input" in ex
        assert "output" in ex


# --- validate_template ---


def test_validate_template_valid(app):
    parsed, error = app.validate_template('{"name": "string"}')
    assert parsed == {"name": "string"}
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


# --- parse_examples ---


def test_parse_examples_valid(app):
    examples_str = json.dumps([{"input": "foo", "output": "bar"}])
    parsed, error = app.parse_examples(examples_str)
    assert parsed == [{"input": "foo", "output": "bar"}]
    assert error is None


def test_parse_examples_empty_string(app):
    parsed, error = app.parse_examples("")
    assert parsed == []
    assert error is None


def test_parse_examples_whitespace(app):
    parsed, error = app.parse_examples("   ")
    assert parsed == []
    assert error is None


def test_parse_examples_not_array(app):
    parsed, error = app.parse_examples('{"not": "array"}')
    assert parsed is None
    assert "array" in error.lower()


def test_parse_examples_missing_keys(app):
    parsed, error = app.parse_examples('[{"input": "foo"}]')
    assert parsed is None
    assert "output" in error.lower()


# --- extract ---


def test_extract_returns_parsed_dict(app):
    output = json.dumps({"company": "Acme", "revenue": "$1B"})
    model, processor = _make_mocks(output)
    result = app.extract(
        "some text", model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES
    )
    assert result == {"company": "Acme", "revenue": "$1B"}


def test_extract_json_failure_returns_none(app):
    model, processor = _make_mocks("not valid json {{{")
    result = app.extract(
        "some text", model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES
    )
    assert result is None


def test_extract_empty_values_returns_dict(app):
    output = json.dumps({"company": "", "revenue": ""})
    model, processor = _make_mocks(output)
    result = app.extract(
        "some text", model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES
    )
    assert result == {"company": "", "revenue": ""}


def test_extract_decodes_only_new_tokens(app):
    output = json.dumps({"company": "Acme", "revenue": "$1B"})
    model, processor = _make_mocks(output)
    app.extract("some text", model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES)

    decode_call = processor.batch_decode.call_args
    decoded_tensor = decode_call[0][0]
    output_tensor = model.generate.return_value
    input_len = 5  # _make_mocks creates input_ids with shape (1, 5)
    expected_trimmed = output_tensor[:, input_len:]
    assert torch.equal(decoded_tensor, expected_trimmed)
    assert decode_call[1]["skip_special_tokens"] is True


def test_extract_generate_error_propagates(app):
    model, processor = _make_mocks(json.dumps({"company": "Acme"}))
    model.generate.side_effect = RuntimeError("out of memory")
    with pytest.raises(RuntimeError, match="out of memory"):
        app.extract("some text", model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES)


def test_extract_passes_template_and_examples(app):
    output = json.dumps({"company": "Acme"})
    model, processor = _make_mocks(output)
    app.extract("test sentence", model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES)

    call_args = processor.tokenizer.apply_chat_template.call_args
    assert call_args[1]["template"] == TEST_TEMPLATE
    assert call_args[1]["examples"] == TEST_EXAMPLES


def test_extract_zero_shot(app):
    output = json.dumps({"company": "Acme"})
    model, processor = _make_mocks(output)
    app.extract("test sentence", model, processor, "cpu", TEST_TEMPLATE, [])

    call_args = processor.tokenizer.apply_chat_template.call_args
    assert call_args[1]["examples"] == []


def test_extract_image_builds_vision_message(app):
    output = json.dumps({"company": "Acme"})
    model, processor = _make_mocks(output)
    fake_image = MagicMock()
    fake_image_inputs = MagicMock()

    with patch(
        "streamlit_app.process_vision_info",
        return_value=(fake_image_inputs, None),
    ) as mock_pvi:
        app.extract(
            "context",
            model,
            processor,
            "cpu",
            TEST_TEMPLATE,
            TEST_EXAMPLES,
            image=fake_image,
        )

        mock_pvi.assert_called_once()
        messages = mock_pvi.call_args[0][0]
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert content[0]["type"] == "image"
        assert content[0]["image"] is fake_image
        assert content[1] == {"type": "text", "text": "context"}

        proc_call = processor.call_args
        assert proc_call[1]["images"] is fake_image_inputs


def test_extract_text_only_passes_images_none(app):
    output = json.dumps({"company": "Acme"})
    model, processor = _make_mocks(output)
    app.extract("some text", model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES)

    proc_call = processor.call_args
    assert proc_call[1]["images"] is None


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
