import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from conftest import make_batch_mocks, make_mocks

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

    tab_mocks = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]

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
        patch.object(
            AutoModelForImageTextToText, "from_pretrained", return_value=MagicMock()
        ),
        patch.object(AutoProcessor, "from_pretrained", return_value=MagicMock()),
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


def test_default_examples_have_required_keys(app):
    parsed = json.loads(app.DEFAULT_EXAMPLES)
    assert isinstance(parsed, list)
    assert len(parsed) > 0
    for ex in parsed:
        assert "input" in ex
        assert "output" in ex


def test_max_input_tokens_constant(app):
    assert app.MAX_INPUT_TOKENS == 10_000


def test_default_max_new_tokens_constant(app):
    assert app.DEFAULT_MAX_NEW_TOKENS == 2048


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


def test_parse_examples_image_input_accepted(app):
    examples_str = json.dumps(
        [
            {
                "input": {"type": "image", "image": "https://example.com/img.png"},
                "output": '{"name": "John"}',
            }
        ]
    )
    parsed, error = app.parse_examples(examples_str)
    assert parsed is not None
    assert error is None


@pytest.mark.parametrize(
    "input_dict",
    [
        {"type": "image"},
        {"type": "image", "image": ""},
        {"type": "image", "image": 12345},
        {"type": "image", "image": "file:///etc/passwd"},
        {"type": "video", "image": "https://example.com/vid.mp4"},
    ],
)
def test_parse_examples_invalid_image_input(app, input_dict):
    examples_str = json.dumps([{"input": input_dict, "output": '{"name": "John"}'}])
    parsed, error = app.parse_examples(examples_str)
    assert parsed is None
    assert error is not None


def test_parse_examples_mixed_text_and_image_accepted(app):
    examples_str = json.dumps(
        [
            {"input": "text example", "output": '{"name": "Alice"}'},
            {
                "input": {"type": "image", "image": "https://example.com/img.png"},
                "output": '{"name": "Bob"}',
            },
        ]
    )
    parsed, error = app.parse_examples(examples_str)
    assert parsed is not None
    assert len(parsed) == 2
    assert error is None


# --- _has_config_errors ---


def test_has_config_errors_template_error(app):
    with patch("streamlit_app.st") as mock_st:
        assert app._has_config_errors("bad json", None, None) is True
        mock_st.error.assert_called_once_with("Fix template: bad json")


def test_has_config_errors_examples_error(app):
    with patch("streamlit_app.st") as mock_st:
        assert app._has_config_errors(None, "missing keys", {"a": "b"}) is True
        mock_st.error.assert_called_once_with("Fix examples: missing keys")


def test_has_config_errors_no_errors(app):
    with patch("streamlit_app.st") as mock_st:
        assert app._has_config_errors(None, None, {"a": "b"}) is False
        mock_st.error.assert_not_called()


def test_has_config_errors_template_takes_priority(app):
    with patch("streamlit_app.st") as mock_st:
        assert app._has_config_errors("bad template", "bad examples", None) is True
        mock_st.error.assert_called_once_with("Fix template: bad template")


def test_has_config_errors_no_template_parsed(app):
    with patch("streamlit_app.st") as mock_st:
        assert app._has_config_errors(None, None, None) is True
        mock_st.error.assert_called_once_with("Generate a JSON template first.")


# --- _convert_template_if_needed ---


@pytest.mark.parametrize("fmt", ["yaml", "pydantic", "pydantic_with_unknown"])
def test_convert_template_converts_non_json(app, fmt):
    with patch("streamlit_app.st") as mock_st:
        mock_st.session_state = {}
        result = app._convert_template_if_needed('{"name": "string"}', fmt)
        assert result == '{"name": "string"}'
        assert mock_st.session_state["template_input"] == '{"name": "string"}'


@pytest.mark.parametrize("fmt", ["json", None])
def test_convert_template_returns_none(app, fmt):
    assert app._convert_template_if_needed('{"name": "string"}', fmt) is None


# --- extract ---


def test_extract_returns_parsed_dict(app):
    output = json.dumps({"company": "Acme", "revenue": "$1B"})
    model, processor = make_mocks(output)
    result, _ = app.extract(
        "some text", model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES
    )
    assert result == {"company": "Acme", "revenue": "$1B"}


def test_extract_json_failure_returns_none(app):
    model, processor = make_mocks("not valid json {{{")
    result, _ = app.extract(
        "some text", model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES
    )
    assert result is None


def test_extract_empty_values_returns_dict(app):
    output = json.dumps({"company": "", "revenue": ""})
    model, processor = make_mocks(output)
    result, _ = app.extract(
        "some text", model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES
    )
    assert result == {"company": "", "revenue": ""}


def test_extract_decodes_only_new_tokens(app):
    output = json.dumps({"company": "Acme", "revenue": "$1B"})
    model, processor = make_mocks(output)
    app.extract("some text", model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES)

    decode_call = processor.batch_decode.call_args
    decoded_tensor = decode_call[0][0]
    output_tensor = model.generate.return_value
    input_len = 5  # _make_mocks creates input_ids with shape (1, 5)
    expected_trimmed = output_tensor[:, input_len:]
    assert torch.equal(decoded_tensor, expected_trimmed)
    assert decode_call[1]["skip_special_tokens"] is True


def test_extract_uses_inference_mode(app):
    output = json.dumps({"company": "Acme"})
    model, processor = make_mocks(output)
    with patch("streamlit_app.torch.inference_mode") as mock_ctx:
        mock_ctx.return_value.__enter__ = MagicMock()
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
        app.extract("some text", model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES)
        mock_ctx.assert_called_once()


def test_extract_generate_error_propagates(app):
    model, processor = make_mocks(json.dumps({"company": "Acme"}))
    model.generate.side_effect = RuntimeError("out of memory")
    with pytest.raises(RuntimeError, match="out of memory"):
        app.extract("some text", model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES)


def test_extract_passes_template_and_examples(app):
    output = json.dumps({"company": "Acme"})
    model, processor = make_mocks(output)
    app.extract("test sentence", model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES)

    call_args = processor.tokenizer.apply_chat_template.call_args
    assert call_args[1]["template"] == TEST_TEMPLATE
    assert call_args[1]["examples"] == TEST_EXAMPLES


def test_extract_zero_shot(app):
    output = json.dumps({"company": "Acme"})
    model, processor = make_mocks(output)
    app.extract("test sentence", model, processor, "cpu", TEST_TEMPLATE, [])

    call_args = processor.tokenizer.apply_chat_template.call_args
    assert call_args[1]["examples"] == []


def test_extract_image_builds_vision_message(app):
    output = json.dumps({"company": "Acme"})
    model, processor = make_mocks(output)
    fake_image = MagicMock()
    fake_image_inputs = [MagicMock()]

    with patch(
        "streamlit_app.process_all_vision_info",
        return_value=fake_image_inputs,
    ) as mock_pavi:
        app.extract(
            "context",
            model,
            processor,
            "cpu",
            TEST_TEMPLATE,
            TEST_EXAMPLES,
            image=fake_image,
        )

        mock_pavi.assert_called_once()
        batched_messages = mock_pavi.call_args[0][0]
        messages = batched_messages[0]
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert content[0]["type"] == "image"
        assert content[0]["image"] is fake_image
        assert content[1] == {"type": "text", "text": "context"}

        proc_call = processor.call_args
        assert proc_call[1]["images"] is fake_image_inputs


def test_extract_text_only_passes_images_none(app):
    output = json.dumps({"company": "Acme"})
    model, processor = make_mocks(output)
    app.extract("some text", model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES)

    proc_call = processor.call_args
    assert proc_call[1]["images"] is None


def test_extract_image_no_context_text(app):
    output = json.dumps({"company": "Acme"})
    model, processor = make_mocks(output)
    fake_image = MagicMock()
    fake_image_inputs = [MagicMock()]

    with patch(
        "streamlit_app.process_all_vision_info",
        return_value=fake_image_inputs,
    ) as mock_pavi:
        app.extract(
            None,
            model,
            processor,
            "cpu",
            TEST_TEMPLATE,
            TEST_EXAMPLES,
            image=fake_image,
        )

        batched_messages = mock_pavi.call_args[0][0]
        messages = batched_messages[0]
        content = messages[0]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "image"


# --- Token limit ---


def test_extract_over_token_limit_raises(app):
    output = json.dumps({"company": "Acme"})
    model, processor = make_mocks(output)
    # Override input_ids to exceed MAX_INPUT_TOKENS
    big_input_ids = torch.ones(1, 10_001, dtype=torch.long)
    proc_result = MagicMock()
    proc_result.to.return_value = {
        "input_ids": big_input_ids,
        "attention_mask": torch.ones_like(big_input_ids),
    }
    processor.return_value = proc_result
    with pytest.raises(ValueError, match="10001.*10000"):
        app.extract("some text", model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES)


def test_extract_at_token_limit_succeeds(app):
    output = json.dumps({"company": "Acme"})
    model, processor = make_mocks(output)
    # Override input_ids to exactly MAX_INPUT_TOKENS
    exact_input_ids = torch.ones(1, 10_000, dtype=torch.long)
    proc_result = MagicMock()
    proc_result.to.return_value = {
        "input_ids": exact_input_ids,
        "attention_mask": torch.ones_like(exact_input_ids),
    }
    processor.return_value = proc_result
    model.generate.return_value = torch.cat(
        [exact_input_ids, torch.tensor([[10, 20, 30]])], dim=1
    )
    result, _ = app.extract(
        "some text", model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES
    )
    assert result == {"company": "Acme"}


def test_extract_uses_custom_max_new_tokens(app):
    output = json.dumps({"company": "Acme"})
    model, processor = make_mocks(output)
    app.extract(
        "some text",
        model,
        processor,
        "cpu",
        TEST_TEMPLATE,
        TEST_EXAMPLES,
        max_new_tokens=512,
    )
    gen_call = model.generate.call_args
    assert gen_call[1]["max_new_tokens"] == 512


def test_extract_default_max_new_tokens_is_2048(app):
    output = json.dumps({"company": "Acme"})
    model, processor = make_mocks(output)
    app.extract("some text", model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES)
    gen_call = model.generate.call_args
    assert gen_call[1]["max_new_tokens"] == 2048


# --- truncation detection ---


def test_extract_detects_truncation(app):
    output = json.dumps({"company": "Acme"})
    model, processor = make_mocks(output)
    # make_mocks generates output tensor of shape (1, 8), input_ids shape (1, 5)
    # trimmed = 3 tokens. Set max_new_tokens=3 to trigger truncation.
    _, was_truncated = app.extract(
        "some text",
        model,
        processor,
        "cpu",
        TEST_TEMPLATE,
        TEST_EXAMPLES,
        max_new_tokens=3,
    )
    assert was_truncated is True


def test_extract_no_truncation_when_output_shorter(app):
    output = json.dumps({"company": "Acme"})
    model, processor = make_mocks(output)
    # trimmed = 3 tokens, max_new_tokens=100 → no truncation
    _, was_truncated = app.extract(
        "some text",
        model,
        processor,
        "cpu",
        TEST_TEMPLATE,
        TEST_EXAMPLES,
        max_new_tokens=100,
    )
    assert was_truncated is False


def test_extract_truncation_with_json_failure(app):
    model, processor = make_mocks("not valid json {{{")
    # trimmed = 3 tokens, max_new_tokens=3 → truncated AND json fails
    result, was_truncated = app.extract(
        "some text",
        model,
        processor,
        "cpu",
        TEST_TEMPLATE,
        TEST_EXAMPLES,
        max_new_tokens=3,
    )
    assert result is None
    assert was_truncated is True


# --- extract with image examples ---


def test_extract_image_with_image_examples_calls_process_all(app):
    output = json.dumps({"company": "Acme"})
    model, processor = make_mocks(output)
    fake_image = MagicMock()
    fake_all_images = [MagicMock(), MagicMock()]
    image_examples = [
        {
            "input": {"type": "image", "image": "https://example.com/ex.png"},
            "output": '{"company": "Test"}',
        }
    ]

    with patch(
        "streamlit_app.process_all_vision_info",
        return_value=fake_all_images,
    ) as mock_pavi:
        app.extract(
            "context",
            model,
            processor,
            "cpu",
            TEST_TEMPLATE,
            image_examples,
            image=fake_image,
        )

        mock_pavi.assert_called_once()
        call_args = mock_pavi.call_args
        assert call_args[0][1] == image_examples

        proc_call = processor.call_args
        assert proc_call[1]["images"] is fake_all_images


# --- get_device ---


@pytest.mark.parametrize(
    "mps,cuda,expected",
    [
        (True, False, "mps"),
        (False, True, "cuda"),
        (False, False, "cpu"),
    ],
)
def test_get_device(app, mps, cuda, expected):
    with (
        patch("torch.backends.mps.is_available", return_value=mps),
        patch("torch.cuda.is_available", return_value=cuda),
    ):
        assert app.get_device() == expected


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


# --- load_presets ---


def test_load_presets_valid_file(app, tmp_path):
    presets_data = [
        {
            "name": "Test",
            "template": {"field": "string"},
            "examples": [],
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
        {
            "name": "Good",
            "template": {"f": "string"},
            "examples": [],
            "sample_text": "x",
        },
        {"name": "Bad"},
        {
            "name": "Also Good",
            "template": {"g": "integer"},
            "examples": [],
            "sample_text": "y",
        },
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
        assert isinstance(p["examples"], list)
        assert isinstance(p["sample_text"], str)


# --- _clear_device_cache ---


@pytest.mark.parametrize("device,attr", [("cuda", "cuda"), ("mps", "mps")])
def test_clear_device_cache(app, device, attr):
    with patch("streamlit_app.torch") as mock_torch:
        setattr(mock_torch, attr, MagicMock())
        app._clear_device_cache(device)
        getattr(mock_torch, attr).empty_cache.assert_called_once()


def test_clear_device_cache_cpu_is_noop(app):
    with patch("streamlit_app.torch") as mock_torch:
        app._clear_device_cache("cpu")
        mock_torch.cuda.empty_cache.assert_not_called()


# --- extract_batch ---


def test_extract_batch_two_text_items(app):
    outputs = [
        json.dumps({"company": "Acme", "revenue": "$1B"}),
        json.dumps({"company": "Beta", "revenue": "$2B"}),
    ]
    model, processor = make_batch_mocks(outputs)
    inputs = [
        {"text": "Acme text", "image": None, "context": None},
        {"text": "Beta text", "image": None, "context": None},
    ]
    results = app.extract_batch(
        inputs, model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES
    )
    assert len(results) == 2
    assert results[0][0] == {"company": "Acme", "revenue": "$1B"}
    assert results[1][0] == {"company": "Beta", "revenue": "$2B"}


def test_extract_batch_empty_inputs(app):
    model, processor = make_batch_mocks([])
    results = app.extract_batch(
        [], model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES
    )
    assert results == []
    model.generate.assert_not_called()


def test_extract_batch_with_image_and_context(app):
    outputs = [json.dumps({"company": "Acme"})]
    model, processor = make_batch_mocks(outputs)
    fake_image = MagicMock()

    with patch(
        "streamlit_app.process_all_vision_info",
        return_value=[MagicMock()],
    ):
        inputs = [{"text": None, "image": fake_image, "context": "some context"}]
        results = app.extract_batch(
            inputs, model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES
        )

    assert results[0][0] == {"company": "Acme"}
    call_args = processor.tokenizer.apply_chat_template.call_args
    messages = call_args[0][0]
    content = messages[0]["content"]
    assert content[0]["type"] == "image"
    assert content[1] == {"type": "text", "text": "some context"}


def test_extract_batch_image_without_context(app):
    outputs = [json.dumps({"company": "Acme"})]
    model, processor = make_batch_mocks(outputs)
    fake_image = MagicMock()

    with patch(
        "streamlit_app.process_all_vision_info",
        return_value=[MagicMock()],
    ):
        inputs = [{"text": None, "image": fake_image, "context": None}]
        results = app.extract_batch(
            inputs, model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES
        )

    assert results[0][0] == {"company": "Acme"}
    call_args = processor.tokenizer.apply_chat_template.call_args
    messages = call_args[0][0]
    content = messages[0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "image"


def test_extract_batch_token_limit_skips_item(app):
    outputs = [
        json.dumps({"company": "Acme"}),
        json.dumps({"company": "Beta"}),
    ]
    model, processor = make_batch_mocks(outputs, input_lengths=[5, 10_001])

    inputs = [
        {"text": "short", "image": None, "context": None},
        {"text": "very long text", "image": None, "context": None},
    ]
    results = app.extract_batch(
        inputs, model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES, chunk_size=1
    )
    assert len(results) == 2
    assert results[0][0] == {"company": "Acme"}
    assert results[1] == (None, False)


def test_extract_batch_truncation_detection(app):
    output = json.dumps({"company": "Acme"})
    model, processor = make_batch_mocks([output])
    results = app.extract_batch(
        [{"text": "text", "image": None, "context": None}],
        model,
        processor,
        "cpu",
        TEST_TEMPLATE,
        TEST_EXAMPLES,
        max_new_tokens=3,
    )
    assert results[0][1] is True


def test_extract_batch_no_truncation(app):
    output = json.dumps({"company": "Acme"})
    model, processor = make_batch_mocks([output])
    results = app.extract_batch(
        [{"text": "text", "image": None, "context": None}],
        model,
        processor,
        "cpu",
        TEST_TEMPLATE,
        TEST_EXAMPLES,
        max_new_tokens=100,
    )
    assert results[0][1] is False


def test_extract_batch_oom_falls_back_to_sequential(app):
    outputs = [
        json.dumps({"company": "Acme"}),
        json.dumps({"company": "Beta"}),
    ]
    model, processor = make_batch_mocks(outputs)

    call_count = [0]
    single_output_a = torch.tensor([[1, 2, 3, 4, 5, 10, 20, 30]])
    single_output_b = torch.tensor([[1, 2, 3, 4, 5, 11, 21, 31]])

    def generate_side_effect(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("CUDA out of memory")
        if call_count[0] == 2:
            return single_output_a
        return single_output_b

    model.generate.side_effect = generate_side_effect

    single_input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    single_inputs = {
        "input_ids": single_input_ids,
        "attention_mask": torch.ones_like(single_input_ids),
    }
    single_proc = MagicMock()
    single_proc.to.return_value = single_inputs

    proc_call_count = [0]
    original_proc = processor.return_value

    def proc_side_effect(*args, **kwargs):
        proc_call_count[0] += 1
        if proc_call_count[0] == 1:
            return original_proc
        return single_proc

    processor.side_effect = proc_side_effect
    processor.batch_decode.side_effect = [
        [json.dumps({"company": "Acme"})],
        [json.dumps({"company": "Beta"})],
    ]

    with patch("streamlit_app._clear_device_cache"):
        inputs = [
            {"text": "text1", "image": None, "context": None},
            {"text": "text2", "image": None, "context": None},
        ]
        results = app.extract_batch(
            inputs, model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES, chunk_size=2
        )

    assert len(results) == 2
    assert results[0][0] == {"company": "Acme"}
    assert results[1][0] == {"company": "Beta"}


def test_extract_batch_single_item(app):
    output = json.dumps({"company": "Acme", "revenue": "$1B"})
    model, processor = make_batch_mocks([output])
    inputs = [{"text": "some text", "image": None, "context": None}]
    results = app.extract_batch(
        inputs, model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES
    )
    assert len(results) == 1
    assert results[0][0] == {"company": "Acme", "revenue": "$1B"}


def test_extract_batch_progress_callback(app):
    outputs = [
        json.dumps({"company": "A"}),
        json.dumps({"company": "B"}),
        json.dumps({"company": "C"}),
    ]
    model, processor = make_batch_mocks(outputs)
    callback = MagicMock()

    inputs = [
        {"text": "t1", "image": None, "context": None},
        {"text": "t2", "image": None, "context": None},
        {"text": "t3", "image": None, "context": None},
    ]
    app.extract_batch(
        inputs,
        model,
        processor,
        "cpu",
        TEST_TEMPLATE,
        TEST_EXAMPLES,
        chunk_size=2,
        progress_callback=callback,
    )
    assert callback.call_count == 2
    callback.assert_any_call(2, 3)
    callback.assert_any_call(3, 3)


def test_extract_batch_mixed_text_and_image(app):
    outputs = [
        json.dumps({"company": "Acme"}),
        json.dumps({"company": "Beta"}),
    ]
    model, processor = make_batch_mocks(outputs)
    fake_image = MagicMock()

    with patch(
        "streamlit_app.process_all_vision_info",
        return_value=[MagicMock()],
    ):
        inputs = [
            {"text": "text only input", "image": None, "context": None},
            {"text": None, "image": fake_image, "context": "image context"},
        ]
        results = app.extract_batch(
            inputs, model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES
        )

    assert len(results) == 2
    assert results[0][0] == {"company": "Acme"}
    assert results[1][0] == {"company": "Beta"}


def test_extract_batch_different_input_lengths(app):
    outputs = [
        json.dumps({"company": "Acme"}),
        json.dumps({"company": "Beta"}),
    ]
    model, processor = make_batch_mocks(outputs, input_lengths=[3, 7])

    inputs = [
        {"text": "short", "image": None, "context": None},
        {"text": "much longer text input", "image": None, "context": None},
    ]
    results = app.extract_batch(
        inputs, model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES
    )
    assert len(results) == 2
    assert results[0][0] == {"company": "Acme"}
    assert results[1][0] == {"company": "Beta"}


def test_extract_batch_passes_template_and_examples(app):
    output = json.dumps({"company": "Acme"})
    model, processor = make_batch_mocks([output])
    inputs = [{"text": "text", "image": None, "context": None}]
    app.extract_batch(inputs, model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES)
    call_args = processor.tokenizer.apply_chat_template.call_args
    assert call_args[1]["template"] == TEST_TEMPLATE
    assert call_args[1]["examples"] == TEST_EXAMPLES


# --- extract() wrapper ---


def test_extract_delegates_to_extract_batch(app):
    output = json.dumps({"company": "Acme", "revenue": "$1B"})
    model, processor = make_mocks(output)
    with patch.object(
        app,
        "extract_batch",
        return_value=[({"company": "Acme", "revenue": "$1B"}, False)],
    ) as mock_eb:
        result, was_truncated = app.extract(
            "some text", model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES
        )
    mock_eb.assert_called_once()
    call_args = mock_eb.call_args
    batch_input = call_args[0][0][0]
    assert batch_input == {"text": "some text", "image": None, "context": None}
    assert call_args[1]["chunk_size"] == 1
    assert result == {"company": "Acme", "revenue": "$1B"}
    assert was_truncated is False


def test_extract_wrapper_image_maps_to_context(app):
    model, processor = make_mocks(json.dumps({"company": "Acme"}))
    fake_image = MagicMock()
    with patch.object(
        app, "extract_batch", return_value=[({"company": "Acme"}, False)]
    ) as mock_eb:
        app.extract(
            "context text",
            model,
            processor,
            "cpu",
            TEST_TEMPLATE,
            TEST_EXAMPLES,
            image=fake_image,
        )
    batch_input = mock_eb.call_args[0][0][0]
    assert batch_input["text"] is None
    assert batch_input["image"] is fake_image
    assert batch_input["context"] == "context text"


# --- CSV image loading ---


def test_load_csv_image_url(app):
    fake_img = MagicMock(name="fetched_img")
    with patch("qwen_vl_utils.fetch_image", return_value=fake_img) as mock_fetch:
        result = app._load_csv_image("https://example.com/img.png")
    assert result is fake_img
    mock_fetch.assert_called_once_with({"image": "https://example.com/img.png"})


def test_load_csv_image_http_url(app):
    fake_img = MagicMock(name="fetched_img")
    with patch("qwen_vl_utils.fetch_image", return_value=fake_img) as mock_fetch:
        result = app._load_csv_image("http://example.com/img.png")
    assert result is fake_img
    mock_fetch.assert_called_once_with({"image": "http://example.com/img.png"})


def test_load_csv_image_file_path(app, tmp_path):
    from PIL import Image as PILImage

    img_path = tmp_path / "test.png"
    PILImage.new("RGB", (10, 10)).save(img_path)
    result = app._load_csv_image(str(img_path))
    assert result is not None
    assert hasattr(result, "size")


def test_load_csv_image_invalid_path(app):
    result = app._load_csv_image("/nonexistent/path/image.png")
    assert result is None


@pytest.mark.parametrize("value", ["", "nan", None])
def test_load_csv_image_returns_none_for_empty(app, value):
    assert app._load_csv_image(value) is None
