import json
from unittest.mock import MagicMock

import torch


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


# --- generate_template ---


def test_generate_template_returns_dict_on_valid_output():
    from utils import generate_template

    output = json.dumps({"name": "string", "age": "integer"})
    model, processor = _make_mocks(output)
    result, error = generate_template("extract name and age", model, processor, "cpu")
    assert result == {"name": "string", "age": "integer"}
    assert error is None


def test_generate_template_passes_template_none():
    from utils import generate_template

    output = json.dumps({"name": "string"})
    model, processor = _make_mocks(output)
    generate_template("extract name", model, processor, "cpu")

    call_args = processor.tokenizer.apply_chat_template.call_args
    assert call_args[1]["template"] is None


def test_generate_template_passes_description_as_message():
    from utils import generate_template

    output = json.dumps({"name": "string"})
    model, processor = _make_mocks(output)
    generate_template("extract the person's name", model, processor, "cpu")

    call_args = processor.tokenizer.apply_chat_template.call_args
    messages = call_args[0][0]
    assert messages == [{"role": "user", "content": "extract the person's name"}]


def test_generate_template_invalid_output_returns_error():
    from utils import generate_template

    model, processor = _make_mocks("not valid json {{{")
    result, error = generate_template("extract stuff", model, processor, "cpu")
    assert result is None
    assert error is not None


def test_generate_template_model_error_propagates():
    import pytest
    from utils import generate_template

    model, processor = _make_mocks(json.dumps({"name": "string"}))
    model.generate.side_effect = RuntimeError("out of memory")
    with pytest.raises(RuntimeError, match="out of memory"):
        generate_template("extract name", model, processor, "cpu")
