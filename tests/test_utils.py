import json
from unittest.mock import MagicMock, patch

import pytest
import torch

from conftest import make_mocks


# --- generate_template ---


def test_generate_template_returns_dict_on_valid_output():
    from utils import generate_template

    output = json.dumps({"name": "string", "age": "integer"})
    model, processor = make_mocks(output)
    result, error = generate_template("extract name and age", model, processor, "cpu")
    assert result == {"name": "string", "age": "integer"}
    assert error is None


def test_generate_template_passes_template_none():
    from utils import generate_template

    output = json.dumps({"name": "string"})
    model, processor = make_mocks(output)
    generate_template("extract name", model, processor, "cpu")

    call_args = processor.tokenizer.apply_chat_template.call_args
    assert call_args[1]["template"] is None


def test_generate_template_passes_description_as_message():
    from utils import generate_template

    output = json.dumps({"name": "string"})
    model, processor = make_mocks(output)
    generate_template("extract the person's name", model, processor, "cpu")

    call_args = processor.tokenizer.apply_chat_template.call_args
    messages = call_args[0][0]
    assert messages == [{"role": "user", "content": "extract the person's name"}]


def test_generate_template_invalid_output_returns_error():
    from utils import generate_template

    model, processor = make_mocks("not valid json {{{")
    result, error = generate_template("extract stuff", model, processor, "cpu")
    assert result is None
    assert error is not None


def test_generate_template_model_error_propagates():
    from utils import generate_template

    model, processor = make_mocks(json.dumps({"name": "string"}))
    model.generate.side_effect = RuntimeError("out of memory")
    with pytest.raises(RuntimeError, match="out of memory"):
        generate_template("extract name", model, processor, "cpu")


def test_generate_template_uses_inference_mode():
    from utils import generate_template

    output = json.dumps({"name": "string"})
    model, processor = make_mocks(output)
    with patch("utils.torch.inference_mode") as mock_ctx:
        mock_ctx.return_value.__enter__ = MagicMock()
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
        generate_template("extract name", model, processor, "cpu")
        mock_ctx.assert_called_once()


def test_generate_template_passes_images_none():
    from utils import generate_template

    output = json.dumps({"name": "string"})
    model, processor = make_mocks(output)
    generate_template("extract name", model, processor, "cpu")

    proc_call = processor.call_args
    assert proc_call[1]["images"] is None


def test_generate_template_decodes_only_new_tokens():
    from utils import generate_template

    output = json.dumps({"name": "string"})
    model, processor = make_mocks(output)
    generate_template("extract name", model, processor, "cpu")

    decode_call = processor.batch_decode.call_args
    decoded_tensor = decode_call[0][0]
    output_tensor = model.generate.return_value
    input_len = 5  # _make_mocks creates input_ids with shape (1, 5)
    expected_trimmed = output_tensor[:, input_len:]
    assert torch.equal(decoded_tensor, expected_trimmed)


def test_generate_template_uses_256_max_new_tokens():
    from utils import generate_template

    output = json.dumps({"name": "string"})
    model, processor = make_mocks(output)
    generate_template("extract name", model, processor, "cpu")
    gen_call = model.generate.call_args
    assert gen_call[1]["max_new_tokens"] == 256


# --- process_all_vision_info ---


def test_process_all_vision_info_example_images_and_message_image():
    from utils import process_all_vision_info

    fake_example_img = MagicMock(name="example_img")
    fake_message_img = MagicMock(name="message_img")

    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": "https://example.com/input.png"}],
        }
    ]
    examples = [
        {
            "input": {"type": "image", "image": "https://example.com/ex.png"},
            "output": '{"name": "John"}',
        }
    ]

    with (
        patch(
            "utils.process_vision_info",
            return_value=([fake_message_img], None),
        ),
        patch("utils.fetch_image", return_value=fake_example_img),
    ):
        result = process_all_vision_info(messages, examples)

    assert result == [fake_example_img, fake_message_img]


def test_process_all_vision_info_empty_examples():
    from utils import process_all_vision_info

    fake_message_img = MagicMock(name="message_img")
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": "https://example.com/input.png"}],
        }
    ]

    with patch(
        "utils.process_vision_info",
        return_value=([fake_message_img], None),
    ):
        result = process_all_vision_info(messages, [])

    assert result == [fake_message_img]


def test_process_all_vision_info_only_example_images():
    from utils import process_all_vision_info

    fake_example_img = MagicMock(name="example_img")
    messages = [{"role": "user", "content": "just text"}]
    examples = [
        {
            "input": {"type": "image", "image": "https://example.com/ex.png"},
            "output": '{"name": "John"}',
        }
    ]

    with (
        patch(
            "utils.process_vision_info",
            return_value=(None, None),
        ),
        patch("utils.fetch_image", return_value=fake_example_img),
    ):
        result = process_all_vision_info(messages, examples)

    assert result == [fake_example_img]


def test_process_all_vision_info_no_images_returns_none():
    from utils import process_all_vision_info

    messages = [{"role": "user", "content": "just text"}]

    with patch(
        "utils.process_vision_info",
        return_value=(None, None),
    ):
        result = process_all_vision_info(messages, None)

    assert result is None


def test_process_all_vision_info_text_examples_ignored():
    from utils import process_all_vision_info

    fake_message_img = MagicMock(name="message_img")
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": "https://example.com/input.png"}],
        }
    ]
    examples = [
        {"input": "just text", "output": '{"name": "John"}'},
    ]

    with patch(
        "utils.process_vision_info",
        return_value=([fake_message_img], None),
    ):
        result = process_all_vision_info(messages, examples)

    assert result == [fake_message_img]


def test_process_all_vision_info_mixed_examples():
    from utils import process_all_vision_info

    fake_example_img = MagicMock(name="example_img")
    fake_message_img = MagicMock(name="message_img")

    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": "https://example.com/input.png"}],
        }
    ]
    examples = [
        {"input": "just text", "output": '{"name": "Alice"}'},
        {
            "input": {"type": "image", "image": "https://example.com/ex.png"},
            "output": '{"name": "Bob"}',
        },
    ]

    with (
        patch(
            "utils.process_vision_info",
            return_value=([fake_message_img], None),
        ),
        patch("utils.fetch_image", return_value=fake_example_img),
    ):
        result = process_all_vision_info(messages, examples)

    assert result == [fake_example_img, fake_message_img]


# --- process_all_vision_info batch support ---


def test_process_all_vision_info_batch_two_items_with_examples():
    from utils import process_all_vision_info

    fake_ex_img1 = MagicMock(name="ex_img1")
    fake_ex_img2 = MagicMock(name="ex_img2")
    fake_msg_img1 = MagicMock(name="msg_img1")
    fake_msg_img2 = MagicMock(name="msg_img2")

    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "https://example.com/input1.png"}
                ],
            }
        ],
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "https://example.com/input2.png"}
                ],
            }
        ],
    ]
    examples = [
        [
            {
                "input": {"type": "image", "image": "https://example.com/ex1.png"},
                "output": '{"name": "A"}',
            }
        ],
        [
            {
                "input": {"type": "image", "image": "https://example.com/ex2.png"},
                "output": '{"name": "B"}',
            }
        ],
    ]

    call_count = [0]
    example_images = [fake_ex_img1, fake_ex_img2]

    def mock_fetch_image(inp):
        img = example_images[call_count[0]]
        call_count[0] += 1
        return img

    msg_images = [[fake_msg_img1], [fake_msg_img2]]
    pvi_call_count = [0]

    def mock_process_vision_info(msgs):
        imgs = msg_images[pvi_call_count[0]]
        pvi_call_count[0] += 1
        return (imgs, None)

    with (
        patch("utils.process_vision_info", side_effect=mock_process_vision_info),
        patch("utils.fetch_image", side_effect=mock_fetch_image),
    ):
        result = process_all_vision_info(messages, examples)

    assert result == [fake_ex_img1, fake_msg_img1, fake_ex_img2, fake_msg_img2]


def test_process_all_vision_info_batch_single_examples_broadcast():
    from utils import process_all_vision_info

    fake_ex_img = MagicMock(name="ex_img")
    fake_msg_img1 = MagicMock(name="msg_img1")
    fake_msg_img2 = MagicMock(name="msg_img2")

    messages = [
        [
            {
                "role": "user",
                "content": [{"type": "image", "image": "https://example.com/1.png"}],
            }
        ],
        [
            {
                "role": "user",
                "content": [{"type": "image", "image": "https://example.com/2.png"}],
            }
        ],
    ]
    examples = [
        {
            "input": {"type": "image", "image": "https://example.com/ex.png"},
            "output": '{"name": "A"}',
        }
    ]

    msg_images = [[fake_msg_img1], [fake_msg_img2]]
    pvi_call_count = [0]

    def mock_process_vision_info(msgs):
        imgs = msg_images[pvi_call_count[0]]
        pvi_call_count[0] += 1
        return (imgs, None)

    with (
        patch("utils.process_vision_info", side_effect=mock_process_vision_info),
        patch("utils.fetch_image", return_value=fake_ex_img),
    ):
        result = process_all_vision_info(messages, examples)

    # Single examples list is broadcast to each batch item
    assert result == [fake_ex_img, fake_msg_img1, fake_ex_img, fake_msg_img2]


def test_process_all_vision_info_batch_mismatched_lengths():
    from utils import process_all_vision_info

    messages = [
        [{"role": "user", "content": "text1"}],
        [{"role": "user", "content": "text2"}],
    ]
    examples = [
        [{"input": "ex1", "output": "out1"}],
        [{"input": "ex2", "output": "out2"}],
        [{"input": "ex3", "output": "out3"}],
    ]

    with pytest.raises(ValueError, match="length"):
        process_all_vision_info(messages, examples)


def test_process_all_vision_info_single_input_backward_compat():
    from utils import process_all_vision_info

    fake_message_img = MagicMock(name="message_img")
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": "https://example.com/input.png"}],
        }
    ]

    with patch(
        "utils.process_vision_info",
        return_value=([fake_message_img], None),
    ):
        result = process_all_vision_info(messages)

    assert result == [fake_message_img]


# --- detect_and_convert_template ---


def test_detect_json_valid():
    from utils import detect_and_convert_template

    json_str, fmt, error = detect_and_convert_template('{"name": "string"}')
    assert fmt == "json"
    assert json.loads(json_str) == {"name": "string"}
    assert error is None


def test_detect_json_empty_object():
    from utils import detect_and_convert_template

    json_str, fmt, error = detect_and_convert_template("{}")
    assert json_str is None
    assert fmt is None
    assert error is not None
    assert "empty" in error.lower()


def test_detect_json_non_dict():
    from utils import detect_and_convert_template

    json_str, fmt, error = detect_and_convert_template("[1, 2, 3]")
    assert json_str is None
    assert fmt is None
    assert error is not None
    assert "object" in error.lower()


@pytest.mark.parametrize("input_str", ["", "   "])
def test_detect_empty_or_whitespace(input_str):
    from utils import detect_and_convert_template

    json_str, fmt, error = detect_and_convert_template(input_str)
    assert json_str is None
    assert fmt is None
    assert "empty" in error.lower()


def test_detect_yaml_valid():
    from utils import detect_and_convert_template

    yaml_input = "name: string\nage: integer\n"
    json_str, fmt, error = detect_and_convert_template(yaml_input)
    assert fmt == "yaml"
    assert json.loads(json_str) == {"name": "string", "age": "integer"}
    assert error is None


def test_detect_yaml_nested():
    from utils import detect_and_convert_template

    yaml_input = "person:\n  name: verbatim-string\n  age: integer\n"
    json_str, fmt, error = detect_and_convert_template(yaml_input)
    assert fmt == "yaml"
    parsed = json.loads(json_str)
    assert parsed == {"person": {"name": "verbatim-string", "age": "integer"}}


def test_detect_yaml_not_dict():
    from utils import detect_and_convert_template

    yaml_input = "- item1\n- item2\n"
    json_str, fmt, error = detect_and_convert_template(yaml_input)
    assert json_str is None
    assert fmt is None
    assert error is None


def test_detect_yaml_invalid():
    from utils import detect_and_convert_template

    yaml_input = "{{{\ninvalid: yaml: content: [["
    json_str, fmt, error = detect_and_convert_template(yaml_input)
    assert json_str is None
    assert fmt is None
    assert error is None


def test_detect_pydantic_flat_model():
    from utils import detect_and_convert_template

    pydantic_input = (
        "class Person(BaseModel):\n"
        "    name: str\n"
        "    age: int\n"
        "    salary: float\n"
        "    active: bool\n"
    )
    json_str, fmt, error = detect_and_convert_template(pydantic_input)
    assert fmt == "pydantic"
    parsed = json.loads(json_str)
    assert parsed == {
        "name": "string",
        "age": "integer",
        "salary": "number",
        "active": "boolean",
    }
    assert error is None


def test_detect_pydantic_list_type():
    from utils import detect_and_convert_template

    pydantic_input = "class Job(BaseModel):\n    title: str\n    skills: list[str]\n"
    json_str, fmt, error = detect_and_convert_template(pydantic_input)
    assert fmt == "pydantic"
    parsed = json.loads(json_str)
    assert parsed == {"title": "string", "skills": ["string"]}


def test_detect_pydantic_optional_type():
    from utils import detect_and_convert_template

    pydantic_input = (
        "class Item(BaseModel):\n    name: str\n    description: Optional[str]\n"
    )
    json_str, fmt, error = detect_and_convert_template(pydantic_input)
    assert fmt == "pydantic"
    parsed = json.loads(json_str)
    assert parsed == {"name": "string", "description": "string"}


def test_detect_pydantic_datetime_type():
    from utils import detect_and_convert_template

    pydantic_input = "class Event(BaseModel):\n    name: str\n    date: datetime\n"
    json_str, fmt, error = detect_and_convert_template(pydantic_input)
    assert fmt == "pydantic"
    parsed = json.loads(json_str)
    assert parsed == {"name": "string", "date": "date-time"}


def test_detect_pydantic_nested_model_falls_back_to_string():
    from utils import detect_and_convert_template

    pydantic_input = (
        "class Address(BaseModel):\n"
        "    street: str\n"
        "\n"
        "class Person(BaseModel):\n"
        "    name: str\n"
        "    address: Address\n"
    )
    json_str, fmt, error = detect_and_convert_template(pydantic_input)
    assert fmt == "pydantic_with_unknown"
    parsed = json.loads(json_str)
    assert parsed == {"name": "string", "address": "string"}


def test_detect_pydantic_nested_list_falls_back():
    from utils import detect_and_convert_template

    pydantic_input = "class Item(BaseModel):\n    tags: list[list[str]]\n"
    json_str, fmt, error = detect_and_convert_template(pydantic_input)
    assert fmt == "pydantic"
    parsed = json.loads(json_str)
    assert parsed == {"tags": ["string"]}


def test_detect_natural_language_fallthrough():
    from utils import detect_and_convert_template

    text = "Extract the person's name and age from the text"
    json_str, fmt, error = detect_and_convert_template(text)
    assert json_str is None
    assert fmt is None
    assert error is None
