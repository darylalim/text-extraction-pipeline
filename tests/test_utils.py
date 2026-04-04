import json

import pytest


# --- detect_and_convert_template ---


def test_detect_json_valid():
    from utils import detect_and_convert_template

    json_str, fmt, error = detect_and_convert_template('{"name": ""}')
    assert fmt == "json"
    assert json.loads(json_str) == {"name": ""}
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

    yaml_input = 'name: ""\nage: ""\n'
    json_str, fmt, error = detect_and_convert_template(yaml_input)
    assert fmt == "yaml"
    assert json.loads(json_str) == {"name": "", "age": ""}
    assert error is None


def test_detect_yaml_nested():
    from utils import detect_and_convert_template

    yaml_input = 'person:\n  name: ""\n  age: ""\n'
    json_str, fmt, error = detect_and_convert_template(yaml_input)
    assert fmt == "yaml"
    parsed = json.loads(json_str)
    assert parsed == {"person": {"name": "", "age": ""}}


def test_detect_yaml_not_dict():
    from utils import detect_and_convert_template

    yaml_input = "- item1\n- item2\n"
    json_str, fmt, error = detect_and_convert_template(yaml_input)
    assert json_str is None
    assert fmt is None
    assert error is not None


def test_detect_yaml_invalid():
    from utils import detect_and_convert_template

    yaml_input = "{{{\ninvalid: yaml: content: [["
    json_str, fmt, error = detect_and_convert_template(yaml_input)
    assert json_str is None
    assert fmt is None
    assert error is not None


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
        "name": "",
        "age": "",
        "salary": "",
        "active": "",
    }
    assert error is None


def test_detect_pydantic_list_type():
    from utils import detect_and_convert_template

    pydantic_input = "class Job(BaseModel):\n    title: str\n    skills: list[str]\n"
    json_str, fmt, error = detect_and_convert_template(pydantic_input)
    assert fmt == "pydantic"
    parsed = json.loads(json_str)
    assert parsed == {"title": "", "skills": []}


def test_detect_pydantic_optional_type():
    from utils import detect_and_convert_template

    pydantic_input = (
        "class Item(BaseModel):\n    name: str\n    description: Optional[str]\n"
    )
    json_str, fmt, error = detect_and_convert_template(pydantic_input)
    assert fmt == "pydantic"
    parsed = json.loads(json_str)
    assert parsed == {"name": "", "description": ""}


def test_detect_pydantic_datetime_type():
    from utils import detect_and_convert_template

    pydantic_input = "class Event(BaseModel):\n    name: str\n    date: datetime\n"
    json_str, fmt, error = detect_and_convert_template(pydantic_input)
    assert fmt == "pydantic"
    parsed = json.loads(json_str)
    assert parsed == {"name": "", "date": ""}


def test_detect_pydantic_nested_model_falls_back_to_empty():
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
    assert parsed == {"name": "", "address": ""}


def test_detect_pydantic_nested_list_falls_back():
    from utils import detect_and_convert_template

    pydantic_input = "class Item(BaseModel):\n    tags: list[list[str]]\n"
    json_str, fmt, error = detect_and_convert_template(pydantic_input)
    assert fmt == "pydantic"
    parsed = json.loads(json_str)
    assert parsed == {"tags": []}


def test_detect_natural_language_returns_error():
    from utils import detect_and_convert_template

    text = "Extract the person's name and age from the text"
    json_str, fmt, error = detect_and_convert_template(text)
    assert json_str is None
    assert fmt is None
    assert error is not None
    assert "unrecognized" in error.lower()
