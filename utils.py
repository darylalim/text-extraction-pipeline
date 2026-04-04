import json
import re

import yaml

DEFAULT_MAX_NEW_TOKENS = 2048


_PYDANTIC_TYPE_MAP = {
    "str": "",
    "int": "",
    "float": "",
    "bool": "",
    "datetime": "",
}


def _parse_pydantic_model(text):
    """Parse the last BaseModel class in text into a JSON-compatible dict.

    Returns (dict, has_unknown_types) or (None, False) if no model found.
    """
    class_pattern = re.compile(
        r"class\s+\w+\s*\(\s*BaseModel\s*\)\s*:(.*?)(?=\nclass\s|\Z)",
        re.DOTALL,
    )
    matches = list(class_pattern.finditer(text))
    if not matches:
        return None, False

    body = matches[-1].group(1)
    field_pattern = re.compile(r"^\s+(\w+)\s*:\s*(.+)$", re.MULTILINE)
    fields = field_pattern.findall(body)
    if not fields:
        return None, False

    result = {}
    has_unknown = False
    for name, type_str in fields:
        type_str = type_str.strip()
        mapped = _map_pydantic_type(type_str)
        if mapped is None:
            mapped = ""
            has_unknown = True
        result[name] = mapped

    return result, has_unknown


def _map_pydantic_type(type_str):
    """Map a Pydantic type annotation to a NuExtract-1.5 placeholder."""
    type_str = type_str.strip()

    if type_str in _PYDANTIC_TYPE_MAP:
        return _PYDANTIC_TYPE_MAP[type_str]

    optional_match = re.match(r"Optional\[(.+)\]$", type_str)
    if optional_match:
        inner = optional_match.group(1).strip()
        return _map_pydantic_type(inner)

    list_match = re.match(r"list\[(.+)\]$", type_str, re.IGNORECASE)
    if list_match:
        return []

    return None


def detect_and_convert_template(template_str):
    """Detect template format and convert to JSON.

    Returns (json_str, source_format, error) where source_format is
    "json", "yaml", "pydantic", "pydantic_with_unknown", or None.
    """
    if not template_str or not template_str.strip():
        return None, None, "Template must not be empty."

    # 1. Try JSON
    try:
        parsed = json.loads(template_str)
        if isinstance(parsed, dict):
            if not parsed:
                return None, None, "Template must not be empty."
            return template_str.strip(), "json", None
        else:
            return None, None, "Template must be a JSON object."
    except json.JSONDecodeError:
        pass

    # 2. Try Pydantic
    parsed, has_unknown = _parse_pydantic_model(template_str)
    if parsed:
        json_out = json.dumps(parsed, indent=2)
        if has_unknown:
            return json_out, "pydantic_with_unknown", None
        return json_out, "pydantic", None

    # 3. Try YAML
    try:
        parsed = yaml.safe_load(template_str)
        if isinstance(parsed, dict) and parsed:
            return json.dumps(parsed, indent=2), "yaml", None
    except yaml.YAMLError:
        pass

    # No format matched
    return None, None, "Unrecognized format. Provide a JSON, YAML, or Pydantic template."
