import json
import re

import torch
import yaml
from qwen_vl_utils import fetch_image, process_vision_info

DEFAULT_MAX_NEW_TOKENS = 2048


_PYDANTIC_TYPE_MAP = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "datetime": "date-time",
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
            mapped = "string"
            has_unknown = True
        result[name] = mapped

    return result, has_unknown


def _map_pydantic_type(type_str):
    """Map a Pydantic type annotation to a NuExtract type."""
    type_str = type_str.strip()

    if type_str in _PYDANTIC_TYPE_MAP:
        return _PYDANTIC_TYPE_MAP[type_str]

    optional_match = re.match(r"Optional\[(.+)\]$", type_str)
    if optional_match:
        inner = optional_match.group(1).strip()
        return _map_pydantic_type(inner)

    list_match = re.match(r"list\[(.+)\]$", type_str, re.IGNORECASE)
    if list_match:
        inner = list_match.group(1).strip()
        inner_mapped = _map_pydantic_type(inner)
        if inner_mapped is None or isinstance(inner_mapped, list):
            return ["string"]
        return [inner_mapped]

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

    # No format matched — treat as natural language
    return None, None, None


def generate_template(description, model, processor, device):
    """Generate a JSON extraction template from a natural language description.

    Uses the NuExtract model's native template generation mode by passing
    template=None to apply_chat_template.

    Returns (dict, None) on success, (None, error_message) on failure.
    """
    messages = [{"role": "user", "content": description}]
    formatted = processor.tokenizer.apply_chat_template(
        messages,
        template=None,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[formatted], images=None, padding=True, return_tensors="pt"
    ).to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=256,
        )

    trimmed = output[:, input_len:]
    decoded = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    try:
        parsed = json.loads(decoded[0])
        return parsed, None
    except (json.JSONDecodeError, IndexError) as e:
        return None, f"Could not parse generated template: {e}"


def process_all_vision_info(messages, examples=None):
    """Extract images from both ICL examples and user messages.

    Supports single input (messages is a list of dicts) or batch input
    (messages is a list of lists of dicts). Returns a flat list of images
    in per-item order (example images then message images for each item),
    or None if no images found.

    Raises ValueError if batched examples length doesn't match messages length.
    """
    if not messages:
        return None

    # Detect single vs batch: single messages is [{"role": ...}, ...],
    # batch is [[{"role": ...}, ...], ...]
    is_batch = isinstance(messages[0], list)
    messages_batch = messages if is_batch else [messages]

    # Normalize examples
    if examples is None:
        examples_batch = [None] * len(messages_batch)
    elif isinstance(examples, list) and examples and isinstance(examples[0], list):
        # Batched examples: [[ex1, ex2], [ex3, ex4]]
        examples_batch = examples
    else:
        # Single examples list: broadcast to each batch item
        examples_batch = [examples for _ in messages_batch]

    if len(examples_batch) != len(messages_batch):
        raise ValueError(
            f"Examples batch length ({len(examples_batch)}) must match "
            f"messages batch length ({len(messages_batch)})."
        )

    all_images = []
    for msg_group, ex_group in zip(messages_batch, examples_batch):
        if ex_group:
            for ex in ex_group:
                inp = ex.get("input")
                if isinstance(inp, dict) and inp.get("type") == "image":
                    all_images.append(fetch_image(inp))

        message_images = process_vision_info(msg_group)[0] or []
        all_images.extend(message_images)

    return all_images if all_images else None
