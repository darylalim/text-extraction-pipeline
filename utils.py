import json

import torch


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
