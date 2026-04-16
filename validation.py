import json
import logging

logger = logging.getLogger(__name__)


def load_icd10_codes(path="data/icd10_cm_2025.json"):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning("Failed to load ICD-10 codes from %s: %s", path, e)
        return set()

    if not isinstance(data, list):
        logger.warning("ICD-10 file is not a list: %s", path)
        return set()

    return {code.upper() for code in data if isinstance(code, str)}


def annotate_icd10(result, codes):
    if isinstance(result, dict):
        return _annotate_dict(result, codes)
    if isinstance(result, list):
        return [annotate_icd10(item, codes) for item in result]
    return result


def _annotate_dict(d, codes):
    annotated = {}
    for key, value in d.items():
        if key == "icd10_code" and isinstance(value, str):
            annotated[key] = value
            normalized = value.strip().upper().replace(".", "")
            annotated["icd10_code_valid"] = bool(normalized) and normalized in codes
        elif isinstance(value, dict):
            annotated[key] = _annotate_dict(value, codes)
        elif isinstance(value, list):
            annotated[key] = [annotate_icd10(item, codes) for item in value]
        else:
            annotated[key] = value
    return annotated


def count_invalid_codes(result):
    count = 0
    if isinstance(result, dict):
        for key, value in result.items():
            if key == "icd10_code_valid" and value is False:
                count += 1
            elif isinstance(value, (dict, list)):
                count += count_invalid_codes(value)
    elif isinstance(result, list):
        for item in result:
            count += count_invalid_codes(item)
    return count
