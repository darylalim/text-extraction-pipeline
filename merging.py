import json


def merge_results(results, template):
    """Merge per-chunk extraction results using template shape.

    - Scalar field (template value is ""): first non-empty value across results wins.
    - List field (template value is []): union + de-dupe. Items are compared by
      canonical JSON serialization (with sorted keys), so de-dupe is exact; near-
      duplicates like "metformin 1000mg" vs "metformin 1000 mg" are not merged.
    - Nested dict field: recurse using the same rules.
    - None results are skipped.
    - All None -> returns None.
    """
    non_none = [r for r in results if r is not None]
    if not non_none:
        return None
    if len(non_none) == 1:
        return non_none[0]
    return _merge_dicts(non_none, template)


def _merge_dicts(results, template):
    merged = {}
    for key, template_val in template.items():
        if isinstance(template_val, list):
            merged[key] = _merge_lists(results, key)
        elif isinstance(template_val, dict):
            sub_results = [r[key] for r in results if isinstance(r.get(key), dict)]
            if sub_results:
                merged[key] = _merge_dicts(sub_results, template_val)
            else:
                merged[key] = _first_non_empty(results, key)
        else:
            merged[key] = _first_non_empty(results, key)
    return merged


def _first_non_empty(results, key):
    for r in results:
        val = r.get(key, "")
        if val != "" and val is not None:
            return val
    return ""


def _merge_lists(results, key):
    all_items = []
    seen = set()
    for r in results:
        items = r.get(key, [])
        if not isinstance(items, list):
            continue
        for item in items:
            item_key = _hash_key(item)
            if item_key not in seen:
                seen.add(item_key)
                all_items.append(item)
    return all_items


def _hash_key(item):
    """Produce a hashable key for de-dupe. Handles dicts with nested values."""
    if isinstance(item, (dict, list)):
        return json.dumps(item, sort_keys=True, default=str)
    return item
