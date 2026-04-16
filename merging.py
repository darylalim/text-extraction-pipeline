def merge_results(results, template):
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
            if isinstance(item, dict):
                item_key = tuple(sorted(item.items()))
            else:
                item_key = item
            if item_key not in seen:
                seen.add(item_key)
                all_items.append(item)
    return all_items
