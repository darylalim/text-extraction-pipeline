"""Convert CMS ICD-10-CM order file to JSON code list.

Download the source from:
https://www.cms.gov/medicare/coding-billing/icd-10-codes/2025-icd-10-cm

Usage:
    python scripts/generate_icd10_data.py icd10cm_order_2025.txt data/icd10_cm_2025.json
"""

import json
import re
import sys


def parse_icd10_cm(input_path, output_path):
    codes = []
    pattern = re.compile(r"^\d{5}\s+([A-Z0-9]{3,7})\s+(\d)\s+")
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line)
            if m and m.group(2) == "0":
                codes.append(m.group(1))

    codes.sort()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(codes, f)

    print(f"Wrote {len(codes)} codes to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python scripts/generate_icd10_data.py <icd10cm_order.txt> <output.json>"
        )
        sys.exit(1)
    parse_icd10_cm(sys.argv[1], sys.argv[2])
