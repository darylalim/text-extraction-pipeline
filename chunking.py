import re
from typing import Any

DEFAULT_CHUNK_TOKENS = 3500
DEFAULT_OVERLAP_TOKENS = 200
HEADER_ATTACH_ZONE = 100

SECTION_HEADERS: frozenset[str] = frozenset(
    {
        "HPI",
        "PMH",
        "PSH",
        "FH",
        "SH",
        "MEDICATIONS",
        "MEDS",
        "ALLERGIES",
        "ROS",
        "REVIEW OF SYSTEMS",
        "PHYSICAL EXAM",
        "EXAM",
        "VITALS",
        "ASSESSMENT",
        "PLAN",
        "ASSESSMENT AND PLAN",
        "CHIEF COMPLAINT",
        "CC",
        "HOSPITAL COURSE",
        "DISPOSITION",
        "SUBJECTIVE",
        "OBJECTIVE",
        "DIAGNOSES",
        "DIAGNOSIS",
        "PROCEDURES",
        "LABS",
        "IMAGING",
        "FOLLOW UP",
        "FOLLOW-UP",
        "DISCHARGE INSTRUCTIONS",
        "PENDING RESULTS",
    }
)

_HEADER_PATTERN = re.compile(
    r"^("
    + "|".join(
        re.escape(h)
        for h in sorted(SECTION_HEADERS, key=lambda h: len(h), reverse=True)
    )
    + r")\s*:?\s*$",
    re.IGNORECASE,
)


def _split_long_line(line: str, tokenizer: Any, max_tokens: int) -> list[str]:
    """Split a single line into sub-lines of at most max_tokens each."""
    # Preserve the trailing newline if present
    trailing = "\n" if line.endswith("\n") else ""
    # Intermediate sub-lines use "\n" only if the original line had a newline,
    # otherwise use a space to avoid injecting newlines into non-newline input.
    intermediate_sep = "\n" if trailing else " "
    stripped = line.rstrip("\n")
    words = stripped.split(" ")

    sub_lines = []
    current_words = []
    current_tokens = 0

    for word in words:
        word_tokens = len(tokenizer.encode(word))
        if current_words and current_tokens + 1 + word_tokens > max_tokens:
            sub_lines.append(" ".join(current_words) + intermediate_sep)
            current_words = [word]
            current_tokens = word_tokens
        else:
            if current_words:
                current_tokens += 1  # space separator token (approximation)
            current_words.append(word)
            current_tokens += word_tokens

    if current_words:
        sub_lines.append(" ".join(current_words) + trailing)

    return sub_lines if sub_lines else [line]


def chunk_text(
    text: str,
    tokenizer: Any,
    max_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap: int = DEFAULT_OVERLAP_TOKENS,
) -> list[str]:
    all_tokens = tokenizer.encode(text)
    if len(all_tokens) <= max_tokens:
        return [text]

    raw_lines = text.splitlines(keepends=True)

    # Split any line that exceeds max_tokens into word-level sub-lines.
    # Cache token counts from the pre-pass to avoid re-tokenizing each line.
    lines: list[str] = []
    line_token_counts: list[int] = []
    for line in raw_lines:
        count = len(tokenizer.encode(line))
        if count > max_tokens:
            sub = _split_long_line(line, tokenizer, max_tokens)
            lines.extend(sub)
            line_token_counts.extend(len(tokenizer.encode(s)) for s in sub)
        else:
            lines.append(line)
            line_token_counts.append(count)

    chunks = []
    start = 0

    while start < len(lines):
        token_sum = 0
        end = start
        while end < len(lines):
            if token_sum + line_token_counts[end] > max_tokens and end > start:
                break
            token_sum += line_token_counts[end]
            end += 1

        if end < len(lines):
            zone_tokens = 0
            for k in range(end - 1, start, -1):
                zone_tokens += line_token_counts[k]
                if zone_tokens > HEADER_ATTACH_ZONE:
                    break
                if _HEADER_PATTERN.match(lines[k].strip()):
                    end = k
                    break

        chunks.append("".join(lines[start:end]))

        if overlap > 0:
            overlap_tokens = 0
            next_start = end
            for k in range(end - 1, start, -1):
                overlap_tokens += line_token_counts[k]
                if overlap_tokens >= overlap:
                    next_start = k
                    break

            if next_start <= start:
                next_start = end
        else:
            next_start = end

        start = next_start

    return chunks if chunks else [text]
