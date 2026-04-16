import re

import pytest


class WordTokenizer:
    """Mock tokenizer: each whitespace-separated word = 1 token."""

    def encode(self, text):
        return text.split() if text and text.strip() else []


@pytest.fixture
def tokenizer():
    return WordTokenizer()


# --- Short text (no chunking needed) ---


def test_short_text_returns_single_chunk(tokenizer):
    from chunking import chunk_text

    text = "hello world"
    result = chunk_text(text, tokenizer, max_tokens=10)
    assert result == [text]


def test_exact_limit_returns_single_chunk(tokenizer):
    from chunking import chunk_text

    text = "one two three"
    result = chunk_text(text, tokenizer, max_tokens=3)
    assert result == [text]


def test_empty_text_returns_single_chunk(tokenizer):
    from chunking import chunk_text

    result = chunk_text("", tokenizer, max_tokens=10)
    assert result == [""]


def test_whitespace_only_returns_single_chunk(tokenizer):
    from chunking import chunk_text

    result = chunk_text("   ", tokenizer, max_tokens=10)
    assert result == ["   "]


# --- Basic chunking ---


def test_over_limit_produces_multiple_chunks(tokenizer):
    from chunking import chunk_text

    # 10 words, max 4 tokens per chunk, overlap 1
    text = "a b c d e f g h i j"
    result = chunk_text(text, tokenizer, max_tokens=4, overlap=1)
    assert len(result) > 1
    # Every chunk should have at most 4 words
    for chunk in result:
        assert len(tokenizer.encode(chunk)) <= 4


def test_all_text_is_covered(tokenizer):
    from chunking import chunk_text

    text = "alpha bravo charlie delta echo foxtrot golf hotel"
    result = chunk_text(text, tokenizer, max_tokens=3, overlap=1)
    combined = " ".join(" ".join(c.split()) for c in result)
    for word in text.split():
        assert word in combined


# --- Overlap ---


def test_overlap_repeats_tokens(tokenizer):
    from chunking import chunk_text

    # 6 lines, max 3 tokens/chunk, overlap 1
    text = "line1\nline2\nline3\nline4\nline5\nline6"
    result = chunk_text(text, tokenizer, max_tokens=3, overlap=1)
    assert len(result) >= 2
    # Last line of chunk N should appear in chunk N+1
    for i in range(len(result) - 1):
        prev_lines = result[i].splitlines()
        next_lines = result[i + 1].splitlines()
        assert prev_lines[-1] == next_lines[0]


def test_zero_overlap_no_repeat(tokenizer):
    from chunking import chunk_text

    text = "a\nb\nc\nd\ne\nf"
    result = chunk_text(text, tokenizer, max_tokens=2, overlap=0)
    total_lines = sum(len(c.splitlines()) for c in result)
    assert total_lines == 6


# --- Header-attach ---


def test_header_at_chunk_end_moves_to_next(tokenizer):
    from chunking import chunk_text

    # 5 lines, max 3 tokens/chunk, overlap 0
    # Line 3 is a header — should be pushed to next chunk
    text = "word1\nword2\nMEDICATIONS:\nmed1\nmed2"
    result = chunk_text(text, tokenizer, max_tokens=3, overlap=0)
    # MEDICATIONS: should not be the last line of any chunk (except the last chunk)
    for chunk in result[:-1]:
        lines = chunk.splitlines()
        assert not any(
            re.match(r"^MEDICATIONS\s*:?", line, re.IGNORECASE) for line in lines[-1:]
        )


def test_header_early_in_chunk_stays(tokenizer):
    from chunking import chunk_text

    text = "MEDICATIONS:\nmed1\nmed2\nmed3\nmed4\nmed5\nmed6"
    result = chunk_text(text, tokenizer, max_tokens=4, overlap=0)
    assert result[0].startswith("MEDICATIONS:")


def test_header_case_insensitive(tokenizer):
    from chunking import chunk_text

    text = "word1\nword2\nassessment:\nfinding1\nfinding2"
    result = chunk_text(text, tokenizer, max_tokens=3, overlap=0)
    for chunk in result[:-1]:
        lines = chunk.splitlines()
        assert not any(
            re.match(r"^assessment\s*:?", line, re.IGNORECASE) for line in lines[-1:]
        )


def test_header_not_matched_mid_sentence(tokenizer):
    from chunking import chunk_text

    # "the assessment was" should NOT trigger header-attach
    text = "word1\nthe assessment was good\nword3\nword4\nword5"
    result = chunk_text(text, tokenizer, max_tokens=3, overlap=0)
    # Should chunk normally without header-attach affecting "the assessment was good"
    assert len(result) >= 2


# --- No headers ---


def test_no_headers_pure_token_window(tokenizer):
    from chunking import chunk_text

    text = "a\nb\nc\nd\ne\nf\ng\nh"
    result = chunk_text(text, tokenizer, max_tokens=3, overlap=1)
    assert len(result) >= 3
    for chunk in result:
        assert len(tokenizer.encode(chunk)) <= 3


# --- Constants ---


def test_section_headers_is_frozenset():
    from chunking import SECTION_HEADERS

    assert isinstance(SECTION_HEADERS, frozenset)
    assert len(SECTION_HEADERS) >= 25


def test_default_constants():
    from chunking import (
        DEFAULT_CHUNK_TOKENS,
        DEFAULT_OVERLAP_TOKENS,
        HEADER_ATTACH_ZONE,
    )

    assert DEFAULT_CHUNK_TOKENS == 3500
    assert DEFAULT_OVERLAP_TOKENS == 200
    assert HEADER_ATTACH_ZONE == 100


# --- _split_long_line ---


def test_split_long_line_short_returns_single(tokenizer):
    from chunking import _split_long_line

    result = _split_long_line("one two three", tokenizer, max_tokens=10)
    assert result == ["one two three"]


def test_split_long_line_splits_at_word_boundary(tokenizer):
    from chunking import _split_long_line

    result = _split_long_line("one two three four five six", tokenizer, max_tokens=3)
    assert len(result) > 1
    # Every sub-line fits within max_tokens
    for sub in result:
        assert len(tokenizer.encode(sub)) <= 3


def test_split_long_line_preserves_trailing_newline(tokenizer):
    from chunking import _split_long_line

    result = _split_long_line("one two three four\n", tokenizer, max_tokens=2)
    # Last sub-line must keep the newline
    assert result[-1].endswith("\n")


def test_split_long_line_uses_newline_between_subs_when_input_has_newline(tokenizer):
    from chunking import _split_long_line

    result = _split_long_line("one two three four\n", tokenizer, max_tokens=2)
    # Intermediate sub-lines get \n, final sub-line keeps the original trailing \n
    for sub in result[:-1]:
        assert sub.endswith("\n")


def test_split_long_line_uses_space_between_subs_when_no_newline(tokenizer):
    from chunking import _split_long_line

    result = _split_long_line("one two three four five six", tokenizer, max_tokens=2)
    # No sub-line should end with a newline since input had no trailing newline
    for sub in result:
        assert not sub.endswith("\n")


def test_split_long_line_reconstructs_without_newlines(tokenizer):
    from chunking import _split_long_line

    original = "alpha bravo charlie delta echo foxtrot"
    result = _split_long_line(original, tokenizer, max_tokens=2)
    # Rejoining with stripping trailing whitespace reproduces the words in order
    all_words = []
    for sub in result:
        all_words.extend(sub.split())
    assert all_words == original.split()


def test_split_long_line_single_word_kept(tokenizer):
    from chunking import _split_long_line

    # Single word still returns one sub-line even if tokenizer says it exceeds limit
    result = _split_long_line("onlyoneword", tokenizer, max_tokens=1)
    assert len(result) == 1
    assert "onlyoneword" in result[0]


# --- Clinical fixture integration ---


def test_chunking_on_clinical_sample(tokenizer, sample_3_chunk_text):
    """Real clinical text from shared fixture splits into multiple chunks."""
    from chunking import chunk_text

    result = chunk_text(sample_3_chunk_text, tokenizer, max_tokens=15, overlap=2)
    assert len(result) >= 2
    # Every word in the original appears somewhere in the combined chunks
    combined_words = set()
    for chunk in result:
        combined_words.update(chunk.split())
    assert set(sample_3_chunk_text.split()).issubset(combined_words)
