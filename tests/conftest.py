import sys
from pathlib import Path
from unittest.mock import MagicMock

_TESTS_DIR = str(Path(__file__).resolve().parent)
_PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)


def make_mock_tokenizer(encode_length=50):
    """Create a mock tokenizer that returns encode_length tokens.

    The tokenizer.encode() method returns a list of ints of the given length.
    """
    tokenizer = MagicMock()
    tokenizer.encode.return_value = list(range(encode_length))
    return tokenizer
