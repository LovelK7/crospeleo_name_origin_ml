"""Shared helpers for text processing and IO."""

import re
from pathlib import Path


def tokenize(text: str) -> set[str]:
    """Lowercase, strip accents-preserving tokenization for Croatian text."""
    if not isinstance(text, str) or not text.strip():
        return set()
    return set(re.findall(r"[a-zA-ZčćšžđČĆŠŽĐ]+", text.lower()))


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
