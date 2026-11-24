"""Hashing and stable identifier helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK = 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    """Hash raw bytes with SHA-256.

    Args:
        data: The bytes to hash.

    Returns:
        The 64-character lowercase hex digest.
    """
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Hash a string with SHA-256 after encoding it as UTF-8.

    Args:
        text: The text to hash.

    Returns:
        The 64-character lowercase hex digest of ``text.encode("utf-8")``.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file's contents with SHA-256, reading it in 1 MiB chunks so large files fit in memory.

    Args:
        path: The file to read (opened in binary mode).

    Returns:
        The 64-character lowercase hex digest of the file's bytes.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def short_hash(*parts: str, length: int = 10) -> str:
    """Build a deterministic short hash of the given parts (joined by NUL).

    The NUL separator keeps ``("ab", "c")`` and ``("a", "bc")`` distinct. Used for stable identifiers.

    Args:
        *parts: Strings to combine, in order.
        length: Number of leading hex characters of the SHA-256 digest to keep.

    Returns:
        The first ``length`` hex characters of the SHA-256 of the joined parts.
    """
    joined = "\x00".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:length]
