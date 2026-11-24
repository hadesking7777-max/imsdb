"""Content-addressed, append-only archive of raw HTTP response bodies.

Layout::

    data/raw/<host>/<page_type>/<sha[:2]>/<sha256>.html[.gz]
    data/raw/_manifest/<YYYY-MM-DD>.jsonl     one line per fetch (self-describing archive)

* Files are named by the SHA-256 of the *uncompressed* body, so identical content is
  stored once and a changed page always lands in a new file - previous versions are
  never overwritten.
* Writes are atomic (temp file + ``os.replace``).
* The JSONL manifest lets the archive be re-indexed without MongoDB.
"""

from __future__ import annotations

import gzip
import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.clock import utcnow
from src.utils.hashing import sha256_bytes

_manifest_lock = threading.Lock()


@dataclass(slots=True)
class StoredArtifact:
    """Where and how a raw body was stored by :meth:`RawStore.save`.

    Attributes:
        sha256: Hex SHA-256 of the uncompressed body (also the file name stem).
        size: Length of the uncompressed body in bytes.
        relative_path: POSIX path of the file relative to the store root.
        compressed: Whether the file on disk is gzip-compressed (``.html.gz``).
        created: True when this call wrote the file; False when identical content already existed.
    """

    sha256: str
    size: int
    relative_path: str
    compressed: bool
    created: bool  # False when identical content already existed


class RawStoreError(RuntimeError):
    """Raised for paths that escape the store root and for checksum mismatches on read."""

    pass


class RawStore:
    """Content-addressed archive of raw response bodies on the local filesystem (see module docstring)."""

    def __init__(self, root: Path, *, gzip_bodies: bool = False) -> None:
        """Open (and create if needed) a raw store.

        Args:
            root: Root directory of the archive, e.g. ``data/raw``; created with parents if missing.
            gzip_bodies: Whether new files are written gzip-compressed. Existing files in the other
                format are still found and reused by :meth:`save`.
        """
        self.root = root
        self.gzip_bodies = gzip_bodies
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths
    def _relative(self, host: str, page_type: str, sha: str, compressed: bool) -> Path:
        """Compute the store-relative path ``<host>/<page_type>/<sha[:2]>/<sha>.html[.gz]``.

        Args:
            host: Source host name; characters other than alphanumerics, ``.`` and ``-`` become ``_``
                (an empty host becomes ``unknown``).
            page_type: Page category used as a directory name; sanitised like ``host`` so it can never
                contain path separators or ``..``.
            sha: Hex SHA-256 of the uncompressed body.
            compressed: Whether to use the ``.html.gz`` suffix instead of ``.html``.

        Returns:
            The relative path (not checked for existence).
        """
        suffix = ".html.gz" if compressed else ".html"
        safe_host = "".join(c if c.isalnum() or c in ".-" else "_" for c in host) or "unknown"
        safe_type = "".join(c if c.isalnum() or c in "_-" else "_" for c in page_type) or "other"
        return Path(safe_host) / safe_type / sha[:2] / f"{sha}{suffix}"

    def absolute(self, relative_path: str) -> Path:
        """Resolve a store-relative path to an absolute path, refusing anything outside the root.

        Args:
            relative_path: Path relative to the store root, as recorded in ``StoredArtifact``.

        Returns:
            The resolved absolute path (it may not exist).

        Raises:
            RawStoreError: If the resolved path escapes the store root (e.g. via ``..``).
        """
        path = (self.root / relative_path).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise RawStoreError(f"path escapes raw store: {relative_path}")
        return path

    # ------------------------------------------------------------------ write
    def save(self, body: bytes, *, host: str, page_type: str) -> StoredArtifact:
        """Store a body under its SHA-256, reusing an existing identical file. Idempotent.

        Both the preferred and the alternate compression format are checked. An existing file whose
        content verifies is reused (``created=False``); a corrupt one is renamed to
        ``<name>.corrupt-<timestamp>`` rather than overwritten. Otherwise the body is written atomically
        (temp file, fsync, ``os.replace``) in the preferred format.

        Args:
            body: Raw (uncompressed) response bytes.
            host: Source host name, used as the top-level directory.
            page_type: Page category, used as the second-level directory.

        Returns:
            Where the body is stored, and whether this call created the file.
        """
        sha = sha256_bytes(body)
        for compressed in (self.gzip_bodies, not self.gzip_bodies):
            rel = self._relative(host, page_type, sha, compressed)
            path = self.root / rel
            if path.exists():
                if self._verify_file(path, sha, compressed):
                    return StoredArtifact(sha, len(body), rel.as_posix(), compressed, created=False)
                # Corrupt file with our name: keep it for inspection, never overwrite silently.
                quarantine = path.with_name(path.name + f".corrupt-{utcnow():%Y%m%dT%H%M%S}")
                os.replace(path, quarantine)

        rel = self._relative(host, page_type, sha, self.gzip_bodies)
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        data = gzip.compress(body, mtime=0) if self.gzip_bodies else body
        fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return StoredArtifact(sha, len(body), rel.as_posix(), self.gzip_bodies, created=True)

    def append_manifest(self, record: dict[str, Any], when: datetime | None = None) -> None:
        """Append one JSON line describing a fetch to the daily manifest ``_manifest/<YYYY-MM-DD>.jsonl``.

        Thread-safe within the process (module-level lock). Keys are sorted and non-JSON values are
        ``str()``-ed.

        Args:
            record: The fetch description to write.
            when: Time whose date selects the manifest file; defaults to now (UTC).
        """
        when = when or utcnow()
        manifest_dir = self.root / "_manifest"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str, ensure_ascii=False, sort_keys=True)
        with _manifest_lock, (manifest_dir / f"{when:%Y-%m-%d}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    # ------------------------------------------------------------------ read / verify
    @staticmethod
    def _verify_file(path: Path, sha: str, compressed: bool) -> bool:
        """Check that a stored file's (decompressed) content has the expected SHA-256.

        Args:
            path: Absolute path of the file.
            sha: Expected hex SHA-256 of the uncompressed content.
            compressed: Whether to gunzip the file before hashing.

        Returns:
            True when the hash matches; False on mismatch, an unreadable file or a broken gzip stream.
        """
        try:
            data = path.read_bytes()
            if compressed:
                data = gzip.decompress(data)
        except (OSError, EOFError, gzip.BadGzipFile):
            return False
        return sha256_bytes(data) == sha

    def read(self, relative_path: str, *, expected_sha256: str | None = None) -> bytes:
        """Read a stored body, decompressing ``.gz`` files.

        Args:
            relative_path: Path relative to the store root.
            expected_sha256: If given, the uncompressed content's hash must match it.

        Returns:
            The uncompressed body bytes.

        Raises:
            RawStoreError: If the path escapes the store root or the checksum does not match.
        """
        path = self.absolute(relative_path)
        data = path.read_bytes()
        if relative_path.endswith(".gz"):
            data = gzip.decompress(data)
        if expected_sha256 and sha256_bytes(data) != expected_sha256:
            raise RawStoreError(f"checksum mismatch for {relative_path}")
        return data

    def verify(self, relative_path: str, expected_sha256: str) -> bool:
        """Check that a stored file exists and its content matches the expected hash.

        Args:
            relative_path: Path relative to the store root; a ``.gz`` suffix means compressed.
            expected_sha256: Expected hex SHA-256 of the uncompressed content.

        Returns:
            True when the file exists and verifies, otherwise False.

        Raises:
            RawStoreError: If the path escapes the store root.
        """
        path = self.absolute(relative_path)
        return path.exists() and self._verify_file(path, expected_sha256, relative_path.endswith(".gz"))

    def total_size(self) -> int:
        """Sum the on-disk size of every file under the store root.

        Returns:
            Total bytes, including manifests and any quarantined ``.corrupt-*`` files.
        """
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())


def artifact_record(stored: StoredArtifact) -> dict[str, Any]:
    """Convert a stored-artifact result to a plain dict (e.g. for MongoDB or the manifest).

    Args:
        stored: The result of :meth:`RawStore.save`.

    Returns:
        A dict with keys ``sha256``, ``size``, ``relative_path``, ``compressed`` and ``created``.
    """
    return asdict(stored)
