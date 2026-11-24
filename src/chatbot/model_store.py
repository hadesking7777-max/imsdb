"""Local model files: a pinned manifest, one-time download, and verification.

Models are downloaded once from Hugging Face into ``data/models`` and verified by SHA-256. After that the
chatbot never touches the network. The revision is pinned, so the same bytes are fetched every time.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx

log = logging.getLogger("imsdb.models")

DOWNLOAD_USER_AGENT = "imsdb-local-chatbot/1.0 (one-time model download)"
MAX_STALLED_ATTEMPTS = 5  # consecutive connections that deliver no data before the download gives up
# The repository stores each model as numbered pieces below GitHub's 100 MB file limit, in data/models/parts.
# They are joined (and SHA-256-verified) into the real .gguf file the first time a model is needed.
PARTS_DIR = "parts"
PART_BYTES = 95 * 1024 * 1024


@dataclass(frozen=True)
class ModelSpec:
    """A pinned GGUF model file on Hugging Face.

    Attributes:
        key: Short identifier used in settings and the CLI, e.g. ``"qwen3-8b-q4km"``.
        role: ``"chat"`` or ``"embedding"``.
        repo: Hugging Face repository, e.g. ``"Qwen/Qwen3-8B-GGUF"``.
        revision: Pinned commit hash, so the same bytes are fetched every time.
        filename: GGUF file name inside the repository (and in ``data/models``).
        size: Exact file size in bytes.
        sha256: Expected SHA-256 hex digest.
        license: License identifier of the model.
        n_layers: Number of transformer blocks; used to split the model between GPU and CPU.
        kv_bytes_per_token: KV-cache bytes per context token (f16), used to estimate GPU memory.
    """

    key: str
    role: str  # "chat" or "embedding"
    repo: str
    revision: str
    filename: str
    size: int
    sha256: str
    license: str
    n_layers: int = 0  # transformer blocks; used to split the model between GPU and CPU
    kv_bytes_per_token: int = 0  # KV-cache size per context token (f16): layers * kv_heads * head_dim * 2 * 2

    @property
    def url(self) -> str:
        """Download URL of the file at the pinned revision.

        Returns:
            The ``https://huggingface.co/<repo>/resolve/<revision>/<filename>`` URL.
        """
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{self.filename}"


MODELS: dict[str, ModelSpec] = {
    "qwen3-8b-q4km": ModelSpec(
        key="qwen3-8b-q4km", role="chat", repo="Qwen/Qwen3-8B-GGUF",
        revision="7c41481f57cb95916b40956ab2f0b139b296d974", filename="Qwen3-8B-Q4_K_M.gguf",
        size=5_027_783_488, sha256="d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785",
        license="Apache-2.0", n_layers=36, kv_bytes_per_token=36 * 8 * 128 * 2 * 2),
    "qwen3-embedding-0.6b-q8": ModelSpec(
        key="qwen3-embedding-0.6b-q8", role="embedding", repo="Qwen/Qwen3-Embedding-0.6B-GGUF",
        revision="370f27d7550e0def9b39c1f16d3fbaa13aa67728", filename="Qwen3-Embedding-0.6B-Q8_0.gguf",
        size=639_150_592, sha256="06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439",
        license="Apache-2.0", n_layers=28, kv_bytes_per_token=28 * 8 * 128 * 2 * 2),
}
DEFAULT_CHAT_MODEL = "qwen3-8b-q4km"
DEFAULT_EMBED_MODEL = "qwen3-embedding-0.6b-q8"


class ModelError(RuntimeError):
    """A model file is missing, incomplete, corrupt, or could not be downloaded."""

    pass


def model_path(models_dir: Path, spec: ModelSpec) -> Path:
    """Local path of a model file.

    Args:
        models_dir: The models folder (``data/models``).
        spec: The model.

    Returns:
        ``models_dir / spec.filename`` (it may not exist).
    """
    return models_dir / spec.filename


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    """Hash a file with SHA-256, reading it in blocks.

    Args:
        path: File to hash.
        chunk: Block size in bytes.

    Returns:
        The lowercase hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def model_status(models_dir: Path, spec: ModelSpec, *, verify_hash: bool = False) -> dict[str, object]:
    """Describe the local state of one model file (for ``models status``).

    Args:
        models_dir: The models folder.
        spec: The model.
        verify_hash: Also compute the SHA-256 (slow for multi-GB files).

    Returns:
        A dict with ``key``, ``role``, ``file``, ``license``, ``expected_size``, ``present``; when present also
        ``size`` and ``size_ok`` (and ``sha256_ok`` if ``verify_hash``); ``partial_bytes`` when an interrupted
        download (``.part`` file) exists.
    """
    path = model_path(models_dir, spec)
    status: dict[str, object] = {"key": spec.key, "role": spec.role, "file": str(path), "license": spec.license,
                                 "expected_size": spec.size, "present": path.exists()}
    if path.exists():
        status["size"] = path.stat().st_size
        status["size_ok"] = path.stat().st_size == spec.size
        if verify_hash:
            status["sha256_ok"] = sha256_file(path) == spec.sha256
    partial = path.with_name(path.name + ".part")
    if partial.exists():
        status["partial_bytes"] = partial.stat().st_size
    return status


def part_files(models_dir: Path, spec: ModelSpec) -> list[Path]:
    """List the stored pieces of a model, in order.

    Args:
        models_dir: The models folder.
        spec: The model.

    Returns:
        ``<models_dir>/parts/<filename>.part001``, ``.part002`` ... (empty when the model has no pieces).
    """
    return sorted((models_dir / PARTS_DIR).glob(f"{spec.filename}.part[0-9][0-9][0-9]"))


def split_model(models_dir: Path, spec: ModelSpec, part_bytes: int = PART_BYTES) -> list[Path]:
    """Cut a verified model file into numbered pieces small enough for GitHub (maintainer tool).

    Existing pieces of the model are replaced. The complete file is kept.

    Args:
        models_dir: The models folder holding the complete model file.
        spec: The model.
        part_bytes: Maximum size of one piece in bytes (default 95 MiB, below GitHub's 100 MB limit).

    Returns:
        The pieces written, in order.

    Raises:
        ModelError: The complete model file is missing or fails its SHA-256 check.
    """
    source = model_path(models_dir, spec)
    if not source.exists() or source.stat().st_size != spec.size or sha256_file(source) != spec.sha256:
        raise ModelError(f"{source} is missing or does not match its SHA-256; fetch it first")
    folder = models_dir / PARTS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    for old in part_files(models_dir, spec):
        old.unlink()
    parts: list[Path] = []
    with source.open("rb") as fh:
        while block := fh.read(part_bytes):
            part = folder / f"{spec.filename}.part{len(parts) + 1:03d}"
            part.write_bytes(block)
            parts.append(part)
    return parts


def assemble_model(models_dir: Path, spec: ModelSpec) -> Path | None:
    """Join a model's stored pieces into the complete file, if the pieces are present.

    The pieces are written to a temporary file that only replaces the target after its SHA-256 matches,
    so an interrupted or corrupt join never leaves a bad model behind.

    Args:
        models_dir: The models folder.
        spec: The model.

    Returns:
        The complete model path (also when it already existed), or None when there are no pieces, or they do
        not add up to the model's size (e.g. an incomplete clone).

    Raises:
        ModelError: The pieces have the right total size but the joined file fails the SHA-256 check.
    """
    final = model_path(models_dir, spec)
    if final.exists() and final.stat().st_size == spec.size:
        return final
    parts = part_files(models_dir, spec)
    if not parts or sum(p.stat().st_size for p in parts) != spec.size:
        return None
    log.info("assembling %s from %d stored pieces (first use only)", spec.filename, len(parts))
    tmp = final.with_name(final.name + ".assembling")
    digest = hashlib.sha256()
    with tmp.open("wb") as out:
        for part in parts:
            with part.open("rb") as fh:
                while block := fh.read(8 * 1024 * 1024):
                    digest.update(block)
                    out.write(block)
    if digest.hexdigest() != spec.sha256:
        tmp.unlink(missing_ok=True)
        raise ModelError(f"the stored pieces of {spec.filename} do not match its SHA-256; "
                         f"re-clone the repository or run: python -m src.main models fetch")
    tmp.replace(final)
    return final


def require_model(models_dir: Path, spec: ModelSpec) -> Path:
    """Return the model path, assembling it from the repository's stored pieces if needed. Never downloads.

    Only the size of an existing file is checked (hashing is too slow for every start-up); a file assembled
    from pieces is SHA-256-verified once, when it is assembled.

    Args:
        models_dir: The models folder.
        spec: The model.

    Returns:
        Path of the complete model file.

    Raises:
        ModelError: The file is missing or its size is wrong.
    """
    path = model_path(models_dir, spec)
    if not path.exists() or path.stat().st_size != spec.size:
        assembled = assemble_model(models_dir, spec)
        if assembled is not None:
            return assembled
        raise ModelError(f"Model file {spec.filename} is missing or incomplete in {models_dir}. "
                         f"Run once (needs internet): python -m src.main models fetch")
    return path


def fetch_model(models_dir: Path, spec: ModelSpec, *, stop_event: threading.Event | None = None,
                progress: Callable[[int, int], None] | None = None, client: httpx.Client | None = None) -> Path:
    """Get one model file: assemble it from the repository's stored pieces, or download (or resume) it; verify SHA-256.

    Bytes are appended to ``<filename>.part`` using HTTP Range requests, resuming after dropped connections;
    the part file is renamed to the final name only after the size and SHA-256 match. An existing complete file
    is only re-verified.

    Args:
        models_dir: The models folder (created if needed).
        spec: The model to fetch.
        stop_event: When set, the download stops between blocks (raised as KeyboardInterrupt); the part file
            is kept for resuming.
        progress: Called as ``progress(bytes_done, bytes_total)`` after each 1 MiB block.
        client: HTTP client to use (for tests); by default a new one with redirects and a user agent is created
            and closed afterwards.

    Returns:
        Path of the verified model file.

    Raises:
        ModelError: HTTP error, no progress for ``MAX_STALLED_ATTEMPTS`` connections in a row, wrong final size,
            or SHA-256 mismatch (a mismatching part file is removed; an
            existing complete file with a wrong hash is left for the user to delete).
        KeyboardInterrupt: ``stop_event`` was set during the download.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    final = model_path(models_dir, spec)
    if assemble_model(models_dir, spec) is not None:  # pieces stored in the repository: no download needed
        return final
    if final.exists() and final.stat().st_size == spec.size:
        log.info("%s already present; verifying checksum", spec.filename)
        if sha256_file(final) == spec.sha256:
            return final
        raise ModelError(f"{final} has the right size but a wrong SHA-256; delete it and fetch again")
    partial = final.with_name(final.name + ".part")
    have = partial.stat().st_size if partial.exists() else 0
    if have > spec.size:
        partial.unlink()
        have = 0
    own_client = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, connect=20.0),
                                    headers={"User-Agent": DOWNLOAD_USER_AGENT})
    stalled = 0
    try:
        while have < spec.size:
            before = have
            headers = {"Range": f"bytes={have}-"} if have else {}
            with client.stream("GET", spec.url, headers=headers) as resp:
                if resp.status_code == 200 and have:
                    have = 0  # server ignored the range; start over
                    partial.unlink(missing_ok=True)
                elif resp.status_code not in (200, 206):
                    raise ModelError(f"download of {spec.filename} failed: HTTP {resp.status_code}")
                with partial.open("ab") as out:
                    for block in resp.iter_bytes(1024 * 1024):
                        if stop_event is not None and stop_event.is_set():
                            raise KeyboardInterrupt
                        out.write(block)
                        have += len(block)
                        if progress:
                            progress(have, spec.size)
            if have < spec.size:
                stalled = stalled + 1 if have <= before else 0
                if stalled >= MAX_STALLED_ATTEMPTS:
                    raise ModelError(f"download of {spec.filename} stopped making progress at {have} of "
                                     f"{spec.size} bytes; try again later (the partial file is kept)")
                log.warning("connection ended early at %d/%d bytes; resuming", have, spec.size)
    finally:
        if own_client:
            client.close()
    if have != spec.size:
        raise ModelError(f"{spec.filename}: expected {spec.size} bytes, got {have}")
    digest = sha256_file(partial)
    if digest != spec.sha256:
        partial.unlink(missing_ok=True)
        raise ModelError(f"{spec.filename}: SHA-256 mismatch ({digest}); the partial file was removed")
    partial.replace(final)
    return final
