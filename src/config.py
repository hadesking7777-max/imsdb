"""Application configuration loaded from environment variables / ``.env``.

All paths are resolved relative to the project root unless absolute. Secrets are kept
in :class:`pydantic.SecretStr` so they never appear in logs or reprs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

from src.errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Environment variable -> settings field
_ENV_MAP: dict[str, str] = {
    "MONGODB_URI": "mongodb_uri",
    "MONGODB_DB": "mongodb_db",
    "MONGODB_USERNAME": "mongodb_username",
    "MONGODB_PASSWORD": "mongodb_password",
    "MONGODB_AUTH_SOURCE": "mongodb_auth_source",
    "MONGODB_TIMEOUT_MS": "mongodb_timeout_ms",
    "MONGODB_TOOLS_DIR": "mongodb_tools_dir",
    "IMSDB_DATA_DIR": "data_dir",
    "IMSDB_LOG_DIR": "log_dir",
    "IMSDB_RAW_GZIP": "raw_gzip",
    "IMSDB_LOG_LEVEL": "log_level",
    "IMSDB_BASE_URL": "base_url",
    "IMSDB_USER_AGENT": "user_agent",
    "IMSDB_MIN_REQUEST_INTERVAL": "min_request_interval",
    "IMSDB_CONCURRENCY": "concurrency",
    "IMSDB_REQUEST_TIMEOUT": "request_timeout",
    "IMSDB_CONNECT_TIMEOUT": "connect_timeout",
    "IMSDB_MAX_RETRIES": "max_retries",
    "IMSDB_BACKOFF_BASE": "backoff_base",
    "IMSDB_BACKOFF_MAX": "backoff_max",
    "IMSDB_MAX_RETRY_AFTER": "max_retry_after",
    "IMSDB_MAX_QUEUE_ATTEMPTS": "max_queue_attempts",
    "IMSDB_LEASE_SECONDS": "lease_seconds",
    "IMSDB_MAX_RESPONSE_BYTES": "max_response_bytes",
    "IMSDB_MODELS_DIR": "models_dir",
    "IMSDB_CHAT_MODEL": "chat_model",
    "IMSDB_EMBED_MODEL": "embed_model",
    "IMSDB_LLM_THREADS": "llm_threads",
    "IMSDB_LLM_CONTEXT": "llm_context",
    "IMSDB_CHAT_HOST": "chat_host",
    "IMSDB_CHAT_PORT": "chat_port",
    "IMSDB_MSVC_RUNTIME_DIR": "msvc_runtime_dir",
    "IMSDB_LLM_DEVICE": "llm_device",
    "IMSDB_LLM_GPU_LAYERS": "llm_gpu_layers",
    "IMSDB_GPU_RUNTIME_DIR": "gpu_runtime_dir",
    "IMSDB_GPU_BACKENDS": "gpu_backends",
}

MIN_ALLOWED_INTERVAL = 1.0  # politeness floor, cannot be configured lower


class Settings(BaseModel):
    """Validated, immutable application settings; build them with ``load_settings``.

    Relative paths are resolved against the project root.

    Attributes:
        mongodb_uri: Connection string of the local MongoDB server (remote hosts are refused when connecting).
        mongodb_db: Database name.
        mongodb_username: User to authenticate as, or None to connect without credentials.
        mongodb_password: Password of that user (kept secret), or None.
        mongodb_auth_source: Database that holds the user's credentials.
        mongodb_timeout_ms: Server-selection and connect timeout in milliseconds (100-120000).
        mongodb_tools_dir: Folder with ``mongodump``/``mongorestore`` for backups, or None to look elsewhere.
        data_dir: Root of the local data folder (raw archive, processed files, exports, backups, models).
        log_dir: Folder for log files.
        raw_gzip: Store raw HTML bodies gzip-compressed.
        log_level: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR`` or ``CRITICAL``.
        base_url: IMSDb base URL, always ending with ``/``.
        user_agent: ``User-Agent`` header sent with every request.
        min_request_interval: Minimum seconds between requests (never below ``MIN_ALLOWED_INTERVAL``).
        concurrency: Number of concurrent crawler workers (1-8).
        request_timeout: HTTP timeout in seconds for reading, writing and pool waits.
        connect_timeout: HTTP connect timeout in seconds.
        max_retries: HTTP retries per fetch after the first attempt (0-10).
        backoff_base: Base delay in seconds of the exponential HTTP retry back-off.
        backoff_max: Longest HTTP retry back-off delay in seconds.
        max_retry_after: Longest server ``Retry-After`` delay honoured, in seconds.
        max_queue_attempts: Attempts per queue item before it is marked ``failed``.
        lease_seconds: How long a claimed queue item stays leased to a worker, in seconds.
        max_response_bytes: Largest response body accepted, in bytes.
        models_dir: Folder with the local model files of the chatbot.
        chat_model: Key of the chat model to load.
        embed_model: Key of the embedding model to load.
        llm_threads: CPU threads used for inference; 0 means the estimated number of physical cores.
        llm_context: Model context window in tokens (2048-32768).
        chat_host: Address the chatbot server binds to; only loopback addresses are accepted.
        chat_port: Port of the chatbot server.
        msvc_runtime_dir: Folder with the app-local MSVC runtime DLLs, or None.
        llm_device: ``auto`` (GPU when usable, otherwise CPU), ``gpu`` (GPU or fail) or ``cpu``.
        llm_gpu_layers: Model layers to offload to the GPU; -1 means as many as fit in GPU memory.
        gpu_runtime_dir: Folder holding the GPU builds as ``<backend>/llama_cpp``.
        gpu_backends: Comma-separated GPU builds to try, in order (``cuda``, ``vulkan``).
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    # MongoDB
    mongodb_uri: str = "mongodb://127.0.0.1:27017/"
    mongodb_db: str = "imsdb_local"
    mongodb_username: str | None = None
    mongodb_password: SecretStr | None = None
    mongodb_auth_source: str = "admin"
    mongodb_timeout_ms: int = Field(default=5000, ge=100, le=120_000)
    mongodb_tools_dir: Path | None = None

    # Storage
    data_dir: Path = PROJECT_ROOT / "data"
    log_dir: Path = PROJECT_ROOT / "logs"
    raw_gzip: bool = False

    # Logging
    log_level: str = "INFO"

    # Crawling
    base_url: str = "https://imsdb.com/"
    user_agent: str = "imsdb-local-research/1.0 (educational research archive; local use)"
    min_request_interval: float = Field(default=5.0, ge=MIN_ALLOWED_INTERVAL)
    concurrency: int = Field(default=2, ge=1, le=8)
    request_timeout: float = Field(default=30.0, gt=0, le=300)
    connect_timeout: float = Field(default=10.0, gt=0, le=120)
    max_retries: int = Field(default=4, ge=0, le=10)
    backoff_base: float = Field(default=2.0, gt=0, le=60)
    backoff_max: float = Field(default=120.0, gt=0, le=3600)
    max_retry_after: float = Field(default=600.0, ge=0, le=86_400)
    max_queue_attempts: int = Field(default=5, ge=1, le=50)
    lease_seconds: int = Field(default=1800, ge=60, le=86_400)
    max_response_bytes: int = Field(default=20_000_000, ge=10_000)

    # Offline chatbot (local models only)
    models_dir: Path = PROJECT_ROOT / "data" / "models"
    chat_model: str = "qwen3-8b-q4km"
    embed_model: str = "qwen3-embedding-0.6b-q8"
    llm_threads: int = Field(default=0, ge=0, le=256)  # 0 = number of physical cores (estimated)
    llm_context: int = Field(default=8192, ge=2048, le=32768)
    chat_host: str = "127.0.0.1"
    chat_port: int = Field(default=8765, ge=1, le=65535)
    msvc_runtime_dir: Path | None = PROJECT_ROOT / "runtime" / "win64"
    # Where the models run. "auto" (default) = GPU when one is usable, otherwise CPU; "gpu" = GPU or fail;
    # "cpu" = never try the GPU. See src/chatbot/device.py.
    llm_device: Literal["auto", "gpu", "cpu"] = "auto"
    llm_gpu_layers: int = Field(default=-1, ge=-1, le=999)  # -1 = as many layers as fit in GPU memory
    gpu_runtime_dir: Path = PROJECT_ROOT / "runtime" / "gpu"  # holds <backend>/llama_cpp (GPU builds)
    gpu_backends: str = "cuda,vulkan"  # GPU builds to try, in order (only installed ones are tried)

    @field_validator("chat_host")
    @classmethod
    def _local_host_only(cls, value: str) -> str:
        """Restrict the chatbot server to this computer.

        Args:
            value: Configured ``chat_host``.

        Returns:
            The value unchanged.

        Raises:
            ValueError: The host is not ``127.0.0.1``, ``localhost`` or ``::1``.
        """
        if value not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("the chatbot binds to this computer only (127.0.0.1, localhost or ::1)")
        return value

    @field_validator("mongodb_username", "mongodb_tools_dir", "msvc_runtime_dir", mode="before")
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        """Treat empty or whitespace-only strings as "not set" for optional fields.

        Applies to ``mongodb_username``, ``mongodb_tools_dir`` and ``msvc_runtime_dir``.

        Args:
            value: Raw value from the environment or a keyword argument.

        Returns:
            None for a blank string, otherwise the value unchanged.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("mongodb_password", mode="before")
    @classmethod
    def _blank_password(cls, value: Any) -> Any:
        """Treat an empty password as no password (whitespace is kept, it may be a real password).

        Args:
            value: Raw value from the environment or a keyword argument.

        Returns:
            None for an empty string, otherwise the value unchanged.
        """
        if isinstance(value, str) and not value:
            return None
        return value

    @field_validator("llm_device", mode="before")
    @classmethod
    def _lower_device(cls, value: Any) -> Any:
        """Accept the device name in any case (``GPU``, ``Cpu`` ...).

        Args:
            value: Raw value from the environment or a keyword argument.

        Returns:
            The value lower-cased and stripped when it is a string, otherwise unchanged.
        """
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("gpu_backends")
    @classmethod
    def _check_backends(cls, value: str) -> str:
        """Validate the comma-separated list of GPU backends.

        Args:
            value: E.g. ``"cuda,vulkan"``. Spaces and case are ignored.

        Returns:
            The normalised list, e.g. ``"cuda,vulkan"``.

        Raises:
            ValueError: If a name other than ``cuda`` or ``vulkan`` is given.
        """
        names = [v.strip().lower() for v in value.split(",") if v.strip()]
        unknown = sorted(set(names) - {"cuda", "vulkan"})
        if unknown:
            raise ValueError(f"unknown GPU backend(s) {unknown}; use cuda and/or vulkan")
        return ",".join(names)

    @field_validator("data_dir", "log_dir", "mongodb_tools_dir", "models_dir", "msvc_runtime_dir",
                     "gpu_runtime_dir", mode="after")
    @classmethod
    def _resolve_path(cls, value: Path | None) -> Path | None:
        """Make a relative path absolute, relative to the project root.

        Args:
            value: Configured path, or None.

        Returns:
            The absolute path, or None when no path is set.
        """
        if value is None:
            return None
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @field_validator("log_level")
    @classmethod
    def _check_level(cls, value: str) -> str:
        """Normalise and validate the log level.

        Args:
            value: Level name in any case.

        Returns:
            The upper-cased level name.

        Raises:
            ValueError: Not one of ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR`` or ``CRITICAL``.
        """
        value = value.upper().strip()
        if value not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"invalid log level {value!r}")
        return value

    @field_validator("base_url")
    @classmethod
    def _check_base(cls, value: str) -> str:
        """Validate the base URL and make sure it ends with a slash.

        Args:
            value: Configured base URL.

        Returns:
            The URL, ending with ``/``.

        Raises:
            ValueError: The URL does not start with ``http://`` or ``https://``.
        """
        if not value.startswith(("http://", "https://")):
            raise ValueError("base_url must be an http(s) URL")
        return value if value.endswith("/") else value + "/"

    # ---------------------------------------------------------------- derived paths
    @property
    def raw_dir(self) -> Path:
        """Folder of the raw HTTP response archive.

        Returns:
            ``<data_dir>/raw``.
        """
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        """Folder for processed files (e.g. script elements offloaded from oversized documents).

        Returns:
            ``<data_dir>/processed``.
        """
        return self.data_dir / "processed"

    @property
    def export_dir(self) -> Path:
        """Folder for exports.

        Returns:
            ``<data_dir>/exports``.
        """
        return self.data_dir / "exports"

    @property
    def backup_dir(self) -> Path:
        """Folder for backups.

        Returns:
            ``<data_dir>/backups``.
        """
        return self.data_dir / "backups"

    @property
    def report_dir(self) -> Path:
        """Folder for validation reports.

        Returns:
            ``<data_dir>/exports/reports``.
        """
        return self.export_dir / "reports"

    @property
    def index_dir(self) -> Path:
        """Folder reserved for index files.

        Returns:
            ``<data_dir>/index``.
        """
        return self.data_dir / "index"

    def ensure_directories(self) -> None:
        """Create the raw, processed, export, backup, report and log folders if they are missing."""
        for path in (self.raw_dir, self.processed_dir, self.export_dir, self.backup_dir,
                     self.report_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)

    def safe_dict(self) -> dict[str, Any]:
        """Settings snapshot without secrets (for manifests and logs).

        Returns:
            All settings as JSON-compatible values, with the password replaced by ``***`` (or None when unset).
        """
        data = self.model_dump(mode="json")
        data["mongodb_password"] = "***" if self.mongodb_password else None
        return data


def load_settings(env_file: str | os.PathLike[str] | None = None, **overrides: Any) -> Settings:
    """Load settings from ``.env`` (if present) and the process environment.

    Variables already set in the environment win over the ``.env`` file. An empty variable means "use the
    default", except for the username, password, tools folder and MSVC runtime folder, where it means "not set".
    Explicit ``overrides`` take precedence over the environment (used by tests).

    Args:
        env_file: Path of the ``.env`` file; None uses ``<project root>/.env``.
        **overrides: Settings field values that replace the environment values.

    Returns:
        The validated settings.

    Raises:
        ConfigError: A value is invalid (every problem is listed in the message).
    """
    path = Path(env_file) if env_file else PROJECT_ROOT / ".env"
    if path.exists():
        load_dotenv(path, override=False)

    values: dict[str, Any] = {}
    for env_name, field_name in _ENV_MAP.items():
        raw = os.environ.get(env_name)
        if raw is not None and raw != "":
            values[field_name] = raw
        elif raw == "" and field_name in {"mongodb_username", "mongodb_password", "mongodb_tools_dir",
                                          "msvc_runtime_dir"}:
            values[field_name] = None
    values.update(overrides)
    try:
        return Settings(**values)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ConfigError(f"Invalid configuration: {problems}") from exc
