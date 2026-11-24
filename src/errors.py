"""Application exceptions and CLI exit codes."""

from __future__ import annotations

from enum import IntEnum


class ExitCode(IntEnum):
    """Process exit codes returned by the CLI.

    Attributes:
        OK: Success.
        ERROR: Unexpected or generic failure.
        USAGE: Invalid command-line usage or configuration.
        VALIDATION_FAILED: ``validate`` found errors.
        PERMISSION_DENIED: The source permission level does not allow the action.
        DATABASE_UNAVAILABLE: MongoDB is unreachable or rejected the credentials.
        NETWORK_ERROR: A network failure, or access restricted by the source.
        BACKUP_FAILED: Creating or verifying a backup failed.
        INTERRUPTED: Stopped by Ctrl+C or a shutdown signal (128 + SIGINT).
    """
    OK = 0
    ERROR = 1
    USAGE = 2
    VALIDATION_FAILED = 3
    PERMISSION_DENIED = 4
    DATABASE_UNAVAILABLE = 5
    NETWORK_ERROR = 6
    BACKUP_FAILED = 7
    INTERRUPTED = 130


class IMSDbError(Exception):
    """Base class for expected, user-facing errors.

    Attributes:
        exit_code: Exit code the CLI returns when this error ends a command.
    """

    exit_code: ExitCode = ExitCode.ERROR


class ConfigError(IMSDbError):
    """Invalid configuration or command-line usage (exit code 2)."""
    exit_code = ExitCode.USAGE


class DatabaseUnavailableError(IMSDbError):
    """MongoDB cannot be reached or rejected the connection (exit code 5)."""
    exit_code = ExitCode.DATABASE_UNAVAILABLE


class PermissionDeniedError(IMSDbError):
    """The source permission level does not allow the requested action (exit code 4)."""
    exit_code = ExitCode.PERMISSION_DENIED


class BackupError(IMSDbError):
    """Creating, verifying or restoring a backup failed (exit code 7)."""
    exit_code = ExitCode.BACKUP_FAILED


class CrawlerBusyError(IMSDbError):
    """Another crawler process holds the crawl lock."""


class FetchError(IMSDbError):
    """An HTTP fetch failed.

    ``retryable`` tells the queue whether to reschedule the URL. ``blocked`` marks
    access-control responses (401/403/451), which the crawler must never try to bypass.

    Attributes:
        url: URL that failed, or None.
        status_code: HTTP status code, or None for network errors.
        retryable: Whether the failure is transient and the URL may be retried.
        blocked: Whether the server denied access.
        retry_after: Delay requested by the server (``Retry-After``) in seconds, or None.
        attempts: Number of attempts made before giving up.
    """

    exit_code = ExitCode.NETWORK_ERROR

    def __init__(
        self,
        message: str,
        *,
        url: str | None = None,
        status_code: int | None = None,
        retryable: bool = False,
        blocked: bool = False,
        retry_after: float | None = None,
        attempts: int = 0,
    ) -> None:
        """Create the error.

        Args:
            message: Human-readable description.
            url: URL that failed, or None.
            status_code: HTTP status code, or None for network errors.
            retryable: Whether the failure is transient.
            blocked: Whether the server denied access (401/403/451).
            retry_after: Delay requested by the server in seconds, or None.
            attempts: Number of attempts made.
        """
        super().__init__(message)
        self.url = url
        self.status_code = status_code
        self.retryable = retryable
        self.blocked = blocked
        self.retry_after = retry_after
        self.attempts = attempts


class FetchCancelled(IMSDbError):
    """Raised when a shutdown was requested while waiting to fetch."""

    exit_code = ExitCode.INTERRUPTED
