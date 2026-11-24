"""MongoDB connection management (local servers only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConfigurationError, InvalidURI, OperationFailure, PyMongoError
from pymongo.uri_parser import parse_uri

from src.config import Settings
from src.errors import ConfigError, DatabaseUnavailableError

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})

START_HINT = (
    "Start MongoDB first:\n"
    "  Windows (service):  net start MongoDB        (portable: python scripts/local_mongodb.py start)\n"
    "  Linux (systemd):    sudo systemctl start mongod\n"
    "  macOS (Homebrew):   brew services start mongodb-community\n"
    "Then check with:      python scripts/verify_environment.py"
)


def assert_local_uri(uri: str) -> None:
    """Refuse anything that is not a local MongoDB server (project rule: no remote databases).

    Args:
        uri: MongoDB connection string.

    Raises:
        ConfigError: The URI uses ``mongodb+srv://``, cannot be parsed, or names a host other than
            ``localhost``, ``127.0.0.1`` or ``::1`` (Unix socket paths are allowed).
    """
    if uri.startswith("mongodb+srv://"):
        raise ConfigError("mongodb+srv:// URIs point to remote clusters and are not allowed; use a local server")
    try:
        parsed = parse_uri(uri, validate=False, warn=True)
    except (InvalidURI, ConfigurationError, ValueError) as exc:
        raise ConfigError(f"Invalid MONGODB_URI: {exc}") from exc
    for host, _port in parsed["nodelist"]:
        if host.endswith(".sock"):
            continue
        if host.lower() not in LOCAL_HOSTS:
            raise ConfigError(
                f"MONGODB_URI host {host!r} is not local. This project only supports a MongoDB "
                "server running on this computer (127.0.0.1 / localhost)."
            )


def build_client(settings: Settings, **extra: Any) -> MongoClient:
    """Create a ``MongoClient`` for the configured local server.

    The client connects lazily: nothing is sent until the first operation (use ``ping`` to check). Both timeouts
    use ``mongodb_timeout_ms`` (milliseconds), returned datetimes are timezone-aware, and credentials are added
    only when ``mongodb_username`` is set.

    Args:
        settings: Application settings (URI, timeout, credentials).
        **extra: Additional ``MongoClient`` keyword arguments; they override the defaults above.

    Returns:
        The new client.
    """
    assert_local_uri(settings.mongodb_uri)
    kwargs: dict[str, Any] = {
        "serverSelectionTimeoutMS": settings.mongodb_timeout_ms,
        "connectTimeoutMS": settings.mongodb_timeout_ms,
        "appname": "imsdb-local",
        "tz_aware": True,
        "uuidRepresentation": "standard",
    }
    if settings.mongodb_username:
        kwargs["username"] = settings.mongodb_username
        kwargs["password"] = settings.mongodb_password.get_secret_value() if settings.mongodb_password else None
        kwargs["authSource"] = settings.mongodb_auth_source
    kwargs.update(extra)
    return MongoClient(settings.mongodb_uri, **kwargs)


def ping(client: MongoClient) -> dict[str, Any]:
    """Verify the server is reachable and return basic server info.

    Args:
        client: Client to check.

    Returns:
        ``{"version": ..., "git_version": ...}`` as reported by the server.

    Raises:
        DatabaseUnavailableError: The server rejected the credentials, or could not be reached (the message
            then explains how to start MongoDB).
    """
    try:
        client.admin.command("ping")
        info = client.server_info()
    except OperationFailure as exc:
        raise DatabaseUnavailableError(
            f"MongoDB rejected the connection ({exc.code_name or exc.code}): check MONGODB_USERNAME/"
            "MONGODB_PASSWORD/MONGODB_AUTH_SOURCE."
        ) from exc
    except PyMongoError as exc:
        raise DatabaseUnavailableError(f"MongoDB is not reachable: {exc.__class__.__name__}.\n{START_HINT}") from exc
    return {"version": info.get("version"), "git_version": info.get("gitVersion")}


@dataclass
class Mongo:
    """Owns the client and exposes the configured database.

    Attributes:
        client: Connected PyMongo client.
        db: Application database on that client.
        server_version: MongoDB server version reported by ``ping``, or None when the connection was not checked.
    """

    client: MongoClient
    db: Database
    server_version: str | None = None

    @classmethod
    def connect(cls, settings: Settings, *, db_name: str | None = None, check: bool = True) -> "Mongo":
        """Connect to the local MongoDB server.

        Args:
            settings: Application settings.
            db_name: Database to use instead of ``settings.mongodb_db``, or None.
            check: Ping the server first; on failure the client is closed and the error re-raised.

        Returns:
            A ``Mongo`` bound to the database.
        """
        client = build_client(settings)
        version = None
        if check:
            try:
                version = ping(client)["version"]
            except Exception:
                client.close()
                raise
        return cls(client=client, db=client[db_name or settings.mongodb_db], server_version=version)

    def close(self) -> None:
        """Close the client and its connection pool."""
        self.client.close()

    def __enter__(self) -> "Mongo":
        """Use the connection as a context manager.

        Returns:
            This object.
        """
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the client when the ``with`` block exits; exceptions are not suppressed.

        Args:
            *exc: Exception type, value and traceback (ignored).
        """
        self.close()
