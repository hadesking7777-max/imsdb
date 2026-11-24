"""Start / stop / check a *portable* (non-service) local mongod.

For installations made with the official installer/package manager, use the OS service
manager instead (see README). This helper is for ZIP/tarball installs without admin rights.

Configuration (.env):
    MONGOD_PATH    full path to the mongod executable
    MONGOD_CONFIG  path to a mongod config file (must bind to 127.0.0.1)

Usage:  python scripts/local_mongodb.py start|stop|status
        (no argument or an unknown command means ``status``)

Exit codes: 0 on success (``status``: server reachable), 1 on failure (``status``: not reachable).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from src.config import load_settings  # noqa: E402
from src.database.connection import build_client  # noqa: E402


def _reachable(timeout_ms: int = 1500) -> str | None:
    """Ping the MongoDB server configured in settings.

    Args:
        timeout_ms: Server selection timeout in milliseconds.

    Returns:
        The server version string if the server answers, otherwise ``None`` (errors are swallowed).
    """
    settings = load_settings(mongodb_timeout_ms=timeout_ms)
    client = build_client(settings)
    try:
        client.admin.command("ping")
        return client.server_info().get("version")
    except Exception:  # noqa: BLE001
        return None
    finally:
        client.close()


def start() -> int:
    """Start mongod from ``MONGOD_PATH`` as a detached background process and wait up to 30 s for it.

    Uses ``--config MONGOD_CONFIG`` when set, else ``--bind_ip 127.0.0.1``. Does nothing if a server
    already answers.

    Returns:
        0 if the server is (or already was) reachable; 1 if ``MONGOD_PATH`` is unset/missing, mongod exits
        early, or it is still unreachable after 30 s.
    """
    if (version := _reachable()) is not None:
        print(f"MongoDB {version} is already running.")
        return 0
    mongod = os.environ.get("MONGOD_PATH")
    config = os.environ.get("MONGOD_CONFIG")
    if not mongod or not Path(mongod).exists():
        print("Set MONGOD_PATH in .env to your mongod executable.", file=sys.stderr)
        return 1
    args = [mongod] + (["--config", config] if config else ["--bind_ip", "127.0.0.1"])
    kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL, "stdin": subprocess.DEVNULL}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(args, **kwargs)
    for _ in range(30):
        time.sleep(1)
        if (version := _reachable()) is not None:
            print(f"MongoDB {version} started (pid {proc.pid}).")
            return 0
        if proc.poll() is not None:
            print(f"mongod exited with code {proc.returncode}; check its log file.", file=sys.stderr)
            return 1
    print("mongod did not become reachable within 30 s.", file=sys.stderr)
    return 1


def stop() -> int:
    """Shut the configured MongoDB server down with the ``shutdown`` admin command.

    Returns:
        0 once the server is gone (the command dropping the connection is expected) or when none was running
        (a message says so); 1 if it is still reachable.
    """
    if _reachable() is None:
        print("MongoDB is not running; nothing to stop.")
        return 0
    settings = load_settings()
    client = build_client(settings)
    try:
        client.admin.command("shutdown")
    except Exception as exc:  # noqa: BLE001 - shutdown closes the connection by design
        if _reachable() is not None:
            print(f"Shutdown failed: {exc}", file=sys.stderr)
            return 1
    finally:
        client.close()
    print("MongoDB stopped.")
    return 0


def status() -> int:
    """Print whether the configured MongoDB server is reachable.

    Returns:
        0 if reachable, 1 otherwise.
    """
    version = _reachable()
    print(f"MongoDB {version} is running." if version else "MongoDB is not reachable.")
    return 0 if version else 1


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    sys.exit({"start": start, "stop": stop, "status": status}.get(command, status)())
