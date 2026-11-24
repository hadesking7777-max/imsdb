"""Backup integrity, verification, restore and CLI exit-code tests."""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path

import pytest

from src.database.connection import Mongo
from src.database.models import Collections as C
from src.main import main
from src.services.backup import create_backup, find_tool, restore_backup, verify_backup
from src.services.ingestion import IngestionService
from tests.conftest import FakeSite, make_settings

pytestmark = pytest.mark.mongodb


def _ingest(app) -> None:
    """Fill the test database: discover the fake site and crawl all five movie records.

    Args:
        app: The ingestion ``AppContext`` to run against.
    """
    service = IngestionService(app)
    service.discover()
    service.crawl(limit=5)


def _counts(db) -> dict[str, int]:
    """Count the documents in the record collections that a backup must preserve.

    Args:
        db: A pymongo ``Database``.

    Returns:
        Document count per collection name (movies, scripts, scenes, dialogues, characters).
    """
    return {name: db[name].count_documents({}) for name in (C.MOVIES, C.SCRIPTS, C.SCENES, C.DIALOGUES, C.CHARACTERS)}


def test_python_backup_verify_and_restore(app, site: FakeSite, tmp_path: Path) -> None:
    """A Python (BSON) backup verifies, restores into an empty database and raw folder, and re-restoring is a no-op.

    Args:
        app: Ingestion context on a throwaway database, fetching from the fake site.
        site: The fake IMSDb behind ``app`` (requested so its pages are served).
        tmp_path: pytest's per-test temporary folder (holds the restore target's data directory).
    """
    _ingest(app)
    result = create_backup(app.settings, app.mongo, method="python")
    directory = Path(result["directory"])
    assert result["method"] == "python-bson" and result["verification"]["ok"]
    assert (directory / "raw-archive.zip").exists() and (directory / "manifest.json.sha256").exists()
    assert result["collections"][C.DIALOGUES] == 16
    assert verify_backup(directory)["ok"]

    # Restore into a fresh database and an empty raw directory.
    target = make_settings(tmp_path / "restore", mongodb_db=app.settings.mongodb_db + "_r")
    try:
        restored = restore_backup(directory, target)
        with Mongo.connect(target) as mongo:
            assert _counts(mongo.db) == _counts(app.db)
            assert "uniq_detail_url" in mongo.db[C.MOVIES].index_information()  # indexes rebuilt
        assert restored["raw"]["restored"] > 0 and restored["raw"]["conflicts_preserved"] == 0
        original_files = sorted(p.relative_to(app.settings.raw_dir) for p in app.settings.raw_dir.rglob("*.html"))
        restored_files = sorted(p.relative_to(target.raw_dir) for p in target.raw_dir.rglob("*.html"))
        assert original_files == restored_files

        again = restore_backup(directory, target)  # idempotent: existing documents/files are kept
        assert sum(again["inserted"].values()) == 0 and again["raw"]["identical_skipped"] > 0
    finally:
        with Mongo.connect(target, check=False) as mongo:
            mongo.client.drop_database(target.mongodb_db)


def test_tampering_is_detected_and_blocks_restore(app, site: FakeSite) -> None:
    """Flipping one byte in a backed-up BSON file fails verification and makes restore raise ``BackupError``.

    Args:
        app: Ingestion context on a throwaway database, fetching from the fake site.
        site: The fake IMSDb behind ``app`` (requested so its pages are served).
    """
    _ingest(app)
    directory = Path(create_backup(app.settings, app.mongo, method="python")["directory"])
    bson_file = next((directory / "mongodb-bson").rglob("scripts.bson"))
    data = bytearray(bson_file.read_bytes())
    data[len(data) // 2] ^= 0xFF
    bson_file.write_bytes(bytes(data))
    result = verify_backup(directory)
    assert not result["ok"] and any("checksum mismatch" in p for p in result["problems"])
    from src.errors import BackupError
    with pytest.raises(BackupError):
        restore_backup(directory, app.settings)


def test_restore_never_overwrites_differing_raw_files(app, site: FakeSite) -> None:
    """A locally modified raw file is left untouched; the backup copy goes to ``_restore_conflicts``.

    Args:
        app: Ingestion context on a throwaway database, fetching from the fake site.
        site: The fake IMSDb behind ``app`` (requested so its pages are served).
    """
    _ingest(app)
    directory = Path(create_backup(app.settings, app.mongo, method="python")["directory"])
    victim = next(app.settings.raw_dir.rglob("*.html"))
    victim.write_bytes(b"locally modified")
    result = restore_backup(directory, app.settings)
    assert result["raw"]["conflicts_preserved"] == 1
    assert victim.read_bytes() == b"locally modified"  # untouched
    assert list((app.settings.raw_dir / "_restore_conflicts").rglob(victim.name))


def _tools_dir() -> Path | None:
    """Locate the MongoDB Database Tools (``mongodump``/``mongorestore``) for the optional tool-based test.

    Returns:
        ``TEST_MONGODB_TOOLS_DIR`` if it exists, else the folder of ``mongodump`` on PATH, else ``None``.
    """
    configured = os.environ.get("TEST_MONGODB_TOOLS_DIR")
    if configured and Path(configured).exists():
        return Path(configured)
    found = shutil.which("mongodump")
    return Path(found).parent if found else None


@pytest.mark.skipif(_tools_dir() is None, reason="MongoDB Database Tools not installed (set TEST_MONGODB_TOOLS_DIR)")
def test_mongodump_backup_and_restore(tmp_path: Path, site: FakeSite) -> None:
    """A ``mongodump`` backup restores into a different database name with identical record counts.

    Skipped when the MongoDB Database Tools are not installed.

    Args:
        tmp_path: pytest's per-test temporary folder for the test's own settings.
        site: The fake IMSDb used to fill the database.
    """
    import threading

    from src.crawler.fetcher import Fetcher, RateLimiter
    from src.services.ingestion import AppContext

    settings = make_settings(tmp_path, mongodb_tools_dir=_tools_dir())
    with Mongo.connect(settings) as mongo:
        try:
            from src.database.indexes import ensure_schema
            ensure_schema(mongo.db)
            stop = threading.Event()
            fetcher = Fetcher(settings, rate_limiter=RateLimiter(0.0), stop_event=stop, transport=site.transport())
            app = AppContext(settings, mongo, stop_event=stop, fetcher=fetcher)
            _ingest(app)
            assert find_tool("mongodump", settings) is not None
            result = create_backup(settings, mongo, method="mongodump")
            directory = Path(result["directory"])
            assert result["method"] == "mongodump" and (directory / "mongodb.archive.gz").exists()
            restored = restore_backup(directory, settings, target_db=settings.mongodb_db + "_md")
            assert restored["target_db"].endswith("_md")
            restored_db = mongo.client[settings.mongodb_db + "_md"]
            assert _counts(restored_db) == _counts(mongo.db)
        finally:
            mongo.client.drop_database(settings.mongodb_db)
            mongo.client.drop_database(settings.mongodb_db + "_md")


# ====================================================================== CLI
def _env_file(tmp_path: Path, settings, **extra: str) -> Path:
    """Write a ``.env`` file that points the CLI at the test database and folders.

    Args:
        tmp_path: Folder in which ``test.env`` is written.
        settings: Settings whose database name and data/log directories are copied.
        **extra: Additional ``NAME=value`` entries (they override the copied ones).

    Returns:
        Path of the written file.
    """
    values = {"MONGODB_DB": settings.mongodb_db, "IMSDB_DATA_DIR": str(settings.data_dir),
              "IMSDB_LOG_DIR": str(settings.log_dir), **extra}
    path = tmp_path / "test.env"
    path.write_text("\n".join(f"{k}={v}" for k, v in values.items()) + "\n", encoding="utf-8")
    return path


def test_cli_exit_codes(app, site: FakeSite, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI commands return the documented exit codes (0 ok, 2 usage, 3 validation, 4 permission, 7 backup).

    Args:
        app: Ingestion context on a throwaway database, fetching from the fake site.
        site: The fake IMSDb behind ``app`` (requested so its pages are served).
        tmp_path: Folder for the ``.env`` file.
        capsys: Captures the CLI's printed output.
    """
    _ingest(app)
    env = str(_env_file(tmp_path, app.settings))
    assert main(["--env-file", env, "stats"]) == 0
    assert "Discovered movie records" in capsys.readouterr().out
    assert main(["--env-file", env, "stats", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["records"]["discovered_movies"] == 5
    assert main(["--env-file", env, "validate"]) == 0
    assert "Validation PASSED" in capsys.readouterr().out
    assert main(["--env-file", env, "export", "--format", "jsonl"]) == 0
    assert main(["--env-file", env, "crawl"]) == 2  # neither --limit nor --all
    assert main(["--env-file", env, "crawl", "--all"]) == 4  # bulk not authorized (robots cached: no network)
    assert main(["--env-file", env, "permit", "--level", "bulk"]) == 2  # requires --confirm
    assert main(["--env-file", env, "permit", "--level", "bulk", "--confirm"]) == 4  # requires evidence
    assert main(["--env-file", env, "backup", "--method", "python", "--no-raw"]) == 0
    out = capsys.readouterr().out
    backup_dir = next(line.split(": ", 1)[1] for line in out.splitlines() if line.startswith("Backup created"))
    assert main(["--env-file", env, "backup", "--verify", backup_dir]) == 0
    manifest = Path(backup_dir) / "manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8").replace('"bytes"', '"bytes" ', 1), encoding="utf-8")
    assert main(["--env-file", env, "backup", "--verify", backup_dir]) == 7  # manifest tampering detected
    app.db[C.SCENES].delete_many({})
    assert main(["--env-file", env, "validate"]) == 3


def test_cli_reports_unreachable_database(tmp_path: Path) -> None:
    """The CLI exits with 5 when the configured MongoDB cannot be reached.

    Args:
        tmp_path: Folder for the settings and ``.env`` file.
    """
    settings = make_settings(tmp_path)
    env = _env_file(tmp_path, settings, MONGODB_URI="mongodb://127.0.0.1:1/", MONGODB_TIMEOUT_MS="300")
    assert main(["--env-file", str(env), "stats"]) == 5


def test_cli_rejects_remote_database(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI refuses a non-local MongoDB URI with exit code 2 and a "not local" message.

    Args:
        tmp_path: Folder for the settings and ``.env`` file.
        capsys: Captures the error message on stderr.
    """
    settings = make_settings(tmp_path)
    env = _env_file(tmp_path, settings, MONGODB_URI="mongodb://db.example.org:27017/")
    assert main(["--env-file", str(env), "stats"]) == 2
    assert "not local" in capsys.readouterr().err


def test_zip_archives_are_valid(app, site: FakeSite) -> None:
    """The raw archive zip opens and contains both the fetch manifest and the HTML bodies.

    Args:
        app: Ingestion context on a throwaway database, fetching from the fake site.
        site: The fake IMSDb behind ``app`` (requested so its pages are served).
    """
    _ingest(app)
    directory = Path(create_backup(app.settings, app.mongo, method="python")["directory"])
    with zipfile.ZipFile(directory / "raw-archive.zip") as zf:
        names = zf.namelist()
    assert any(n.startswith("_manifest/") for n in names)  # fetch manifest travels with raw bodies
    assert any(n.endswith(".html") for n in names)
