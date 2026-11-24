"""Local backup, verification and restore.

Backup layout (``data/backups/backup-<UTC stamp>/``)::

    mongodb.archive.gz         mongodump --archive --gzip         (when MongoDB Database Tools exist)
    mongodb-bson/<db>/*.bson   mongorestore-compatible BSON dump   (pure-Python fallback)
    raw-archive.zip            data/raw (original HTTP bodies + manifests)
    processed-archive.zip      data/processed (offloaded large arrays), if any
    manifest.json              counts, sizes, SHA-256 of every file, tool versions, restore steps
    manifest.json.sha256       checksum of the manifest itself

Nothing is uploaded anywhere. Restore never overwrites existing raw files; conflicting
files are written to ``data/raw/_restore_conflicts/<stamp>/`` instead.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import bson
from bson import json_util
from pymongo.database import Database
from pymongo.errors import BulkWriteError

from src.config import Settings
from src.database.connection import Mongo
from src.database.indexes import ensure_schema
from src.errors import BackupError
from src.utils.clock import stamp, utcnow
from src.utils.hashing import sha256_file
from src.utils.logging import get_logger

log = get_logger("backup")

ARCHIVE = "mongodb.archive.gz"
BSON_DIR = "mongodb-bson"
RAW_ZIP = "raw-archive.zip"
PROCESSED_ZIP = "processed-archive.zip"
MANIFEST = "manifest.json"


# ====================================================================== tools
def find_tool(name: str, settings: Settings) -> Path | None:
    """Locate a MongoDB Database Tools executable.

    Looks in ``settings.mongodb_tools_dir`` first, then on ``PATH``.

    Args:
        name: Tool name without extension, e.g. ``"mongodump"`` (``.exe`` is added on Windows).
        settings: Application settings.

    Returns:
        Path to the executable, or None when it is not installed.
    """
    exe = name + (".exe" if os.name == "nt" else "")
    if settings.mongodb_tools_dir:
        candidate = settings.mongodb_tools_dir / exe
        if candidate.exists():
            return candidate
    found = shutil.which(name)
    return Path(found) if found else None


def _tool_version(tool: Path) -> str | None:
    """Ask a tool for its version (runs ``<tool> --version`` with a 30 s timeout).

    Args:
        tool: Path to the executable.

    Returns:
        The first line of its output, or None when it cannot be run or prints nothing.
    """
    try:
        out = subprocess.run([str(tool), "--version"], capture_output=True, text=True, timeout=30, check=False)
    except OSError:
        return None
    return (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr) else None


def _auth_args(settings: Settings, workdir: Path) -> list[str]:
    """Build the connection arguments for mongodump/mongorestore.

    Credentials go into a temporary --config file, never onto the command line.

    Args:
        settings: Application settings (URI, username, password, auth source).
        workdir: Temporary folder where the password config file is written; delete it after the tool runs.

    Returns:
        Arguments: ``--uri``, plus ``--username``/``--authenticationDatabase`` and ``--config`` when set.
    """
    args = [f"--uri={settings.mongodb_uri}"]
    if settings.mongodb_username:
        args += [f"--username={settings.mongodb_username}", f"--authenticationDatabase={settings.mongodb_auth_source}"]
        if settings.mongodb_password:
            cfg = workdir / ".mongo-tools.yaml"
            cfg.write_text(f"password: {json.dumps(settings.mongodb_password.get_secret_value())}\n", encoding="utf-8")
            try:
                os.chmod(cfg, 0o600)
            except OSError:  # pragma: no cover - Windows ACLs
                pass
            args.append(f"--config={cfg}")
    return args


def _run_tool(cmd: list[str], *, what: str) -> str:
    """Run an external tool to completion and capture its output.

    Args:
        cmd: Command line (executable and arguments).
        what: Short name for log messages and errors.

    Returns:
        The tool's stdout followed by its stderr.

    Raises:
        BackupError: The tool exited with a non-zero code (message includes the last 4000 characters of output).
    """
    log.info("running %s", what)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise BackupError(f"{what} failed (exit {proc.returncode}):\n{output[-4000:]}")
    return output


# ====================================================================== python BSON dump
def python_bson_dump(db: Database, target: Path) -> dict[str, int]:
    """Write ``<target>/<db>/<collection>.bson`` + ``.metadata.json`` (mongorestore --dir compatible).

    Documents are written in ``_id`` order; the metadata holds the collection options and index specs.

    Args:
        db: Database to dump (every collection).
        target: Dump root folder (created if missing).

    Returns:
        Number of documents written per collection name.
    """
    base = target / db.name
    base.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name in db.list_collection_names():
        count = 0
        with (base / f"{name}.bson").open("wb") as handle:
            for doc in db[name].find({}, sort=[("_id", 1)]).batch_size(500):
                handle.write(bson.encode(doc))
                count += 1
        info = db.command("listCollections", filter={"name": name})["cursor"]["firstBatch"][0]
        indexes = [{"v": 2, **{k: v for k, v in spec.items() if k != "ns"}}
                   for spec in db[name].list_indexes()]
        metadata = {"options": info.get("options", {}), "indexes": indexes, "collectionName": name,
                    "type": info.get("type", "collection")}
        (base / f"{name}.metadata.json").write_text(json_util.dumps(metadata), encoding="utf-8")
        counts[name] = count
    return counts


def _count_bson(path: Path) -> int:
    """Count the documents in a BSON dump file by decoding it.

    Args:
        path: ``.bson`` file of concatenated documents.

    Returns:
        Number of documents.
    """
    with path.open("rb") as handle:
        return sum(1 for _ in bson.decode_file_iter(handle))


# ====================================================================== zip helpers
def _zip_dir(source: Path, target: Path, *, exclude_prefixes: tuple[str, ...] = ()) -> int:
    """Zip every file under a folder (deflate, level 6), with paths relative to the folder.

    Args:
        source: Folder to archive.
        target: Zip file to create (overwritten if present).
        exclude_prefixes: Relative POSIX path prefixes to leave out, e.g. ``("_restore_conflicts/",)``.

    Returns:
        Number of files written to the archive.
    """
    files = 0
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(source).as_posix()
            if rel.startswith(exclude_prefixes):
                continue
            zf.write(path, rel)
            files += 1
    return files


def _safe_member_target(root: Path, member: str) -> Path:
    """Resolve an archive member's destination, refusing paths that escape the root (zip-slip).

    Args:
        root: Folder files are extracted into.
        member: Member name from the archive.

    Returns:
        The resolved destination path inside ``root``.

    Raises:
        BackupError: The member would be written outside ``root``.
    """
    target = (root / member).resolve()
    if not target.is_relative_to(root.resolve()):
        raise BackupError(f"unsafe path in archive: {member}")
    return target


# ====================================================================== create
def create_backup(settings: Settings, mongo: Mongo, *, include_raw: bool = True, include_exports: bool = False,
                  method: str = "auto") -> dict[str, Any]:
    """Create a timestamped local backup of the database and data folders, then verify it.

    Writes ``backup-<stamp>/`` under ``settings.backup_dir`` (see the module docstring for the layout):
    a ``mongodump`` gzip archive when the tool is available (or required), otherwise the pure-Python BSON
    dump; zips of the raw store (without restore conflicts), the processed folder (if non-empty) and
    optionally the exports; and a checksummed manifest. The verification result is added to the manifest.

    Args:
        settings: Application settings (folders, MongoDB connection and tools location).
        mongo: Open connection to the database to back up.
        include_raw: Include the raw HTTP archive (``data/raw``).
        include_exports: Include the exports folder.
        method: ``auto`` (mongodump if found, else Python), ``mongodump`` (required) or ``python``.

    Returns:
        The manifest (backup id, collection counts, files with sizes and SHA-256, method, tool versions,
        restore steps, verification) plus ``directory``.

    Raises:
        BackupError: Invalid method, mongodump required but missing, mongodump failed, or verification failed
            (the backup folder is left in place).
    """
    if method not in {"auto", "mongodump", "python"}:
        raise BackupError("method must be auto, mongodump or python")
    db = mongo.db
    backup_id = f"backup-{stamp()}"
    directory = settings.backup_dir / backup_id
    directory.mkdir(parents=True, exist_ok=False)
    counts = {name: db[name].count_documents({}) for name in db.list_collection_names()}
    tools: dict[str, Any] = {}
    mongodump = find_tool("mongodump", settings) if method in {"auto", "mongodump"} else None
    if method == "mongodump" and mongodump is None:
        raise BackupError("mongodump not found; set MONGODB_TOOLS_DIR or install MongoDB Database Tools")

    used: str
    if mongodump is not None:
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [str(mongodump), *_auth_args(settings, Path(tmp)), f"--db={db.name}",
                   f"--archive={directory / ARCHIVE}", "--gzip"]
            _run_tool(cmd, what="mongodump")
        used = "mongodump"
        tools["mongodump"] = _tool_version(mongodump)
    else:
        python_bson_dump(db, directory / BSON_DIR)
        used = "python-bson"

    if include_raw and settings.raw_dir.exists():
        _zip_dir(settings.raw_dir, directory / RAW_ZIP, exclude_prefixes=("_restore_conflicts/",))
    if settings.processed_dir.exists() and any(settings.processed_dir.rglob("*")):
        _zip_dir(settings.processed_dir, directory / PROCESSED_ZIP)
    if include_exports and settings.export_dir.exists():
        _zip_dir(settings.export_dir, directory / "exports-archive.zip")

    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name not in {MANIFEST, MANIFEST + ".sha256"}:
            files.append({"path": path.relative_to(directory).as_posix(), "bytes": path.stat().st_size,
                          "sha256": sha256_file(path)})
    manifest = {
        "backup_id": backup_id,
        "created_at": utcnow().isoformat(),
        "database": db.name,
        "mongodb_server_version": mongo.server_version,
        "method": used,
        "tools": tools,
        "collections": counts,
        "includes": {"raw": include_raw, "exports": include_exports},
        "files": files,
        "settings": {k: v for k, v in settings.safe_dict().items() if k.startswith(("mongodb_db", "data_dir", "raw"))},
        "restore": [
            "python -m src.main backup --verify <backup dir>",
            "python -m src.main restore <backup dir> --confirm [--drop] [--target-db NAME]",
            "Manual: mongorestore --uri=mongodb://127.0.0.1:27017/ --archive=mongodb.archive.gz --gzip "
            f"--nsInclude='{db.name}.*'  (python-bson backups: mongorestore --dir=mongodb-bson)",
        ],
    }
    _write_manifest(directory, manifest)
    verification = verify_backup(directory)
    manifest["verification"] = {"ok": verification["ok"], "checked_at": utcnow().isoformat(),
                                "problems": verification["problems"]}
    _write_manifest(directory, manifest)
    if not verification["ok"]:
        raise BackupError(f"backup verification failed: {verification['problems']}")
    return {"directory": str(directory), **manifest}


def _write_manifest(directory: Path, manifest: dict[str, Any]) -> None:
    """Write ``manifest.json`` (pretty JSON) and its ``manifest.json.sha256`` checksum file.

    Args:
        directory: Backup folder.
        manifest: Manifest data; non-JSON values are written with ``str()``.
    """
    path = directory / MANIFEST
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (directory / (MANIFEST + ".sha256")).write_text(f"{sha256_file(path)}  {MANIFEST}\n", encoding="utf-8")


# ====================================================================== verify
def verify_backup(directory: Path) -> dict[str, Any]:
    """Check a backup folder against its manifest without modifying anything.

    Checks the manifest checksum, each listed file's size and SHA-256, zip integrity, and either the
    document count of every BSON dump (Python method) or the presence of the mongodump archive.

    Args:
        directory: Backup folder.

    Returns:
        ``ok`` (no problems), ``problems`` (messages), ``checked_files``, ``backup_id`` and ``method``; only the
        first three when the manifest is missing.
    """
    problems: list[str] = []
    manifest_path = directory / MANIFEST
    if not manifest_path.exists():
        return {"ok": False, "problems": [f"{MANIFEST} missing"], "checked_files": 0}
    side = directory / (MANIFEST + ".sha256")
    if side.exists():
        expected = side.read_text(encoding="utf-8").split()[0]
        if sha256_file(manifest_path) != expected:
            problems.append("manifest.json checksum mismatch (manifest altered)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked = 0
    for entry in manifest.get("files", []):
        path = directory / entry["path"]
        if not path.exists():
            problems.append(f"missing file: {entry['path']}")
            continue
        if path.stat().st_size != entry["bytes"]:
            problems.append(f"size mismatch: {entry['path']}")
        elif sha256_file(path) != entry["sha256"]:
            problems.append(f"checksum mismatch: {entry['path']}")
        checked += 1
        if path.suffix == ".zip":
            try:
                with zipfile.ZipFile(path) as zf:
                    bad = zf.testzip()
                if bad:
                    problems.append(f"corrupt member {bad} in {entry['path']}")
            except zipfile.BadZipFile as exc:
                problems.append(f"bad zip {entry['path']}: {exc}")
    if manifest.get("method") == "python-bson":
        base = directory / BSON_DIR / manifest["database"]
        for name, count in manifest.get("collections", {}).items():
            path = base / f"{name}.bson"
            if not path.exists():
                problems.append(f"missing BSON dump for {name}")
                continue
            try:
                actual = _count_bson(path)
            except Exception as exc:  # noqa: BLE001 - corrupt BSON of any kind
                problems.append(f"unreadable BSON for {name}: {exc}")
                continue
            if actual != count:
                problems.append(f"document count mismatch for {name}: {actual} != {count}")
    elif manifest.get("method") == "mongodump" and not (directory / ARCHIVE).exists():
        problems.append("mongodump archive missing")
    return {"ok": not problems, "problems": problems, "checked_files": checked,
            "backup_id": manifest.get("backup_id"), "method": manifest.get("method")}


# ====================================================================== restore
def restore_backup(directory: Path, settings: Settings, *, drop: bool = False, target_db: str | None = None,
                   restore_raw: bool = True) -> dict[str, Any]:
    """Restore a verified backup into MongoDB and the local data folders.

    mongodump backups are restored with ``mongorestore`` (renaming the namespace when the target database
    differs). Python BSON backups are inserted in batches of 500; documents whose ``_id`` already exists
    are skipped, so without ``drop`` existing data is kept. Indexes are ensured afterwards. Raw and
    processed zips are extracted without overwriting existing files (see ``_restore_zip``).

    Args:
        directory: Backup folder.
        settings: Application settings.
        drop: Drop each restored collection first (``--drop`` for mongorestore).
        target_db: Database to restore into; ``settings.mongodb_db`` when None.
        restore_raw: Also restore the raw HTTP archive.

    Returns:
        ``backup_id``, ``target_db``, ``method``; then ``tool_output`` (mongorestore, last 2000 characters) or
        ``inserted``/``skipped_existing`` per collection (Python); plus ``raw``/``processed`` file counts when
        restored.

    Raises:
        BackupError: Verification failed, mongorestore is missing or failed, or a non-duplicate insert error.
    """
    verification = verify_backup(directory)
    if not verification["ok"]:
        raise BackupError(f"refusing to restore an unverified backup: {verification['problems']}")
    manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    source_db = manifest["database"]
    target = target_db or settings.mongodb_db  # default: the database this installation is configured for
    result: dict[str, Any] = {"backup_id": manifest["backup_id"], "target_db": target, "method": manifest["method"]}

    if manifest["method"] == "mongodump":
        mongorestore = find_tool("mongorestore", settings)
        if mongorestore is None:
            raise BackupError("mongorestore not found; set MONGODB_TOOLS_DIR or install MongoDB Database Tools")
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [str(mongorestore), *_auth_args(settings, Path(tmp)), f"--archive={directory / ARCHIVE}", "--gzip",
                   f"--nsInclude={source_db}.*"]
            if target != source_db:
                cmd += [f"--nsFrom={source_db}.*", f"--nsTo={target}.*"]
            if drop:
                cmd.append("--drop")
            result["tool_output"] = _run_tool(cmd, what="mongorestore")[-2000:]
        with Mongo.connect(settings, db_name=target) as mongo:
            ensure_schema(mongo.db)
    else:
        inserted: dict[str, int] = {}
        skipped: dict[str, int] = {}
        with Mongo.connect(settings, db_name=target) as mongo:
            base = directory / BSON_DIR / source_db
            for path in sorted(base.glob("*.bson")):
                name = path.stem
                if drop:
                    mongo.db.drop_collection(name)
                batch: list[dict[str, Any]] = []
                ins = skip = 0
                with path.open("rb") as handle:
                    for doc in bson.decode_file_iter(handle):
                        batch.append(doc)
                        if len(batch) >= 500:
                            i, s = _insert_batch(mongo.db, name, batch)
                            ins, skip, batch = ins + i, skip + s, []
                if batch:
                    i, s = _insert_batch(mongo.db, name, batch)
                    ins, skip = ins + i, skip + s
                inserted[name], skipped[name] = ins, skip
            ensure_schema(mongo.db)
        result.update({"inserted": inserted, "skipped_existing": skipped})

    if restore_raw and (directory / RAW_ZIP).exists():
        result["raw"] = _restore_zip(directory / RAW_ZIP, settings.raw_dir)
    if (directory / PROCESSED_ZIP).exists():
        result["processed"] = _restore_zip(directory / PROCESSED_ZIP, settings.processed_dir)
    return result


def _insert_batch(db: Database, name: str, batch: list[dict[str, Any]]) -> tuple[int, int]:
    """Insert documents unordered, tolerating duplicate ``_id`` errors.

    Args:
        db: Target database.
        name: Collection name.
        batch: Documents to insert.

    Returns:
        ``(inserted, duplicates_skipped)``.

    Raises:
        BackupError: Any write error other than a duplicate key (code 11000).
    """
    try:
        res = db[name].insert_many(batch, ordered=False)
        return len(res.inserted_ids), 0
    except BulkWriteError as exc:
        details = exc.details
        dupes = sum(1 for e in details.get("writeErrors", []) if e.get("code") == 11000)
        others = [e for e in details.get("writeErrors", []) if e.get("code") != 11000]
        if others:
            raise BackupError(f"restore of {name} failed: {others[:3]}") from exc
        return details.get("nInserted", 0), dupes


def _restore_zip(archive: Path, root: Path) -> dict[str, int]:
    """Extract a zip into a folder without ever overwriting existing files.

    Identical existing files are skipped; a differing archived copy is written under
    ``<root>/_restore_conflicts/<stamp>/`` instead.

    Args:
        archive: Zip file from the backup.
        root: Destination folder (created if missing).

    Returns:
        ``restored`` (files written to their original place), ``identical_skipped`` and ``conflicts_preserved``
        (differing files whose archived copy was written to the conflict folder instead).

    Raises:
        BackupError: A member path would escape the destination.
    """
    restored = identical = conflicts = 0
    conflict_root = root / "_restore_conflicts" / stamp()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            dest = _safe_member_target(root, info.filename)
            data = zf.read(info)
            if dest.exists():
                if dest.read_bytes() == data:
                    identical += 1
                    continue
                # Never overwrite: the differing archived copy goes to the conflict folder.
                dest = _safe_member_target(conflict_root, info.filename)
                conflicts += 1
            else:
                restored += 1  # conflict copies are counted only under conflicts_preserved
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
    return {"restored": restored, "identical_skipped": identical, "conflicts_preserved": conflicts}


def list_backups(settings: Settings) -> list[dict[str, Any]]:
    """List the backups in ``settings.backup_dir`` that have a manifest.

    Args:
        settings: Application settings.

    Returns:
        One dict per backup, oldest first: ``directory``, ``created_at``, ``method`` and total ``bytes`` of the
        files listed in the manifest.
    """
    out = []
    if not settings.backup_dir.exists():
        return out
    for directory in sorted(settings.backup_dir.glob("backup-*")):
        manifest_path = directory / MANIFEST
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            out.append({"directory": str(directory), "created_at": manifest.get("created_at"),
                        "method": manifest.get("method"),
                        "bytes": sum(f["bytes"] for f in manifest.get("files", []))})
    return out
