"""Export collected records to local JSON / JSON Lines files with a provenance manifest."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from bson import ObjectId
from pymongo.database import Database

from src.database.models import Collections as C
from src.errors import PermissionDeniedError
from src.services.permissions import ATTRIBUTION, USAGE_RESTRICTION
from src.utils.clock import stamp, utcnow
from src.utils.hashing import sha256_file

DEFAULT_COLLECTIONS = [C.MOVIES, C.SCRIPTS, C.SCENES, C.CHARACTERS, C.DIALOGUES, C.SOURCE_PERMISSIONS]
OPERATIONAL = [C.SCRIPT_VERSIONS, C.CRAWL_QUEUE, C.CRAWL_LOGS, C.CRAWL_JOBS, C.RAW_ARTIFACTS]
# Records derived from screenplays are exportable only if collected under a recorded permission.
AUTHORIZED_FILTER = {"rights.permission_level_at_collection": {"$in": ["sample", "bulk"]}}
# Fields holding screenplay wording, removed when exporting with include_text=False (``export --no-text``).
TEXT_FIELDS = {
    C.SCRIPTS: ["text", "source_markup", "elements", "formatting_spans"],
    C.SCENES: ["text", "description"],
    C.DIALOGUES: ["text", "original_text", "parentheticals", "segments"],
}


def _json_default(value: Any) -> Any:
    """``json.dumps`` fallback for BSON/Python types JSON cannot represent.

    Args:
        value: Value the JSON encoder could not serialize.

    Returns:
        ISO-8601 string for datetimes/dates, hex string for ObjectIds and bytes.

    Raises:
        TypeError: Any other type.
    """
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def export_records(db: Database, export_dir: Path, *, fmt: str = "json", collections: list[str] | None = None,
                   include_text: bool = True, include_operational: bool = False) -> dict[str, Any]:
    """Export collections to ``export-<stamp>/`` as JSON arrays or JSON Lines, with a manifest and attribution.

    Only scripts collected under the ``sample`` or ``bulk`` level are exported, and scenes, characters and
    dialogues only for those scripts. Documents are written in ``_id`` order. ``manifest.json`` records
    per-file record counts, sizes and SHA-256, the permission level and the attribution/usage notice;
    ``ATTRIBUTION.txt`` is written alongside.

    Args:
        db: Project database.
        export_dir: Parent folder; a new timestamped sub-folder is created.
        fmt: ``"json"`` (one array per file) or ``"jsonl"`` (one document per line).
        collections: Collections to export; ``DEFAULT_COLLECTIONS`` when None.
        include_text: Include screenplay wording. When False, the fields in ``TEXT_FIELDS`` are left out of
            ``scripts`` (and of the script copies archived in ``script_versions``), ``scenes`` and ``dialogues``.
        include_operational: Also export the operational collections (``OPERATIONAL``). Archived
            ``script_versions`` are filtered by the same permission rule as ``scripts``.

    Returns:
        The manifest plus ``directory`` (path of the export folder).

    Raises:
        ValueError: Unknown format or collection name.
        PermissionDeniedError: No IMSDb permission record exists, or its level is ``denied``.
    """
    if fmt not in {"json", "jsonl"}:
        raise ValueError("format must be 'json' or 'jsonl'")
    perm = db[C.SOURCE_PERMISSIONS].find_one({"_id": "imsdb"})
    if perm is None:
        raise PermissionDeniedError("No source permission record found; run 'python -m src.main inspect-source' first.")
    if perm.get("level") == "denied":
        raise PermissionDeniedError("Source permission level is 'denied'; exporting IMSDb records is not allowed.")

    names = list(collections or DEFAULT_COLLECTIONS)
    if include_operational:
        names += [n for n in OPERATIONAL if n not in names]
    unknown = [n for n in names if n not in C.ALL]
    if unknown:
        raise ValueError(f"unknown collection(s): {', '.join(unknown)}")

    authorized_scripts = {d["_id"] for d in db[C.SCRIPTS].find(AUTHORIZED_FILTER, {"_id": 1})}
    out_dir = export_dir / f"export-{stamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)
    files: list[dict[str, Any]] = []
    for name in names:
        query: dict[str, Any] = {}
        projection: dict[str, int] | None = None
        if name == C.SCRIPTS:
            query = AUTHORIZED_FILTER
            if not include_text:
                projection = {field: 0 for field in TEXT_FIELDS[C.SCRIPTS]}
        elif name in {C.SCENES, C.CHARACTERS, C.DIALOGUES}:
            query = {"script_id": {"$in": list(authorized_scripts)}}
            if not include_text and name in TEXT_FIELDS:
                projection = {field: 0 for field in TEXT_FIELDS[name]}
        elif name == C.SCRIPT_VERSIONS:
            # Each archived version embeds a complete earlier script document, text included: apply the same
            # permission filter and text removal as for ``scripts``.
            query = {"document." + key: value for key, value in AUTHORIZED_FILTER.items()}
            if not include_text:
                projection = {"document." + field: 0 for field in TEXT_FIELDS[C.SCRIPTS]}
        path = out_dir / f"{name}.{fmt}"
        count = 0
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            if fmt == "json":
                handle.write("[\n")
            for doc in db[name].find(query, projection, no_cursor_timeout=False).sort("_id", 1):
                line = json.dumps(doc, default=_json_default, ensure_ascii=False)
                if fmt == "json":
                    handle.write(("," if count else "") + line + "\n")
                else:
                    handle.write(line + "\n")
                count += 1
            if fmt == "json":
                handle.write("]\n")
        files.append({"file": path.name, "collection": name, "records": count, "bytes": path.stat().st_size,
                      "sha256": sha256_file(path)})

    manifest = {
        "created_at": utcnow().isoformat(),
        "database": db.name,
        "format": fmt,
        "include_text": include_text,
        "permission": {"level": perm.get("level"), "version": perm.get("version"),
                       "sample_limit": perm.get("sample_limit")},
        "attribution": ATTRIBUTION,
        "usage_restriction": USAGE_RESTRICTION,
        "notice": "Local export for educational research. Do not redistribute screenplay text.",
        "files": files,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "ATTRIBUTION.txt").write_text(f"{ATTRIBUTION}\n\n{USAGE_RESTRICTION}\n", encoding="utf-8")
    return {"directory": str(out_dir), **manifest}
