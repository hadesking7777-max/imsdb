"""Collection statistics, coverage report and CLI dashboard rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pymongo.database import Database

from src.database.models import Collections as C
from src.services.deduplication import duplicate_script_contents, duplicate_movie_titles
from src.utils.clock import as_utc


def _dir_size(path: Path) -> tuple[int, int]:
    """Measure the files under a folder, recursively.

    Args:
        path: Folder to measure; a missing folder counts as empty.

    Returns:
        ``(total_bytes, file_count)``.
    """
    total = files = 0
    if path.exists():
        for item in path.rglob("*"):
            if item.is_file():
                files += 1
                total += item.stat().st_size
    return total, files


def human_bytes(value: float) -> str:
    """Format a byte count with binary (1024-based) units.

    Args:
        value: Size in bytes.

    Returns:
        E.g. ``"512 B"``, ``"1.5 KB"``, ``"2,048.0 TB"`` (TB is the largest unit).
    """
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024 or unit == "TB":
            return f"{value:,.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"  # pragma: no cover


def collect_stats(db: Database, *, raw_dir: Path, data_dir: Path) -> dict[str, Any]:
    """Gather collection, queue, quality, storage and coverage statistics (read-only).

    Reads ``movies``, ``scripts``, ``crawl_queue``, ``crawl_jobs``, ``crawl_logs``, ``source_permissions``
    and MongoDB ``dbStats``, and walks the raw, processed, backups and exports folders to size them.

    Args:
        db: Project database.
        raw_dir: Raw HTTP archive folder.
        data_dir: Data root containing ``processed``, ``backups`` and ``exports``.

    Returns:
        Nested dict with sections ``permission`` (level, sample limit, version, robots status), ``records``
        (movie/script/derived counts and status breakdowns), ``queue`` (pending/in-progress/failed counts and
        per-type status counts), ``quality`` (soft duplicates from the first 100 groups, parse errors,
        warning events, flagged scripts), ``storage`` (sizes in bytes; ``total_bytes`` = MongoDB storage +
        indexes + raw + processed), ``latest_successful_ingestion`` (ISO time or None), ``last_job`` (or None)
        and ``coverage`` (records measured against discovered movies, with ``collected_pct_of_discovered``).
    """
    movies = db[C.MOVIES]
    scripts = db[C.SCRIPTS]
    queue = db[C.CRAWL_QUEUE]

    status_counts = {r["_id"]: r["n"] for r in movies.aggregate([{"$group": {"_id": "$status", "n": {"$sum": 1}}}])}
    script_status = {r["_id"]: r["n"] for r in scripts.aggregate([{"$group": {"_id": "$status", "n": {"$sum": 1}}}])}
    queue_counts: dict[str, dict[str, int]] = {}
    for row in queue.aggregate([{"$group": {"_id": {"t": "$page_type", "s": "$status"}, "n": {"$sum": 1}}}]):
        queue_counts.setdefault(row["_id"]["t"], {})[row["_id"]["s"]] = row["n"]

    def q(status: str, types: list[str] | None = None) -> int:
        """Count queue items with a status, optionally restricted to page types.

        Args:
            status: Queue status, e.g. ``pending``.
            types: Page types to count, or None for all.

        Returns:
            Number of matching ``crawl_queue`` documents.
        """
        query: dict[str, Any] = {"status": status}
        if types:
            query["page_type"] = {"$in": types}
        return queue.count_documents(query)

    discovered = movies.count_documents({})
    detail_collected = movies.count_documents({"detail_fetched_at": {"$exists": True}})
    extracted_ok = scripts.count_documents({"status": {"$in": ["parsed", "parsed_with_warnings"]}})
    parse_errors = scripts.count_documents({"status": {"$in": ["parse_failed", "extraction_failed"]}})
    unavailable = movies.count_documents({"status": "script_unavailable"})
    empty_at_source = movies.count_documents({"status": "script_empty_at_source"})
    shared = movies.count_documents({"status": "script_shared"})
    failed_records = movies.count_documents({"status": {"$in": ["script_failed", "detail_failed"]}})
    record_types = ["movie_detail", "script_html"]
    failed_urls = q("failed") + q("blocked")
    failed_detail_movies = 0
    for item in queue.find({"page_type": "movie_detail", "status": {"$in": ["failed", "blocked"]}}, {"url": 1}):
        doc = movies.find_one({"detail_url": item["url"]}, {"detail_fetched_at": 1})
        if doc is not None and "detail_fetched_at" not in doc:
            failed_detail_movies += 1

    dup_movies = duplicate_movie_titles(db)
    dup_scripts = duplicate_script_contents(db)
    latest = scripts.find_one({"status": {"$in": ["parsed", "parsed_with_warnings"]}}, {"parsed_at": 1},
                              sort=[("parsed_at", -1)])
    last_job = db[C.CRAWL_JOBS].find_one({"command": {"$exists": True}}, sort=[("started_at", -1)])
    db_stats = db.command("dbStats")
    raw_bytes, raw_files = _dir_size(raw_dir)
    processed_bytes, _ = _dir_size(data_dir / "processed")
    backup_bytes, _ = _dir_size(data_dir / "backups")
    export_bytes, _ = _dir_size(data_dir / "exports")
    perm = db[C.SOURCE_PERMISSIONS].find_one({"_id": "imsdb"}) or {}

    coverage = {
        "discovered_records": discovered,
        "detail_pages_collected": detail_collected,
        "scripts_collected_successfully": extracted_ok,
        "scripts_unavailable_on_source": unavailable,
        "scripts_empty_on_source": empty_at_source,
        "scripts_shared_with_other_movie": shared,
        # Movies only: a failed script already marks its movie 'script_failed', so scripts are not added again.
        "failed_records": failed_records + failed_detail_movies,
        "unresolved_records": max(0, discovered - detail_collected - failed_detail_movies)
                              + movies.count_documents({"status": "detail_collected"}),
    }
    coverage["collected_pct_of_discovered"] = round(100.0 * extracted_ok / discovered, 2) if discovered else 0.0

    return {
        "permission": {"level": perm.get("level"), "sample_limit": perm.get("sample_limit"),
                       "version": perm.get("version"), "robots_status": (perm.get("robots") or {}).get("status")},
        "records": {
            "discovered_movies": discovered,
            "processed_movies": detail_collected,
            "movie_status": status_counts,
            "script_records": scripts.count_documents({}),
            "scripts_extracted_ok": extracted_ok,
            "script_status": script_status,
            "script_versions_archived": db[C.SCRIPT_VERSIONS].count_documents({}),
            "scenes": db[C.SCENES].estimated_document_count(),
            "characters": db[C.CHARACTERS].estimated_document_count(),
            "dialogue_blocks": db[C.DIALOGUES].estimated_document_count(),
        },
        "queue": {
            "pending_urls": q("pending"),
            "in_progress_urls": q("in_progress"),
            "failed_urls": failed_urls,
            "failed_record_urls": q("failed", record_types) + q("blocked", record_types),
            "by_type": queue_counts,
        },
        "quality": {
            "duplicate_movie_title_groups": len(dup_movies),
            "duplicate_script_content_groups": len(dup_scripts),
            "duplicate_records": sum(g["count"] - 1 for g in dup_movies) + sum(g["count"] - 1 for g in dup_scripts),
            "parsing_errors": parse_errors,
            "parse_warning_events": db[C.CRAWL_LOGS].count_documents({"event": "parse_warning"}),
            "scripts_with_quality_flags": scripts.count_documents({"quality_flags.0": {"$exists": True}}),
        },
        "storage": {
            "mongodb_data_bytes": int(db_stats.get("dataSize", 0)),
            "mongodb_storage_bytes": int(db_stats.get("storageSize", 0) + db_stats.get("indexSize", 0)),
            "raw_archive_bytes": raw_bytes,
            "raw_archive_files": raw_files,
            "processed_bytes": processed_bytes,
            "exports_bytes": export_bytes,
            "backups_bytes": backup_bytes,
            "total_bytes": int(db_stats.get("storageSize", 0) + db_stats.get("indexSize", 0)) + raw_bytes
                           + processed_bytes,
        },
        "latest_successful_ingestion": as_utc(latest["parsed_at"]).isoformat() if latest else None,
        "last_job": {"id": last_job["_id"], "command": last_job.get("command"), "status": last_job.get("status"),
                     "started_at": as_utc(last_job.get("started_at")).isoformat() if last_job.get("started_at") else None,
                     "counters": last_job.get("counters", {})} if last_job else None,
        "coverage": coverage,
    }


def render_dashboard(stats: dict[str, Any]) -> str:
    """Render statistics as a fixed-width (64-column) plain-text dashboard for the CLI.

    Args:
        stats: Output of ``collect_stats``.

    Returns:
        The dashboard text, lines joined with newlines.
    """
    r, qd, ql, st, cov = stats["records"], stats["queue"], stats["quality"], stats["storage"], stats["coverage"]
    perm = stats["permission"]
    width = 64
    line = "=" * width

    def row(label: str, value: Any) -> str:
        """Format one dashboard line: label left-aligned, value right-aligned.

        Args:
            label: Row label (padded to 40 characters).
            value: Value shown with ``str()`` (padded to 22 characters).

        Returns:
            The formatted line.
        """
        return f"  {label:<40}{value!s:>22}"

    out = [line, " IMSDb local collection - statistics".center(width), line,
           row("Permission level", f"{perm['level']} (limit {perm['sample_limit']})"
               if perm["level"] == "sample" else perm["level"]),
           row("robots.txt HTTP status", perm["robots_status"]),
           "", " RECORDS (discovered != collected)",
           row("Discovered movie records", r["discovered_movies"]),
           row("Processed movie records (detail page)", r["processed_movies"]),
           row("Screenplay records", r["script_records"]),
           row("Successfully extracted scripts", r["scripts_extracted_ok"]),
           row("Archived previous script versions", r["script_versions_archived"]),
           row("Scenes", r["scenes"]),
           row("Characters", r["characters"]),
           row("Dialogue blocks", r["dialogue_blocks"]),
           "", " QUEUE",
           row("Pending URLs", qd["pending_urls"]),
           row("In-progress URLs", qd["in_progress_urls"]),
           row("Failed / blocked URLs", qd["failed_urls"]),
           "", " QUALITY",
           row("Duplicate records (soft duplicates)", ql["duplicate_records"]),
           row("Parsing errors", ql["parsing_errors"]),
           row("Scripts with quality flags", ql["scripts_with_quality_flags"]),
           row("Parse-warning log events", ql["parse_warning_events"]),
           "", " STORAGE",
           row("MongoDB (data + indexes on disk)", human_bytes(st["mongodb_storage_bytes"])),
           row("Raw archive", f"{human_bytes(st['raw_archive_bytes'])} / {st['raw_archive_files']} files"),
           row("Total stored data", human_bytes(st["total_bytes"])),
           "", row("Latest successful ingestion", stats["latest_successful_ingestion"] or "never"),
           "", " COVERAGE REPORT",
           row("Discovered records", cov["discovered_records"]),
           row("Successfully collected records", cov["scripts_collected_successfully"]),
           row("Script not offered by source (PDF/none)", cov["scripts_unavailable_on_source"]),
           row("Script page empty on source", cov["scripts_empty_on_source"]),
           row("Script shared with another movie", cov["scripts_shared_with_other_movie"]),
           row("Failed records", cov["failed_records"]),
           row("Unresolved records (not yet attempted)", cov["unresolved_records"]),
           row("Collected / discovered", f"{cov['collected_pct_of_discovered']} %"),
           line,
           " Note: coverage is measured against records *discovered* from IMSDb's listing",
           " pages. An empty queue does not prove the collection is complete; re-run",
           " 'discover --refresh' to detect new or changed listings.",
           line]
    if stats.get("last_job"):
        job = stats["last_job"]
        # -5 = just before the separator that precedes the 3-line note (keeps it inside the coverage block)
        out.insert(-5, row("Last job", f"{job['command']} / {job['status']}"))
    return "\n".join(out)
