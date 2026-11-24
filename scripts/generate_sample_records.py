"""Write docs/SAMPLE_RECORDS.md: one example document per collection from the local database.

Screenplay text is truncated (copyrighted material must not be republished in docs);
the full records stay in the local database only.

Usage:  python scripts/generate_sample_records.py [--movie-id ID] [--output PATH]
Exit codes: 0 when the file was written, 1 when no collected movie was found, 2 for invalid arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_settings  # noqa: E402
from src.database.connection import Mongo  # noqa: E402
from src.database.models import Collections as C  # noqa: E402
from src.services.stats import collect_stats  # noqa: E402
from src.services.validation import run_validation  # noqa: E402
from src.storage.raw_store import RawStore  # noqa: E402

TEXT_LIMIT = 160
LIST_LIMIT = 3


def _trim(value: Any, key: str = "") -> Any:
    """Shorten a document for display: long strings are cut, long lists capped, dates made ISO strings.

    Args:
        value: Any JSON-like value (dicts and lists are processed recursively).
        key: Name of the field holding ``value`` (passed down the recursion; not used otherwise).

    Returns:
        A copy with strings longer than ``TEXT_LIMIT`` truncated (with the original length noted) and lists cut to
        ``LIST_LIMIT`` items plus a "... (N more)" marker.
    """
    if isinstance(value, dict):
        return {k: _trim(v, k) for k, v in value.items()}
    if isinstance(value, list):
        trimmed = [_trim(v, key) for v in value[:LIST_LIMIT]]
        if len(value) > LIST_LIMIT:
            trimmed.append(f"... ({len(value) - LIST_LIMIT} more)")
        return trimmed
    if isinstance(value, str) and len(value) > TEXT_LIMIT:
        return value[:TEXT_LIMIT] + f"... [truncated, {len(value):,} chars]"
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _block(title: str, doc: dict[str, Any] | None) -> str:
    """Render one Markdown section with a trimmed document as a JSON code block.

    Args:
        title: Section heading (collection name).
        doc: The document, or ``None`` if the collection had no matching document.

    Returns:
        The Markdown text of the section.
    """
    if doc is None:
        return f"### {title}\n\n_(no document)_\n"
    return f"### {title}\n\n```json\n{json.dumps(_trim(doc), indent=2, ensure_ascii=False, default=str)}\n```\n"


def main() -> int:
    """Pick a collected movie and its related records, and write them with run statistics to Markdown.

    Returns:
        0 after writing the output file, 1 if no collected movie (or the ``--movie-id`` given) was found.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--movie-id")
    parser.add_argument("--output", default=str(ROOT / "docs" / "SAMPLE_RECORDS.md"))
    args = parser.parse_args()
    settings = load_settings()
    with Mongo.connect(settings) as mongo:
        db = mongo.db
        query = {"_id": args.movie_id} if args.movie_id else {"status": "script_collected"}
        movie = db[C.MOVIES].find_one(query, sort=[("detail_fetched_at", 1)])
        if movie is None:
            print("No collected movie found; run a crawl first.", file=sys.stderr)
            return 1
        script = db[C.SCRIPTS].find_one({"movie_id": movie["_id"]})
        sid = script["_id"] if script else None
        scene = db[C.SCENES].find_one({"script_id": sid, "kind": "scene"}, sort=[("scene_number", 1)])
        dialogue = db[C.DIALOGUES].find_one({"script_id": sid, "parentheticals.0": {"$exists": True}}) or \
            db[C.DIALOGUES].find_one({"script_id": sid})
        character = db[C.CHARACTERS].find_one({"script_id": sid}, sort=[("dialogue_count", -1)])
        queue_item = db[C.CRAWL_QUEUE].find_one({"url": script["script_url"]}) if script else None
        raw = db[C.RAW_ARTIFACTS].find_one({"_id": script["raw"]["artifact_id"]}) if script else None
        log = db[C.CRAWL_LOGS].find_one({"event": "script_parsed", "record_id": sid})
        job = db[C.CRAWL_JOBS].find_one({"command": "crawl"}, sort=[("started_at", -1)])
        perm = db[C.SOURCE_PERMISSIONS].find_one({"_id": "imsdb"})
        if perm:
            perm = {k: v for k, v in perm.items() if k not in {"policy_documents"}} | {
                "policy_documents": [{k: d.get(k) for k in ("url", "final_url", "status_code", "sha256", "raw_path")}
                                     for d in perm.get("policy_documents", [])]}
        stats = collect_stats(db, raw_dir=settings.raw_dir, data_dir=settings.data_dir)
        report = run_validation(db, raw_store=RawStore(settings.raw_dir), deep=True)
        collected = list(db[C.MOVIES].find({"detail_fetched_at": {"$exists": True}},
                                           {"display_title": 1, "status": 1, "script_availability": 1}))
        scripts = {s["movie_id"]: s for s in db[C.SCRIPTS].find({}, {"movie_id": 1, "status": 1, "structure_stats": 1,
                                                                     "quality_flags": 1, "text_length": 1})}

        lines = [
            "# Sample database records (live sample run)", "",
            f"Generated {datetime.now().astimezone().isoformat(timespec='seconds')} from the local `{db.name}` "
            "database by `scripts/generate_sample_records.py`.",
            "", "> Screenplay text is **truncated** here on purpose: the scripts are copyrighted by their owners. "
            "Full records exist only in the local database and raw archive.", "",
            "## Live run", "",
            f"* Permission level: `{stats['permission']['level']}` (limit {stats['permission']['sample_limit']}); "
            f"robots.txt HTTP {stats['permission']['robots_status']}",
            f"* Discovered movie records: **{stats['records']['discovered_movies']}**; processed (detail page): "
            f"**{stats['records']['processed_movies']}**; scripts extracted: **{stats['records']['scripts_extracted_ok']}**",
            f"* Scenes {stats['records']['scenes']}, dialogue blocks {stats['records']['dialogue_blocks']}, "
            f"characters {stats['records']['characters']}",
            f"* Raw archive: {stats['storage']['raw_archive_files']} files, "
            f"{stats['storage']['raw_archive_bytes']:,} bytes",
            f"* Deep validation: **{'PASSED' if report.passed else 'FAILED'}** "
            f"({report.errors} error checks, {report.warnings} warning checks)",
            "", "| Movie | Status | Script | Scenes | Dialogue blocks | Characters | Uncertain | Flags |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
        for m in collected:
            s = scripts.get(m["_id"])
            st = (s or {}).get("structure_stats", {})
            lines.append(f"| {m.get('display_title')} | {m.get('status')} | {(s or {}).get('status', m.get('script_availability'))} "
                         f"| {st.get('scenes', '-')} | {st.get('dialogue_blocks', '-')} | {st.get('characters', '-')} "
                         f"| {st.get('uncertain_ratio', '-')} | {', '.join((s or {}).get('quality_flags', [])) or '-'} |")
        warn = [i for i in report.issues if i.count and i.severity != "info"]
        if warn:
            lines += ["", "Validation findings (kept for inspection, nothing was discarded):", ""]
            lines += [f"* `{i.check}` ({i.severity}): {i.count} - {i.message}" for i in warn]
        lines += ["", "## One document per collection", ""]
        if script:
            script = dict(script)
            script["elements"] = (script.get("elements") or [])[:4]
            script["formatting_spans"] = (script.get("formatting_spans") or [])[:3]
        for title, doc in (("movies", movie), ("scripts (elements/spans abbreviated)", script), ("scenes", scene),
                           ("dialogues", dialogue), ("characters", character), ("raw_artifacts", raw),
                           ("crawl_queue", queue_item), ("crawl_logs", log), ("crawl_jobs", job),
                           ("source_permissions", perm)):
            lines.append(_block(title, doc))
        Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
