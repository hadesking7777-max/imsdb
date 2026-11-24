"""Database consistency validation and reporting.

Validation is read-only: problematic records are listed (with ids) for inspection and
are never modified or deleted. Severity ``error`` means the data model is inconsistent;
``warning`` means data quality needs review (e.g. uncertain parsing, encoding problems).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pymongo.database import Database

from src.crawler.url_manager import PageType, classify_url, normalize_url
from src.database.models import Collections as C
from src.services.deduplication import duplicate_key_violations, duplicate_movie_titles, duplicate_script_contents
from src.storage.raw_store import RawStore
from src.utils.clock import stamp, utcnow

PARSED = ["parsed", "parsed_with_warnings"]
EXAMPLES = 15


@dataclass
class Issue:
    """Result of one validation check.

    Attributes:
        check: Check name, e.g. ``broken_scene_references``.
        severity: ``error`` or ``warning`` when problems were found; ``info`` when ``count`` is 0.
        count: Number of problems found (records, groups or URLs, depending on the check).
        message: What the check looks for.
        examples: Up to ``EXAMPLES`` problem records (ids and relevant fields) for inspection.
    """
    check: str
    severity: str  # error | warning | info
    count: int
    message: str
    examples: list[Any] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Full validation result with JSON and Markdown renderings.

    Attributes:
        generated_at: ISO-8601 UTC time the report was produced.
        database: Name of the validated database.
        deep: Whether deep mode was used (all scripts' offsets, raw checksums).
        counts: Estimated document count per collection.
        issues: One entry per check, in the order run.
        checks_run: Names of the check groups that ran.
    """
    generated_at: str
    database: str
    deep: bool
    counts: dict[str, int]
    issues: list[Issue]
    checks_run: list[str]

    @property
    def errors(self) -> int:
        """Number of error-severity checks that found problems.

        Returns:
            Count of issues with severity ``error`` and a non-zero count.
        """
        return sum(1 for i in self.issues if i.severity == "error" and i.count)

    @property
    def warnings(self) -> int:
        """Number of warning-severity checks that found problems.

        Returns:
            Count of issues with severity ``warning`` and a non-zero count.
        """
        return sum(1 for i in self.issues if i.severity == "warning" and i.count)

    @property
    def passed(self) -> bool:
        """Whether validation passed (warnings do not fail it).

        Returns:
            True when no error check found problems.
        """
        return self.errors == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to a JSON-ready dict.

        Returns:
            All fields (issues as dicts) plus ``passed``, ``error_checks`` and ``warning_checks``.
        """
        data = asdict(self)
        data.update({"passed": self.passed, "error_checks": self.errors, "warning_checks": self.warnings})
        return data

    def to_markdown(self) -> str:
        """Render the report as Markdown.

        Returns:
            Markdown with a result summary, a collection-count table, a checks table and a JSON block of
            examples for each check that found problems.
        """
        lines = [f"# Validation report - {self.generated_at}", "",
                 f"* Database: `{self.database}`  |  deep: {self.deep}",
                 f"* Result: **{'PASSED' if self.passed else 'FAILED'}** "
                 f"({self.errors} error check(s), {self.warnings} warning check(s))", "",
                 "## Collection counts", "", "| collection | documents |", "|---|---:|"]
        lines += [f"| {k} | {v} |" for k, v in self.counts.items()]
        lines += ["", "## Checks", "", "| check | severity | count | message |", "|---|---|---:|---|"]
        for issue in self.issues:
            lines.append(f"| {issue.check} | {issue.severity} | {issue.count} | {issue.message} |")
        for issue in self.issues:
            if issue.count and issue.examples:
                lines += ["", f"### {issue.check}", "", "```json",
                          json.dumps(issue.examples[:EXAMPLES], indent=2, default=str), "```"]
        return "\n".join(lines) + "\n"


def _orphans(db: Database, collection: str, local: str, target: str, match: dict[str, Any] | None = None
             ) -> tuple[int, list[Any]]:
    """Find documents whose reference field points to a missing document (via ``$lookup`` on ``_id``).

    Args:
        db: Project database.
        collection: Collection to check.
        local: Reference field in ``collection``, e.g. ``script_id``.
        target: Collection the field should reference.
        match: Optional pre-filter, e.g. to skip null references.

    Returns:
        ``(orphan_count, examples)`` where examples are up to ``EXAMPLES`` documents with ``_id`` and the field.
    """
    pipeline: list[dict[str, Any]] = []
    if match:
        pipeline.append({"$match": match})
    pipeline += [
        {"$lookup": {"from": target, "localField": local, "foreignField": "_id", "as": "_ref",
                     "pipeline": [{"$project": {"_id": 1}}]}},
        {"$match": {"_ref": {"$size": 0}}},
        {"$facet": {"n": [{"$count": "n"}], "examples": [{"$limit": EXAMPLES}, {"$project": {"_id": 1, local: 1}}]}},
    ]
    result = next(db[collection].aggregate(pipeline, allowDiskUse=True))
    return (result["n"][0]["n"] if result["n"] else 0), result["examples"]


def run_validation(db: Database, *, raw_store: RawStore | None = None, deep: bool = False,
                   offset_sample: int = 25) -> ValidationReport:
    """Run every consistency and quality check on the database (read-only).

    Checks: duplicate keys/movies/scripts; missing, short, empty and failed screenplay text; broken shared
    script references and uncollected scripts; empty dialogue blocks; non-canonical or wrongly typed URLs;
    orphaned scripts/scenes/dialogues and character references; dialogue counts per character; scene and
    dialogue counts against ``structure_stats``; scene/dialogue text against script offsets; encoding
    problems; unusual HTML structures; incomplete extraction flags; raw artifacts on disk (when
    ``raw_store`` is given); records beyond the sample limit; scripts without rights metadata.

    Args:
        db: Project database.
        raw_store: Raw file store used to check artifacts; None skips the raw-artifact check.
        deep: Check offsets for all parsed scripts and verify raw checksums instead of only existence.
        offset_sample: Number of parsed scripts whose offsets are checked when not ``deep``.

    Returns:
        The report; it passes when no error-severity check found problems.
    """
    issues: list[Issue] = []
    checks: list[str] = []

    def add(check: str, severity: str, count: int, message: str, examples: list[Any] | None = None) -> None:
        """Record a check result (severity becomes ``info`` when nothing was found).

        Args:
            check: Check name.
            severity: ``error`` or ``warning``, used when ``count`` is non-zero.
            count: Number of problems found.
            message: Description of the check.
            examples: Problem records; truncated to ``EXAMPLES``.
        """
        issues.append(Issue(check, severity if count else "info", count, message, (examples or [])[:EXAMPLES]))

    counts = {name: db[name].estimated_document_count() for name in C.ALL}

    # ---------------------------------------------------------------- duplicates
    checks.append("duplicates")
    violations = duplicate_key_violations(db)
    add("duplicate_natural_keys", "error", sum(violations.values()),
        "natural keys appearing more than once (should be impossible with unique indexes)",
        [{"collection": k, "groups": v} for k, v in violations.items() if v])
    movie_groups = duplicate_movie_titles(db)
    add("duplicate_movies", "warning", len(movie_groups),
        "movies sharing a normalized title+year under different URLs (kept separate for review)",
        [{"title_key": g["_id"].get("title_key"), "year": g["_id"].get("year"), "ids": g["ids"]} for g in movie_groups])
    script_groups = duplicate_script_contents(db)
    add("duplicate_scripts", "warning", len(script_groups),
        "different script URLs with byte-identical extracted text",
        [{"content_hash": g["_id"], "ids": g["ids"]} for g in script_groups])

    # ---------------------------------------------------------------- missing text
    checks.append("missing_screenplay_text")
    failed = list(db[C.SCRIPTS].find({"status": {"$in": ["extraction_failed", "parse_failed"]}},
                                     {"_id": 1, "status": 1, "error": 1}).limit(EXAMPLES))
    add("missing_or_failed_screenplay_text", "error",
        db[C.SCRIPTS].count_documents({"status": {"$in": ["extraction_failed", "parse_failed"]}}),
        "scripts whose text could not be extracted or parsed (raw HTML preserved; try 'reparse')", failed)
    short_q = {"status": {"$in": PARSED}, "text_length": {"$lt": 2000}}
    add("very_short_screenplay_text", "warning", db[C.SCRIPTS].count_documents(short_q),
        "parsed scripts with fewer than 2000 characters (possibly incomplete page)",
        list(db[C.SCRIPTS].find(short_q, {"_id": 1, "text_length": 1}).limit(EXAMPLES)))
    empty_q = {"status": "empty_at_source"}
    add("scripts_empty_at_source", "warning", db[C.SCRIPTS].count_documents(empty_q),
        "script pages whose IMSDb script container is present but empty (a source-side gap; raw HTML preserved)",
        list(db[C.SCRIPTS].find(empty_q, {"_id": 1, "script_url": 1}).limit(EXAMPLES)))
    shared_q = {"status": "script_shared"}
    shared_missing = db[C.MOVIES].count_documents(
        {**shared_q, "$or": [{"script_id": {"$exists": False}}, {"shared_script_owner_id": {"$exists": False}}]})
    for m in db[C.MOVIES].find(shared_q, {"script_id": 1}):
        if m.get("script_id") and db[C.SCRIPTS].count_documents({"_id": m["script_id"]}, limit=1) == 0:
            shared_missing += 1
    add("broken_shared_script_references", "error", shared_missing,
        "movies marked as sharing another movie's script whose referenced script is missing",
        list(db[C.MOVIES].find(shared_q, {"_id": 1, "script_id": 1}).limit(EXAMPLES)) if shared_missing else [])
    pending_q = {"status": "detail_collected"}
    add("scripts_not_yet_collected", "warning", db[C.MOVIES].count_documents(pending_q),
        "movies whose detail page was collected but whose screenplay is not stored yet",
        list(db[C.MOVIES].find(pending_q, {"_id": 1, "script_url": 1}).limit(EXAMPLES)))

    # ---------------------------------------------------------------- dialogue blocks
    checks.append("empty_dialogue_blocks")
    empty_q = {"text": ""}
    add("empty_dialogue_blocks", "warning", db[C.DIALOGUES].count_documents(empty_q),
        "dialogue blocks with a character cue but no spoken text",
        list(db[C.DIALOGUES].find(empty_q, {"_id": 1, "speaker_raw": 1, "original_text": 1}).limit(EXAMPLES)))

    # ---------------------------------------------------------------- URLs
    checks.append("invalid_source_urls")
    bad_urls: list[dict[str, Any]] = []
    for collection, field_name, expected in ((C.MOVIES, "detail_url", PageType.MOVIE_DETAIL),
                                             (C.SCRIPTS, "script_url", PageType.SCRIPT_HTML)):
        for doc in db[collection].find({}, {field_name: 1}):
            url = doc.get(field_name)
            if not url or normalize_url(url) != url or classify_url(url) != expected:
                bad_urls.append({"collection": collection, "_id": doc["_id"], "url": url})
    for doc in db[C.MOVIES].find({"script_url": {"$ne": None, "$exists": True}}, {"script_url": 1}):
        url = doc["script_url"]
        if normalize_url(url) != url or classify_url(url) not in {PageType.SCRIPT_HTML, PageType.SCRIPT_PDF}:
            bad_urls.append({"collection": C.MOVIES, "_id": doc["_id"], "script_url": url})
    add("invalid_source_urls", "error", len(bad_urls), "URLs that are not canonical or have the wrong page type",
        bad_urls)

    # ---------------------------------------------------------------- references
    checks.append("references")
    n, ex = _orphans(db, C.SCRIPTS, "movie_id", C.MOVIES)
    add("scripts_without_movie", "error", n, "scripts referencing a missing movie", ex)
    n, ex = _orphans(db, C.SCENES, "script_id", C.SCRIPTS)
    add("broken_scene_references", "error", n, "scenes referencing a missing script", ex)
    n, ex = _orphans(db, C.DIALOGUES, "scene_id", C.SCENES, {"scene_id": {"$ne": None}})
    add("dialogues_with_broken_scene", "error", n, "dialogues referencing a missing scene", ex)
    n_null = db[C.DIALOGUES].count_documents({"scene_id": None})
    add("dialogues_without_scene", "warning", n_null, "dialogues not assigned to any scene",
        list(db[C.DIALOGUES].find({"scene_id": None}, {"_id": 1}).limit(EXAMPLES)))
    n, ex = _orphans(db, C.DIALOGUES, "script_id", C.SCRIPTS)
    add("dialogues_without_script", "error", n, "dialogues referencing a missing script", ex)

    checks.append("character_references")
    n, ex = _orphans(db, C.DIALOGUES, "character_id", C.CHARACTERS, {"character_id": {"$ne": None}})
    add("dialogues_with_broken_character", "error", n, "dialogues referencing a missing character", ex)
    mismatched: list[dict[str, Any]] = []
    actual = {row["_id"]: row["n"] for row in db[C.DIALOGUES].aggregate(
        [{"$match": {"character_id": {"$ne": None}}}, {"$group": {"_id": "$character_id", "n": {"$sum": 1}}}],
        allowDiskUse=True)}
    for char in db[C.CHARACTERS].find({}, {"dialogue_count": 1, "name": 1}):
        if actual.get(char["_id"], 0) != char.get("dialogue_count", 0):
            mismatched.append({"_id": char["_id"], "stored": char.get("dialogue_count"),
                               "actual": actual.get(char["_id"], 0)})
    add("inconsistent_character_references", "error", len(mismatched),
        "characters whose stored dialogue_count differs from their dialogue records", mismatched)
    unnamed = db[C.DIALOGUES].count_documents({"character_id": None})
    add("dialogues_without_character", "warning", unnamed, "dialogue blocks whose speaker could not be normalized",
        list(db[C.DIALOGUES].find({"character_id": None}, {"_id": 1, "speaker_raw": 1}).limit(EXAMPLES)))

    # ---------------------------------------------------------------- structure counts
    checks.append("structure_counts")
    scene_counts = Counter({r["_id"]: r["n"] for r in db[C.SCENES].aggregate(
        [{"$group": {"_id": "$script_id", "n": {"$sum": 1}}}])})
    dialogue_counts = Counter({r["_id"]: r["n"] for r in db[C.DIALOGUES].aggregate(
        [{"$group": {"_id": "$script_id", "n": {"$sum": 1}}}], allowDiskUse=True)})
    count_mismatch: list[dict[str, Any]] = []
    for script in db[C.SCRIPTS].find({"status": {"$in": PARSED}}, {"structure_stats": 1}):
        stats = script.get("structure_stats") or {}
        expected_scenes = stats.get("scenes", 0) + (1 if stats.get("has_prelude") else 0)
        if stats.get("scenes", 0) == 0 and stats.get("elements"):
            expected_scenes = 1  # unsegmented
        if scene_counts.get(script["_id"], 0) != expected_scenes or \
                dialogue_counts.get(script["_id"], 0) != stats.get("dialogue_blocks", 0):
            count_mismatch.append({"_id": script["_id"], "scenes": [scene_counts.get(script["_id"], 0), expected_scenes],
                                   "dialogues": [dialogue_counts.get(script["_id"], 0), stats.get("dialogue_blocks")]})
    add("structure_count_mismatch", "error", len(count_mismatch),
        "scripts whose stored scene/dialogue counts differ from the derived records ([actual, expected])",
        count_mismatch)

    # ---------------------------------------------------------------- offsets
    checks.append("text_offsets")
    offset_errors: list[dict[str, Any]] = []
    cursor = db[C.SCRIPTS].find({"status": {"$in": PARSED}}, {"text": 1})
    if not deep:
        cursor = cursor.limit(offset_sample)
    for script in cursor:
        text = script["text"]
        for d in db[C.DIALOGUES].find({"script_id": script["_id"]}, {"original_text": 1, "char_start": 1, "char_end": 1}):
            if text[d["char_start"]:d["char_end"]] != d["original_text"]:
                offset_errors.append({"_id": d["_id"], "kind": "dialogue"})
        for s in db[C.SCENES].find({"script_id": script["_id"]}, {"text": 1, "char_start": 1, "char_end": 1}):
            if text[s["char_start"]:s["char_end"]] != s["text"]:
                offset_errors.append({"_id": s["_id"], "kind": "scene"})
    add("text_offset_mismatch", "error", len(offset_errors),
        "scene/dialogue text that does not match its offsets in the script text" +
        ("" if deep else f" (sampled {offset_sample} scripts; use --deep for all)"), offset_errors)

    # ---------------------------------------------------------------- encoding / html structure / completeness
    checks.append("encoding_problems")
    enc_q = {"$or": [{"quality_flags": {"$in": ["encoding_suspect", "encoding_fallback"]}}, {"encoding.strict": False}]}
    add("encoding_problems", "warning", db[C.SCRIPTS].count_documents(enc_q),
        "scripts with replacement characters, mojibake, control characters or fallback decoding",
        list(db[C.SCRIPTS].find(enc_q, {"_id": 1, "encoding": 1, "structure_stats.encoding_problems": 1}).limit(EXAMPLES)))

    checks.append("unexpected_html_structures")
    method_counts = {r["_id"]: r["n"] for r in db[C.SCRIPTS].aggregate(
        [{"$group": {"_id": "$extraction.method", "n": {"$sum": 1}}}])}
    unusual = {k: v for k, v in method_counts.items() if k not in {"pre"}}
    add("unexpected_script_html", "warning", sum(unusual.values()),
        f"scripts not extracted from a single <pre> block (by method: {unusual or {}})",
        list(db[C.SCRIPTS].find({"extraction.method": {"$ne": "pre"}}, {"_id": 1, "extraction.method": 1}).limit(EXAMPLES)))
    det_q = {"detail_warnings": {"$elemMatch": {"$regex": "not found"}}}
    add("unexpected_detail_html", "warning", db[C.MOVIES].count_documents(det_q),
        "movie detail pages missing the expected script-details structure",
        list(db[C.MOVIES].find(det_q, {"_id": 1, "detail_warnings": 1}).limit(EXAMPLES)))

    checks.append("incomplete_extraction")
    flags = ["no_scene_headings", "no_dialogue", "low_extraction_ratio", "very_short_text", "high_uncertainty",
             "fallback_extraction"]
    inc_q = {"quality_flags": {"$in": flags}}
    add("incomplete_or_uncertain_extraction", "warning", db[C.SCRIPTS].count_documents(inc_q),
        "scripts flagged as possibly incomplete or uncertain: " + ", ".join(flags),
        list(db[C.SCRIPTS].find(inc_q, {"_id": 1, "quality_flags": 1, "structure_stats.uncertain_ratio": 1}).limit(EXAMPLES)))

    # ---------------------------------------------------------------- raw artifacts
    checks.append("raw_artifacts")
    if raw_store is not None:
        missing: list[dict[str, Any]] = []
        for art in db[C.RAW_ARTIFACTS].find({}, {"path": 1, "sha256": 1}):
            try:
                ok = raw_store.verify(art["path"], art["sha256"]) if deep else raw_store.absolute(art["path"]).exists()
            except Exception:  # noqa: BLE001 - report any filesystem problem
                ok = False
            if not ok:
                missing.append({"_id": art["_id"], "path": art["path"]})
        add("raw_artifact_integrity", "error", len(missing),
            "raw artifacts missing on disk" + (" or failing checksum" if deep else " (use --deep to verify checksums)"),
            missing)

    # ---------------------------------------------------------------- permission consistency
    checks.append("permission_consistency")
    perm = db[C.SOURCE_PERMISSIONS].find_one({"_id": "imsdb"}) or {}
    collected = db[C.MOVIES].count_documents({"detail_fetched_at": {"$exists": True}})
    level = perm.get("level")
    over = level == "sample" and collected > int(perm.get("sample_limit", 0))
    if level == "sample":
        message = (f"records collected beyond the sample limit ({collected} collected, "
                   f"limit {perm.get('sample_limit')}, current level 'sample')")
    else:
        message = f"records collected beyond the sample limit (not applicable: current level '{level}')"
    add("records_exceed_sample_limit", "warning", collected if over else 0, message)
    no_rights = db[C.SCRIPTS].count_documents({"rights": {"$exists": False}})
    add("scripts_without_rights_metadata", "error", no_rights, "scripts lacking attribution/rights metadata")

    return ValidationReport(generated_at=utcnow().isoformat(), database=db.name, deep=deep, counts=counts,
                            issues=issues, checks_run=checks)


def write_report(report: ValidationReport, directory: Path) -> tuple[Path, Path]:
    """Write a report as ``validation-<stamp>.json`` and ``validation-<stamp>.md``.

    Args:
        report: Validation report.
        directory: Output folder (created if missing).

    Returns:
        ``(json_path, markdown_path)``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    base = directory / f"validation-{stamp()}"
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    json_path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    return json_path, md_path
