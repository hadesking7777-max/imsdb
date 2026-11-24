"""Command-line interface.

Usage: ``python -m src.main <command> [options]`` - run with ``--help`` for details.

Exit codes: 0 ok, 1 error, 2 usage/config, 3 validation failed, 4 permission denied,
5 database unavailable, 6 network/access error, 7 backup failed, 130 interrupted.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src import __version__
from src.config import Settings, load_settings
from src.database.connection import Mongo
from src.database.indexes import ensure_schema
from src.errors import ConfigError, ExitCode, IMSDbError, PermissionDeniedError
from src.services.permissions import LEVELS, SOURCE_ID, default_imsdb_permission
from src.utils.logging import get_logger, setup_logging

log = get_logger("cli")


# ====================================================================== helpers
def _print(text: str = "") -> None:
    """Write one line to stdout and flush immediately.

    Args:
        text: Line without the trailing newline; empty prints a blank line.
    """
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _print_json(data: Any) -> None:
    """Print data as indented JSON (non-JSON values are converted with ``str``, non-ASCII text is kept).

    Args:
        data: JSON-like value to print.
    """
    _print(json.dumps(data, indent=2, default=str, ensure_ascii=False))


def _kv(title: str, data: dict[str, Any]) -> None:
    """Print a title followed by aligned ``key  value`` lines.

    Args:
        title: Heading line.
        data: Values to print; nested dicts are shown as ``k=v, ...`` (``-`` when empty).
    """
    _print(title)
    for key, value in data.items():
        if isinstance(value, dict):
            value = ", ".join(f"{k}={v}" for k, v in value.items()) or "-"
        _print(f"  {key:<32} {value}")


class Runtime:
    """Settings + database + graceful-shutdown wiring for one command."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Load settings, create the data folders and configure logging. MongoDB is connected lazily.

        Args:
            args: Parsed CLI arguments (uses ``env_file`` and ``log_level``).
        """
        self.settings: Settings = load_settings(args.env_file)
        self.settings.ensure_directories()
        setup_logging(self.settings.log_dir, args.log_level or self.settings.log_level)
        self.stop_event = threading.Event()
        self._mongo: Mongo | None = None
        self._ctx: Any = None
        self._signals = 0

    def install_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM/SIGBREAK handlers for a graceful shutdown.

        The first signal sets ``stop_event`` so running jobs finish their current items; the next one raises
        ``KeyboardInterrupt`` to abort. Signals unavailable on this platform (or outside the main thread) are skipped.
        """
        def handler(signum: int, _frame: Any) -> None:
            """Handle a shutdown signal.

            Args:
                signum: Signal number received.
                _frame: Current stack frame (unused).

            Raises:
                KeyboardInterrupt: On the second and later signals.
            """
            self._signals += 1
            if self._signals == 1:
                log.warning("shutdown requested (signal %s): finishing current items, press Ctrl+C again to abort",
                            signum)
                self.stop_event.set()
            else:
                raise KeyboardInterrupt
        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is not None:
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):  # not in main thread / unsupported
                    pass

    @property
    def mongo(self) -> Mongo:
        """MongoDB connection, opened on first use.

        The first access connects (with a ping), ensures collections, validators and indexes, and seeds the default
        IMSDb permission record if it is missing.

        Returns:
            The shared connection.
        """
        if self._mongo is None:
            self._mongo = Mongo.connect(self.settings)
            ensure_schema(self._mongo.db)
            from src.database.repositories import PermissionRepository
            PermissionRepository(self._mongo.db).seed(default_imsdb_permission())
        return self._mongo

    @property
    def ctx(self) -> Any:
        """Ingestion application context, created on first use (connects to MongoDB if needed).

        Returns:
            The shared ``AppContext``.
        """
        if self._ctx is None:
            from src.services.ingestion import AppContext
            self._ctx = AppContext(self.settings, self.mongo, stop_event=self.stop_event)
        return self._ctx

    def close(self) -> None:
        """Release resources: close the app context (which owns the connection), or only the connection."""
        if self._ctx is not None:
            self._ctx.close()
        elif self._mongo is not None:
            self._mongo.close()


def _job_exit(summary: dict[str, Any]) -> int:
    """Map a job summary to a CLI exit code.

    Args:
        summary: Result of an ingestion job (uses ``status`` and ``reason``).

    Returns:
        ``INTERRUPTED`` for an interrupted job, ``NETWORK_ERROR`` when it stopped because access was restricted,
        ``PERMISSION_DENIED`` when it stopped for any other reason, otherwise ``OK``.
    """
    status = summary.get("status")
    if status == "interrupted":
        return ExitCode.INTERRUPTED
    reason = summary.get("reason") or ""
    if status == "stopped" and "access restricted" in reason:
        return ExitCode.NETWORK_ERROR
    if status == "stopped" and reason:
        return ExitCode.PERMISSION_DENIED
    return ExitCode.OK


def _print_job(summary: dict[str, Any]) -> None:
    """Print a job summary: notices, job id/status/reason, sorted counters and the remaining fields.

    Args:
        summary: Result of an ingestion job.
    """
    for notice in summary.get("notices", []):
        _print(f"NOTE: {notice}")
    _print(f"Job {summary.get('job_id')} -> {summary.get('status')}"
           + (f" ({summary['reason']})" if summary.get("reason") else ""))
    counters = summary.get("counters") or {}
    if counters:
        _kv("Counters:", dict(sorted(counters.items())))
    extra = {k: v for k, v in summary.items() if k not in {"job_id", "status", "reason", "counters", "notices"}}
    if extra:
        _kv("Summary:", extra)


# ====================================================================== commands
def cmd_init_db(rt: Runtime, args: argparse.Namespace) -> int:
    """``init-db``: create collections, validators and indexes, then list the indexes.

    Args:
        rt: Runtime of this command.
        args: Parsed arguments (unused).

    Returns:
        ``ExitCode.OK``.
    """
    created = ensure_schema(rt.mongo.db)
    _print(f"Database '{rt.mongo.db.name}' ready on MongoDB {rt.mongo.server_version}.")
    for name, idx in created.items():
        _print(f"  {name:<20} indexes: {', '.join(idx) if idx else '(default _id only)'}")
    return ExitCode.OK


def cmd_inspect_source(rt: Runtime, args: argparse.Namespace) -> int:
    """``inspect-source``: fetch robots.txt and the policy pages, store the inspection and print the report.

    Accesses the network. At ``sample`` level the used part of the sample quota is shown too.

    Args:
        rt: Runtime of this command.
        args: Parsed arguments (unused).

    Returns:
        ``ExitCode.OK``.
    """
    from src.services.ingestion import IngestionService

    report = IngestionService(rt.ctx).inspect_source()
    perm = report["permission"]
    robots = report["robots"]
    assessment = report["assessment"]
    _print("=" * 72)
    _print(" Source policy inspection - IMSDb")
    _print("=" * 72)
    robots_note = ("no robots.txt: no machine-readable restrictions (not a licence)" if robots.allow_all else
                   "disallows ALL crawling" if robots.disallow_all else "rules parsed")
    _print(f"robots.txt          HTTP {robots.status} - {robots_note}")
    if robots.crawl_delay:
        _print(f"Crawl-delay         {robots.crawl_delay}s (applied)")
    for doc in report["documents"]:
        status = doc.get("status_code")
        _print(f"policy page         {doc['url']}  HTTP {status}  sha256={str(doc.get('sha256'))[:12]}"
               + (f"  ERROR {doc['error']}" if doc.get("error") else ""))
    _print("")
    _print(f"Fair-use claim found          {assessment.get('fair_use_claim_found')}")
    _print(f"Restrictive language found    {assessment.get('restriction_found')}")
    for phrase in assessment.get("restrictive_phrases", [])[:5]:
        _print(f"    - '{phrase['phrase']}' at {phrase['url']}")
    _print(f"Explicit grant found          {assessment.get('explicit_grant_found')}")
    _print(f"Bulk collection authorized    {perm.assessment.get('bulk_authorized', False) or perm.level == 'bulk'}")
    _print("")
    _print(f"Current permission level      {perm.level} (version {perm.version})")
    if perm.level == "sample":
        from src.database.repositories import MovieRepository
        used = MovieRepository(rt.mongo.db).count_records_collected()
        _print(f"Sample quota                  {used} of {perm.sample_limit} records used")
    _print(f"Recommendation                {assessment.get('recommendation')}")
    _print("Usage restrictions:")
    for item in perm.usage_restrictions:
        _print(f"  - {item}")
    _print("Full assessment: docs/SOURCE_PERMISSIONS.md   Evidence: source_permissions collection")
    return ExitCode.OK


def cmd_permit(rt: Runtime, args: argparse.Namespace) -> int:
    """``permit``: change the IMSDb source permission level (operator action, recorded with evidence).

    With ``--override-restriction`` it also sets ``assessment.restriction_overridden`` on the record.

    Args:
        rt: Runtime of this command.
        args: Uses ``level``, ``sample_limit``, ``granted_by``, ``evidence``, ``note``, ``override_restriction``
            and ``confirm``.

    Returns:
        ``ExitCode.OK``.

    Raises:
        ConfigError: ``--confirm`` was not given.
        PermissionDeniedError: ``bulk`` was requested without both ``--evidence`` and ``--granted-by``.
    """
    from src.database.repositories import PermissionRepository

    if not args.confirm:
        raise ConfigError("Changing source permissions requires --confirm.")
    if args.level == "bulk" and (not args.evidence or not args.granted_by):
        raise PermissionDeniedError("'bulk' requires --evidence describing the authorization and --granted-by.")
    repo = PermissionRepository(rt.mongo.db)
    record = repo.change_level(SOURCE_ID, level=args.level, sample_limit=args.sample_limit,
                               granted_by=args.granted_by or "operator", evidence=args.evidence or "(none given)",
                               note=args.note)
    if args.override_restriction:
        repo.col.update_one({"_id": SOURCE_ID}, {"$set": {"assessment.restriction_overridden": True}})
    _print(f"Permission for '{SOURCE_ID}' is now '{record.level}' (version {record.version}, "
           f"sample limit {record.sample_limit}). History entries: {len(record.history)}.")
    return ExitCode.OK


def cmd_discover(rt: Runtime, args: argparse.Namespace) -> int:
    """``discover``: read listing pages to discover movie records (no screenplay downloads).

    Args:
        rt: Runtime of this command.
        args: Uses ``quick`` (all-scripts page only) and ``refresh`` (re-fetch listing pages).

    Returns:
        Exit code derived from the job summary (see ``_job_exit``).
    """
    from src.services.ingestion import IngestionService

    rt.install_signal_handlers()
    summary = IngestionService(rt.ctx).discover(quick=args.quick, refresh=args.refresh)
    _print_job(summary)
    return _job_exit(summary)


def cmd_crawl(rt: Runtime, args: argparse.Namespace) -> int:
    """``crawl``: collect movie detail pages and screenplays.

    Args:
        rt: Runtime of this command.
        args: Uses ``limit``, ``all``, ``refresh_days`` and ``retry_failed``.

    Returns:
        Exit code derived from the job summary (see ``_job_exit``).

    Raises:
        ConfigError: None of ``--limit``, ``--all`` or ``--refresh-days`` was given.
    """
    from src.services.ingestion import IngestionService

    if not args.all and args.limit is None and args.refresh_days is None:
        raise ConfigError("Specify --limit N (sample collection) or --all (requires bulk authorization).")
    rt.install_signal_handlers()
    summary = IngestionService(rt.ctx).crawl(limit=args.limit, all_records=args.all,
                                             refresh_days=args.refresh_days, retry_failed=args.retry_failed)
    _print_job(summary)
    return _job_exit(summary)


def cmd_resume(rt: Runtime, args: argparse.Namespace) -> int:
    """``resume``: continue the most recent interrupted or crashed job.

    Args:
        rt: Runtime of this command.
        args: Parsed arguments (unused).

    Returns:
        ``ExitCode.OK`` when there is nothing to resume, otherwise the exit code of the resumed job.
    """
    from src.services.ingestion import IngestionService

    rt.install_signal_handlers()
    summary = IngestionService(rt.ctx).resume()
    if not summary.get("resumed"):
        _print(summary["message"])
        return ExitCode.OK
    _print(f"Resumed job {summary['resumed_from']} (released {summary['released_items']} stale item(s)).")
    _print_job(summary)
    return _job_exit(summary)


def cmd_reparse(rt: Runtime, args: argparse.Namespace) -> int:
    """``reparse``: re-run parsing from the local raw archive (no network).

    Args:
        rt: Runtime of this command.
        args: Uses ``listings`` (re-process listing pages instead of scripts), ``force`` and ``script_id``.

    Returns:
        ``ExitCode.OK``.
    """
    from src.services.ingestion import IngestionService

    rt.install_signal_handlers()
    service = IngestionService(rt.ctx)
    if args.listings:
        _kv("Listing pages re-processed from the raw archive:", service.reparse_listings())
        return ExitCode.OK
    result = service.reparse(force=args.force, script_ids=args.script_id)
    _kv(f"Re-parse with parser {result['parser_version']}:", result["counts"] or {"scripts": 0})
    return ExitCode.OK


def cmd_stats(rt: Runtime, args: argparse.Namespace) -> int:
    """``stats``: print the collection dashboard, or the statistics as JSON.

    Args:
        rt: Runtime of this command.
        args: Uses ``json``.

    Returns:
        ``ExitCode.OK``.
    """
    from src.services.stats import collect_stats, render_dashboard

    stats = collect_stats(rt.mongo.db, raw_dir=rt.settings.raw_dir, data_dir=rt.settings.data_dir)
    if args.json:
        _print_json(stats)
    else:
        _print(render_dashboard(stats))
    return ExitCode.OK


def cmd_validate(rt: Runtime, args: argparse.Namespace) -> int:
    """``validate``: check database consistency, write JSON and Markdown reports to ``report_dir`` and print them.

    Args:
        rt: Runtime of this command.
        args: Uses ``deep`` (verify all offsets and raw-file checksums) and ``json``.

    Returns:
        ``ExitCode.OK`` if validation passed, otherwise ``ExitCode.VALIDATION_FAILED``.
    """
    from src.services.validation import run_validation, write_report
    from src.storage.raw_store import RawStore

    report = run_validation(rt.mongo.db, raw_store=RawStore(rt.settings.raw_dir), deep=args.deep)
    json_path, md_path = write_report(report, rt.settings.report_dir)
    if args.json:
        _print_json(report.to_dict())
    else:
        _print(f"Validation {'PASSED' if report.passed else 'FAILED'}: {report.errors} error check(s), "
               f"{report.warnings} warning check(s)")
        for issue in report.issues:
            marker = {"error": "ERROR", "warning": "warn ", "info": " ok  "}[issue.severity]
            _print(f"  [{marker}] {issue.check:<40} {issue.count:>7}  {issue.message}")
        _print(f"Report: {md_path}\n        {json_path}")
    return ExitCode.OK if report.passed else ExitCode.VALIDATION_FAILED


def cmd_export(rt: Runtime, args: argparse.Namespace) -> int:
    """``export``: write records to JSON/JSONL files in a new folder under ``export_dir``.

    Args:
        rt: Runtime of this command.
        args: Uses ``format``, ``collections``, ``no_text`` and ``include_operational``.

    Returns:
        ``ExitCode.OK``.
    """
    from src.services.export import export_records

    result = export_records(rt.mongo.db, rt.settings.export_dir, fmt=args.format,
                            collections=args.collections, include_text=not args.no_text,
                            include_operational=args.include_operational)
    _print(f"Exported to {result['directory']}")
    for item in result["files"]:
        _print(f"  {item['file']:<28} {item['records']:>8} records  {item['bytes']:>12,} bytes")
    _print(f"Attribution notice written to {Path(result['directory']) / 'ATTRIBUTION.txt'}")
    return ExitCode.OK


def cmd_backup(rt: Runtime, args: argparse.Namespace) -> int:
    """``backup``: list backups (``--list``), verify one (``--verify DIR``) or create a new one.

    Args:
        rt: Runtime of this command.
        args: Uses ``list``, ``verify``, ``no_raw``, ``include_exports`` and ``method``.

    Returns:
        ``ExitCode.OK``, or ``ExitCode.BACKUP_FAILED`` when ``--verify`` finds problems. A newly created
        backup that fails its own verification raises ``BackupError`` in ``create_backup`` (exit code 7).
    """
    from src.services import backup

    if args.list:
        for item in backup.list_backups(rt.settings):
            _print(f"{item['directory']}  {item['created_at']}  {item['method']}  {item['bytes']:,} bytes")
        return ExitCode.OK
    if args.verify:
        result = backup.verify_backup(Path(args.verify))
        _print(f"Backup {result.get('backup_id')}: {'OK' if result['ok'] else 'FAILED'} "
               f"({result['checked_files']} file(s) checked)")
        for problem in result["problems"]:
            _print(f"  - {problem}")
        return ExitCode.OK if result["ok"] else ExitCode.BACKUP_FAILED
    result = backup.create_backup(rt.settings, rt.mongo, include_raw=not args.no_raw,
                                  include_exports=args.include_exports, method=args.method)
    _print(f"Backup created: {result['directory']}")
    _print(f"  method: {result['method']}   verified: {result['verification']['ok']}")
    for item in result["files"]:
        _print(f"  {item['path']:<40} {item['bytes']:>14,} bytes  sha256={item['sha256'][:16]}")
    return ExitCode.OK


def cmd_restore(rt: Runtime, args: argparse.Namespace) -> int:
    """``restore``: restore a backup into the local MongoDB (and the raw archive).

    Args:
        rt: Runtime of this command.
        args: Uses ``path``, ``confirm``, ``drop``, ``target_db`` and ``no_raw``.

    Returns:
        ``ExitCode.OK``.

    Raises:
        ConfigError: ``--confirm`` was not given.
    """
    from src.services import backup

    if not args.confirm:
        raise ConfigError("Restoring modifies the database; re-run with --confirm.")
    result = backup.restore_backup(Path(args.path), rt.settings, drop=args.drop, target_db=args.target_db,
                                   restore_raw=not args.no_raw)
    _print_json(result)
    return ExitCode.OK


# ====================================================================== parser
def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with every subcommand, including the chatbot ones from ``src.chatbot.cli``.

    Each subcommand stores its handler in ``args.func``.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(prog="python -m src.main", description="IMSDb local screenplay database and offline chatbot")
    parser.add_argument("--version", action="version", version=f"imsdb-local {__version__}")
    parser.add_argument("--env-file", help="path to a .env file (default: ./.env)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="console log level")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def add(name: str, func: Callable[[Runtime, argparse.Namespace], int], help_text: str) -> argparse.ArgumentParser:
        """Register a subcommand.

        Args:
            name: Command name as typed on the command line.
            func: Handler called as ``func(runtime, args)``; returns the exit code.
            help_text: Help and description text.

        Returns:
            The subparser, so options can be added to it.
        """
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.set_defaults(func=func)
        return p

    add("init-db", cmd_init_db, "create collections, validators and indexes")
    add("inspect-source", cmd_inspect_source, "fetch robots.txt and policy pages; report collection permissions")

    p = add("permit", cmd_permit, "record a change of the source permission level (operator action)")
    p.add_argument("--level", required=True, choices=LEVELS)
    p.add_argument("--sample-limit", type=int, help="total records allowed at 'sample' level")
    p.add_argument("--granted-by", help="who authorized the change")
    p.add_argument("--evidence", help="description/location of the authorization evidence")
    p.add_argument("--note")
    p.add_argument("--override-restriction", action="store_true",
                   help="acknowledge restrictive policy language found by inspect-source")
    p.add_argument("--confirm", action="store_true")

    p = add("discover", cmd_discover, "discover records from listing pages (no screenplay downloads)")
    p.add_argument("--quick", action="store_true", help="use only the all-scripts page (1 request)")
    p.add_argument("--refresh", action="store_true", help="re-fetch listing pages to detect new/changed records")

    p = add("crawl", cmd_crawl, "collect movie records and screenplays")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--limit", type=int, help="maximum number of records to collect in this run")
    group.add_argument("--all", action="store_true", help="collect every discovered record (requires bulk permission)")
    p.add_argument("--refresh-days", type=float, help="re-check already collected records older than N days")
    p.add_argument("--retry-failed", action="store_true", help="re-queue URLs that failed permanently")

    add("resume", cmd_resume, "resume the most recent interrupted or crashed job")

    p = add("reparse", cmd_reparse, "re-run parsing from the local raw archive (no network)")
    p.add_argument("--force", action="store_true", help="re-parse every script, not only outdated/failed ones")
    p.add_argument("--script-id", action="append", help="only this script id (repeatable)")
    p.add_argument("--listings", action="store_true",
                   help="re-process fetched listing pages (discovery) instead of scripts")

    p = add("stats", cmd_stats, "show the collection dashboard and coverage report")
    p.add_argument("--json", action="store_true")

    p = add("validate", cmd_validate, "check database consistency and write a validation report")
    p.add_argument("--deep", action="store_true", help="verify all offsets and raw-file checksums")
    p.add_argument("--json", action="store_true")

    p = add("export", cmd_export, "export authorized records to local JSON files")
    p.add_argument("--format", choices=["json", "jsonl"], default="json")
    p.add_argument("--collections", nargs="+", help="collections to export")
    p.add_argument("--no-text", action="store_true", help="omit full screenplay/scene text")
    p.add_argument("--include-operational", action="store_true", help="also export queue/logs/jobs/raw index")

    p = add("backup", cmd_backup, "create, verify or list local backups")
    p.add_argument("--no-raw", action="store_true", help="skip the raw HTML archive")
    p.add_argument("--include-exports", action="store_true")
    p.add_argument("--method", choices=["auto", "mongodump", "python"], default="auto")
    p.add_argument("--verify", metavar="BACKUP_DIR", help="verify an existing backup instead of creating one")
    p.add_argument("--list", action="store_true", help="list existing backups")

    p = add("restore", cmd_restore, "restore a backup into local MongoDB (and raw archive)")
    p.add_argument("path", help="backup directory")
    p.add_argument("--confirm", action="store_true")
    p.add_argument("--drop", action="store_true", help="drop collections before restoring")
    p.add_argument("--target-db", help="restore into a different database name")
    p.add_argument("--no-raw", action="store_true")

    from src.chatbot.cli import register as register_chatbot
    register_chatbot(add)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: parse the arguments, run the command and map errors to exit codes.

    An ``IMSDbError`` prints ``error: ...`` and returns its exit code; Ctrl+C returns 130; any other exception is
    logged with its traceback and returns 1. The runtime is always closed. Invalid arguments make argparse exit
    with code 2.

    Args:
        argv: Arguments without the program name; None uses ``sys.argv``.

    Returns:
        The process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    runtime: Runtime | None = None
    try:
        runtime = Runtime(args)
        return int(args.func(runtime, args))
    except IMSDbError as exc:
        sys.stderr.write(f"error: {exc}\n")
        log.debug("command failed", exc_info=True)
        return int(exc.exit_code)
    except KeyboardInterrupt:
        sys.stderr.write("aborted\n")
        return int(ExitCode.INTERRUPTED)
    except Exception as exc:  # unexpected: show a concise message, full trace in the log file
        log.exception("unexpected failure")
        sys.stderr.write(f"unexpected error: {exc.__class__.__name__}: {exc} (details in logs/imsdb.log)\n")
        return int(ExitCode.ERROR)
    finally:
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    sys.exit(main())
