"""End-to-end ingestion tests against an in-memory fake IMSDb (no network access)."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from src.database.models import Collections as C
from src.database.repositories import Repositories
from src.errors import PermissionDeniedError
from src.main import _job_exit
from src.services.export import export_records
from src.services.ingestion import IngestionService
from src.services.stats import collect_stats, render_dashboard
from src.services.validation import run_validation, write_report
from src.utils.clock import utcnow
from tests.conftest import BASE_URL, FakeSite

pytestmark = pytest.mark.mongodb

DETAIL = {
    "lighthouse": "/Movie Scripts/Lighthouse Keeper, The Script.html",
    "paper": "/Movie Scripts/Paper Moons Script.html",
    "orchard": "/Movie Scripts/Quiet Orchard Script.html",
    "night": "/Movie Scripts/Night Shift at Dawn Script.html",
    "ember": "/Movie Scripts/Ember Road Script.html",
}
SCRIPT = {"lighthouse": "/scripts/Lighthouse-Keeper,-The.html", "paper": "/scripts/Paper-Moons.html",
          "orchard": "/scripts/Quiet-Orchard.html"}


def _set_level(app, level: str, sample_limit: int | None = None) -> None:
    """Change the stored IMSDb permission level, as the ``permit`` command would.

    Args:
        app: Ingestion context whose database holds the permission record.
        level: New permission level.
        sample_limit: New sample quota, or ``None`` to keep the stored one.
    """
    Repositories(app.db).permissions.change_level("imsdb", level=level, sample_limit=sample_limit,
                                                  granted_by="test", evidence="test fixture")


def _service(app) -> IngestionService:
    """Create an ingestion service and load its permission gate without fetching robots.txt.

    Args:
        app: Ingestion context to run against.

    Returns:
        The service; the default permission record (level ``sample``) is now seeded.
    """
    service = IngestionService(app)
    service.load_gate(refresh_robots=False)  # seeds the default permission record
    return service


# ====================================================================== inspect-source
def test_inspect_source_records_policy_evidence(app, site: FakeSite) -> None:
    """``inspect-source`` stores robots/policy evidence, never raises the level, and fetches no record pages.

    Args:
        app: Ingestion context on a throwaway database, fetching from the fake site.
        site: The fake IMSDb; its request log shows what was fetched.
    """
    report = IngestionService(app).inspect_source()
    assert report["robots"].status == 404 and report["robots"].allow_all
    assessment = report["assessment"]
    assert assessment["fair_use_claim_found"] and not assessment["restriction_found"]
    assert report["permission"].level == "sample"  # never raised automatically
    privacy = next(d for d in report["documents"] if d["url"].endswith("/privacy"))
    assert privacy["final_url"] == "https://imsdb.com/privacy/"  # live-site redirect shape, no false loop
    perm = app.db[C.SOURCE_PERMISSIONS].find_one({"_id": "imsdb"})
    assert len(perm["policy_documents"]) == 4 and perm["inspections"]
    assert app.db[C.RAW_ARTIFACTS].count_documents({"page_type": "policy"}) == 4
    assert not any("/scripts/" in r or "Movie%20Scripts" in r for r in site.requests)


# ====================================================================== discovery
def test_discover_finds_records_without_downloading_screenplays(app, site: FakeSite) -> None:
    """Discovery records all five movies from listing pages and queues their detail pages without fetching them.

    Args:
        app: Ingestion context on a throwaway database, fetching from the fake site.
        site: The fake IMSDb; its request log shows what was fetched.
    """
    summary = _service(app).discover()
    assert summary["status"] == "completed"
    assert app.db[C.MOVIES].count_documents({}) == 5
    assert app.db[C.CRAWL_QUEUE].count_documents({"page_type": "movie_detail", "status": "pending"}) == 5
    fetched = set(site.requests)
    assert {BASE_URL(p) for p in ("/", "/all-scripts.html", "/alphabetical/L", "/alphabetical/P", "/genre/Drama")} <= fetched
    assert not any("Movie%20Scripts" in u or "/scripts/" in u for u in fetched)  # no detail/script downloads
    assert not any("writer.php" in u or "/TV/" in u or "example.org" in u for u in fetched)  # out of scope
    lighthouse = app.db[C.MOVIES].find_one({"detail_url": BASE_URL(DETAIL["lighthouse"])})
    assert {"all-scripts", "alphabetical:L", "genre:Drama"} <= set(lighthouse["listing_sources"])
    assert lighthouse["status"] == "discovered" and "detail_fetched_at" not in lighthouse


def test_rediscovery_is_idempotent(app, site: FakeSite) -> None:
    """Repeated discovery re-fetches nothing unless ``refresh=True`` and never duplicates movies or queue items.

    Args:
        app: Ingestion context on a throwaway database, fetching from the fake site.
        site: The fake IMSDb; its request log shows what was fetched.
    """
    service = _service(app)
    service.discover()
    requests_after_first = len(site.requests)
    service.discover()
    assert len(site.requests) == requests_after_first  # done pages are not re-fetched
    service.discover(refresh=True)
    assert len(site.requests) > requests_after_first  # explicit refresh re-checks listings
    assert app.db[C.MOVIES].count_documents({}) == 5
    assert app.db[C.CRAWL_QUEUE].count_documents({}) == 10  # 5 listing pages + 5 detail pages


# ====================================================================== crawl
def test_crawl_collects_records_and_rerun_is_idempotent(app, site: FakeSite) -> None:
    """A crawl stores movies, scripts and derived records (PDFs skipped, raw kept); a rerun adds nothing.

    Args:
        app: Ingestion context on a throwaway database, fetching from the fake site.
        site: The fake IMSDb; its request log shows what was fetched.
    """
    service = _service(app)
    service.discover()
    summary = service.crawl(limit=5)
    assert summary["status"] == "completed" and summary["counters"]["records_processed"] == 5
    movies = {m["detail_url"]: m for m in app.db[C.MOVIES].find()}
    assert movies[BASE_URL(DETAIL["lighthouse"])]["status"] == "script_collected"
    assert movies[BASE_URL(DETAIL["night"])]["status"] == "script_unavailable"
    assert movies[BASE_URL(DETAIL["ember"])]["script_availability"] == "pdf"
    assert not any(u.endswith(".pdf") for u in site.requests)  # PDFs are not fetched
    assert app.db[C.SCRIPTS].count_documents({"status": {"$in": ["parsed", "parsed_with_warnings"]}}) == 3
    counts = {c: app.db[c].count_documents({}) for c in (C.MOVIES, C.SCRIPTS, C.SCENES, C.DIALOGUES, C.CHARACTERS,
                                                         C.RAW_ARTIFACTS)}
    assert counts[C.DIALOGUES] == 8 + 5 + 3 and counts[C.CHARACTERS] == 2 + 2 + 2

    lighthouse = movies[BASE_URL(DETAIL["lighthouse"])]
    assert lighthouse["writers"][0]["name"] == "Ada Fenwick" and lighthouse["genres"] == ["Drama", "Mystery"]
    assert lighthouse["rights"]["permission_level_at_collection"] == "sample"
    script = app.db[C.SCRIPTS].find_one({"movie_id": lighthouse["_id"]})
    raw_path = app.settings.raw_dir / script["raw"]["path"]
    assert raw_path.exists() and b"<pre>" in raw_path.read_bytes()  # original response preserved

    second = service.crawl(limit=5)  # nothing new to collect; must not duplicate anything
    assert second["counters"].get("records_processed", 0) == 0
    assert {c: app.db[c].count_documents({}) for c in counts} == counts

    report = run_validation(app.db, raw_store=app.raw_store, deep=True)
    assert report.passed, [i for i in report.issues if i.severity == "error"]


def test_crawl_without_discovery_runs_quick_discovery(app, site: FakeSite) -> None:
    """Crawling an empty database first runs quick discovery from all-scripts.html only.

    Args:
        app: Ingestion context on a throwaway database, fetching from the fake site.
        site: The fake IMSDb; its request log shows what was fetched.
    """
    summary = _service(app).crawl(limit=1)
    assert any("quick discovery" in n for n in summary["notices"])
    assert BASE_URL("/") not in site.requests  # only all-scripts.html was used
    assert app.db[C.MOVIES].count_documents({"detail_fetched_at": {"$exists": True}}) == 1


def test_sample_quota_is_enforced_cumulatively(app, site: FakeSite) -> None:
    """The sample quota limits a run and, once used up across runs, refuses further crawls.

    Args:
        app: Ingestion context on a throwaway database, fetching from the fake site.
        site: The fake IMSDb; its request log shows what was fetched.
    """
    service = _service(app)
    _set_level(app, "sample", sample_limit=2)
    service.discover()
    summary = service.crawl(limit=5)
    assert any("limiting this run to 2" in n for n in summary["notices"])
    assert app.db[C.MOVIES].count_documents({"detail_fetched_at": {"$exists": True}}) == 2
    assert sum(1 for u in site.requests if "Movie%20Scripts" in u) == 2
    with pytest.raises(PermissionDeniedError, match="quota is used up"):
        service.crawl(limit=1)


def test_crawl_all_requires_bulk_authorization(app, site: FakeSite) -> None:
    """``crawl --all`` is refused (with no record fetches) until the level is ``bulk``, then collects everything.

    Args:
        app: Ingestion context on a throwaway database, fetching from the fake site.
        site: The fake IMSDb; its request log shows what was fetched.
    """
    service = _service(app)
    with pytest.raises(PermissionDeniedError, match="bulk"):
        service.crawl(all_records=True)
    assert not any("Movie%20Scripts" in u for u in site.requests)
    _set_level(app, "bulk")
    summary = service.crawl(all_records=True)
    assert summary["status"] == "completed"
    assert app.db[C.MOVIES].count_documents({"detail_fetched_at": {"$exists": True}}) == 5


def test_discovery_only_level_blocks_record_collection(app, site: FakeSite) -> None:
    """At ``discovery_only`` discovery works but crawling records is refused.

    Args:
        app: Ingestion context on a throwaway database, fetching from the fake site.
        site: The fake IMSDb; its request log shows what was fetched.
    """
    service = _service(app)
    _set_level(app, "discovery_only")
    service.discover()
    with pytest.raises(PermissionDeniedError):
        service.crawl(limit=1)


def test_robots_disallow_is_respected(app, site: FakeSite) -> None:
    """Script pages disallowed by robots.txt are never fetched and their queue items are skipped.

    Args:
        app: Ingestion context on a throwaway database, fetching from the fake site.
        site: The fake IMSDb; its request log shows what was fetched.
    """
    site.add("/robots.txt", "User-agent: *\nDisallow: /scripts/\n", headers={"content-type": "text/plain"})
    service = IngestionService(app)
    service.discover()
    summary = service.crawl(limit=5)
    assert summary["status"] == "completed"
    assert not any("/scripts/" in u for u in site.requests)
    assert app.db[C.CRAWL_QUEUE].count_documents({"page_type": "script_html", "status": "skipped"}) == 3
    assert app.db[C.SCRIPTS].count_documents({}) == 0


def test_access_restriction_stops_the_crawl(app, site: FakeSite) -> None:
    """A 403 on record pages stops the crawl at once (exit code 6) without retrying.

    Args:
        app: Ingestion context on a throwaway database, fetching from the fake site.
        site: The fake IMSDb; its request log shows what was fetched.
    """
    for path in DETAIL.values():
        site.add(path, b"Forbidden", status=403)
    service = _service(app)
    service.discover()
    summary = service.crawl(limit=5)
    assert summary["status"] == "stopped" and "access restricted" in summary["reason"]
    assert _job_exit(summary) == 6
    assert app.db[C.CRAWL_QUEUE].count_documents({"status": "blocked"}) >= 1
    blocked_requests = sum(1 for u in site.requests if "Movie%20Scripts" in u)
    assert blocked_requests <= 2  # stopped immediately (one per worker at most), never retried


def test_transient_errors_are_retried_transparently(app, site: FakeSite) -> None:
    """A single 503 on a detail page is retried and the record is still collected.

    Args:
        app: Ingestion context on a throwaway database, fetching from the fake site.
        site: The fake IMSDb; its request log shows what was fetched.
    """
    site.script(DETAIL["lighthouse"], [(503, b"busy", None)])
    service = _service(app)
    service.discover()
    service.crawl(limit=5)
    assert site.count(DETAIL["lighthouse"]) == 2
    assert app.db[C.MOVIES].find_one({"detail_url": BASE_URL(DETAIL["lighthouse"])})["status"] == "script_collected"


# ====================================================================== interruption / resume
def test_interrupted_crawl_resumes_without_duplicates(app, site: FakeSite, monkeypatch: pytest.MonkeyPatch) -> None:
    """An interrupted crawl releases its leases (exit 130), and ``resume`` finishes it fetching each page at most once.

    Args:
        app: Ingestion context on a throwaway database, fetching from the fake site.
        site: The fake IMSDb; its request log shows what was fetched.
        monkeypatch: Wraps ``IngestionService.handle_detail`` to set the stop event after the first record.
    """
    service = _service(app)
    service.discover()
    original = IngestionService.handle_detail

    def interrupting(self, item, gate, job_id):
        """Process the record normally, then request shutdown as Ctrl+C would.

        Args:
            self: The ingestion service (this replaces a method, so ``self`` is passed explicitly).
            item: The claimed queue item.
            gate: The permission gate.
            job_id: Id of the running crawl job.

        Returns:
            The outcome of the original ``handle_detail``.
        """
        outcome = original(self, item, gate, job_id)
        app.stop_event.set()  # simulate Ctrl+C after the first record
        return outcome

    monkeypatch.setattr(IngestionService, "handle_detail", interrupting)
    first = service.crawl(limit=4)
    assert first["status"] == "interrupted" and _job_exit(first) == 130
    assert app.db[C.CRAWL_QUEUE].count_documents({"status": "in_progress"}) == 0  # leases released
    job = app.db[C.CRAWL_JOBS].find_one({"_id": first["job_id"]})
    assert job["status"] == "interrupted"
    processed_before = first["counters"]["records_processed"]

    monkeypatch.setattr(IngestionService, "handle_detail", original)
    app.stop_event.clear()
    resumed = IngestionService(app).resume()
    assert resumed["resumed"] and resumed["resumed_from"] == first["job_id"]
    assert resumed["counters"].get("records_processed", 0) == 4 - processed_before
    assert app.db[C.MOVIES].count_documents({"detail_fetched_at": {"$exists": True}}) == 4
    assert app.db[C.CRAWL_JOBS].find_one({"_id": first["job_id"]})["status"] == "stopped"
    for path in DETAIL.values():
        assert site.count(path) <= 1  # every detail page fetched at most once
    assert IngestionService(app).resume()["resumed"] is False  # nothing left to resume


def test_resume_recovers_from_a_crashed_process(app, site: FakeSite) -> None:
    """``resume`` adopts a job with a stale heartbeat, releases its leased items and completes the work.

    Args:
        app: Ingestion context on a throwaway database, fetching from the fake site.
        site: The fake IMSDb; its request log shows what was fetched.
    """
    service = _service(app)
    service.discover()
    repos = Repositories(app.db)
    job = repos.jobs.start("crawl", {"limit": 2, "all": False, "refresh_days": None, "effective_limit": 2})
    claimed = repos.queue.claim_next(job_id=job.id, worker_id="dead-worker", page_types=["movie_detail"],
                                     lease_seconds=3600)
    app.db[C.CRAWL_JOBS].update_one({"_id": job.id}, {"$set": {"heartbeat_at": utcnow() - timedelta(hours=1)}})
    result = IngestionService(app).resume()
    assert result["resumed"] and result["released_items"] >= 1
    assert repos.queue.get_by_url(claimed["url"])["status"] == "done"
    assert app.db[C.MOVIES].count_documents({"detail_fetched_at": {"$exists": True}}) == 2


# ====================================================================== change detection / reparse
def test_refresh_detects_changes_and_preserves_previous_version(app, site: FakeSite) -> None:
    """A refresh crawl re-fetches pages; changed scripts get version 2 with version 1 archived, others stay.

    Args:
        app: Ingestion context on a throwaway database, fetching from the fake site.
        site: The fake IMSDb; its request log shows what was fetched.
    """
    service = _service(app)
    service.discover()
    service.crawl(limit=5)
    original_html = site.pages[BASE_URL(SCRIPT["lighthouse"])][1]
    site.add(SCRIPT["lighthouse"], original_html.replace(b"Steady now. Steady.", b"Steady now. Hold steady."))
    summary = service.crawl(refresh_days=0)
    assert summary["status"] == "completed"
    script = app.db[C.SCRIPTS].find_one({"script_url": BASE_URL(SCRIPT["lighthouse"])})
    assert script["version"] == 2 and "Hold steady." in script["text"]
    archived = app.db[C.SCRIPT_VERSIONS].find_one({"script_id": script["_id"]})
    assert archived["version"] == 1 and "Steady now. Steady." in archived["document"]["text"]
    unchanged = app.db[C.SCRIPTS].find_one({"script_url": BASE_URL(SCRIPT["paper"])})
    assert unchanged["version"] == 1
    assert app.db[C.RAW_ARTIFACTS].count_documents({"url": BASE_URL(SCRIPT["lighthouse"])}) == 2
    assert app.db[C.MOVIES].count_documents({"detail_versions.0": {"$exists": True}}) == 0  # details unchanged
    assert run_validation(app.db, raw_store=app.raw_store).passed


def test_reparse_uses_local_archive_only(app, site: FakeSite) -> None:
    """Forced reparse rebuilds every script from the raw archive with no requests and identical counts.

    Args:
        app: Ingestion context on a throwaway database, fetching from the fake site.
        site: The fake IMSDb; its request log shows what was fetched.
    """
    service = _service(app)
    service.discover()
    service.crawl(limit=5)
    before = {c: app.db[c].count_documents({}) for c in (C.SCENES, C.DIALOGUES, C.CHARACTERS)}
    requests_before = len(site.requests)
    result = service.reparse(force=True)
    assert sum(result["counts"].values()) == 3
    assert len(site.requests) == requests_before
    assert {c: app.db[c].count_documents({}) for c in before} == before


# ====================================================================== validation / stats / export
def test_validation_detects_broken_references(app, site: FakeSite) -> None:
    """Validation fails on a deleted scene and an emptied dialogue, and the written report says FAILED.

    Args:
        app: Ingestion context on a throwaway database, fetching from the fake site.
        site: The fake IMSDb; its request log shows what was fetched.
    """
    service = _service(app)
    service.discover()
    service.crawl(limit=5)
    app.db[C.SCENES].delete_one({"scene_number": 1})
    app.db[C.DIALOGUES].update_one({}, {"$set": {"text": ""}})
    report = run_validation(app.db, raw_store=app.raw_store)
    failing = {i.check for i in report.issues if i.count and i.severity == "error"}
    assert not report.passed
    assert {"dialogues_with_broken_scene", "structure_count_mismatch"} <= failing
    assert any(i.check == "empty_dialogue_blocks" and i.count >= 1 for i in report.issues)
    json_path, md_path = write_report(report, app.settings.report_dir)
    assert json_path.exists() and "FAILED" in md_path.read_text(encoding="utf-8")


def test_stats_distinguish_discovered_from_collected(app, site: FakeSite) -> None:
    """Stats separate discovered from processed movies and the dashboard warns that coverage is unproven.

    Args:
        app: Ingestion context on a throwaway database, fetching from the fake site.
        site: The fake IMSDb; its request log shows what was fetched.
    """
    service = _service(app)
    service.discover()
    service.crawl(limit=2)
    stats = collect_stats(app.db, raw_dir=app.settings.raw_dir, data_dir=app.settings.data_dir)
    assert stats["records"]["discovered_movies"] == 5
    assert stats["records"]["processed_movies"] == 2
    assert stats["coverage"]["unresolved_records"] >= 3
    assert stats["queue"]["pending_urls"] >= 3
    dashboard = render_dashboard(stats)
    assert "does not prove the collection is complete" in dashboard


EMPTY_SCRIPT_PAGE = (
    "<html><head><title>Quiet Orchard Script at IMSDb.</title></head><body><table><tr><td>"
    "<table width=\"100%\"><tr><td class=\"scrtext\">\n<pre></pre>\n"
    "<table><tr><td><h1>Quiet Orchard</h1><b>Writers</b> : <a href=\"/writer.php?w=June Park\">June Park</a>"
    "<br><b>Genres</b> : <a href=\"/genre/Drama\">Drama</a></td></tr></table>"
    "</td></tr></table></td></tr></table></body></html>"
)


def test_empty_source_script_is_not_a_failure(app, site: FakeSite) -> None:
    """A script page with an empty ``<pre>`` is stored as ``empty_at_source``, not counted as a parse error.

    Args:
        app: Ingestion context on a throwaway database, fetching from the fake site.
        site: The fake IMSDb; its request log shows what was fetched.
    """
    site.add(SCRIPT["orchard"], EMPTY_SCRIPT_PAGE, headers={"content-type": "text/html; charset=UTF-8"})
    service = _service(app)
    service.discover()
    summary = service.crawl(limit=5)
    assert summary["counters"].get("scripts_empty_at_source") == 1 and "parse_errors" not in summary["counters"]
    script = app.db[C.SCRIPTS].find_one({"script_url": BASE_URL(SCRIPT["orchard"])})
    assert script["status"] == "empty_at_source" and script["text"] == ""
    assert (app.settings.raw_dir / script["raw"]["path"]).exists()  # the empty page itself is preserved
    movie = app.db[C.MOVIES].find_one({"detail_url": BASE_URL(DETAIL["orchard"])})
    assert movie["status"] == "script_empty_at_source"
    assert app.db[C.CRAWL_LOGS].count_documents({"event": "parse_error"}) == 0

    cov = collect_stats(app.db, raw_dir=app.settings.raw_dir, data_dir=app.settings.data_dir)["coverage"]
    assert cov["failed_records"] == 0 and cov["scripts_empty_on_source"] == 1 and cov["unresolved_records"] == 0
    report = run_validation(app.db, raw_store=app.raw_store, deep=True)
    assert report.passed, [i for i in report.issues if i.severity == "error"]
    assert next(i for i in report.issues if i.check == "scripts_empty_at_source").count == 1


def test_movie_linking_another_movies_script_is_marked_shared(app, site: FakeSite) -> None:
    """Two movies linking one script share a single script document; the second is marked ``script_shared``.

    Args:
        app: Ingestion context on a throwaway database, fetching from the fake site.
        site: The fake IMSDb; its request log shows what was fetched.
    """
    # Mirrors IMSDb's "Deadpool & Wolverine" page, which links to the original "Deadpool" script.
    detail = (Path(__file__).parent / "fixtures" / "detail_paper_moons.html").read_text(encoding="utf-8")
    site.add(DETAIL["paper"], detail.replace(SCRIPT["paper"], SCRIPT["lighthouse"]),
             headers={"content-type": "text/html; charset=UTF-8"})
    service = _service(app)
    service.discover()
    service.crawl(limit=5)
    assert not any(u.endswith(SCRIPT["paper"]) for u in site.requests)
    pair = [app.db[C.MOVIES].find_one({"detail_url": BASE_URL(DETAIL[k])}) for k in ("lighthouse", "paper")]
    owner = next(m for m in pair if m["status"] == "script_collected")
    sharer = next(m for m in pair if m is not owner)
    script = app.db[C.SCRIPTS].find_one({"script_url": BASE_URL(SCRIPT["lighthouse"])})
    assert app.db[C.SCRIPTS].count_documents({"script_url": BASE_URL(SCRIPT["lighthouse"])}) == 1
    assert script["movie_id"] == owner["_id"]
    assert sharer["status"] == "script_shared"
    assert sharer["script_id"] == script["_id"] and sharer["shared_script_owner_id"] == owner["_id"]

    cov = collect_stats(app.db, raw_dir=app.settings.raw_dir, data_dir=app.settings.data_dir)["coverage"]
    assert cov["unresolved_records"] == 0 and cov["scripts_shared_with_other_movie"] == 1
    assert run_validation(app.db, raw_store=app.raw_store).passed


def test_validation_message_reflects_current_permission_level(app, site: FakeSite) -> None:
    """The sample-limit validation message names the current permission level.

    Args:
        app: Ingestion context on a throwaway database, fetching from the fake site.
        site: The fake IMSDb; its request log shows what was fetched.
    """
    _service(app)
    _set_level(app, "bulk")
    report = run_validation(app.db, raw_store=app.raw_store)
    issue = next(i for i in report.issues if i.check == "records_exceed_sample_limit")
    assert "'bulk'" in issue.message and "'sample'" not in issue.message


@pytest.mark.parametrize("fmt", ["json", "jsonl"])
def test_export_authorized_records(app, site: FakeSite, fmt: str) -> None:
    """Export writes records, a manifest and attribution in JSON/JSONL, and is refused at level ``denied``.

    Args:
        app: Ingestion context on a throwaway database, fetching from the fake site.
        site: The fake IMSDb; its request log shows what was fetched.
        fmt: Export format, ``"json"`` or ``"jsonl"``.
    """
    import json

    service = _service(app)
    service.discover()
    service.crawl(limit=5)
    result = export_records(app.db, app.settings.export_dir, fmt=fmt)
    out = Path(result["directory"])
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    files = {f["collection"]: f for f in manifest["files"]}
    assert files["scripts"]["records"] == 3 and files["dialogues"]["records"] == 16
    assert "IMSDb" in (out / "ATTRIBUTION.txt").read_text(encoding="utf-8")
    text = (out / f"scripts.{fmt}").read_text(encoding="utf-8")
    rows = json.loads(text) if fmt == "json" else [json.loads(line) for line in text.splitlines()]
    assert len(rows) == 3 and all(r["rights"]["attribution"] for r in rows)
    _set_level(app, "denied")
    with pytest.raises(PermissionDeniedError):
        export_records(app.db, app.settings.export_dir, fmt=fmt)


def test_url_specific_refusal_skips_but_level_refusal_stops(app, site: FakeSite) -> None:
    """A queue item refused for a reason that concerns only its URL (external host, non-crawlable page type)
    is skipped and the crawl continues; a refusal by the permission level still stops the crawl.

    Args:
        app: Ingestion context on a throwaway database, fetching from the fake site.
        site: The fake IMSDb; its request log shows that nothing was fetched.
    """
    service = _service(app)
    gate = service.load_gate(refresh_robots=False)
    handle = service._handler(gate, "job-test")
    for url, page_type in (("https://example.org/page.html", "external"),
                           (BASE_URL("/some/unknown/thing"), "other")):
        outcome = handle({"url": url, "page_type": page_type})
        assert outcome.status == "skipped" and outcome.note
    assert app.db.crawl_logs.count_documents({"event": "url_skipped"}) == 2
    _set_level(app, "discovery_only")
    gate = service.load_gate(refresh_robots=False)
    with pytest.raises(PermissionDeniedError, match="does not allow record collection"):
        service._handler(gate, "job-test")({"url": BASE_URL("/scripts/Some-Film.html"), "page_type": "script_html"})


def test_export_without_text_and_archived_versions(app, site: FakeSite) -> None:
    """``include_text=False`` removes all screenplay wording, including dialogue and archived script versions,
    and archived versions of scripts collected without permission are never exported.

    Args:
        app: Ingestion context on a throwaway database, fetching from the fake site.
        site: The fake IMSDb.
    """
    import json

    service = _service(app)
    service.discover()
    service.crawl(limit=5)
    script = app.db.scripts.find_one()
    allowed = {k: v for k, v in script.items() if k != "_id"}
    now = script["created_at"]
    denied = {**allowed, "rights": {**allowed["rights"], "permission_level_at_collection": "discovery_only"}}
    app.db.script_versions.insert_many([
        {"_id": f"{script['_id']}:v1", "script_id": script["_id"], "version": 1, "archived_at": now,
         "document": allowed},
        {"_id": "sc-other:v1", "script_id": "sc-other", "version": 1, "archived_at": now, "document": denied},
    ])
    result = export_records(app.db, app.settings.export_dir, fmt="jsonl", include_text=False,
                            include_operational=True)
    out = Path(result["directory"])

    def rows(name: str) -> list[dict]:
        """Read one exported collection.

        Args:
            name: Collection name.

        Returns:
            The exported documents.
        """
        return [json.loads(line) for line in (out / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]

    versions = rows("script_versions")
    assert [v["_id"] for v in versions] == [f"{script['_id']}:v1"]
    assert "text" not in versions[0]["document"] and "elements" not in versions[0]["document"]
    dialogues = rows("dialogues")
    assert dialogues and not any({"text", "original_text", "segments"} & d.keys() for d in dialogues)
    assert not any("text" in s or "description" in s for s in rows("scenes"))


def test_listing_reparse_is_offline_and_idempotent(app, site: FakeSite) -> None:
    """Listing reparse restores lost movies and detail queue items from the raw archive, offline and idempotently.

    Args:
        app: Ingestion context on a throwaway database, fetching from the fake site.
        site: The fake IMSDb; its request log shows what was fetched.
    """
    service = _service(app)
    service.discover()
    app.db[C.MOVIES].delete_many({})  # simulate discovery results lost to an old parser bug
    app.db[C.CRAWL_QUEUE].delete_many({"page_type": "movie_detail"})
    requests_before = len(site.requests)
    totals = service.reparse_listings()
    assert len(site.requests) == requests_before  # raw archive only
    assert totals["movies_discovered_total"] == 5 and totals["detail_urls_pending"] == 5
    service.reparse_listings()
    assert app.db[C.MOVIES].count_documents({}) == 5


def test_real_site_banner_heading_does_not_hide_entries() -> None:
    """Regression: live IMSDb pages start with a banner <h1> (found during the live run)."""
    from src.parsers.movie_parser import parse_listing_page
    from tests.conftest import fixture_text
    page = parse_listing_page(fixture_text("all_scripts.html"), BASE_URL("/all-scripts.html"))
    assert page.heading == "All Movie Scripts on IMSDb (A-Z)" and len(page.entries) == 5
