"""Ingestion orchestration: inspect-source, discover, crawl, resume, reparse.

Data flow for one record::

    crawl_queue(movie_detail) --fetch--> raw store + raw_artifacts --parse--> movies
           |                                                                   |
           +--> crawl_queue(script_html) --fetch--> raw store --extract/parse--> scripts
                                                                   +--> scenes / dialogues / characters
"""

from __future__ import annotations

import os
import socket
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from src.config import Settings
from src.crawler.discovery import detail_items, listing_followups, make_queue_item, seed_items
from src.crawler.fetcher import FetchResult, Fetcher
from src.crawler.scheduler import Counters, CrawlScheduler, ItemOutcome, RecordBudget, SchedulerResult
from src.crawler.url_manager import (
    LISTING_TYPES,
    PageType,
    classify_url,
    detail_title_from_url,
    host_of,
    listing_label,
    normalize_url,
)
from src.database.connection import Mongo
from src.database.models import RawArtifactDoc, RawRef, Rights
from src.database.repositories import RawArtifactRepository, Repositories
from src.errors import FetchError, PermissionDeniedError
from src.parsers import PARSER_VERSION
from src.parsers.movie_parser import parse_listing_page, parse_movie_detail
from src.parsers.screenplay_parser import ExtractedScript, extract_screenplay, parse_screenplay
from src.services.deduplication import canonicalize_movie_urls
from src.services.documents import build_script_bundle, make_rights, movie_detail_fields, script_id_from_url
from src.services.permissions import (
    SOURCE_ID,
    PermissionGate,
    RobotsPolicy,
    assess_policy_text,
    default_imsdb_permission,
    policy_main_text,
)
from src.storage.raw_store import RawStore, StoredArtifact
from src.utils.clock import utcnow
from src.utils.hashing import sha256_text
from src.utils.logging import get_logger
from src.utils.text_processing import DecodedText, decode_html_bytes

log = get_logger("ingestion")

POLICY_PATHS = ("disclaimer.html", "contact.html", "privacy", "submit.html")
RECORD_PAGE_TYPES = [PageType.SCRIPT_HTML.value, PageType.MOVIE_DETAIL.value]
LISTING_PAGE_TYPES = [t.value for t in LISTING_TYPES]


class AppContext:
    """Long-lived resources shared by services in one CLI invocation.

    Attributes:
        settings: Application settings.
        mongo: MongoDB connection wrapper.
        db: The project database.
        repos: Repositories for all collections.
        raw_store: Content-addressed archive of fetched bodies.
        stop_event: Shared shutdown event.
    """

    def __init__(self, settings: Settings, mongo: Mongo, *, stop_event: threading.Event | None = None,
                 fetcher: Fetcher | None = None) -> None:
        """Open the shared resources (MongoDB repositories and the raw file store); the fetcher is created lazily.

        Args:
            settings: Application settings.
            mongo: Open MongoDB connection; closed by ``close``.
            stop_event: Shared shutdown event passed to the fetcher and scheduler; a new one when None.
            fetcher: Pre-built fetcher (e.g. with a mock transport in tests); created on first use when None.
        """
        self.settings = settings
        self.mongo = mongo
        self.db = mongo.db
        self.repos = Repositories(mongo.db)
        self.raw_store = RawStore(settings.raw_dir, gzip_bodies=settings.raw_gzip)
        self.stop_event = stop_event or threading.Event()
        self._fetcher = fetcher

    @property
    def fetcher(self) -> Fetcher:
        """The HTTP fetcher, created on first access so commands that never go online make no client.

        Returns:
            The shared ``Fetcher`` bound to this context's stop event.
        """
        if self._fetcher is None:
            self._fetcher = Fetcher(self.settings, stop_event=self.stop_event)
        return self._fetcher

    def close(self) -> None:
        """Close the fetcher (if one was created) and the MongoDB connection."""
        if self._fetcher is not None:
            self._fetcher.close()
        self.mongo.close()


@dataclass(slots=True)
class FetchBundle:
    """A fetched page after it was archived: the HTTP result plus where and how it was stored.

    Attributes:
        result: The fetch result (status, headers, body, timing).
        stored: Location, hash and size of the body in the raw file store.
        raw_ref: Reference to the ``raw_artifacts`` record, embedded in movie/script documents.
        decoded: Body decoded to text, with the encoding that was used.
    """
    result: FetchResult
    stored: StoredArtifact
    raw_ref: RawRef
    decoded: DecodedText


def _empty_extraction(reason: str) -> ExtractedScript:
    """Build a placeholder extraction for a screenplay page the extractor crashed on.

    Args:
        reason: Error description, stored as the only extraction warning.

    Returns:
        An ``ExtractedScript`` with no text (``method="error"``), so the script is saved as failed while the
        raw page is kept.
    """
    return ExtractedScript(text="", lines=[], markup="", method="error", formatting_spans=[], footer_metadata={},
                           page_title=None, warnings=[reason], normalizations=[], visible_page_chars=0,
                           extraction_ratio=None, removed_tags={})


class IngestionService:
    """Orchestrates source inspection, discovery, crawling, resuming and offline reparsing.

    Every network command first loads the permission gate; crawl runs hold the global crawl lock in
    ``crawl_jobs`` and are recorded as jobs there.
    """
    def __init__(self, ctx: AppContext) -> None:
        """Bind the service to the shared resources.

        Args:
            ctx: Application context (settings, repositories, raw store, fetcher, stop event).
        """
        self.ctx = ctx
        self.settings = ctx.settings
        self.repos = ctx.repos
        self.lock_owner = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"

    # ================================================================== permissions
    def load_gate(self, *, refresh_robots: bool = True) -> PermissionGate:
        """Build the permission gate from the ``source_permissions`` record.

        Seeds the default IMSDb record (sample level) when none exists. If ``refresh_robots`` is set and the
        stored robots.txt is missing or older than 24 hours, robots.txt is fetched again and saved to the
        record. A robots.txt ``Crawl-delay`` raises the fetcher's minimum request interval.

        Args:
            refresh_robots: Whether a missing or stale robots.txt may be re-fetched (a network request).

        Returns:
            A gate combining the permission level, robots.txt rules and the collected-record count.
        """
        perms = self.repos.permissions
        record = perms.get(SOURCE_ID)
        if record is None:
            perms.seed(default_imsdb_permission())
            record = perms.get(SOURCE_ID)
            assert record is not None
        robots = RobotsPolicy.from_record(record.robots, self.settings.user_agent)
        if refresh_robots and (robots is None or robots.is_stale()):
            robots = self.fetch_robots()
            perms.update_robots(SOURCE_ID, robots.to_record())
            record = perms.get(SOURCE_ID)
            assert record is not None
        if robots is not None and robots.crawl_delay:
            self.ctx.fetcher.rate_limiter.set_interval(robots.crawl_delay)
        return PermissionGate(record, robots, self.settings.user_agent, self.repos.movies.count_records_collected)

    def fetch_robots(self) -> RobotsPolicy:
        """Fetch robots.txt from the source and interpret it.

        A successful response is archived in the raw store and ``raw_artifacts``. A failed fetch is logged to
        ``crawl_logs`` and interpreted from its status (404 -> no restrictions, 401/403 or unreachable ->
        disallow everything). Does not save the policy to the permission record; callers do that.

        Returns:
            The robots.txt policy.
        """
        url = normalize_url("robots.txt", base=self.settings.base_url)
        assert url is not None
        try:
            result = self.ctx.fetcher.fetch(url)
        except FetchError as exc:
            self.repos.logs.log("robots_fetch", level="INFO", url=url, page_type="robots",
                                message=str(exc), details={"status_code": exc.status_code})
            return RobotsPolicy.from_response(exc.status_code, "", fetched_at=utcnow())
        bundle = self._store(url, PageType.ROBOTS.value, result, job_id=None)
        return RobotsPolicy.from_response(result.status_code, bundle.decoded.text, fetched_at=result.fetched_at,
                                          sha256=bundle.stored.sha256, user_agent=self.settings.user_agent)

    def inspect_source(self) -> dict[str, Any]:
        """Fetch robots.txt and policy pages, record evidence, and report the assessment.

        Network requests: robots.txt plus each page in ``POLICY_PATHS`` (archived like any fetch). The
        heuristic assessment is merged into the ``source_permissions`` record together with the robots.txt
        result and the policy texts; if restrictive language is found, any earlier operator override is
        cleared, which blocks record collection. Never raises the permission level.

        Returns:
            ``permission`` (updated permission record), ``robots`` (``RobotsPolicy``), ``documents`` (one dict
            per policy page: URL, status, hash, raw path and main text, or the error) and ``assessment``
            (the merged assessment dict).
        """
        robots = self.fetch_robots()
        self.repos.permissions.update_robots(SOURCE_ID, robots.to_record())
        gate = self.load_gate(refresh_robots=False)  # built after the update, so it uses the fresh robots.txt
        documents: list[dict[str, Any]] = []
        for path in POLICY_PATHS:
            url = normalize_url(path, base=self.settings.base_url)
            assert url is not None
            try:
                bundle = self.fetch_and_store(url, PageType.POLICY.value, gate, job_id=None)
            except FetchError as exc:
                documents.append({"url": url, "status_code": exc.status_code, "error": str(exc)})
                continue
            except PermissionDeniedError as exc:  # e.g. robots.txt disallows the policy page: not fetched
                documents.append({"url": url, "status_code": None, "error": f"not fetched: {exc}"})
                continue
            documents.append({
                "url": url, "final_url": bundle.result.final_url, "status_code": bundle.result.status_code,
                "sha256": bundle.stored.sha256, "raw_path": bundle.stored.relative_path,
                "fetched_at": bundle.result.fetched_at, "text": policy_main_text(bundle.decoded.text)[:20_000],
            })
        assessment = assess_policy_text([d for d in documents if d.get("text")])
        record = self.repos.permissions.get(SOURCE_ID)
        assert record is not None
        merged = {**record.assessment, **assessment, "inspected_at": utcnow(),
                  "robots_status": robots.status, "robots_allows_all": robots.allow_all,
                  "robots_disallows_all": robots.disallow_all, "assessed_by": "inspect-source (automated heuristics)"}
        if assessment["restriction_found"]:
            merged["restriction_overridden"] = False
        self.repos.permissions.record_inspection(SOURCE_ID, robots=robots.to_record(),
                                                 policy_documents=documents, assessment=merged)
        self.repos.logs.log("inspect_source", details={"robots_status": robots.status,
                                                       "restriction_found": assessment["restriction_found"],
                                                       "documents": len(documents)})
        record = self.repos.permissions.get(SOURCE_ID)
        assert record is not None
        return {"permission": record, "robots": robots, "documents": documents, "assessment": merged}

    # ================================================================== fetch + store
    def _store(self, url: str, page_type: str, result: FetchResult, job_id: str | None) -> FetchBundle:
        """Archive a fetched body and record it.

        Saves the body to the raw file store (content-addressed by SHA-256), upserts a ``raw_artifacts``
        document, appends a line to the raw-store manifest and logs a ``fetch_ok`` event to ``crawl_logs``.

        Args:
            url: URL that was requested.
            page_type: Page type of the URL (``PageType`` value).
            result: Successful fetch result.
            job_id: Crawl job the fetch belongs to, or None outside a job.

        Returns:
            The fetch result together with its storage location, raw reference and decoded text.
        """
        host = host_of(result.final_url) or host_of(url)
        stored = self.ctx.raw_store.save(result.body, host=host, page_type=page_type)
        decoded = decode_html_bytes(result.body, result.content_type)
        artifact_id = RawArtifactRepository.artifact_id(url, stored.sha256)
        self.repos.raw.record(RawArtifactDoc(
            _id=artifact_id, url=url, final_url=result.final_url, page_type=page_type, sha256=stored.sha256,
            size=stored.size, path=stored.relative_path, compressed=stored.compressed,
            status_code=result.status_code, headers=result.headers, http_version=result.http_version,
            encoding=decoded.encoding, first_fetched_at=result.fetched_at, last_fetched_at=result.fetched_at,
        ))
        self.ctx.raw_store.append_manifest({
            "url": url, "final_url": result.final_url, "page_type": page_type, "sha256": stored.sha256,
            "size": stored.size, "path": stored.relative_path, "status_code": result.status_code,
            "headers": result.headers, "fetched_at": result.fetched_at.isoformat(), "job_id": job_id,
            "encoding": decoded.encoding, "new_file": stored.created,
        }, when=result.fetched_at)
        raw_ref = RawRef(artifact_id=artifact_id, sha256=stored.sha256, path=stored.relative_path, size=stored.size,
                         fetched_at=result.fetched_at, status_code=result.status_code)
        self.repos.logs.log("fetch_ok", job_id=job_id, url=url, page_type=page_type, details={
            "status_code": result.status_code, "bytes": stored.size, "sha256": stored.sha256,
            "elapsed_ms": result.elapsed_ms, "attempts": result.attempts, "new_file": stored.created,
            "redirects": result.redirects})
        return FetchBundle(result, stored, raw_ref, decoded)

    def fetch_and_store(self, url: str, page_type: str, gate: PermissionGate, *, job_id: str | None) -> FetchBundle:
        """Check permission, fetch a page (redirect targets vetted by the gate) and archive it.

        Args:
            url: Normalized URL to fetch.
            page_type: Page type of the URL (``PageType`` value).
            gate: Permission gate to check the URL and redirect targets against.
            job_id: Crawl job the fetch belongs to, or None.

        Returns:
            The archived fetch.

        Raises:
            PermissionDeniedError: The gate does not allow the URL.
        """
        gate.require(url, page_type)
        result = self.ctx.fetcher.fetch(url, allow_url=lambda target: gate.check(target).allowed)
        return self._store(url, page_type, result, job_id)

    # ================================================================== handlers
    def _handler(self, gate: PermissionGate, job_id: str):  # noqa: ANN202 - returns a closure
        """Create the queue-item handler the scheduler calls for each claimed item.

        Args:
            gate: Permission gate checked before every fetch.
            job_id: Crawl job id, used in log events.

        Returns:
            A function taking a queue document and returning an ``ItemOutcome``.
        """
        def handle(item: dict[str, Any]) -> ItemOutcome:
            """Check one queue item against the gate and dispatch it to the handler for its page type.

            URLs refused for a reason that concerns only that URL (``Decision.url_specific``: external host,
            robots.txt, excluded or non-crawlable page type) are skipped and logged as ``url_skipped``; a
            refusal that applies to the whole collection (permission level, restrictive policy) stops the crawl.

            Args:
                item: Claimed ``crawl_queue`` document.

            Returns:
                The outcome of the listing, detail or script handler, or a ``skipped`` outcome.

            Raises:
                PermissionDeniedError: The gate refuses the URL for a collection-wide reason.
            """
            page_type = item["page_type"]
            decision = gate.check(item["url"], page_type)
            if not decision.allowed:
                if decision.url_specific:
                    self.repos.logs.log("url_skipped", level="INFO", job_id=job_id, url=item["url"],
                                        page_type=page_type, message=decision.reason)
                    return ItemOutcome(status="skipped", note=decision.reason)
                raise PermissionDeniedError(f"Not permitted to fetch {item['url']}: {decision.reason}")
            if page_type in LISTING_PAGE_TYPES:
                return self.handle_listing(item, gate, job_id)
            if page_type == PageType.MOVIE_DETAIL.value:
                return self.handle_detail(item, gate, job_id)
            if page_type == PageType.SCRIPT_HTML.value:
                return self.handle_script(item, gate, job_id)
            return ItemOutcome(status="skipped", note=f"page type {page_type} is not processed")
        return handle

    def handle_listing(self, item: dict[str, Any], gate: PermissionGate, job_id: str) -> ItemOutcome:
        """Fetch a listing page, then upsert its movies and enqueue the pages it links to.

        Args:
            item: Claimed ``crawl_queue`` document of a listing page.
            gate: Permission gate.
            job_id: Crawl job id.

        Returns:
            A ``done`` outcome; ``changed`` is True when the page hash differs from the last fetch, and the
            counters are those of ``process_listing`` plus ``listing_pages_fetched``.
        """
        bundle = self.fetch_and_store(item["url"], item["page_type"], gate, job_id=job_id)
        counters = self.process_listing(item, bundle.decoded.text, job_id=job_id)
        counters["listing_pages_fetched"] = 1
        return ItemOutcome(changed=item.get("content_hash") != bundle.stored.sha256,
                           content_hash=bundle.stored.sha256, status_code=bundle.result.status_code,
                           counters=counters)

    def process_listing(self, item: dict[str, Any], html: str, *, job_id: str | None) -> dict[str, int]:
        """Parse a listing page and upsert/enqueue what it lists (shared by crawl and offline reparse).

        Upserts listed movies into ``movies``, enqueues their detail pages and linked listing pages in
        ``crawl_queue`` (existing URLs are left alone) and logs parser warnings to ``crawl_logs``.

        Args:
            item: ``crawl_queue`` document of the listing page (``url``, ``page_type``, ``depth``).
            html: Decoded HTML of the page.
            job_id: Crawl job id for log events, or None for offline reparse.

        Returns:
            Counters: ``listing_entries``, ``listing_pages_enqueued`` and, when entries were found,
            ``movies_new`` and ``detail_urls_enqueued``.
        """
        url = item["url"]
        page = parse_listing_page(html, url)
        counters: dict[str, int] = {"listing_entries": len(page.entries)}
        if page.entries:
            res = self.repos.movies.upsert_from_listing(page.entries, source_label=listing_label(url), listing_url=url)
            counters["movies_new"] = res["inserted"]
            urls = [e.detail_url for e in page.entries]
            ids = {d["detail_url"]: d["_id"] for d in
                   self.repos.movies.col.find({"detail_url": {"$in": urls}}, {"detail_url": 1})}
            counters["detail_urls_enqueued"] = self.repos.queue.enqueue_many(
                detail_items(((u, ids.get(u)) for u in urls), source_url=url, depth=item.get("depth", 0)))
        counters["listing_pages_enqueued"] = self.repos.queue.enqueue_many(
            listing_followups(page.listing_links, source_url=url, depth=item.get("depth", 0)))
        if page.warnings:
            self.repos.logs.log("parse_warning", level="WARNING", job_id=job_id, url=url, page_type=item["page_type"],
                                details={"warnings": page.warnings[:50], "entries": len(page.entries)})
        return counters

    def reparse_listings(self) -> dict[str, Any]:
        """Re-process already-fetched listing pages from the raw archive (no network).

        Reads every ``done`` listing item's archived body (shallowest first), runs ``process_listing`` on it,
        then merges duplicate movie URLs with ``canonicalize_movie_urls``. Results are logged to
        ``crawl_logs``.

        Returns:
            Summed ``process_listing`` counters plus ``pages``, ``raw_missing`` (when archives are missing),
            ``records_merged_into_canonical_url``, ``records_flagged_for_review``,
            ``movies_discovered_total`` and ``detail_urls_pending``.

        Raises:
            PermissionDeniedError: The permission level does not allow discovery.
        """
        gate = self.load_gate(refresh_robots=False)
        gate.require_discovery()
        totals: dict[str, int] = {"pages": 0}
        for item in self.repos.queue.col.find({"page_type": {"$in": LISTING_PAGE_TYPES}, "status": "done",
                                               "content_hash": {"$ne": None}}).sort([("depth", 1), ("position", 1)]):
            artifact = self.repos.raw.col.find_one({"url": item["url"], "sha256": item["content_hash"]})
            if artifact is None:
                totals["raw_missing"] = totals.get("raw_missing", 0) + 1
                continue
            body = self.ctx.raw_store.read(artifact["path"], expected_sha256=artifact["sha256"])
            decoded = decode_html_bytes(body, (artifact.get("headers") or {}).get("content-type"))
            for key, value in self.process_listing(item, decoded.text, job_id=None).items():
                totals[key] = totals.get(key, 0) + value
            totals["pages"] += 1
        canon = canonicalize_movie_urls(self.repos.db)
        totals["records_merged_into_canonical_url"] = len(canon["merged"])
        totals["records_flagged_for_review"] = len(canon["flagged_for_review"])
        if canon["merged"] or canon["flagged_for_review"]:
            self.repos.logs.log("url_canonicalization", level="WARNING", details=canon)
        self.repos.logs.log("listings_reparsed", details=totals)
        totals["movies_discovered_total"] = self.repos.movies.col.count_documents({})
        totals["detail_urls_pending"] = self.repos.queue.count({"page_type": "movie_detail", "status": "pending"})
        return totals

    def handle_detail(self, item: dict[str, Any], gate: PermissionGate, job_id: str) -> ItemOutcome:
        """Fetch and parse a movie detail page, update the ``movies`` document and enqueue its screenplay page.

        Creates the movie from the URL if no listing produced it. The movie status becomes
        ``script_unavailable`` (no HTML script), ``script_shared`` (the script is already stored under another
        movie; ``script_id``/``shared_script_owner_id`` are set), ``script_collected`` (kept when the same script
        was already collected) or ``detail_collected``. Parse warnings are logged to ``crawl_logs``.

        Args:
            item: Claimed ``crawl_queue`` document of a movie detail page.
            gate: Permission gate (its level and version are recorded as the record's rights).
            job_id: Crawl job id.

        Returns:
            A ``done`` outcome with the movie id and counters ``records_new``/``records_refreshed`` plus one of
            ``script_urls_enqueued``, ``scripts_pdf_only`` or ``scripts_missing``.
        """
        url = item["url"]
        movie = self.repos.movies.get_by_detail_url(url)
        if movie is None:
            self.repos.movies.ensure_from_detail_url(url, detail_title_from_url(url) or url)
            movie = self.repos.movies.get_by_detail_url(url)
            assert movie is not None
        first_collection = "detail_fetched_at" not in movie
        bundle = self.fetch_and_store(url, item["page_type"], gate, job_id=job_id)
        detail = parse_movie_detail(bundle.decoded.text, url)
        rights = make_rights(gate.level, gate.record.version)
        fields = movie_detail_fields(detail, listing_title=movie.get("listing_title") or movie.get("title"),
                                     raw=bundle.raw_ref, rights=rights)
        owner = None
        if detail.script_url and detail.script_availability == "html":
            owner = self.repos.scripts.col.find_one({"script_url": detail.script_url}, {"movie_id": 1})
            if owner is not None and owner["movie_id"] != movie["_id"]:
                status = "script_shared"  # the source links this movie to a script stored under another movie
            else:
                owner = None
                keep = movie.get("status") == "script_collected" and movie.get("script_url") == detail.script_url
                status = "script_collected" if keep else "detail_collected"
        else:
            status = "script_unavailable"
        changed = self.repos.movies.apply_detail(movie["_id"], fields, status=status)
        if owner is not None:
            self.repos.movies.col.update_one({"_id": movie["_id"]}, {"$set": {
                "script_id": owner["_id"], "shared_script_owner_id": owner["movie_id"]}})
        else:
            self.repos.movies.col.update_one({"_id": movie["_id"], "shared_script_owner_id": {"$exists": True}},
                                             {"$unset": {"shared_script_owner_id": ""}})
        counters = {"records_new": 1 if first_collection else 0, "records_refreshed": 0 if first_collection else 1}
        if detail.warnings:
            self.repos.logs.log("parse_warning", level="WARNING", job_id=job_id, url=url, page_type="movie_detail",
                                record_id=movie["_id"], details={"warnings": detail.warnings})
        if status != "script_unavailable" and detail.script_url:
            script_item = make_queue_item(detail.script_url, discovered_from=url, depth=item.get("depth", 0) + 1,
                                          movie_id=movie["_id"])
            if script_item is not None:
                counters["script_urls_enqueued"] = self.repos.queue.enqueue_many([script_item])
        elif detail.script_availability == "pdf":
            counters["scripts_pdf_only"] = 1
        else:
            counters["scripts_missing"] = 1
        return ItemOutcome(changed=changed, content_hash=bundle.stored.sha256, status_code=bundle.result.status_code,
                           movie_id=movie["_id"], counters=counters)

    def handle_script(self, item: dict[str, Any], gate: PermissionGate, job_id: str) -> ItemOutcome:
        """Fetch a screenplay page and extract/parse it into the script and its derived documents.

        The owning movie is found via the item's ``movie_id`` or, failing that, the movie whose ``script_url``
        is this URL.

        Args:
            item: Claimed ``crawl_queue`` document of a screenplay page.
            gate: Permission gate.
            job_id: Crawl job id.

        Returns:
            A ``skipped`` outcome when no movie references the URL; otherwise a ``done`` outcome counting
            ``scripts_<action>`` and, when relevant, ``parse_errors`` or ``scripts_empty_at_source``.
        """
        url = item["url"]
        movie = self.repos.movies.get(item["movie_id"]) if item.get("movie_id") else None
        if movie is None:
            movie = self.repos.movies.col.find_one({"script_url": url})
        if movie is None:
            return ItemOutcome(status="skipped", note="no movie record references this script URL")
        bundle = self.fetch_and_store(url, item["page_type"], gate, job_id=job_id)
        result = self.process_script(movie, url, bundle.raw_ref, bundle.decoded, fetched_at=bundle.result.fetched_at,
                                     rights=make_rights(gate.level, gate.record.version), job_id=job_id)
        counters = {f"scripts_{result['action']}": 1}
        if result.get("status") in {"parse_failed", "extraction_failed"}:
            counters["parse_errors"] = 1
        elif result.get("status") == "empty_at_source":
            counters["scripts_empty_at_source"] = 1
        return ItemOutcome(changed=result["action"] in {"created", "updated"}, content_hash=bundle.stored.sha256,
                           status_code=bundle.result.status_code, movie_id=movie["_id"], counters=counters)

    def _link_sharing_movies(self, script_url: str, script_id: str, owner_movie_id: str, now: datetime) -> int:
        """Mark other movies whose detail page links to this same script URL as sharing it.

        IMSDb occasionally points a movie at another movie's screenplay (e.g. a sequel's page links to the
        original). The script is stored once, under the movie whose queue item fetched it; the others reference it.
        Only movies with status ``detail_collected`` or ``script_shared`` are updated.

        Args:
            script_url: Screenplay URL shared by the movies.
            script_id: Id of the stored script.
            owner_movie_id: Movie the script is stored under (not modified).
            now: Timestamp written to ``updated_at``.

        Returns:
            Number of movie documents modified.
        """
        result = self.repos.movies.col.update_many(
            {"script_url": script_url, "_id": {"$ne": owner_movie_id},
             "status": {"$in": ["detail_collected", "script_shared"]}},
            {"$set": {"status": "script_shared", "script_id": script_id, "shared_script_owner_id": owner_movie_id,
                      "updated_at": now}})
        return result.modified_count

    def process_script(self, movie: dict[str, Any], script_url: str, raw_ref: RawRef, decoded: DecodedText, *,
                       fetched_at: datetime, rights: Rights, job_id: str | None, force: bool = False) -> dict[str, Any]:
        """Extract and parse a screenplay page and store the results (shared by crawl and offline reparse).

        Unless ``force`` is set, a script whose extracted-text hash and parser version match an already
        successfully parsed document is only marked as checked. Otherwise the script is saved to ``scripts``
        (older content is versioned by the repository), its ``scenes``/``dialogues``/``characters`` are
        replaced, the movie's status becomes ``script_collected``, ``script_empty_at_source`` or
        ``script_failed``, other movies linking to the same script are marked as sharing it, and the result is
        logged to ``crawl_logs``. Extractor or parser crashes are logged and stored as a failed script.

        Args:
            movie: ``movies`` document the script belongs to.
            script_url: Normalized screenplay URL.
            raw_ref: Reference to the archived raw page.
            decoded: Decoded page text.
            fetched_at: When the raw page was fetched.
            rights: Rights metadata to store on the script.
            job_id: Crawl job id for log events, or None.
            force: Re-parse and rewrite even if nothing changed.

        Returns:
            ``action`` (``unchanged``, ``created``, ``updated`` or ``reparsed``), ``status`` (script status),
            ``script_id``, and after a write the ``<collection>_written``/``<collection>_removed_stale`` counts.
        """
        now = utcnow()
        title = movie.get("title") or script_url
        script_id = script_id_from_url(script_url, title)
        try:
            extracted = extract_screenplay(decoded.text)
        except Exception as exc:  # malformed beyond what the parser tolerates: keep raw, flag the record
            log.exception("extraction crashed for %s", script_url)
            extracted = _empty_extraction(f"extraction error: {exc.__class__.__name__}: {exc}")
        content_hash = sha256_text(extracted.text) if extracted.found else None
        if not force and not self.repos.scripts.needs_parse(script_id, content_hash, PARSER_VERSION):
            self.repos.scripts.touch_checked(script_id, raw_ref.to_mongo(), now)
            self.repos.movies.col.update_one({"_id": movie["_id"], "status": {"$ne": "script_collected"}},
                                             {"$set": {"status": "script_collected", "updated_at": now}})
            self._link_sharing_movies(script_url, script_id, movie["_id"], now)
            return {"action": "unchanged", "status": "unchanged", "script_id": script_id}

        parsed = None
        error = None
        if extracted.found:
            try:
                parsed = parse_screenplay(extracted)
            except Exception as exc:
                log.exception("parse crashed for %s", script_url)
                error = f"{exc.__class__.__name__}: {exc}"
        script, scenes, dialogues, characters = build_script_bundle(
            movie=movie, script_url=script_url, raw=raw_ref, decoded=decoded, extracted=extracted, parsed=parsed,
            rights=rights, fetched_at=fetched_at, now=now, processed_dir=self.settings.processed_dir, error=error)
        parse_id = uuid.uuid4().hex
        outcome = self.repos.scripts.save(script, parse_id=parse_id, now=now)
        written = self.repos.derived.replace_for_script(script.id, parse_id, scenes=scenes, dialogues=dialogues,
                                                        characters=characters)
        ok = script.status in {"parsed", "parsed_with_warnings"}
        empty = script.status == "empty_at_source"
        movie_status = "script_collected" if ok else "script_empty_at_source" if empty else "script_failed"
        self.repos.movies.col.update_one({"_id": movie["_id"]}, {"$set": {
            "status": movie_status, "script_id": script.id,
            "script_status": script.status, "script_content_hash": script.content_hash, "updated_at": now}})
        self._link_sharing_movies(script_url, script.id, movie["_id"], now)
        details = {"action": outcome.action, "version": outcome.version, "status": script.status,
                   "stats": {k: script.structure_stats.get(k) for k in
                             ("scenes", "dialogue_blocks", "characters", "uncertain_ratio")},
                   "quality_flags": script.quality_flags, **written}
        event, level = (("script_parsed", "INFO") if ok else ("script_empty_at_source", "WARNING") if empty
                        else ("parse_error", "ERROR"))
        self.repos.logs.log(event, level=level, job_id=job_id, url=script_url, page_type="script_html",
                            record_id=script.id, message=error, details=details)
        if script.parse_warnings or script.quality_flags:
            self.repos.logs.log("parse_warning", level="WARNING", job_id=job_id, url=script_url,
                                page_type="script_html", record_id=script.id,
                                details={"warnings": script.parse_warnings[:50], "flags": script.quality_flags})
        return {"action": outcome.action, "status": script.status, "script_id": script.id, **written}

    # ================================================================== jobs
    def _run(self, *, command: str, params: dict[str, Any], gate: PermissionGate, page_types: list[str],
             budget: RecordBudget | None, claim_filter: dict[str, Any] | None, resumed_from: str | None,
             initial_counters: dict[str, int] | None = None) -> tuple[dict[str, Any], SchedulerResult]:
        """Run one crawl job: take the crawl lock, record the job, run the scheduler and record the result.

        Expired queue leases are recovered first. The job is finished as ``completed``, ``interrupted`` or
        ``stopped`` (blocked or permission denied), or ``failed`` if an exception escapes (then re-raised).
        The crawl lock is always released.

        Args:
            command: Job command name stored on the job (``discover`` or ``crawl``).
            params: Command parameters stored on the job (used by ``resume``).
            gate: Permission gate for every fetch.
            page_types: Page types the scheduler may claim.
            budget: Record budget, or None for unlimited.
            claim_filter: Extra queue predicate for claims, or None.
            resumed_from: Id of the job this run resumes, or None.
            initial_counters: Starting counter values, or None.

        Returns:
            A summary dict (``job_id``, ``status``, ``counters``, ``reason``) and the raw scheduler result.
        """
        repos = self.repos
        repos.jobs.acquire_lock(self.lock_owner)
        try:
            job = repos.jobs.start(command, params, resumed_from=resumed_from)
            repos.logs.log("job_started", job_id=job.id, details={"command": command, "params": params,
                                                                   "permission_level": gate.level})
        except BaseException:
            repos.jobs.release_lock(self.lock_owner)  # otherwise the lock blocks crawls until it times out
            raise
        counters = Counters(initial_counters)
        try:
            reset = repos.queue.reset_expired_leases()
            if reset:
                repos.logs.log("leases_recovered", job_id=job.id, details={"count": reset})

            def is_new_record(item: dict[str, Any]) -> bool:
                """Decide whether a movie detail item would collect a new record (counts against the sample quota).

                Args:
                    item: Claimed movie detail queue document.

                Returns:
                    True when no movie has this detail URL or its detail page was never fetched.
                """
                doc = repos.movies.col.find_one({"detail_url": item["url"]}, {"detail_fetched_at": 1})
                return doc is None or "detail_fetched_at" not in doc

            scheduler = CrawlScheduler(
                queue=repos.queue, jobs=repos.jobs, logs=repos.logs, handler=self._handler(gate, job.id),
                job_id=job.id, page_types=page_types, concurrency=self.settings.concurrency,
                lease_seconds=self.settings.lease_seconds, max_attempts=self.settings.max_queue_attempts,
                stop_event=self.ctx.stop_event, counters=counters, budget=budget, is_new_record=is_new_record,
                claim_filter=claim_filter, lock_owner=self.lock_owner)
            result = scheduler.run()
            status = {"completed": "completed", "interrupted": "interrupted"}.get(result.state, "stopped")
            repos.jobs.finish(job.id, status, result.counters, error=result.stop_reason)
            repos.logs.log("job_finished", job_id=job.id, level="INFO" if status == "completed" else "WARNING",
                           details={"status": status, "counters": result.counters, "reason": result.stop_reason})
        except BaseException as exc:
            repos.jobs.finish(job.id, "failed", counters.snapshot(), error=f"{exc.__class__.__name__}: {exc}")
            raise
        finally:
            repos.jobs.release_lock(self.lock_owner)
        return {"job_id": job.id, "status": status, "counters": result.counters, "reason": result.stop_reason}, result

    def discover(self, *, quick: bool = False, refresh: bool = False, resumed_from: str | None = None) -> dict[str, Any]:
        """Crawl listing pages to discover movies, without collecting records.

        Refreshes robots.txt if stale, enqueues the seed pages and runs a ``discover`` job over listing page
        types only.

        Args:
            quick: Only the home and all-scripts pages (minimal requests).
            refresh: Re-queue listing pages that are already done or failed.
            resumed_from: Id of the job being resumed, or None.

        Returns:
            The job summary plus ``seeds_enqueued``, ``listing_pages_requeued``, ``movies_discovered_total`` and
            ``detail_urls_pending``.

        Raises:
            PermissionDeniedError: The permission level does not allow discovery.
        """
        gate = self.load_gate(refresh_robots=True)
        gate.require_discovery()
        seeds = seed_items(self.settings.base_url, quick=quick)
        enqueued = self.repos.queue.enqueue_many(seeds)
        requeued = 0
        if refresh:
            requeued = self.repos.queue.requeue(page_types=LISTING_PAGE_TYPES, statuses=["done", "failed"])
        types = [PageType.HOME.value, PageType.LISTING_ALL.value] if quick else LISTING_PAGE_TYPES
        summary, _ = self._run(command="discover", params={"quick": quick, "refresh": refresh}, gate=gate,
                               page_types=types, budget=None, claim_filter=None, resumed_from=resumed_from)
        summary.update({"seeds_enqueued": enqueued, "listing_pages_requeued": requeued,
                        "movies_discovered_total": self.repos.movies.col.count_documents({}),
                        "detail_urls_pending": self.repos.queue.count({"page_type": "movie_detail",
                                                                       "status": "pending"})})
        return summary

    def crawl(self, *, limit: int | None = None, all_records: bool = False, refresh_days: float | None = None,
              retry_failed: bool = False, resumed_from: str | None = None,
              finishing_only: bool = False) -> dict[str, Any]:
        """Collect movie records (detail and screenplay pages) within the permission limits.

        Modes: normal (``limit`` new or existing records, capped by the sample quota), ``all_records`` (bulk
        level only), refresh (re-fetch records completed more than ``refresh_days`` ago, never new ones) and
        finishing-only (budget 0: only pending screenplay pages of already started records). If no movies are
        known yet, a quick discovery is included in the same job.

        Args:
            limit: Maximum records to start in this run; required unless ``all_records`` or refreshing.
            all_records: Collect every record (requires the ``bulk`` level).
            refresh_days: Refresh records older than this many days, or None for a normal crawl.
            retry_failed: Re-queue record pages that previously failed.
            resumed_from: Id of the job being resumed, or None.
            finishing_only: Only finish started records (used when resuming a run whose limit is used up).

        Returns:
            The job summary plus ``notices`` (messages also logged) and ``remaining_sample_quota``
            (None when unlimited).

        Raises:
            PermissionDeniedError: The level, quota or a detected policy restriction does not allow the request.
        """
        gate = self.load_gate(refresh_robots=True)
        notices: list[str] = []
        claim_filter: dict[str, Any] | None = None
        page_types = list(RECORD_PAGE_TYPES)
        if finishing_only:
            run_limit, quota = 0, 0
        elif refresh_days is not None:
            if gate.level not in {"sample", "bulk"}:
                raise PermissionDeniedError(f"Refreshing records requires 'sample' or 'bulk' level (now {gate.level!r}).")
            run_limit, quota = limit, 0  # refresh never collects new records
            cutoff = utcnow() - timedelta(days=refresh_days)
            refreshed = self.repos.queue.request_refresh(page_types=RECORD_PAGE_TYPES, completed_before=cutoff)
            notices.append(f"{refreshed} collected URL(s) older than {refresh_days} day(s) queued for refresh.")
            # Pending script pages always belong to already-collected records, so they are safe to include.
            claim_filter = {"$or": [{"refresh_requested_at": {"$exists": True}}, {"page_type": "script_html"}]}
        else:
            run_limit, quota, notices = gate.plan_crawl(limit=limit, all_records=all_records)
        if retry_failed:
            count = self.repos.queue.requeue(page_types=RECORD_PAGE_TYPES, statuses=["failed"])
            notices.append(f"{count} failed URL(s) re-queued.")
        if self.repos.movies.col.estimated_document_count() == 0 and not finishing_only:
            gate.require_discovery()
            self.repos.queue.enqueue_many(seed_items(self.settings.base_url, quick=True))
            page_types = [PageType.HOME.value, PageType.LISTING_ALL.value] + page_types
            notices.append("No records discovered yet: running quick discovery (all-scripts page) first.")
        for notice in notices:
            log.info(notice)
        budget = RecordBudget(run_limit, quota)
        params = {"limit": limit, "all": all_records, "refresh_days": refresh_days, "retry_failed": retry_failed,
                  "effective_limit": run_limit}
        summary, _ = self._run(command="crawl", params=params, gate=gate, page_types=page_types, budget=budget,
                               claim_filter=claim_filter, resumed_from=resumed_from)
        summary["notices"] = notices
        summary["remaining_sample_quota"] = gate.remaining_sample_quota()
        return summary

    def resume(self) -> dict[str, Any]:
        """Continue the most recent interrupted or crashed job with its original parameters.

        Releases that job's in-progress items and expired leases, re-runs ``discover`` or ``crawl`` with the
        remaining limit (``effective_limit`` minus ``records_processed``), then marks the old job ``stopped``
        and superseded by the new one.

        Returns:
            ``{"resumed": False, "message": ...}`` when there is nothing to resume; otherwise the new job's
            summary plus ``resumed``, ``resumed_from`` and ``released_items``.
        """
        job = self.repos.jobs.latest_resumable()
        if job is None:
            return {"resumed": False, "message": "No interrupted or crashed job found; nothing to resume."}
        released = self.repos.queue.release_job(job["_id"]) + self.repos.queue.reset_expired_leases()
        params = job.get("params", {})
        counters = job.get("counters", {})
        self.repos.logs.log("job_resumed", job_id=job["_id"], details={"released_items": released, "params": params})
        if job["command"] == "discover":
            summary = self.discover(quick=params.get("quick", False), refresh=False, resumed_from=job["_id"])
        else:
            limit = params.get("effective_limit", params.get("limit"))
            remaining = None if limit is None else max(0, int(limit) - int(counters.get("records_processed", 0)))
            if params.get("refresh_days") is not None:
                summary = self._resume_refresh(job, remaining)
            elif params.get("all"):
                summary = self.crawl(all_records=True, resumed_from=job["_id"])
            elif remaining == 0:
                summary = self.crawl(finishing_only=True, resumed_from=job["_id"])
            else:
                summary = self.crawl(limit=remaining, resumed_from=job["_id"])
        self.repos.jobs.mark_superseded(job["_id"], by_job=summary.get("job_id", "unknown"))
        summary.update({"resumed": True, "resumed_from": job["_id"], "released_items": released})
        return summary

    def _resume_refresh(self, job: dict[str, Any], remaining: int | None) -> dict[str, Any]:
        """Continue a refresh run: only items still flagged for refresh are processed.

        No new records are collected (quota 0); pending screenplay pages are also claimed.

        Args:
            job: The interrupted ``crawl_jobs`` document.
            remaining: Records still allowed by the original limit, or None for no limit.

        Returns:
            The new job's summary (see ``_run``).
        """
        gate = self.load_gate(refresh_robots=True)
        budget = RecordBudget(remaining, 0)
        summary, _ = self._run(command="crawl", params={**job.get("params", {}), "effective_limit": remaining},
                               gate=gate, page_types=list(RECORD_PAGE_TYPES), budget=budget,
                               claim_filter={"$or": [{"refresh_requested_at": {"$exists": True}},
                                                     {"page_type": "script_html"}]},
                               resumed_from=job["_id"])
        return summary

    def reparse(self, *, force: bool = False, script_ids: list[str] | None = None) -> dict[str, Any]:
        """Re-run extraction/parsing from the local raw archive (no network).

        Each selected script is re-processed with ``process_script(force=True)``. Stops early when the stop
        event is set. Unreadable archives are logged to ``crawl_logs`` as ``reparse_error``.

        Args:
            force: Re-parse every script, not only outdated or failed ones.
            script_ids: Only these script ids (takes precedence over ``force``); None for the default selection
                (parser version differs from ``PARSER_VERSION`` or status ``parse_failed``/``extraction_failed``).

        Returns:
            ``parser_version`` and ``counts``: number of scripts per ``process_script`` action, plus
            ``raw_missing``, ``orphans`` (no movie) and ``interrupted`` when they occur.
        """
        query: dict[str, Any] = {}
        if script_ids:
            query["_id"] = {"$in": script_ids}
        elif not force:
            query["$or"] = [{"parser_version": {"$ne": PARSER_VERSION}},
                            {"status": {"$in": ["parse_failed", "extraction_failed"]}}]
        counts: dict[str, int] = {}
        projection = {"movie_id": 1, "script_url": 1, "raw": 1, "rights": 1, "fetched_at": 1}
        for script in self.repos.scripts.col.find(query, projection):
            if self.ctx.stop_event.is_set():
                counts["interrupted"] = 1
                break
            raw = script["raw"]
            try:
                body = self.ctx.raw_store.read(raw["path"], expected_sha256=raw["sha256"])
            except (OSError, ValueError) as exc:
                counts["raw_missing"] = counts.get("raw_missing", 0) + 1
                self.repos.logs.log("reparse_error", level="ERROR", record_id=script["_id"], message=str(exc))
                continue
            artifact = self.repos.raw.col.find_one({"_id": raw["artifact_id"]}, {"headers": 1}) or {}
            decoded = decode_html_bytes(body, (artifact.get("headers") or {}).get("content-type"))
            movie = self.repos.movies.get(script["movie_id"])
            if movie is None:
                counts["orphans"] = counts.get("orphans", 0) + 1
                continue
            result = self.process_script(movie, script["script_url"], RawRef.model_validate(raw), decoded,
                                         fetched_at=script["fetched_at"], rights=Rights.model_validate(script["rights"]),
                                         job_id=None, force=True)
            counts[result["action"]] = counts.get(result["action"], 0) + 1
        return {"parser_version": PARSER_VERSION, "counts": counts}
