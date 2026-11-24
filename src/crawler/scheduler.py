"""Worker pool over the persistent crawl queue.

* Items are claimed atomically with a lease (``find_one_and_update``), so a crash leaves
  them recoverable: expired leases are returned to ``pending`` on the next run/resume.
* A :class:`RecordBudget` caps how many movie records a run may start (``--limit`` and the
  permission sample quota).
* Shutdown is cooperative: the shared ``stop_event`` interrupts waits; claimed but
  unprocessed items are released without counting an attempt.
* An access-control response (401/403/407/451) stops the whole crawl - restrictions are never
  worked around.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from pymongo import ReturnDocument

from src.database.repositories import JobRepository, LogRepository, QueueRepository
from src.errors import FetchCancelled, FetchError, PermissionDeniedError
from src.utils.clock import as_utc, utcnow
from src.utils.logging import get_logger

log = get_logger("scheduler")


class Counters:
    """Thread-safe named integer counters shared by the crawl workers (e.g. ``movie_detail_done``)."""
    def __init__(self, initial: dict[str, int] | None = None) -> None:
        """Create the counters.

        Args:
            initial: Starting values (copied), e.g. counters of a resumed job; empty when None.
        """
        self._lock = threading.Lock()
        self._values: dict[str, int] = dict(initial or {})

    def inc(self, key: str, amount: int = 1) -> None:
        """Add to a counter, creating it at 0 first if needed.

        Args:
            key: Counter name.
            amount: Value to add (may be negative).
        """
        with self._lock:
            self._values[key] = self._values.get(key, 0) + amount

    def get(self, key: str) -> int:
        """Read one counter.

        Args:
            key: Counter name.

        Returns:
            The current value, or 0 when the counter was never incremented.
        """
        with self._lock:
            return self._values.get(key, 0)

    def snapshot(self) -> dict[str, int]:
        """Copy all counters atomically.

        Returns:
            A new dict mapping counter name to value; later increments do not affect it.
        """
        with self._lock:
            return dict(self._values)


class RecordBudget:
    """Run limit (all record starts) plus permission quota (new records only).

    Thread-safe; shared by all workers of a run.

    Attributes:
        run_limit: Maximum record starts in this run, or None for no limit.
        quota: Maximum new-record starts, or None for no quota.
        run_used: Record starts taken so far.
        new_used: New-record starts taken so far.
        quota_hit: Set once a new record was refused because the quota was reached.
    """

    def __init__(self, run_limit: int | None, new_record_quota: int | None) -> None:
        """Create an unused budget.

        Args:
            run_limit: Maximum record starts in this run (``--limit``), or None for no limit.
            new_record_quota: Maximum records not yet in the database that may be started (permission sample
                quota), or None for no quota.
        """
        self.run_limit = run_limit
        self.quota = new_record_quota
        self.run_used = 0
        self.new_used = 0
        self.quota_hit = False
        self._lock = threading.Lock()

    def try_acquire(self, new_record: bool) -> bool:
        """Try to take one unit of budget before starting a record.

        Args:
            new_record: Whether the record is new (counts against the quota as well as the run limit).

        Returns:
            True when the record may be started (counters incremented); False when the run limit is reached
            or, for a new record, the quota is reached (which also sets ``quota_hit``).
        """
        with self._lock:
            if self.run_limit is not None and self.run_used >= self.run_limit:
                return False
            if new_record and self.quota is not None and self.new_used >= self.quota:
                self.quota_hit = True
                return False
            self.run_used += 1
            if new_record:
                self.new_used += 1
            return True

    def release(self, new_record: bool) -> None:
        """Give back a unit taken by ``try_acquire`` for a record that was not processed (never below 0).

        Args:
            new_record: The value passed to the matching ``try_acquire`` call.
        """
        with self._lock:
            self.run_used = max(0, self.run_used - 1)
            if new_record:
                self.new_used = max(0, self.new_used - 1)

    @property
    def exhausted(self) -> bool:
        """Whether no further budgeted records should be claimed.

        Returns:
            True when the run limit is used up or a new record has been refused by the quota. After a quota
            refusal this stays True, so existing (non-new) records are no longer claimed either.
        """
        with self._lock:
            return (self.run_limit is not None and self.run_used >= self.run_limit) or self.quota_hit


@dataclass(slots=True)
class ItemOutcome:
    """What a page handler reports back to the scheduler for one successfully handled queue item.

    Attributes:
        status: ``"done"`` (item completed) or ``"skipped"`` (item marked skipped with ``note`` as reason).
        changed: Whether the stored content changed (increments the queue item's ``change_count``).
        content_hash: Hash of the fetched content, stored on the queue item; None to keep the old one.
        status_code: HTTP status of the fetch, stored as ``last_status_code``.
        etag: ``ETag`` to store for the next conditional request, or None.
        last_modified: ``Last-Modified`` to store for the next conditional request, or None.
        movie_id: Movie the page was linked to, stored on the queue item, or None.
        note: Skip reason (used when ``status`` is ``"skipped"``).
        counters: Extra counter increments to add to the run's ``Counters``.
    """
    status: str = "done"  # done | skipped
    changed: bool = False
    content_hash: str | None = None
    status_code: int | None = None
    etag: str | None = None
    last_modified: str | None = None
    movie_id: str | None = None
    note: str | None = None
    counters: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class SchedulerResult:
    """Final state of a scheduler run.

    Attributes:
        state: ``completed``, ``interrupted`` (shutdown requested), ``blocked`` (access-control response) or
            ``permission_denied`` (permission gate refused a URL).
        counters: Snapshot of the run's counters at the end.
        stop_reason: Human-readable reason the run stopped early, or None when it completed.
    """
    state: str  # completed | interrupted | blocked | permission_denied
    counters: dict[str, int]
    stop_reason: str | None = None


class CrawlScheduler:
    """Runs a pool of worker threads that claim, handle and settle items of the persistent crawl queue.

    Each worker claims the highest-priority due item of the requested page types from ``crawl_queue``,
    passes it to ``handler`` and records the outcome there (done/skipped/pending retry/failed/blocked).
    Workers exit when no due work is left, when the remaining retries are further away than
    ``max_retry_wait``, or when ``stop_event`` is set.
    """
    def __init__(
        self,
        *,
        queue: QueueRepository,
        jobs: JobRepository,
        logs: LogRepository,
        handler: Callable[[dict[str, Any]], ItemOutcome],
        job_id: str,
        page_types: Sequence[str],
        concurrency: int,
        lease_seconds: int,
        max_attempts: int,
        stop_event: threading.Event,
        counters: Counters,
        budget: RecordBudget | None = None,
        budget_types: frozenset[str] = frozenset({"movie_detail"}),
        is_new_record: Callable[[dict[str, Any]], bool] | None = None,
        claim_filter: dict[str, Any] | None = None,
        lock_owner: str | None = None,
        idle_poll: float = 1.0,
        max_retry_wait: float = 300.0,
        heartbeat_every: float = 15.0,
    ) -> None:
        """Configure the scheduler; no threads are started until ``run``.

        Args:
            queue: Repository of the ``crawl_queue`` collection.
            jobs: Repository of ``crawl_jobs``, used for heartbeats and refreshing the crawl lock.
            logs: Repository of ``crawl_logs``, where fetch errors and permission refusals are recorded.
            handler: Processes one claimed queue document and returns its outcome; may raise ``FetchError``,
                ``FetchCancelled`` or ``PermissionDeniedError``.
            job_id: Id of the crawl job, stored on claimed items.
            page_types: Page types (``PageType`` values) this run may claim.
            concurrency: Number of worker threads.
            lease_seconds: Lease length for a claimed item, in seconds; expired leases are recoverable.
            max_attempts: Attempts after which a retryable failure becomes ``failed``.
            stop_event: Shared shutdown event; also set by the scheduler when it halts the crawl.
            counters: Shared run counters.
            budget: Record budget; None for unlimited.
            budget_types: Page types that consume budget (movie detail pages by default).
            is_new_record: Decides whether a claimed item is a record not yet in the database; every item
                counts as new when None.
            claim_filter: Extra MongoDB predicate for claims (e.g. refresh runs), or None.
            lock_owner: Owner id of the global crawl lock to keep refreshed, or None.
            idle_poll: Maximum time an idle worker waits before trying to claim again, in seconds.
            max_retry_wait: Workers stop instead of waiting when the next due item is further away, in seconds.
            heartbeat_every: Interval between job heartbeats, in seconds.
        """
        self.queue = queue
        self.jobs = jobs
        self.logs = logs
        self.handler = handler
        self.job_id = job_id
        self.page_types = list(page_types)
        self.concurrency = concurrency
        self.lease_seconds = lease_seconds
        self.max_attempts = max_attempts
        self.stop_event = stop_event
        self.counters = counters
        self.budget = budget
        self.budget_types = budget_types
        self.is_new_record = is_new_record or (lambda _item: True)
        self.claim_filter = claim_filter
        self.lock_owner = lock_owner
        self.idle_poll = idle_poll
        self.max_retry_wait = max_retry_wait
        self.heartbeat_every = heartbeat_every
        self._busy = 0
        self._wake = threading.Condition()
        self.stop_reason: str | None = None
        self.state = "completed"

    # ------------------------------------------------------------------ helpers
    def _done_with_item(self) -> None:
        """Mark this worker idle and wake idle siblings (new work may have been enqueued)."""
        with self._wake:
            self._busy -= 1
            self._wake.notify_all()

    def _idle_or_exit(self) -> bool:
        """Called after an empty claim: mark this worker idle and decide whether to keep polling.

        If other workers are still busy (they may enqueue new items), waits up to ``idle_poll`` seconds
        or until a sibling finishes an item.

        Returns:
            True to retry claiming, False if nobody else is busy.
        """
        with self._wake:
            self._busy -= 1
            if self._busy > 0:
                self._wake.wait(timeout=self.idle_poll)
                return True
            return False

    def _types_now(self) -> list[str]:
        """Page types that may be claimed right now.

        Returns:
            ``page_types`` without the budgeted types once the budget is exhausted, otherwise ``page_types``.
        """
        if self.budget is not None and self.budget.exhausted:
            return [t for t in self.page_types if t not in self.budget_types]
        return self.page_types

    def _halt(self, state: str, reason: str) -> None:
        """Stop the whole crawl: record why, set the stop event and wake idle workers.

        Only the first halt's state and reason are kept.

        Args:
            state: Final state to report, e.g. ``blocked`` or ``permission_denied``.
            reason: Human-readable explanation stored as ``stop_reason``.
        """
        if self.stop_reason is None:
            self.state = state
            self.stop_reason = reason
        self.stop_event.set()
        with self._wake:
            self._wake.notify_all()

    def _claim(self, worker_id: str, types: list[str]) -> dict[str, Any] | None:
        """Atomically claim the next due pending item and lease it to a worker.

        The item becomes ``in_progress`` with a lease, its ``attempts`` is incremented, and it is chosen by
        priority (desc), discovery time, position and ``_id``. ``claim_filter``, when set, is added to the query.

        Args:
            worker_id: Lease owner id of the claiming worker.
            types: Page types that may be claimed.

        Returns:
            The claimed queue document after the update, or None when nothing is due.
        """
        if self.claim_filter is None:
            return self.queue.claim_next(job_id=self.job_id, worker_id=worker_id, page_types=types,
                                         lease_seconds=self.lease_seconds)
        # Filtered claim (e.g. refresh runs): same atomic semantics, extra predicate.
        now = utcnow()
        query = {"status": "pending", "page_type": {"$in": types}, "next_attempt_at": {"$lte": now},
                 **self.claim_filter}
        return self.queue.col.find_one_and_update(
            query,
            {"$set": {"status": "in_progress", "lease_owner": worker_id,
                      "lease_expires_at": now + timedelta(seconds=self.lease_seconds),
                      "job_id": self.job_id, "last_attempt_at": now},
             "$inc": {"attempts": 1}},
            sort=[("priority", -1), ("discovered_at", 1), ("position", 1), ("_id", 1)],
            return_document=ReturnDocument.AFTER,
        )

    # ------------------------------------------------------------------ worker
    def _worker(self, number: int) -> None:
        """Worker thread loop: claim an item, check the budget, run the handler and record the outcome.

        Outcomes: success -> ``complete``/``skip`` and ``<page_type>_done``/``_skipped`` counters;
        ``FetchError`` -> ``fail`` (retry later, failed or blocked), logged to ``crawl_logs``, and a blocked
        error halts the crawl; ``PermissionDeniedError`` -> item released, logged, crawl halted;
        ``FetchCancelled`` -> item and budget released, worker exits; any other exception -> item marked
        failed, ``unexpected_errors`` incremented, logged. Items refused by the budget are released.

        Args:
            number: Index of the worker, used in its thread name and worker id.
        """
        worker_id = f"{self.job_id}:w{number}:{uuid.uuid4().hex[:6]}"
        while not self.stop_event.is_set():
            types = self._types_now()
            if not types:
                return
            with self._wake:
                self._busy += 1  # counted as busy from before the claim: siblings never see a false "idle"
            item = self._claim(worker_id, types)
            if item is None:
                if self._idle_or_exit():
                    continue
                next_at = self.queue.next_ready_at(types)
                if next_at is None:
                    return
                wait = (next_at - utcnow()).total_seconds()
                if wait > self.max_retry_wait:
                    log.info("remaining items are scheduled for later retry (%.0fs); stopping worker", wait)
                    return
                self.stop_event.wait(max(0.05, min(wait, self.idle_poll)))
                continue

            needs_budget = self.budget is not None and item["page_type"] in self.budget_types
            new_record = False
            if needs_budget:
                new_record = self.is_new_record(item)
                assert self.budget is not None
                if not self.budget.try_acquire(new_record):
                    self.queue.release(item["_id"], worker_id)
                    self._done_with_item()
                    continue

            try:
                outcome = self.handler(item)
                if outcome.status == "skipped":
                    self.queue.skip(item["_id"], worker_id, outcome.note or "skipped")
                    self.counters.inc(f"{item['page_type']}_skipped")
                else:
                    ok = self.queue.complete(item["_id"], worker_id, status_code=outcome.status_code,
                                             content_hash=outcome.content_hash, changed=outcome.changed,
                                             etag=outcome.etag, last_modified=outcome.last_modified,
                                             movie_id=outcome.movie_id)
                    if not ok:
                        log.warning("lease lost before completion of %s", item["url"])
                    self.counters.inc(f"{item['page_type']}_done")
                    if item["page_type"] in self.budget_types:
                        self.counters.inc("records_processed")
                for key, value in outcome.counters.items():
                    self.counters.inc(key, value)
            except FetchCancelled:
                self.queue.release(item["_id"], worker_id)
                if needs_budget and self.budget is not None:
                    self.budget.release(new_record)
                return
            except PermissionDeniedError as exc:
                self.queue.release(item["_id"], worker_id)
                if needs_budget and self.budget is not None:
                    self.budget.release(new_record)
                self.logs.log("permission_denied", level="WARNING", job_id=self.job_id, url=item["url"],
                              page_type=item["page_type"], message=str(exc))
                self._halt("permission_denied", str(exc))
                return
            except FetchError as exc:
                status = self.queue.fail(item, worker_id, error=str(exc), status_code=exc.status_code,
                                         retryable=exc.retryable, blocked=exc.blocked,
                                         retry_after=exc.retry_after, max_attempts=self.max_attempts)
                self.counters.inc(f"{item['page_type']}_{'retry' if status == 'pending' else status}")
                if item["page_type"] in self.budget_types and status not in {"pending", "lease_lost"}:
                    self.counters.inc("records_processed")
                self.logs.log("fetch_error", level="WARNING" if status == "pending" else "ERROR",
                              job_id=self.job_id, url=item["url"], page_type=item["page_type"],
                              message=str(exc), details={"status_code": exc.status_code, "queue_status": status,
                                                         "attempts": item.get("attempts")})
                if exc.blocked:
                    self._halt("blocked", f"access restricted at {item['url']} (HTTP {exc.status_code}); "
                                          "crawl stopped without bypassing")
                    return
            except Exception as exc:  # unexpected bug: record it, keep the item for inspection
                log.exception("unexpected error processing %s", item["url"])
                self.queue.fail(item, worker_id, error=f"{exc.__class__.__name__}: {exc}", status_code=None,
                                retryable=False, blocked=False, retry_after=None, max_attempts=self.max_attempts)
                self.counters.inc("unexpected_errors")
                self.logs.log("unexpected_error", level="ERROR", job_id=self.job_id, url=item["url"],
                              page_type=item["page_type"], message=f"{exc.__class__.__name__}: {exc}")
            finally:
                self._done_with_item()

    # ------------------------------------------------------------------ run
    def run(self) -> SchedulerResult:
        """Start the workers and wait for all of them to finish.

        While waiting, writes a job heartbeat (with a counters snapshot) and refreshes the crawl lock every
        ``heartbeat_every`` seconds. Afterwards, any items still ``in_progress`` for this job are returned to
        ``pending``.

        Returns:
            The final state (``interrupted`` if the stop event was set without a halt reason), counters and
            stop reason.
        """
        threads = [threading.Thread(target=self._worker, args=(n,), name=f"crawler-{n}", daemon=True)
                   for n in range(self.concurrency)]
        for thread in threads:
            thread.start()
        last_beat = time.monotonic()
        while any(t.is_alive() for t in threads):
            for thread in threads:
                thread.join(timeout=0.5)
            if time.monotonic() - last_beat >= self.heartbeat_every:
                last_beat = time.monotonic()
                self.jobs.heartbeat(self.job_id, self.counters.snapshot())
                if self.lock_owner:
                    self.jobs.refresh_lock(self.lock_owner)
        if self.stop_reason is None and self.stop_event.is_set():
            self.state = "interrupted"
            self.stop_reason = "shutdown requested"
        self.queue.release_job(self.job_id)
        return SchedulerResult(self.state, self.counters.snapshot(), self.stop_reason)


def lease_is_expired(item: dict[str, Any]) -> bool:
    """Check whether a queue item's lease has run out.

    Args:
        item: Queue document, possibly with a ``lease_expires_at`` datetime.

    Returns:
        True when the item has a lease expiry in the past; False when it has none or it is still valid.
    """
    expires = as_utc(item.get("lease_expires_at"))
    return expires is not None and expires < utcnow()
