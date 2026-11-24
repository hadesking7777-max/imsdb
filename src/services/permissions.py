"""Source permission model and enforcement.

Levels (ordered): ``denied`` < ``discovery_only`` < ``sample`` < ``bulk``.

* ``discovery_only`` - listing pages only (titles, writers, draft dates).
* ``sample`` - discovery plus at most ``sample_limit`` movie records (detail + script),
  counted cumulatively across all runs.
* ``bulk`` - all records; only set by an operator who records authorization evidence.

The application never raises the level by itself. ``robots.txt`` rules are applied on
top of the level, and restrictive policy language found by ``inspect-source`` blocks
record collection until an operator re-confirms.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from urllib.robotparser import RobotFileParser

from src.crawler.url_manager import LISTING_TYPES, RECORD_TYPES, PageType, classify_url
from src.database.models import SourcePermissionDoc
from src.errors import PermissionDeniedError
from src.utils.clock import as_utc, utcnow

SOURCE_ID = "imsdb"
DEFAULT_SAMPLE_LIMIT = 10
LEVELS = ("denied", "discovery_only", "sample", "bulk")
ROBOTS_MAX_AGE = timedelta(hours=24)

ATTRIBUTION = ("Screenplay text retrieved from The Internet Movie Script Database (IMSDb), https://imsdb.com. "
               "All rights belong to the respective copyright holders (studios and writers).")
USAGE_RESTRICTION = ("Educational, scholarship, research and criticism use only (per the source's fair-use "
                     "disclaimer). Not for sale or redistribution. Local storage only.")
SOURCE_BASIS = "Source hosts scripts under a fair-use claim (17 U.S.C. 107); no licence is granted by IMSDb."

EXCLUDED_PAGE_TYPES = [PageType.WRITER.value, PageType.TV.value, PageType.FORM.value, PageType.ASSET.value,
                       PageType.SCRIPT_PDF.value, PageType.OTHER.value, PageType.EXTERNAL.value]

_RESTRICTIVE = re.compile(
    r"(scrap(?:e|er|ers|ing)\b|crawl(?:er|ers|ing)\b|web spiders?|spidering|robots? (?:may|must|are|is) not|"
    r"automated (?:access|means|collection|queries|requests)|bulk download\w*|data mining|"
    r"may not (?:copy|reproduce|redistribute|download)|(?:strictly )?prohibited|not permitted|"
    r"without (?:prior )?(?:written )?(?:permission|consent)|all rights reserved)",
    re.IGNORECASE,
)
_PERMISSIVE = re.compile(
    r"(creative commons|public domain|freely (?:use|copy|distribute)|permission is (?:hereby )?granted|"
    r"open licen[cs]e|may be (?:freely )?(?:copied|reproduced|redistributed))",
    re.IGNORECASE,
)
_FAIR_USE = re.compile(r"(fair use|section 107|educational purposes?)", re.IGNORECASE)


def level_rank(level: str) -> int:
    """Position of a permission level in the ordering ``denied < discovery_only < sample < bulk``.

    Args:
        level: One of ``LEVELS``.

    Returns:
        0 for ``denied`` up to 3 for ``bulk``.

    Raises:
        ValueError: ``level`` is not a known level.
    """
    return LEVELS.index(level)


def policy_main_text(html: str) -> str:
    """Visible text of a policy page's main content cell (sidebar/navigation excluded).

    Scripts, styles and forms are dropped; the text is taken from the parent of the last ``<h1>`` (or
    ``<h2>``), falling back to the whole body.

    Args:
        html: Decoded HTML of the policy page.

    Returns:
        The text with whitespace collapsed to single spaces.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "ins", "form"]):
        tag.decompose()
    # IMSDb pages carry the site banner as the first <h1>; the page's own heading comes last.
    headings = soup.find_all("h1") or soup.find_all("h2")
    heading = headings[-1] if headings else None
    scope = heading.parent if heading is not None and heading.parent is not None else (soup.body or soup)
    return " ".join(scope.get_text(" ").split())


def default_imsdb_permission(now: datetime | None = None) -> SourcePermissionDoc:
    """Build the initial IMSDb permission record seeded when ``source_permissions`` has none.

    The default is the ``sample`` level (``DEFAULT_SAMPLE_LIMIT`` records), bulk authorization not verified,
    with the usage restrictions, attribution, excluded page types and a first evidence entry.

    Args:
        now: Timestamp for ``created_at``/``updated_at`` and the evidence entry; current UTC time when None.

    Returns:
        The permission document (not yet saved).
    """
    now = now or utcnow()
    return SourcePermissionDoc(
        _id=SOURCE_ID,
        source_name="The Internet Movie Script Database (IMSDb)",
        base_url="https://imsdb.com/",
        level="sample",
        sample_limit=DEFAULT_SAMPLE_LIMIT,
        version=1,
        robots={},
        policy_documents=[],
        assessment={
            "summary": ("No robots.txt (HTTP 404) and no terms of service found. The disclaimer claims fair use "
                        "for educational purposes. IMSDb is not the copyright holder and grants no licence. "
                        "Bulk-collection authorization is NOT verified."),
            "bulk_authorized": False,
            "restriction_found": False,
            "assessed_by": "initial project assessment (docs/SOURCE_PERMISSIONS.md)",
        },
        usage_restrictions=[USAGE_RESTRICTION,
                            "Do not crawl forms, comment endpoints, writer pages or TV transcripts.",
                            "Do not bypass authentication, rate limits or anti-bot protections.",
                            "User-comment author names are not extracted into structured fields."],
        attribution={"text": ATTRIBUTION, "source_url": "https://imsdb.com/"},
        rights={"copyright_holder": "Respective studios and writers", "source_basis": SOURCE_BASIS},
        evidence=[{"recorded_at": now, "recorded_by": "system",
                   "statement": "Default sample-level permission pending operator authorization.",
                   "level": "sample"}],
        history=[],
        excluded_page_types=EXCLUDED_PAGE_TYPES,
        created_at=now,
        updated_at=now,
    )


# ====================================================================== robots
@dataclass(slots=True)
class RobotsPolicy:
    """Interpreted robots.txt of the source.

    Attributes:
        status: HTTP status of the robots.txt fetch, or None when it was unreachable.
        allow_all: No restrictions apply (robots.txt missing with a 4xx other than 401/403).
        disallow_all: Nothing may be fetched (401/403, 5xx or unreachable).
        crawl_delay: ``Crawl-delay`` for our user agent in seconds, or None.
        body: robots.txt text (empty when not fetched successfully).
        fetched_at: When robots.txt was fetched, or None.
        sha256: Hash of the archived robots.txt body, or None.
        _parser: Parsed rules for a 2xx response; None otherwise.
    """
    status: int | None
    allow_all: bool
    disallow_all: bool
    crawl_delay: float | None
    body: str
    fetched_at: datetime | None
    sha256: str | None
    _parser: RobotFileParser | None = None

    @classmethod
    def from_response(cls, status: int | None, body: str, *, fetched_at: datetime | None = None,
                      sha256: str | None = None, user_agent: str = "*") -> "RobotsPolicy":
        """Interpret a robots.txt response.

        RFC 9309: 2xx parse; 4xx -> no restrictions (401/403 treated as disallow, conservatively);
        5xx / unreachable -> complete disallow.

        Args:
            status: HTTP status of the robots.txt response, or None when it could not be fetched.
            body: robots.txt text (parsed only for 2xx).
            fetched_at: When it was fetched.
            sha256: Hash of the archived body.
            user_agent: User agent whose ``Crawl-delay`` is read.

        Returns:
            The interpreted policy.
        """
        if status is None or status >= 500:
            return cls(status, False, True, None, body, fetched_at, sha256)
        if status in (401, 403):
            return cls(status, False, True, None, body, fetched_at, sha256)
        if 400 <= status < 500:
            return cls(status, True, False, None, body, fetched_at, sha256)
        parser = RobotFileParser()
        parser.parse(body.splitlines())
        delay = parser.crawl_delay(user_agent)
        return cls(status, False, False, float(delay) if delay is not None else None, body, fetched_at, sha256,
                   parser)

    @classmethod
    def from_record(cls, record: dict[str, Any], user_agent: str = "*") -> "RobotsPolicy | None":
        """Rebuild a policy from the ``robots`` sub-document stored on the permission record.

        Args:
            record: Dict as produced by ``to_record`` (may be empty).
            user_agent: User agent whose ``Crawl-delay`` is read.

        Returns:
            The policy, or None when no robots.txt result has been stored yet.
        """
        if not record or "status" not in record:
            return None
        return cls.from_response(record.get("status"), record.get("body", ""),
                                 fetched_at=as_utc(record.get("fetched_at")), sha256=record.get("sha256"),
                                 user_agent=user_agent)

    def to_record(self) -> dict[str, Any]:
        """Serialize the policy for storage on the permission record.

        Returns:
            Status, flags, crawl delay, body (truncated to 100,000 characters), fetch time and hash.
        """
        return {"status": self.status, "allow_all": self.allow_all, "disallow_all": self.disallow_all,
                "crawl_delay": self.crawl_delay, "body": self.body[:100_000], "fetched_at": self.fetched_at,
                "sha256": self.sha256}

    def is_stale(self, now: datetime | None = None) -> bool:
        """Whether robots.txt should be fetched again.

        Args:
            now: Reference time; current UTC time when None.

        Returns:
            True when the fetch time is unknown or older than ``ROBOTS_MAX_AGE`` (24 hours).
        """
        return self.fetched_at is None or (now or utcnow()) - self.fetched_at > ROBOTS_MAX_AGE

    def allowed(self, url: str, user_agent: str) -> bool:
        """Check a URL against the robots.txt rules.

        Args:
            url: Absolute URL to check.
            user_agent: Our crawler's user agent.

        Returns:
            False when everything is disallowed, True when there are no rules, otherwise the parser's verdict.
        """
        if self.disallow_all:
            return False
        if self.allow_all or self._parser is None:
            return True
        return self._parser.can_fetch(user_agent, url)


# ====================================================================== policy assessment
def assess_policy_text(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Heuristic scan of policy pages. Never grants permission by itself.

    Matches restrictive, permissive and fair-use phrases. "All rights reserved" is reported but does not
    count as a restriction.

    Args:
        documents: Policy pages as dicts with ``url`` and ``text`` (main text).

    Returns:
        ``fair_use_claim_found``, ``restrictive_phrases`` and ``permissive_phrases`` (up to 20 each, with
        URL, phrase and ~80 characters of context on each side), ``restriction_found``,
        ``explicit_grant_found``, ``bulk_authorized`` (always False) and a ``recommendation`` sentence.
    """
    restrictive: list[dict[str, str]] = []
    permissive: list[dict[str, str]] = []
    fair_use = False
    for doc in documents:
        text = doc.get("text") or ""
        for match in _RESTRICTIVE.finditer(text):
            restrictive.append({"url": doc["url"], "phrase": match.group(0),
                                "context": text[max(0, match.start() - 80):match.end() + 80]})
        for match in _PERMISSIVE.finditer(text):
            permissive.append({"url": doc["url"], "phrase": match.group(0),
                               "context": text[max(0, match.start() - 80):match.end() + 80]})
        fair_use = fair_use or bool(_FAIR_USE.search(text))
    # "All rights reserved" is a copyright notice, not an access rule: reported, but not blocking.
    hard = [r for r in restrictive if "rights reserved" not in r["phrase"].lower()]
    return {
        "fair_use_claim_found": fair_use,
        "restrictive_phrases": hard[:20],
        "permissive_phrases": permissive[:20],
        "restriction_found": bool(hard),
        "explicit_grant_found": bool(permissive),
        "bulk_authorized": False,
        "recommendation": (
            "Restrictive language found: review the evidence before any record collection."
            if hard else
            "No explicit grant found: keep the 'sample' level unless you obtain written authorization."
            if not permissive else
            "Possible grant language found: verify it and record evidence with the 'permit' command."
        ),
    }


# ====================================================================== gate
@dataclass(slots=True)
class Decision:
    """Result of a permission check.

    Attributes:
        allowed: Whether the URL may be fetched.
        reason: Human-readable explanation (logged and shown in errors).
        url_specific: For a refusal, True when it concerns only this URL (external host, robots.txt,
            excluded or non-crawlable page type): the crawler skips the URL and continues. False when the
            refusal applies to the whole collection (permission level, restrictive source policy): the
            crawl must stop.
    """
    allowed: bool
    reason: str
    url_specific: bool = False


class PermissionGate:
    """Decides, for every URL, whether the current permission record and robots.txt allow fetching it.

    Also enforces the cumulative sample quota when planning a crawl.
    """
    def __init__(self, record: SourcePermissionDoc, robots: RobotsPolicy | None, user_agent: str,
                 collected_count: Callable[[], int]) -> None:
        """Create the gate.

        Args:
            record: Current ``source_permissions`` record.
            robots: Interpreted robots.txt, or None when unknown (then no robots rules are applied).
            user_agent: Our crawler's user agent, matched against robots.txt.
            collected_count: Returns how many movie records have been collected in total (for the quota).
        """
        self.record = record
        self.robots = robots
        self.user_agent = user_agent
        self._collected_count = collected_count

    @property
    def level(self) -> str:
        """The permission level of the record.

        Returns:
            ``denied``, ``discovery_only``, ``sample`` or ``bulk``.
        """
        return self.record.level

    @property
    def restriction_found(self) -> bool:
        """Whether inspect-source found restrictive policy language that an operator has not overridden.

        Returns:
            True when record collection must be blocked for that reason.
        """
        return bool(self.record.assessment.get("restriction_found")) and \
            not self.record.assessment.get("restriction_overridden")

    def check(self, url: str, page_type: PageType | str | None = None) -> Decision:
        """Decide whether a URL may be fetched.

        Order: external hosts are refused; robots.txt itself is always allowed; then robots.txt rules apply
        (to policy pages too, which are otherwise allowed at every level); then excluded page types and the
        level (listings need ``discovery_only``, records need ``sample`` and no unresolved policy restriction).
        Anything else is refused as not crawlable.

        Args:
            url: Normalized URL.
            page_type: Its page type, or None to classify the URL.

        Returns:
            The decision with its reason.
        """
        page_type = PageType(page_type) if page_type else classify_url(url)
        if page_type == PageType.EXTERNAL:
            return Decision(False, "external host", url_specific=True)
        if page_type == PageType.ROBOTS:
            return Decision(True, "reading robots.txt is always permitted")
        if self.robots is not None and not self.robots.allowed(url, self.user_agent):
            return Decision(False, f"robots.txt disallows {url}", url_specific=True)
        if page_type == PageType.POLICY:
            return Decision(True, "policy inspection is permitted at every level")
        if page_type.value in self.record.excluded_page_types:
            return Decision(False, f"page type {page_type.value!r} is excluded by the permission record",
                            url_specific=True)
        if page_type in LISTING_TYPES:
            if level_rank(self.level) >= level_rank("discovery_only"):
                return Decision(True, "discovery permitted")
            return Decision(False, f"permission level {self.level!r} does not allow discovery")
        if page_type in RECORD_TYPES:
            if self.restriction_found:
                return Decision(False, "restrictive source policy detected by inspect-source; operator review required")
            if level_rank(self.level) >= level_rank("sample"):
                return Decision(True, f"record collection permitted at level {self.level!r}")
            return Decision(False, f"permission level {self.level!r} does not allow record collection")
        return Decision(False, f"page type {page_type.value!r} is not crawlable", url_specific=True)

    def require(self, url: str, page_type: PageType | str | None = None) -> None:
        """Like ``check``, but raise when the URL is not allowed.

        Args:
            url: Normalized URL.
            page_type: Its page type, or None to classify the URL.

        Raises:
            PermissionDeniedError: The gate refuses the URL; the message includes the reason.
        """
        decision = self.check(url, page_type)
        if not decision.allowed:
            raise PermissionDeniedError(f"Not permitted to fetch {url}: {decision.reason}")

    def remaining_sample_quota(self) -> int | None:
        """How many more new records the permission allows, counted across all runs.

        Returns:
            None for ``bulk`` (unlimited), 0 below ``sample``, otherwise ``sample_limit`` minus the records
            already collected (never negative).
        """
        if self.level == "bulk":
            return None
        if self.level != "sample":
            return 0
        return max(0, self.record.sample_limit - self._collected_count())

    def require_discovery(self) -> None:
        """Ensure the permission level allows crawling listing pages.

        Raises:
            PermissionDeniedError: The level is below ``discovery_only``.
        """
        if level_rank(self.level) < level_rank("discovery_only"):
            raise PermissionDeniedError(
                f"Source permission level is {self.level!r}; discovery is not allowed. "
                "Use 'python -m src.main permit' to change it if you are authorized.")

    def plan_crawl(self, *, limit: int | None, all_records: bool) -> tuple[int | None, int | None, list[str]]:
        """Validate a crawl request and work out its limits.

        Args:
            limit: Requested maximum number of records (``--limit``); must be positive unless ``all_records``.
            all_records: Whether ``--all`` was requested (bulk level only).

        Returns:
            ``(run_limit, new_record_quota, notices)``: ``(None, None, [])`` for ``--all``; otherwise the limit
            capped at the remaining sample quota (with a notice when capped), the quota (None if unlimited)
            and the notices.

        Raises:
            PermissionDeniedError: An unresolved policy restriction exists, ``--all`` without bulk level, no
                positive limit, or the sample quota is used up.
        """
        notices: list[str] = []
        if self.restriction_found:
            raise PermissionDeniedError(
                "inspect-source found restrictive policy language. Review source_permissions.assessment and "
                "re-confirm with 'permit --override-restriction' if you are authorized.")
        if all_records:
            if self.level != "bulk":
                raise PermissionDeniedError(
                    f"'crawl --all' requires bulk authorization; current level is {self.level!r}.\n"
                    "Bulk-collection permission for IMSDb has not been verified (see docs/SOURCE_PERMISSIONS.md).\n"
                    "If you hold written authorization, record it with:\n"
                    "  python -m src.main permit --level bulk --granted-by \"...\" --evidence \"...\" --confirm")
            return None, None, notices
        if limit is None or limit < 1:
            raise PermissionDeniedError("A positive --limit is required unless --all is used.")
        quota = self.remaining_sample_quota()
        if quota is not None and quota <= 0:
            raise PermissionDeniedError(
                f"The {self.level!r} permission allows {self.record.sample_limit} records in total and the quota "
                "is used up. Existing records can still be refreshed with --refresh-days. Collecting more requires "
                "recorded authorization (permit --level bulk ...).")
        if quota is not None and limit > quota:
            notices.append(f"Sample permission allows {quota} more new record(s); limiting this run to {quota}.")
            limit = quota
        return limit, quota, notices
