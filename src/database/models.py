"""Pydantic document models for every MongoDB collection.

Every document written by the application passes through one of these models so
malformed records are rejected before they reach the database. ``extra="allow"``
is used where the source may expose fields we do not know yet (they are preserved).

Identifier scheme (stable across re-runs, derived from canonical source URLs):

* movie      ``mv-<slug>-<hash8(detail_url)>``
* script     ``sc-<slug>-<hash8(script_url)>``
* scene      ``<script_id>:s<NNNN>``         (0 = prelude before the first heading)
* dialogue   ``<script_id>:d<NNNNN>``         (global order within the script)
* character  ``<script_id>:c:<name-slug>``
* queue item ``u_<hash32(url)>``
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.utils.hashing import short_hash
from src.utils.text_processing import slugify


class Collections:
    """Names of the MongoDB collections used by the application.

    Attributes:
        MOVIES: One document per IMSDb movie detail page.
        SCRIPTS: Current version of each screenplay (full text plus parse output).
        SCRIPT_VERSIONS: Archived earlier versions of scripts whose source content changed.
        SCENES: Scenes split out of each script.
        CHARACTERS: Speaking characters of each script.
        DIALOGUES: Individual dialogue blocks of each script.
        CRAWL_QUEUE: URLs to fetch, with their lease, retry and change-tracking state.
        CRAWL_LOGS: Structured crawl log events.
        CRAWL_JOBS: One document per CLI job run, plus the crawler lock document.
        SOURCE_PERMISSIONS: Collection permission level and evidence for each source site.
        RAW_ARTIFACTS: Index of the raw HTTP responses stored on disk.
        ALL: Every collection name above, in creation order.
    """
    MOVIES = "movies"
    SCRIPTS = "scripts"
    SCRIPT_VERSIONS = "script_versions"
    SCENES = "scenes"
    CHARACTERS = "characters"
    DIALOGUES = "dialogues"
    CRAWL_QUEUE = "crawl_queue"
    CRAWL_LOGS = "crawl_logs"
    CRAWL_JOBS = "crawl_jobs"
    SOURCE_PERMISSIONS = "source_permissions"
    RAW_ARTIFACTS = "raw_artifacts"

    ALL = (MOVIES, SCRIPTS, SCRIPT_VERSIONS, SCENES, CHARACTERS, DIALOGUES, CRAWL_QUEUE,
           CRAWL_LOGS, CRAWL_JOBS, SOURCE_PERMISSIONS, RAW_ARTIFACTS)


# ---------------------------------------------------------------- identifiers
def movie_id_for(detail_url: str, title: str) -> str:
    """Build the stable movie ``_id`` (``mv-<slug>-<hash8>``).

    Args:
        detail_url: Canonical IMSDb detail-page URL; its hash makes the id unique per URL.
        title: Movie title used for the readable slug (truncated to 60 characters).

    Returns:
        The identifier, e.g. ``mv-the-dark-knight-1a2b3c4d``.
    """
    return f"mv-{slugify(title, 60)}-{short_hash(detail_url, length=8)}"


def script_id_for(script_url: str, title: str) -> str:
    """Build the stable script ``_id`` (``sc-<slug>-<hash8>``).

    Args:
        script_url: Canonical screenplay URL; its hash makes the id unique per URL.
        title: Movie title used for the readable slug (truncated to 60 characters).

    Returns:
        The identifier, e.g. ``sc-the-dark-knight-5e6f7a8b``.
    """
    return f"sc-{slugify(title, 60)}-{short_hash(script_url, length=8)}"


def scene_id_for(script_id: str, scene_number: int) -> str:
    """Build a scene ``_id`` (``<script_id>:sNNNN``).

    Args:
        script_id: ``_id`` of the script the scene belongs to.
        scene_number: Position of the scene in the script; 0 is the prelude before the first heading.

    Returns:
        The identifier, with the scene number zero-padded to 4 digits.
    """
    return f"{script_id}:s{scene_number:04d}"


def dialogue_id_for(script_id: str, order: int) -> str:
    """Build a dialogue ``_id`` (``<script_id>:dNNNNN``).

    Args:
        script_id: ``_id`` of the script the dialogue belongs to.
        order: Global position of the dialogue block within the script.

    Returns:
        The identifier, with the order zero-padded to 5 digits.
    """
    return f"{script_id}:d{order:05d}"


def character_id_for(script_id: str, key: str) -> str:
    """Build a character ``_id`` (``<script_id>:c:<key>``).

    Args:
        script_id: ``_id`` of the script the character appears in.
        key: Normalised name slug of the character (``CharacterDoc.name_key``).

    Returns:
        The identifier.
    """
    return f"{script_id}:c:{key}"


class MongoModel(BaseModel):
    """Base class of all MongoDB document models.

    Unknown fields are rejected (``extra="forbid"``) and fields accept either their name or their alias
    (``id`` / ``_id``).
    """
    model_config = ConfigDict(populate_by_name=True, extra="forbid", arbitrary_types_allowed=True)

    def to_mongo(self, **kwargs: Any) -> dict[str, Any]:
        """Convert the model into a dict ready to be written to MongoDB.

        Args:
            **kwargs: Extra options passed to ``model_dump`` (e.g. ``exclude``).

        Returns:
            The document with aliases applied (``id`` becomes ``_id``) and Python types kept (datetimes stay
            ``datetime`` so PyMongo stores them as BSON dates).
        """
        return self.model_dump(by_alias=True, mode="python", **kwargs)


# ---------------------------------------------------------------- shared parts
class Person(MongoModel):
    """A named person (writer or director) linked from a detail page.

    Attributes:
        name: Name as shown on IMSDb.
        url: Link to the person's IMSDb page, or None when the name is not linked.
    """
    name: str
    url: str | None = None


class Rights(MongoModel):
    """Rights and attribution metadata stored with every collected movie and script.

    Attributes:
        source: Source site id (always ``imsdb``).
        copyright_holder: Who owns the copyright (studios and writers, not IMSDb).
        attribution: Attribution text to show with the content.
        usage_restriction: How the content may be used.
        source_basis: Why the record was allowed to be collected.
        permission_level_at_collection: Source permission level in force at collection time
            (``denied``, ``discovery_only``, ``sample`` or ``bulk``).
        permission_version: Version of the ``source_permissions`` record in force at collection time.
    """
    source: str = "imsdb"
    copyright_holder: str = "Respective copyright holders (studios and writers); not IMSDb"
    attribution: str
    usage_restriction: str
    source_basis: str
    permission_level_at_collection: str
    permission_version: int


class RawRef(MongoModel):
    """Pointer from a parsed record to the raw HTTP response it was parsed from.

    Attributes:
        artifact_id: ``_id`` of the matching ``raw_artifacts`` document.
        sha256: SHA-256 hex digest of the uncompressed response body.
        path: Location of the stored body, relative to the raw archive folder (POSIX separators).
        size: Uncompressed body size in bytes.
        fetched_at: When the response was fetched (UTC).
        status_code: HTTP status code of the response.
    """
    artifact_id: str
    sha256: str
    path: str
    size: int
    fetched_at: datetime
    status_code: int


class DraftInfo(MongoModel):
    """Draft information shown next to a title on a listing page, e.g. ``(1994-06 Draft)``.

    Attributes:
        text: Text inside the parentheses, e.g. ``1994-06 Draft``; None when absent.
        date: Date part, e.g. ``1994-06``; None when undated.
        label: Remaining label, e.g. ``Draft``; None when there is none.
    """
    text: str | None = None
    date: str | None = None
    label: str | None = None


class ListingAppearance(MongoModel):
    """One appearance of a movie on a listing page.

    Attributes:
        source: Listing label, e.g. ``genre:Action``, ``alphabetical:A`` or ``all-scripts``.
        url: URL of the listing page.
        title: Title as shown on the listing.
        draft_text: Draft text shown next to the title, or None.
        writers_text: Writers text shown on the listing, or None.
        position: Order of the entry on the listing page.
    """
    source: str  # e.g. "genre:Action", "alphabetical:A", "all-scripts"
    url: str
    title: str
    draft_text: str | None = None
    writers_text: str | None = None
    position: int


# ---------------------------------------------------------------- movies
MovieStatus = Literal["discovered", "detail_collected", "script_collected", "script_unavailable",
                      "script_empty_at_source", "script_shared", "script_failed", "detail_failed"]


class MovieListingFields(MongoModel):
    """Fields derived from listing pages (never overwrite detail-page data).

    Attributes:
        listing_title: Title as shown on the listing page.
        listing_draft: Draft information shown next to the title.
        listing_writers: Writer names shown on the listing page.
    """

    listing_title: str
    listing_draft: DraftInfo
    listing_writers: list[str]


class MovieDetailFields(MongoModel):
    """Movie metadata parsed from an IMSDb detail page, written with ``$set`` by ``MovieRepository.apply_detail``.

    Attributes:
        title: Title as given on the detail page.
        display_title: Title with a trailing article moved to the front (``Dark Knight, The`` -> ``The Dark Knight``).
        title_key: Normalised slug of the display title, used to detect duplicate titles.
        script_url: Canonical URL of the screenplay, or None when there is no script link.
        script_availability: ``html`` (readable script page), ``pdf``, ``missing`` or ``unknown``.
        script_link_text: Text of the script link, or None.
        release_year: Year taken from the movie release date, or None.
        movie_release_date: Release date text as shown, or None.
        script_date: Script date text as shown, or None.
        genres: Genre names.
        writers: Credited writers.
        directors: Credited directors.
        description: Synopsis text, or None.
        imdb_ids: IMDb title ids linked from the page (sorted, unique).
        ratings: IMSDb opinion/rating and average user rating values and texts.
        poster_url: Poster image URL, or None.
        user_comment_count: Number of user comments with a rating on the page.
        page_meta: HTML ``<title>`` and ``<meta>`` contents of the page.
        extra_fields: Labelled fields the parser does not know yet, preserved as found.
        detail_warnings: Parser warnings for this page.
        detail_raw: Raw response the metadata was parsed from (its hash detects content changes).
        detail_parser_version: Version of the detail-page parser.
        rights: Rights and attribution metadata.
    """
    title: str
    display_title: str
    title_key: str
    script_url: str | None
    script_availability: Literal["html", "pdf", "missing", "unknown"]
    script_link_text: str | None
    release_year: int | None
    movie_release_date: str | None
    script_date: str | None
    genres: list[str]
    writers: list[Person]
    directors: list[Person]
    description: str | None
    imdb_ids: list[str]
    ratings: dict[str, Any]
    poster_url: str | None
    user_comment_count: int
    page_meta: dict[str, str]
    extra_fields: dict[str, Any]
    detail_warnings: list[str]
    detail_raw: RawRef
    detail_parser_version: str
    rights: Rights


# ---------------------------------------------------------------- scripts
ScriptStatus = Literal["parsed", "parsed_with_warnings", "empty_at_source", "extraction_failed", "parse_failed",
                       "pdf_only"]


class ScriptDoc(MongoModel):
    """A screenplay: extracted text plus structure, stored in ``scripts`` (one document per script URL).

    Attributes:
        id: Script ``_id`` (``sc-<slug>-<hash8>``, see ``script_id_for``).
        movie_id: ``_id`` of the movie the script belongs to.
        source: Source site id (always ``imsdb``).
        title: Movie title.
        script_url: Canonical screenplay URL (unique).
        status: ``parsed``, ``parsed_with_warnings``, ``empty_at_source``, ``extraction_failed``, ``parse_failed``
            or ``pdf_only``.
        version: Content version, starting at 1 and incremented each time the source text changes.
        content_hash: SHA-256 of the extracted text, or None when no text was extracted.
        raw: Raw response the script was extracted from.
        encoding: How the page was decoded (encoding used, strict decoding, HTTP/meta declared charsets).
        text: Full extracted screenplay text; all offsets of derived records point into it.
        text_length: Length of ``text`` in characters.
        line_count: Number of lines in ``text``.
        source_markup: Original HTML of the screenplay container, or None (also when dropped for size).
        formatting_spans: Bold/italic/etc. spans mapped onto ``text``, or None when offloaded.
        extraction: Extraction details (method, normalisations, extraction ratio, removed tags, page title).
        layout: Detected indentation/layout profile, or None when not parsed.
        structure_stats: Element counts and other parse statistics.
        elements: Classified screenplay elements (headings, action, dialogue...), or None when offloaded.
        elements_file: Where ``elements``/``formatting_spans`` were offloaded (path, sha256, bytes, element count)
            when the document would exceed the BSON size limit; None otherwise.
        footer_metadata: Metadata found in the page footer below the script.
        parse_warnings: Human-readable parser warnings.
        quality_flags: Sorted machine-readable quality flags.
        parser_version: Version of the screenplay parser that produced the document.
        rights: Rights and attribution metadata.
        fetched_at: When the raw page was fetched (UTC).
        parsed_at: When the parse ran (UTC).
        last_checked_at: When the source was last re-checked (UTC).
        created_at: When the script was first stored (UTC).
        updated_at: When the document was last written (UTC).
        previous_versions: Number of earlier versions archived in ``script_versions``.
        error: Error message for failed statuses, or None.
    """
    id: str = Field(alias="_id")
    movie_id: str
    source: str = "imsdb"
    title: str
    script_url: str
    status: ScriptStatus
    version: int = Field(ge=1)
    content_hash: str | None  # sha256 of the extracted text
    raw: RawRef
    encoding: dict[str, Any]
    text: str
    text_length: int
    line_count: int
    source_markup: str | None  # original HTML of the screenplay container
    formatting_spans: list[dict[str, Any]] | None
    extraction: dict[str, Any]
    layout: dict[str, Any] | None
    structure_stats: dict[str, Any]
    elements: list[dict[str, Any]] | None
    elements_file: dict[str, Any] | None = None  # offloaded when too large for one BSON doc
    footer_metadata: dict[str, Any]
    parse_warnings: list[str]
    quality_flags: list[str]
    parser_version: str
    rights: Rights
    fetched_at: datetime
    parsed_at: datetime
    last_checked_at: datetime
    created_at: datetime
    updated_at: datetime
    previous_versions: int = 0
    error: str | None = None


class SceneDoc(MongoModel):
    """One scene of a script, stored in ``scenes``.

    Offsets are into ``ScriptDoc.text`` (characters, end exclusive), its lines and its ``elements`` list.

    Attributes:
        id: Scene ``_id`` (``<script_id>:sNNNN``).
        script_id: ``_id`` of the script.
        movie_id: ``_id`` of the movie.
        scene_number: Position in the script; 0 is the prelude before the first heading.
        kind: ``prelude`` (text before the first heading), ``scene``, or ``unsegmented`` (no headings found).
        heading: Scene heading line, e.g. ``INT. HOUSE - NIGHT``; None for preludes.
        heading_parts: Heading split into parts (interior/exterior, location, time of day...), or None.
        act: Act label the scene falls in, or None.
        page_start: Script page number where the scene starts, or None.
        page_end: Script page number where the scene ends, or None.
        description: Action/description text of the scene.
        text: Full scene text (``text[char_start:char_end]`` of the script).
        char_start: Start character offset.
        char_end: End character offset (exclusive).
        line_start: First line number.
        line_end: Last line number.
        element_start: Index of the first element.
        element_end: Index of the last element (inclusive).
        characters: Names of the characters speaking in the scene.
        character_ids: ``_id`` values of those characters, in order of first appearance.
        dialogue_count: Number of dialogue blocks.
        shots: Shot directions found in the scene.
        transitions: Transitions found in the scene (``CUT TO:`` ...).
        element_counts: Number of elements per element type.
        confidence: Segmentation confidence from 0 to 1.
        uncertain: True when ``confidence`` is below 0.7.
        warnings: Parser warnings for the scene.
        content_hash: SHA-256 of ``text``.
        script_version_hash: ``content_hash`` of the script version the scene was derived from.
        parser_version: Version of the parser that produced the scene.
        created_at: When the scene was written (UTC).
    """
    id: str = Field(alias="_id")
    script_id: str
    movie_id: str
    scene_number: int
    kind: Literal["prelude", "scene", "unsegmented"]
    heading: str | None
    heading_parts: dict[str, Any] | None
    act: str | None
    page_start: str | None
    page_end: str | None
    description: str
    text: str
    char_start: int
    char_end: int
    line_start: int
    line_end: int
    element_start: int
    element_end: int
    characters: list[str]
    character_ids: list[str]
    dialogue_count: int
    shots: list[str]
    transitions: list[str]
    element_counts: dict[str, int]
    confidence: float
    uncertain: bool
    warnings: list[str]
    content_hash: str
    script_version_hash: str
    parser_version: str
    created_at: datetime


class DialogueDoc(MongoModel):
    """One dialogue block (speaker cue plus speech) of a script, stored in ``dialogues``.

    Attributes:
        id: Dialogue ``_id`` (``<script_id>:dNNNNN``).
        script_id: ``_id`` of the script.
        movie_id: ``_id`` of the movie.
        scene_id: ``_id`` of the scene containing the block, or None.
        character_id: ``_id`` of the speaking character, or None when not resolved.
        order: Global position within the script, from 0.
        order_in_scene: Position within its scene, from 0.
        speaker: Normalised speaker name.
        speaker_raw: Speaker cue exactly as written.
        extensions: Cue extensions such as ``V.O.`` or ``O.S.``.
        dual: True when the cue carried the dual-dialogue marker.
        text: Spoken words only (no cue or parentheticals), whitespace-normalised.
        original_text: Exact slice of the script text, including cue, parentheticals and indentation.
        parentheticals: Parenthetical directions, e.g. ``(whispering)``.
        segments: All elements after the cue (``dialogue``, ``parenthetical``, ``continuation``), in order.
        char_start: Start character offset into the script text.
        char_end: End character offset into the script text (exclusive).
        line_start: First line number.
        line_end: Last line number.
        element_start: Index of the first element.
        element_end: Index of the last element.
        word_count: Number of whitespace-separated words in ``text``.
        confidence: Lowest confidence (0 to 1) among the cue and its segments.
        uncertain: True when ``confidence`` is below 0.7.
        warnings: Parser warnings for the block.
        script_version_hash: ``content_hash`` of the script version the block was derived from.
        parser_version: Version of the parser that produced the block.
        created_at: When the block was written (UTC).
    """
    id: str = Field(alias="_id")
    script_id: str
    movie_id: str
    scene_id: str | None
    character_id: str | None
    order: int
    order_in_scene: int
    speaker: str
    speaker_raw: str
    extensions: list[str]
    dual: bool
    text: str
    original_text: str
    parentheticals: list[str]
    segments: list[dict[str, Any]]
    char_start: int
    char_end: int
    line_start: int
    line_end: int
    element_start: int
    element_end: int
    word_count: int
    confidence: float
    uncertain: bool
    warnings: list[str]
    script_version_hash: str
    parser_version: str
    created_at: datetime


class CharacterDoc(MongoModel):
    """A speaking character of one script, stored in ``characters``.

    Attributes:
        id: Character ``_id`` (``<script_id>:c:<name_key>``).
        script_id: ``_id`` of the script.
        movie_id: ``_id`` of the movie.
        name: Display name.
        name_key: Normalised name slug (unique per script).
        variants: Spellings of the name found in speaker cues.
        extensions: Cue extensions used with the name (``V.O.`` ...).
        dialogue_count: Number of dialogue blocks spoken.
        word_count: Total words spoken.
        first_line: Line number of the first appearance.
        first_dialogue_id: ``_id`` of the first dialogue block, or None.
        scene_ids: Scenes the character speaks in.
        dialogue_ids: Dialogue blocks spoken by the character, in script order.
        script_version_hash: ``content_hash`` of the script version the record was derived from.
        parser_version: Version of the parser that produced the record.
        created_at: When the record was written (UTC).
    """
    id: str = Field(alias="_id")
    script_id: str
    movie_id: str
    name: str
    name_key: str
    variants: list[str]
    extensions: list[str]
    dialogue_count: int
    word_count: int
    first_line: int
    first_dialogue_id: str | None
    scene_ids: list[str]
    dialogue_ids: list[str]
    script_version_hash: str
    parser_version: str
    created_at: datetime


# ---------------------------------------------------------------- crawl state
QueueStatus = Literal["pending", "in_progress", "done", "failed", "blocked", "skipped"]


class QueueItem(MongoModel):
    """A URL in the crawl queue (``crawl_queue``), keyed by canonical URL.

    Attributes:
        id: Queue item ``_id`` (``u_<hash32(url)>``).
        url: Canonical URL to fetch (unique).
        page_type: Kind of page, which decides the handler (e.g. ``movie_detail``, ``script_html``).
        status: ``pending`` (waits until ``next_attempt_at``), ``in_progress`` (leased to a worker), ``done``,
            ``failed`` (gave up), ``blocked`` (access denied, not retried) or ``skipped``.
        priority: Higher values are claimed first.
        depth: Link distance from the start pages.
        position: Order on the page that discovered it (deterministic claim order).
        discovered_from: URL of the page that linked to this one, or None.
        discovered_at: When the URL was first queued (UTC).
        movie_id: Movie the page belongs to, when known.
        attempts: Attempts since the last success or requeue.
        next_attempt_at: Earliest time the item may be claimed (UTC).
        last_attempt_at: When the item was last claimed (UTC), or None.
        completed_at: When the item reached ``done``, ``failed``, ``blocked`` or ``skipped`` (UTC), or None.
        lease_owner: Worker id holding the lease, or None.
        lease_expires_at: When the lease expires (UTC), or None.
        job_id: Job that last claimed the item, or None.
        last_error: Last error message (or skip reason), or None.
        last_status_code: HTTP status of the last attempt, or None.
        content_hash: SHA-256 of the last fetched body, or None.
        etag: ``ETag`` header of the last response, or None.
        last_modified: ``Last-Modified`` header of the last response, or None.
        fetch_count: Number of successful fetches.
        change_count: Number of successful fetches whose content had changed.
    """
    id: str = Field(alias="_id")
    url: str
    page_type: str
    status: QueueStatus = "pending"
    priority: int = 0
    depth: int = 0
    position: int = 0  # order on the page that discovered it (deterministic claim order)
    discovered_from: str | None = None
    discovered_at: datetime
    movie_id: str | None = None
    attempts: int = 0
    next_attempt_at: datetime
    last_attempt_at: datetime | None = None
    completed_at: datetime | None = None
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    job_id: str | None = None
    last_error: str | None = None
    last_status_code: int | None = None
    content_hash: str | None = None
    etag: str | None = None
    last_modified: str | None = None
    fetch_count: int = 0
    change_count: int = 0


class CrawlLogEvent(MongoModel):
    """A structured log event stored in ``crawl_logs``.

    Attributes:
        ts: Event time (UTC).
        level: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR`` or ``CRITICAL``.
        event: Short machine-readable event name.
        job_id: Job the event belongs to, or None.
        url: URL concerned, or None.
        page_type: Page type concerned, or None.
        record_id: Id of the affected record (movie, script...), or None.
        message: Human-readable message, or None.
        details: Extra structured data.
    """
    ts: datetime
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    event: str
    job_id: str | None = None
    url: str | None = None
    page_type: str | None = None
    record_id: str | None = None
    message: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class CrawlJob(MongoModel):
    """One run of a CLI job (discover, crawl, reparse...), stored in ``crawl_jobs``.

    Attributes:
        id: Job ``_id`` (``job-<UTC timestamp>-<6 hex>``).
        command: CLI command that started the job.
        params: Command parameters, kept so the job can be resumed.
        status: ``running``, ``interrupted`` (stopped early or crashed, resumable), ``completed``, ``failed``
            or ``stopped`` (ended on purpose, or superseded by a resuming job).
        started_at: Start time (UTC).
        heartbeat_at: Last sign of life (UTC); a stale heartbeat marks a crashed process.
        finished_at: End time (UTC), or None while running.
        host: Host name of the machine running the job.
        pid: Process id of the job.
        counters: Progress counters (fetched, parsed, failed...).
        error: Error message, or None.
        resumed_from: Id of the interrupted job this one continues, or None.
    """
    id: str = Field(alias="_id")
    command: str
    params: dict[str, Any]
    status: Literal["running", "interrupted", "completed", "failed", "stopped"]
    started_at: datetime
    heartbeat_at: datetime
    finished_at: datetime | None = None
    host: str
    pid: int
    counters: dict[str, int] = Field(default_factory=dict)
    error: str | None = None
    resumed_from: str | None = None


class RawArtifactDoc(MongoModel):
    """Index entry for a raw HTTP response stored on disk, kept in ``raw_artifacts`` (one per URL and body hash).

    Attributes:
        id: Artifact ``_id`` (``raw-<hash24>``, see ``RawArtifactRepository.artifact_id``).
        url: Requested URL.
        final_url: URL after redirects.
        page_type: Kind of page fetched.
        sha256: SHA-256 hex digest of the uncompressed body.
        size: Uncompressed body size in bytes.
        path: Location of the stored file, relative to the raw archive folder.
        compressed: True when the file is gzip-compressed.
        status_code: HTTP status code.
        headers: Response headers.
        http_version: HTTP version used, e.g. ``HTTP/1.1``.
        encoding: Character encoding reported for the response, or None.
        first_fetched_at: When this exact body was first fetched (UTC).
        last_fetched_at: When this exact body was last fetched (UTC).
        fetch_count: Number of fetches that returned this body.
    """
    id: str = Field(alias="_id")
    url: str
    final_url: str
    page_type: str
    sha256: str
    size: int
    path: str
    compressed: bool
    status_code: int
    headers: dict[str, str]
    http_version: str
    encoding: str | None = None
    first_fetched_at: datetime
    last_fetched_at: datetime
    fetch_count: int = 1


# ---------------------------------------------------------------- permissions
PermissionLevel = Literal["denied", "discovery_only", "sample", "bulk"]


class SourcePermissionDoc(MongoModel):
    """Collection permission record of a source site, stored in ``source_permissions``.

    Unknown fields are preserved (``extra="allow"``), e.g. ``inspections`` and ``inspected_at``.

    Attributes:
        id: Source id, e.g. ``imsdb``.
        source_name: Human-readable source name.
        base_url: Base URL of the source.
        level: ``denied`` (nothing), ``discovery_only`` (listing pages only), ``sample`` (up to ``sample_limit``
            records) or ``bulk`` (everything; requires recorded evidence).
        sample_limit: Total records allowed at ``sample`` level.
        version: Incremented on every level change.
        robots: Last parsed robots.txt summary.
        policy_documents: Last fetched policy pages (URL, hash, status).
        assessment: Result of the policy text assessment.
        usage_restrictions: Restrictions shown to the user.
        attribution: Attribution texts.
        rights: Rights information applied to collected records.
        evidence: Recorded authorisation statements.
        history: Snapshots of the previous state before each level change.
        excluded_page_types: Page types that must never be fetched.
        created_at: When the record was created (UTC).
        updated_at: When the record was last changed (UTC).
    """
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    id: str = Field(alias="_id")
    source_name: str
    base_url: str
    level: PermissionLevel
    sample_limit: int = Field(ge=0)
    version: int = Field(ge=1)
    robots: dict[str, Any] = Field(default_factory=dict)
    policy_documents: list[dict[str, Any]] = Field(default_factory=list)
    assessment: dict[str, Any] = Field(default_factory=dict)
    usage_restrictions: list[str] = Field(default_factory=list)
    attribution: dict[str, str] = Field(default_factory=dict)
    rights: dict[str, Any] = Field(default_factory=dict)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    history: list[dict[str, Any]] = Field(default_factory=list)
    excluded_page_types: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
