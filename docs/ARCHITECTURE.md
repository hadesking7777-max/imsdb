# Architecture and Data Model

## 1. Pipeline

```
                       ┌─────────────── permission gate (level + robots.txt + exclusions) ───────────────┐
                       │                                                                                  │
 seeds ──► crawl_queue ──► Fetcher (rate limit, retries, Retry-After, size cap, same-host redirects)      │
   ▲          │  leased claims                  │                                                         │
   │          │                                 ▼                                                         │
   │          │                     RawStore  data/raw/<host>/<type>/<sha[:2]>/<sha>.html  (+ JSONL manifest)
   │          │                     raw_artifacts (url, sha256, path, headers, status, timestamps)
   │          ▼                                 │
   │   listing page ──► parse_listing_page ──► movies (discovered) + queue(movie_detail)
   │   movie detail ──► parse_movie_detail ──► movies (detail fields, rights) + queue(script_html)
   └── script page  ──► extract_screenplay ──► parse_screenplay ──► scripts (+ script_versions)
                                                                  ──► scenes / dialogues / characters
```

* **Queue-driven and finite.** Only allow-listed page types are enqueued: home → listing pages
  (all-scripts, alphabetical, genre, language) → movie detail → screenplay HTML. Listing follow-ups are
  depth-capped. Writer pages, TV transcripts, forms, PDFs and external hosts are never enqueued.
* **Leases.** Workers claim items atomically (`find_one_and_update`) with a lease. A crash leaves
  `in_progress` items with expired leases, and `resume` (or the next run) returns them to `pending`.
* **Jobs and the lock.** Each run is a `crawl_jobs` document with parameters, counters and a heartbeat.
  A lock document prevents two crawler processes from running at the same time. `resume` continues the
  latest interrupted or stale job with its original parameters and the remaining record budget.
* **Graceful shutdown.** SIGINT, SIGTERM and SIGBREAK set a shared stop event. Every wait (rate limit, backoff,
  idle) is interruptible. Claimed-but-unprocessed items are released without counting an attempt. The job is
  marked `interrupted` (exit code 130).
* **Retries.** At the HTTP level: exponential backoff with jitter for timeouts, connection errors and
  408/425/429/5xx, honouring `Retry-After`. At the queue level: up to `IMSDB_MAX_QUEUE_ATTEMPTS`, with backoff
  rescheduling (`next_attempt_at`). 401/403/451 mark the item `blocked` and **stop the crawl**.
* **Incremental sync.** `discover --refresh` re-checks listing pages. `crawl --refresh-days N` re-fetches
  collected records older than N days. Unchanged content (same SHA-256) is a no-op. Changed content archives
  the previous version (`script_versions`, `movies.detail_versions`), and the raw store keeps every distinct body.
  IMSDb sends no ETag or Last-Modified headers, so change detection relies on content hashes. Conditional
  request support exists in the fetcher for sources that do send them.

## 2. Project structure

The requested layout is kept, with these additions:

| Addition | Why |
|---|---|
| `src/errors.py` | One place for exception types and CLI exit codes |
| `src/storage/raw_store.py` | Raw-data preservation is a separate concern from MongoDB (files plus checksums) |
| `src/services/documents.py` | Converts parser output into validated documents; keeps `ingestion.py` readable |
| `src/services/stats.py`, `export.py`, `backup.py` | One module per CLI capability |
| `src/utils/clock.py` | Timezone-aware, millisecond-truncated UTC times (BSON precision) |
| `scripts/local_mongodb.py` | Start, stop and status for portable (non-service) MongoDB installs |
| Collections `raw_artifacts`, `script_versions`, `crawl_jobs` | Raw-artifact index, preserved previous versions, resumable job state |

## 3. Collections (database `imsdb_local`)

All documents are validated by Pydantic models (`src/database/models.py`) before writing. Server-side
`$jsonSchema` validators (`validationLevel: moderate`, `validationAction: error`) check required fields.

| Collection | `_id` | Unique key(s) | Purpose |
|---|---|---|---|
| `movies` | `mv-<slug>-<hash8(detail_url)>` | `detail_url` | Title, display title, draft info, writers, directors (*null when unknown*), genres, dates, ratings, poster, IMDb ids (*only if linked*), extra fields, rights, raw reference, status |
| `scripts` | `sc-<slug>-<hash8(script_url)>` | `script_url` | Canonical extracted `text`, original `source_markup`, `formatting_spans` (bold/italic/underline offsets), `elements` (typed, with offsets and confidence), encoding info, extraction method and ratio, layout profile, stats, quality flags, warnings, version, parser version, rights |
| `script_versions` | `<script_id>:v<N>` | `(script_id, version)` | The full previous script document, archived before replacement |
| `scenes` | `<script_id>:s<NNNN>` | `(script_id, scene_number)` | Heading and its parsed parts (INT/EXT, location, time of day, scene number), act, pages, description, full scene `text`, char and line offsets, characters, shots, transitions, confidence |
| `dialogues` | `<script_id>:d<NNNNN>` | `(script_id, order)` | Speaker, raw cue, extensions (V.O., O.S., CONT'D …), spoken `text`, exact `original_text`, parentheticals, ordered segments, scene and character refs, offsets, confidence |
| `characters` | `<script_id>:c:<name-slug>` | `(script_id, name_key)` | Normalized name, cue variants, extensions, dialogue count, word count, scene ids, dialogue ids |
| `crawl_queue` | `u_<hash32(url)>` | `url` | Page type, status, priority, position, attempts, lease, `next_attempt_at`, last error/status, content hash, fetch and change counts |
| `crawl_logs` | ObjectId | – | Append-only events: fetch_ok, fetch_error, parse_warning, script_parsed, parse_error, job_* … |
| `crawl_jobs` | `job-<stamp>-<rand>` | – | Command, parameters, status, heartbeat, counters; also the crawler lock document |
| `source_permissions` | `imsdb` | – | Level, sample limit, robots.txt record, policy documents with hashes, assessment, restrictions, attribution, evidence and history (append-only) |
| `raw_artifacts` | `raw-<hash24(url,sha)>` | `(url, sha256)` | Path, size, SHA-256, HTTP status, selected headers, encoding, first and last fetch times, fetch count |

### Record statuses

| `movies.status` | Meaning |
|---|---|
| `discovered` | Found on a listing page; detail page not fetched yet |
| `detail_collected` | Detail page stored; screenplay not stored yet |
| `script_collected` | Screenplay stored and parsed (`scripts.status` is `parsed` or `parsed_with_warnings`) |
| `script_unavailable` | The source offers no HTML screenplay (PDF only, or no script link) |
| `script_empty_at_source` | The source's script page exists but its script container is empty (`scripts.status = empty_at_source`). Not a failure; the raw page is kept |
| `script_shared` | The detail page links to a screenplay already stored under another movie (e.g. a sequel's page linking to the original). `script_id` and `shared_script_owner_id` point to it; no duplicate is stored |
| `script_failed` | The page has no recognizable script container (`extraction_failed`) or parsing crashed (`parse_failed`). Raw HTML kept; try `reparse` |

### Relationships

```
movies._id ◄── scripts.movie_id ◄── scenes.script_id ◄── dialogues.scene_id
                         ▲                    ▲                   │
                         └── characters.script_id ◄───────────────┘ dialogues.character_id
scripts.raw.artifact_id ──► raw_artifacts._id ──► file in data/raw (sha256-verified)
```

### Re-run idempotency

* Natural keys are canonical URLs (normalized: https, no `www`, consistent percent-encoding, no fragment, no
  trailing slash). Derived records use deterministic ids built from the script id and the position.
* Each parse carries a `parse_id`. New derived records are upserted first, and leftovers from earlier parses
  are deleted only afterwards. The scripts collection never loses data: a changed source is archived first.
* MongoDB runs as a standalone server (no multi-document transactions), so every step is designed to be
  safely repeatable after a crash.

## 4. Screenplay parsing

1. **Extraction** (`extract_screenplay`). Locates `td.scrtext > pre` (IMSDb's layout). Falls back, in order,
   to several `<pre>` blocks, the container's HTML with `<br>` line breaks, the largest `<pre>`, and finally
   the page body. Each fallback is recorded in `extraction.method` and in `quality_flags`. The HTML walk is
   iterative, so deeply nested malformed markup cannot overflow the stack. Bold/italic/underline become
   character-offset spans. The footer table (writers/genres) is captured separately.
2. **Layout profiling.** Measures the indentation of action, dialogue and cue columns. The script is treated as
   *indented* (classic layout) or *flush* (no indentation), and bold usage is recorded.
3. **Line labelling.** A state machine labels each line as one of: scene heading, transition, act marker,
   character cue (including inline `NAME: speech`), dialogue, parenthetical (multi-line), shot or sub-heading,
   page marker, continuation (`(MORE)`, `CONTINUED`), or action. Each label carries a `confidence` and a `basis`.
4. **Elements.** Consecutive lines of one type are merged. Every element stores exact `char_start`/`char_end`
   into `scripts.text`, so `text[char_start:char_end]` reproduces the original wording and indentation.
5. **Front matter.** Content before the first heading, act marker or FADE IN becomes `title_page`.
6. **Scenes, dialogue blocks, characters.** Built from the elements. Cue extensions are parsed and names
   normalized.
7. **Uncertainty.** Elements below confidence 0.7 are `uncertain: true`. Script-level `quality_flags`
   (`no_scene_headings`, `high_uncertainty`, `low_extraction_ratio`, `encoding_suspect`,
   `possible_dual_dialogue`, …) feed validation. Nothing is silently dropped.

`PARSER_VERSION` (in `src/parsers/__init__.py`) is stored on every record. `python -m src.main reparse`
re-processes outdated records from the local raw archive, with no network access. `reparse --listings` does
the same for discovery: it re-parses archived listing pages and then runs URL canonicalization. Uncollected
records whose URL is no longer canonical (e.g. IMSDb's double-encoded `%2526` links) are merged into their
canonical twin. Their original hrefs are kept in `source_hrefs`/`url_aliases`, superseded queue items are marked
`skipped` with the reason, and every merge is written to `crawl_logs`. Collected records are flagged for review,
never merged.

## 5. Future RAG compatibility (Phase 2 hooks)

* Stable ids for movies, scripts, scenes, dialogues and characters, plus `content_hash` per scene and
  `script_version_hash` everywhere, so embeddings can be keyed and invalidated precisely.
* Scene `text` with offsets is a natural first chunk. Elements and dialogues provide finer chunks that keep
  speaker and scene context.
* Rights and attribution are stored on every script and movie, so a chatbot can cite its sources.
* No embeddings or AI calls are made in Phase 1.
