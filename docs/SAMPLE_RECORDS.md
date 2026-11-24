# Sample database records (live sample run)

Generated 2026-11-17T04:10:54-08:00 from the local `imsdb_local` database by `scripts/generate_sample_records.py`.

> Screenplay text is **truncated** here on purpose: the scripts are copyrighted by their owners. Full records exist only in the local database and raw archive.

## Live run

* Permission level: `sample` (limit 10); robots.txt HTTP 404
* Discovered movie records: **1295**; processed (detail page): **3**; scripts extracted: **3**
* Scenes 452, dialogue blocks 2342, characters 120
* Raw archive: 60 files, 2,229,289 bytes
* Deep validation: **PASSED** (0 error checks, 2 warning checks)

| Movie | Status | Script | Scenes | Dialogue blocks | Characters | Uncertain | Flags |
|---|---|---|---:|---:|---:|---:|---|
| 10 Things I Hate About You | script_collected | parsed_with_warnings | 88 | 994 | 34 | 0.0222 | empty_dialogue_blocks |
| 12 | script_collected | parsed_with_warnings | 199 | 526 | 54 | 0.0235 | empty_dialogue_blocks |
| 12 and Holding | script_collected | parsed | 162 | 822 | 32 | 0.0067 | - |

Validation findings (kept for inspection, nothing was discarded):

* `duplicate_movies` (warning): 1 - movies sharing a normalized title+year under different URLs (kept separate for review)
* `empty_dialogue_blocks` (warning): 4 - dialogue blocks with a character cue but no spoken text

## One document per collection

### movies

```json
{
  "_id": "mv-12-2326067d",
  "detail_url": "https://imsdb.com/Movie%20Scripts/12%20Script.html",
  "created_at": "2026-11-17T11:59:25.626000+00:00",
  "display_title": "12",
  "first_discovered_at": "2026-11-17T11:59:25.626000+00:00",
  "first_listing_url": "https://imsdb.com/all-scripts.html",
  "last_seen_at": "2026-11-17T12:04:10.641000+00:00",
  "listing_draft": {
    "text": "Undated Draft",
    "date": null,
    "label": "Draft"
  },
  "listing_sources": [
    "all-scripts",
    "alphabetical:0",
    "genre:Comedy"
  ],
  "listing_title": "12",
  "listing_writers": [
    "Lawrence Bridges"
  ],
  "source": "imsdb",
  "status": "script_collected",
  "title": "12",
  "title_key": "12",
  "updated_at": "2026-11-17T12:10:44.461000+00:00",
  "listing_genres": [
    "Comedy"
  ],
  "description": null,
  "detail_checked_at": "2026-11-17T11:59:38.993000+00:00",
  "detail_fetched_at": "2026-11-17T11:59:38.993000+00:00",
  "detail_parser_version": "1.0.0",
  "detail_raw": {
    "artifact_id": "raw-f72945c444da040a0e3ec1a0",
    "sha256": "3c50d63228a4573015a8f50fe9d2f524500f3cdf9bed8bff50a564e61949bab4",
    "path": "imsdb.com/movie_detail/3c/3c50d63228a4573015a8f50fe9d2f524500f3cdf9bed8bff50a564e61949bab4.html",
    "size": 11341,
    "fetched_at": "2026-11-17T11:59:38.962000+00:00",
    "status_code": 200
  },
  "detail_warnings": [],
  "directors": [],
  "extra_fields": {
    "detail_page_title": "12"
  },
  "genres": [
    "Comedy"
  ],
  "imdb_ids": [],
  "movie_release_date": null,
  "page_meta": {
    "title": "12 Script at IMSDb.",
    "description": "12 script at the Internet Movie Script Database.",
    "keywords": "12 script, 12 movie script, 12 film script"
  },
  "poster_url": "https://imsdb.com/images/no-poster.gif",
  "ratings": {
    "imsdb_opinion": null,
    "imsdb_rating": null,
    "imsdb_rating_text": "Not available",
    "average_user_rating": null,
    "average_user_rating_text": "None available IMSDb Box"
  },
  "release_year": null,
  "rights": {
    "source": "imsdb",
    "copyright_holder": "Respective copyright holders (studios and writers); not IMSDb",
    "attribution": "Screenplay text retrieved from The Internet Movie Script Database (IMSDb), https://imsdb.com. All rights belong to the respective copyright holders (studios and... [truncated, 170 chars]",
    "usage_restriction": "Educational, scholarship, research and criticism use only (per the source's fair-use disclaimer). Not for sale or redistribution. Local storage only.",
    "source_basis": "Source hosts scripts under a fair-use claim (17 U.S.C. 107); no licence is granted by IMSDb.",
    "permission_level_at_collection": "sample",
    "permission_version": 1
  },
  "script_availability": "html",
  "script_date": null,
  "script_link_text": "Read \"12\" Script",
  "script_url": "https://imsdb.com/scripts/12.html",
  "user_comment_count": 0,
  "writers": [
    {
      "name": "Lawrence Bridges",
      "url": "https://imsdb.com/writer.php?w=Lawrence%20Bridges"
    }
  ],
  "script_content_hash": "1766e71df48a7a33a7a3a2c85ed22fea6743e8539ac43ff24336d1e88bb66c3f",
  "script_id": "sc-12-682c87ef",
  "script_status": "parsed_with_warnings",
  "source_hrefs": [
    "/Movie Scripts/12 Script.html"
  ]
}
```

### scripts (elements/spans abbreviated)

```json
{
  "_id": "sc-12-682c87ef",
  "movie_id": "mv-12-2326067d",
  "source": "imsdb",
  "title": "12",
  "script_url": "https://imsdb.com/scripts/12.html",
  "status": "parsed_with_warnings",
  "version": 1,
  "content_hash": "1766e71df48a7a33a7a3a2c85ed22fea6743e8539ac43ff24336d1e88bb66c3f",
  "raw": {
    "artifact_id": "raw-c9e92a03369b56661fe7658e",
    "sha256": "45e63ffb20ff91998ba1dcf0ac2baf4798aa1b1eed6530049e973c74350ca106",
    "path": "imsdb.com/script_html/45/45e63ffb20ff91998ba1dcf0ac2baf4798aa1b1eed6530049e973c74350ca106.html",
    "size": 211594,
    "fetched_at": "2026-11-17T11:59:43.899000+00:00",
    "status_code": 200
  },
  "encoding": {
    "used": "utf-8",
    "strict": true,
    "declared_http": "UTF-8",
    "declared_meta": "iso-8859-1"
  },
  "text": "\n\n  \n  12 - Script\n\n\n\n\nCUT FROM BLACK\n\nTITLE: FIN\n\nEXTERIOR - LA - DAY\nFin of red 1957 Chevy Impala convertible driving \nsomewhere in the West. A car passes goi... [truncated, 81,729 chars]",
  "text_length": 81729,
  "line_count": 3882,
  "source_markup": "<pre><b>\r\n</b><b>\r\n</b>  <meta content=\"Adobe PageMill 2.0 Mac\" name=\"GENERATOR\"/>\r\n  <title>12 - Script</title>\r\n<b>\r\n</b>\r\n\r\n<b><pre><b><tt><font size=\"-1\"></... [truncated, 211,154 chars]",
  "formatting_spans": [
    {
      "style": "bold",
      "start": 0,
      "end": 2
    },
    {
      "style": "bold",
      "start": 19,
      "end": 20
    },
    {
      "style": "bold",
      "start": 22,
      "end": 71
    }
  ],
  "extraction": {
    "method": "pre",
    "normalizations": [
      "HTML entities decoded",
      "markup tags removed (formatting kept in formatting_spans/markup)",
      "CRLF/CR line endings converted to LF",
      "... (1 more)"
    ],
    "visible_page_chars": 62983,
    "extraction_ratio": 0.9925,
    "removed_tags": {},
    "page_title": "12 - Script"
  },
  "layout": {
    "mode": "indented",
    "action_indent": 0,
    "dialogue_indent": 8,
    "cue_indent": 24,
    "bold_ratio": 0.2941,
    "bold_meaningful": true
  },
  "structure_stats": {
    "lines": 3882,
    "nonblank_lines": 2645,
    "characters_text": 81729,
    "words": 13916,
    "elements": 1789,
    "element_counts": {
      "action": 478,
      "character": 526,
      "dialogue": 526,
      "parenthetical": 6,
      "scene_heading": 199,
      "shot": 50,
      "title_page": 3,
      "transition": 1
    },
    "scenes": 199,
    "has_prelude": true,
    "dialogue_blocks": 526,
    "characters": 54,
    "parentheticals": 6,
    "transitions": 1,
    "act_markers": 0,
    "uncertain_elements": 42,
    "uncertain_ratio": 0.0235,
    "page_markers": 0,
    "last_page": null,
    "empty_dialogue_blocks": 2,
    "encoding_problems": {
      "replacement_chars": 0,
      "mojibake_sequences": 0,
      "control_chars": 0
    }
  },
  "elements": [
    {
      "index": 0,
      "type": "title_page",
      "line_start": 3,
      "line_end": 3,
      "char_start": 7,
      "char_end": 18,
      "text": "12 - Script",
      "content": "12 - Script",
      "confidence": 0.6,
      "basis": [
        "action-indent",
        "front-matter"
      ],
      "bold": false,
      "indent": 2,
      "in_dialogue": false,
      "cue_text": null,
      "scene_index": 0,
      "notes": [],
      "uncertain": true
    },
    {
      "index": 1,
      "type": "title_page",
      "line_start": 8,
      "line_end": 8,
      "char_start": 23,
      "char_end": 37,
      "text": "CUT FROM BLACK",
      "content": "CUT FROM BLACK",
      "confidence": 0.6,
      "basis": [
        "bold",
        "bold-subheading",
        "front-matter"
      ],
      "bold": true,
      "indent": 0,
      "in_dialogue": false,
      "cue_text": null,
      "scene_index": 0,
      "notes": [
        "upper-case bold line treated as secondary heading"
      ],
      "uncertain": true
    },
    {
      "index": 2,
      "type": "title_page",
      "line_start": 10,
      "line_end": 10,
      "char_start": 39,
      "char_end": 49,
      "text": "TITLE: FIN",
      "content": "TITLE: FIN",
      "confidence": 0.6,
      "basis": [
        "bold",
        "front-matter",
        "shot-pattern"
      ],
      "bold": true,
      "indent": 0,
      "in_dialogue": false,
      "cue_text": null,
      "scene_index": 0,
      "notes": [],
      "uncertain": true
    },
    "... (1 more)"
  ],
  "elements_file": null,
  "footer_metadata": {
    "title": "12",
    "writers": [
      "Lawrence Bridges"
    ],
    "genres": [
      "Comedy"
    ]
  },
  "parse_warnings": [],
  "quality_flags": [
    "empty_dialogue_blocks"
  ],
  "parser_version": "1.2.0",
  "rights": {
    "source": "imsdb",
    "copyright_holder": "Respective copyright holders (studios and writers); not IMSDb",
    "attribution": "Screenplay text retrieved from The Internet Movie Script Database (IMSDb), https://imsdb.com. All rights belong to the respective copyright holders (studios and... [truncated, 170 chars]",
    "usage_restriction": "Educational, scholarship, research and criticism use only (per the source's fair-use disclaimer). Not for sale or redistribution. Local storage only.",
    "source_basis": "Source hosts scripts under a fair-use claim (17 U.S.C. 107); no licence is granted by IMSDb.",
    "permission_level_at_collection": "sample",
    "permission_version": 1
  },
  "fetched_at": "2026-11-17T11:59:43.899000+00:00",
  "parsed_at": "2026-11-17T12:10:44.461000+00:00",
  "last_checked_at": "2026-11-17T12:10:44.461000+00:00",
  "created_at": "2026-11-17T11:59:43.912000+00:00",
  "updated_at": "2026-11-17T12:10:44.461000+00:00",
  "previous_versions": 0,
  "error": null,
  "parse_id": "0786e9c24b3b4743b437f50c42aa4768"
}
```

### scenes

```json
{
  "_id": "sc-12-682c87ef:s0001",
  "script_id": "sc-12-682c87ef",
  "movie_id": "mv-12-2326067d",
  "scene_number": 1,
  "kind": "scene",
  "heading": "EXTERIOR - LA - DAY",
  "heading_parts": {
    "raw": "EXTERIOR - LA - DAY",
    "scene_number": null,
    "int_ext": "EXT",
    "location": "LA",
    "time_of_day": "DAY",
    "extra": []
  },
  "act": null,
  "page_start": null,
  "page_end": null,
  "description": "Fin of red 1957 Chevy Impala convertible driving somewhere in the West. A car passes going the other way.\n\nTITLE: PLACE: Los Angeles\n\nMUSIC: Shadowy Men On A Sh... [truncated, 193 chars]",
  "text": "EXTERIOR - LA - DAY\nFin of red 1957 Chevy Impala convertible driving \nsomewhere in the West. A car passes going the other \nway.\n\nTITLE: PLACE: Los Angeles\n\nMUSI... [truncated, 215 chars]",
  "char_start": 51,
  "char_end": 266,
  "line_start": 12,
  "line_end": 20,
  "element_start": 3,
  "element_end": 6,
  "characters": [],
  "character_ids": [],
  "dialogue_count": 0,
  "shots": [],
  "transitions": [],
  "element_counts": {
    "scene_heading": 1,
    "action": 3
  },
  "confidence": 0.99,
  "uncertain": false,
  "warnings": [],
  "content_hash": "feccd04e00c7493576fc86afc935840539be964ed389914f11185aa651423acb",
  "script_version_hash": "1766e71df48a7a33a7a3a2c85ed22fea6743e8539ac43ff24336d1e88bb66c3f",
  "parser_version": "1.2.0",
  "created_at": "2026-11-17T12:10:44.461000+00:00",
  "parse_id": "0786e9c24b3b4743b437f50c42aa4768"
}
```

### dialogues

```json
{
  "_id": "sc-12-682c87ef:d00007",
  "script_id": "sc-12-682c87ef",
  "movie_id": "mv-12-2326067d",
  "scene_id": "sc-12-682c87ef:s0021",
  "character_id": "sc-12-682c87ef:c:talk-show-host",
  "order": 7,
  "order_in_scene": 4,
  "speaker": "TALK SHOW HOST",
  "speaker_raw": "TALK SHOW HOST",
  "extensions": [],
  "dual": false,
  "text": "I'm on hold..",
  "original_text": "TALK SHOW HOST\n\t(aside) I'm on hold..",
  "parentheticals": [
    "(aside)"
  ],
  "segments": [
    {
      "type": "parenthetical",
      "text": "(aside)",
      "element_index": 67
    },
    {
      "type": "dialogue",
      "text": "I'm on hold..",
      "element_index": 68
    }
  ],
  "char_start": 2764,
  "char_end": 2801,
  "line_start": 140,
  "line_end": 141,
  "element_start": 66,
  "element_end": 68,
  "word_count": 3,
  "confidence": 0.92,
  "uncertain": false,
  "warnings": [],
  "script_version_hash": "1766e71df48a7a33a7a3a2c85ed22fea6743e8539ac43ff24336d1e88bb66c3f",
  "parser_version": "1.2.0",
  "created_at": "2026-11-17T12:10:44.461000+00:00",
  "parse_id": "0786e9c24b3b4743b437f50c42aa4768"
}
```

### characters

```json
{
  "_id": "sc-12-682c87ef:c:allen",
  "script_id": "sc-12-682c87ef",
  "movie_id": "mv-12-2326067d",
  "name": "ALLEN",
  "name_key": "allen",
  "variants": [
    "ALLEN",
    "ALLEN (aside to H. TOM)",
    "ALLEN (V.O.)"
  ],
  "extensions": [
    "aside to H. TOM",
    "V.O."
  ],
  "dialogue_count": 129,
  "word_count": 1269,
  "first_line": 127,
  "first_dialogue_id": "sc-12-682c87ef:d00004",
  "scene_ids": [
    "sc-12-682c87ef:s0021",
    "sc-12-682c87ef:s0022",
    "sc-12-682c87ef:s0023",
    "... (39 more)"
  ],
  "dialogue_ids": [
    "sc-12-682c87ef:d00004",
    "sc-12-682c87ef:d00006",
    "sc-12-682c87ef:d00009",
    "... (126 more)"
  ],
  "script_version_hash": "1766e71df48a7a33a7a3a2c85ed22fea6743e8539ac43ff24336d1e88bb66c3f",
  "parser_version": "1.2.0",
  "created_at": "2026-11-17T12:10:44.461000+00:00",
  "parse_id": "0786e9c24b3b4743b437f50c42aa4768"
}
```

### raw_artifacts

```json
{
  "_id": "raw-c9e92a03369b56661fe7658e",
  "url": "https://imsdb.com/scripts/12.html",
  "sha256": "45e63ffb20ff91998ba1dcf0ac2baf4798aa1b1eed6530049e973c74350ca106",
  "compressed": false,
  "encoding": "utf-8",
  "fetch_count": 1,
  "final_url": "https://imsdb.com/scripts/12.html",
  "first_fetched_at": "2026-11-17T11:59:43.899000+00:00",
  "headers": {
    "content-type": "text/html; charset=UTF-8",
    "content-encoding": "gzip",
    "date": "Tue, 17 Nov 2026 11:59:44 GMT",
    "server": "Apache",
    "vary": "Accept-Encoding",
    "via": "1.1 google"
  },
  "http_version": "HTTP/1.1",
  "last_fetched_at": "2026-11-17T11:59:43.899000+00:00",
  "page_type": "script_html",
  "path": "imsdb.com/script_html/45/45e63ffb20ff91998ba1dcf0ac2baf4798aa1b1eed6530049e973c74350ca106.html",
  "size": 211594,
  "status_code": 200
}
```

### crawl_queue

```json
{
  "_id": "u_682c87efc9a3fc2a672e9b96db413226",
  "url": "https://imsdb.com/scripts/12.html",
  "attempts": 0,
  "change_count": 1,
  "completed_at": "2026-11-17T11:59:44.398000+00:00",
  "content_hash": "45e63ffb20ff91998ba1dcf0ac2baf4798aa1b1eed6530049e973c74350ca106",
  "depth": 3,
  "discovered_at": "2026-11-17T11:59:38.995000+00:00",
  "discovered_from": "https://imsdb.com/Movie%20Scripts/12%20Script.html",
  "etag": null,
  "fetch_count": 1,
  "job_id": "job-20261117T115938-0280a0",
  "last_attempt_at": "2026-11-17T11:59:38.996000+00:00",
  "last_error": null,
  "last_modified": null,
  "last_status_code": 200,
  "lease_expires_at": null,
  "lease_owner": null,
  "movie_id": "mv-12-2326067d",
  "next_attempt_at": "2026-11-17T11:59:38.995000+00:00",
  "page_type": "script_html",
  "position": 0,
  "priority": 100,
  "status": "done"
}
```

### crawl_logs

```json
{
  "_id": "6afc444431b7ed119b742b72",
  "ts": "2026-11-17T12:10:44.945000+00:00",
  "level": "INFO",
  "event": "script_parsed",
  "job_id": null,
  "url": "https://imsdb.com/scripts/12.html",
  "page_type": "script_html",
  "record_id": "sc-12-682c87ef",
  "message": null,
  "details": {
    "action": "reparsed",
    "version": 1,
    "status": "parsed_with_warnings",
    "stats": {
      "scenes": 199,
      "dialogue_blocks": 526,
      "characters": 54,
      "uncertain_ratio": 0.0235
    },
    "quality_flags": [
      "empty_dialogue_blocks"
    ],
    "scenes_written": 200,
    "dialogues_written": 526,
    "characters_written": 54,
    "scenes_removed_stale": 0,
    "dialogues_removed_stale": 0,
    "characters_removed_stale": 0
  }
}
```

### crawl_jobs

```json
{
  "_id": "job-20261117T115938-0280a0",
  "command": "crawl",
  "params": {
    "limit": 3,
    "all": false,
    "refresh_days": null,
    "retry_failed": false,
    "effective_limit": 3
  },
  "status": "completed",
  "started_at": "2026-11-17T11:59:38.437000+00:00",
  "heartbeat_at": "2026-11-17T12:00:00.507000+00:00",
  "finished_at": "2026-11-17T12:00:00.507000+00:00",
  "host": "SERVER",
  "pid": 35576,
  "counters": {
    "movie_detail_done": 3,
    "records_processed": 3,
    "records_new": 3,
    "records_refreshed": 0,
    "script_urls_enqueued": 3,
    "script_html_done": 3,
    "scripts_created": 3
  },
  "error": null,
  "resumed_from": null
}
```

### source_permissions

```json
{
  "_id": "imsdb",
  "assessment": {
    "summary": "No robots.txt (HTTP 404) and no terms of service found. The disclaimer claims fair use for educational purposes. IMSDb is not the copyright holder and grants no... [truncated, 216 chars]",
    "bulk_authorized": false,
    "restriction_found": false,
    "assessed_by": "inspect-source (automated heuristics)",
    "fair_use_claim_found": true,
    "restrictive_phrases": [],
    "permissive_phrases": [],
    "explicit_grant_found": false,
    "recommendation": "No explicit grant found: keep the 'sample' level unless you obtain written authorization.",
    "inspected_at": "2026-11-17T11:52:37.568000+00:00",
    "robots_status": 404,
    "robots_allows_all": true,
    "robots_disallows_all": false
  },
  "attribution": {
    "text": "Screenplay text retrieved from The Internet Movie Script Database (IMSDb), https://imsdb.com. All rights belong to the respective copyright holders (studios and... [truncated, 170 chars]",
    "source_url": "https://imsdb.com/"
  },
  "base_url": "https://imsdb.com/",
  "created_at": "2026-11-17T11:34:58.124000+00:00",
  "evidence": [
    {
      "recorded_at": "2026-11-17T11:34:58.124000+00:00",
      "recorded_by": "system",
      "statement": "Default sample-level permission pending operator authorization.",
      "level": "sample"
    }
  ],
  "excluded_page_types": [
    "writer",
    "tv",
    "form",
    "... (4 more)"
  ],
  "history": [],
  "level": "sample",
  "rights": {
    "copyright_holder": "Respective studios and writers",
    "source_basis": "Source hosts scripts under a fair-use claim (17 U.S.C. 107); no licence is granted by IMSDb."
  },
  "robots": {
    "status": 404,
    "allow_all": true,
    "disallow_all": false,
    "crawl_delay": null,
    "body": "",
    "fetched_at": "2026-11-17T11:52:12.557000+00:00",
    "sha256": null
  },
  "sample_limit": 10,
  "source_name": "The Internet Movie Script Database (IMSDb)",
  "updated_at": "2026-11-17T11:52:37.568000+00:00",
  "usage_restrictions": [
    "Educational, scholarship, research and criticism use only (per the source's fair-use disclaimer). Not for sale or redistribution. Local storage only.",
    "Do not crawl forms, comment endpoints, writer pages or TV transcripts.",
    "Do not bypass authentication, rate limits or anti-bot protections.",
    "... (1 more)"
  ],
  "version": 1,
  "inspected_at": "2026-11-17T11:52:37.568000+00:00",
  "inspections": [
    {
      "at": "2026-11-17T11:50:12.696000+00:00",
      "robots_status": 404,
      "robots_sha256": null,
      "documents": [
        {
          "url": "https://imsdb.com/disclaimer.html",
          "sha256": "e9fe408300030ed87c0dbc8cd050ef7c122e0e2e5b6b0930f5871889df23acf0",
          "status": 200
        },
        {
          "url": "https://imsdb.com/contact.html",
          "sha256": "8555dbed17c2401a918662db121815af16726301ca07cc3a9b2184bab615804f",
          "status": 200
        },
        {
          "url": "https://imsdb.com/privacy",
          "sha256": null,
          "status": 301
        },
        "... (1 more)"
      ]
    },
    {
      "at": "2026-11-17T11:52:37.568000+00:00",
      "robots_status": 404,
      "robots_sha256": null,
      "documents": [
        {
          "url": "https://imsdb.com/disclaimer.html",
          "sha256": "e9fe408300030ed87c0dbc8cd050ef7c122e0e2e5b6b0930f5871889df23acf0",
          "status": 200
        },
        {
          "url": "https://imsdb.com/contact.html",
          "sha256": "8555dbed17c2401a918662db121815af16726301ca07cc3a9b2184bab615804f",
          "status": 200
        },
        {
          "url": "https://imsdb.com/privacy",
          "sha256": "4e79029bd9809479b112c37c61fe835156e6a41c73f94f326b2f67e39b0c8d1f",
          "status": 200
        },
        "... (1 more)"
      ]
    }
  ],
  "policy_documents": [
    {
      "url": "https://imsdb.com/disclaimer.html",
      "final_url": "https://imsdb.com/disclaimer.html",
      "status_code": 200,
      "sha256": "e9fe408300030ed87c0dbc8cd050ef7c122e0e2e5b6b0930f5871889df23acf0",
      "raw_path": "imsdb.com/policy/e9/e9fe408300030ed87c0dbc8cd050ef7c122e0e2e5b6b0930f5871889df23acf0.html"
    },
    {
      "url": "https://imsdb.com/contact.html",
      "final_url": "https://imsdb.com/contact.html",
      "status_code": 200,
      "sha256": "8555dbed17c2401a918662db121815af16726301ca07cc3a9b2184bab615804f",
      "raw_path": "imsdb.com/policy/85/8555dbed17c2401a918662db121815af16726301ca07cc3a9b2184bab615804f.html"
    },
    {
      "url": "https://imsdb.com/privacy",
      "final_url": "https://imsdb.com/privacy/",
      "status_code": 200,
      "sha256": "4e79029bd9809479b112c37c61fe835156e6a41c73f94f326b2f67e39b0c8d1f",
      "raw_path": "imsdb.com/policy/4e/4e79029bd9809479b112c37c61fe835156e6a41c73f94f326b2f67e39b0c8d1f.html"
    },
    "... (1 more)"
  ]
}
```

