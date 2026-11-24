"""Build validated database documents from parser output."""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import bson

from src.crawler.url_manager import script_name_from_url
from src.database.models import (
    CharacterDoc,
    DialogueDoc,
    MovieDetailFields,
    Person,
    RawRef,
    Rights,
    SceneDoc,
    ScriptDoc,
    character_id_for,
    dialogue_id_for,
    scene_id_for,
    script_id_for,
)
from src.parsers import PARSER_VERSION
from src.parsers.movie_parser import MovieDetail
from src.parsers.screenplay_parser import ExtractedScript, ParsedScreenplay
from src.services.permissions import ATTRIBUTION, SOURCE_BASIS, USAGE_RESTRICTION
from src.utils.hashing import sha256_bytes, sha256_text
from src.utils.text_processing import DecodedText, display_title, title_key

MAX_DOC_BYTES = 14 * 1024 * 1024  # stay well below MongoDB's 16 MiB document limit
# Extraction methods that mean IMSDb's own script container (td.scrtext > pre) was found on the page.
SOURCE_CONTAINER_METHODS = frozenset({"pre", "multi_pre"})


def make_rights(level: str, permission_version: int) -> Rights:
    """Build the rights/provenance block stored on every collected movie and script.

    Args:
        level: Permission level in force when the record was collected.
        permission_version: Version of the ``source_permissions`` record at collection time.

    Returns:
        Rights with the IMSDb attribution, usage restriction and fair-use source basis.
    """
    return Rights(attribution=ATTRIBUTION, usage_restriction=USAGE_RESTRICTION, source_basis=SOURCE_BASIS,
                  permission_level_at_collection=level, permission_version=permission_version)


def script_id_from_url(script_url: str, fallback_title: str) -> str:
    """Derive the deterministic script id for a screenplay URL.

    Args:
        script_url: Normalized screenplay URL.
        fallback_title: Name used when the URL has no ``/scripts/<name>`` part.

    Returns:
        The id from ``script_id_for`` (URL plus the script name with hyphens turned into spaces).
    """
    name = script_name_from_url(script_url) or fallback_title
    return script_id_for(script_url, name.replace("-", " "))


def movie_detail_fields(detail: MovieDetail, *, listing_title: str | None, raw: RawRef,
                        rights: Rights) -> MovieDetailFields:
    """Map a parsed movie detail page to the fields stored on its ``movies`` document.

    Args:
        detail: Parser output for the detail page.
        listing_title: Title as shown on the listing page, preferred over the detail page's title; None if
            unknown.
        raw: Reference to the archived detail page.
        rights: Rights block for the record.

    Returns:
        The validated detail fields (titles and title key, script link, dates, genres, people, ratings,
        warnings, raw reference, parser version, rights).
    """
    title = listing_title or detail.title or ""
    return MovieDetailFields(
        title=title,
        display_title=display_title(title),
        title_key=title_key(title),
        script_url=detail.script_url,
        script_availability=detail.script_availability,  # type: ignore[arg-type]
        script_link_text=detail.script_link_text,
        release_year=detail.release_year,
        movie_release_date=detail.movie_release_date,
        script_date=detail.script_date,
        genres=detail.genres,
        writers=[Person(name=w.name, url=w.url) for w in detail.writers],
        directors=[Person(name=d.name, url=d.url) for d in detail.directors],
        description=detail.description,
        imdb_ids=detail.imdb_ids,
        ratings={
            "imsdb_opinion": detail.imsdb_opinion,
            "imsdb_rating": detail.imsdb_rating,
            "imsdb_rating_text": detail.imsdb_rating_text,
            "average_user_rating": detail.average_user_rating,
            "average_user_rating_text": detail.average_user_rating_text,
        },
        poster_url=detail.poster_url,
        user_comment_count=detail.user_comment_count,
        page_meta=detail.page_meta,
        extra_fields={**detail.extra_fields, "detail_page_title": detail.title},
        detail_warnings=detail.warnings,
        detail_raw=raw,
        detail_parser_version=PARSER_VERSION,
        rights=rights,
    )


def build_script_bundle(
    *,
    movie: dict[str, Any],
    script_url: str,
    raw: RawRef,
    decoded: DecodedText,
    extracted: ExtractedScript,
    parsed: ParsedScreenplay | None,
    rights: Rights,
    fetched_at: datetime,
    now: datetime,
    processed_dir: Path,
    error: str | None = None,
) -> tuple[ScriptDoc, list[SceneDoc], list[DialogueDoc], list[CharacterDoc]]:
    """Build the script document and its derived scene, dialogue and character documents.

    Status: ``parse_failed`` when ``error`` is set; ``empty_at_source`` when IMSDb's script container was
    found but empty; ``extraction_failed`` when no text was found; ``parsed_with_warnings`` when there are
    warnings or quality flags (including ``encoding_fallback`` for a non-strict decode); else ``parsed``.
    Oversized script documents are shrunk by ``_fit_document`` (may write a file under ``processed_dir``).
    Nothing is written to MongoDB.

    Args:
        movie: ``movies`` document the script belongs to (``_id`` and ``title`` are used).
        script_url: Normalized screenplay URL.
        raw: Reference to the archived screenplay page.
        decoded: Decoded page (encoding details and decoding warnings are stored).
        extracted: Extracted screenplay text, markup and formatting.
        parsed: Parsed structure, or None when parsing was skipped or failed.
        rights: Rights block for the record.
        fetched_at: When the raw page was fetched.
        now: Timestamp for ``parsed_at``/``created_at``/``updated_at`` and derived documents.
        processed_dir: Folder for offloaded large arrays.
        error: Parser error message, or None.

    Returns:
        ``(script, scenes, dialogues, characters)``; the three lists are empty when there is no parsed
        structure or no extracted text.

    Raises:
        ValueError: The script text alone exceeds the document size limit.
    """
    title = movie.get("title") or extracted.footer_metadata.get("title") or script_name_from_url(script_url) or "untitled"
    script_id = script_id_from_url(script_url, title)
    movie_id = movie["_id"]
    content_hash = sha256_text(extracted.text) if extracted.found else None

    warnings = list(decoded.warnings) + list(extracted.warnings)
    flags: list[str] = []
    if parsed is not None:
        warnings += parsed.warnings
        flags = list(parsed.quality_flags)
    if not decoded.strict:
        flags.append("encoding_fallback")

    if error:
        status = "parse_failed"
    elif not extracted.found and extracted.method in SOURCE_CONTAINER_METHODS:
        # The page has IMSDb's script container, but it holds no text: the source itself is empty.
        status = "empty_at_source"
        warnings.append("source page's script container is present but empty")
    elif not extracted.found:
        status = "extraction_failed"
    elif flags or warnings:
        status = "parsed_with_warnings"
    else:
        status = "parsed"

    elements = [e.to_dict() for e in parsed.elements] if parsed else None
    spans = [asdict(s) for s in extracted.formatting_spans]
    script = ScriptDoc(
        _id=script_id,
        movie_id=movie_id,
        title=title,
        script_url=script_url,
        status=status,  # type: ignore[arg-type]
        version=1,
        content_hash=content_hash,
        raw=raw,
        encoding={"used": decoded.encoding, "strict": decoded.strict, "declared_http": decoded.declared_http,
                  "declared_meta": decoded.declared_meta},
        text=extracted.text,
        text_length=len(extracted.text),
        line_count=len(extracted.lines),
        source_markup=extracted.markup,
        formatting_spans=spans,
        extraction={"method": extracted.method, "normalizations": extracted.normalizations,
                    "visible_page_chars": extracted.visible_page_chars,
                    "extraction_ratio": extracted.extraction_ratio, "removed_tags": extracted.removed_tags,
                    "page_title": extracted.page_title},
        layout=parsed.layout.to_dict() if parsed else None,
        structure_stats=parsed.stats if parsed else {},
        elements=elements,
        footer_metadata=extracted.footer_metadata,
        parse_warnings=warnings,
        quality_flags=sorted(set(flags)),
        parser_version=PARSER_VERSION,
        rights=rights,
        fetched_at=fetched_at,
        parsed_at=now,
        last_checked_at=now,
        created_at=now,
        updated_at=now,
        error=error,
    )
    script = _fit_document(script, processed_dir)
    if parsed is None or content_hash is None:
        return script, [], [], []

    scene_ids = [scene_id_for(script_id, s.scene_number) for s in parsed.scenes]
    character_ids = {c.key: character_id_for(script_id, c.key) for c in parsed.characters}
    dialogue_ids = [dialogue_id_for(script_id, b.order) for b in parsed.dialogues]
    text = extracted.text

    scenes: list[SceneDoc] = []
    for idx, scene in enumerate(parsed.scenes):
        scene_text = text[scene.char_start:scene.char_end]
        scenes.append(SceneDoc(
            _id=scene_ids[idx], script_id=script_id, movie_id=movie_id, scene_number=scene.scene_number,
            kind=scene.kind,  # type: ignore[arg-type]
            heading=scene.heading, heading_parts=scene.heading_parts, act=scene.act,
            page_start=scene.page_start, page_end=scene.page_end, description=scene.description,
            text=scene_text, char_start=scene.char_start, char_end=scene.char_end, line_start=scene.line_start,
            line_end=scene.line_end, element_start=scene.element_start, element_end=scene.element_end,
            characters=scene.characters,
            character_ids=[character_ids[k] for k in dict.fromkeys(
                b.speaker_key for b in parsed.dialogues if b.scene_index == idx and b.speaker_key in character_ids)],
            dialogue_count=scene.dialogue_count, shots=scene.shots, transitions=scene.transitions,
            element_counts=scene.element_counts, confidence=scene.confidence, uncertain=scene.confidence < 0.7,
            warnings=scene.warnings, content_hash=sha256_text(scene_text), script_version_hash=content_hash,
            parser_version=PARSER_VERSION, created_at=now,
        ))

    dialogues: list[DialogueDoc] = []
    per_scene: dict[int | None, int] = {}
    for block in parsed.dialogues:
        order_in_scene = per_scene.get(block.scene_index, 0)
        per_scene[block.scene_index] = order_in_scene + 1
        dialogues.append(DialogueDoc(
            _id=dialogue_ids[block.order], script_id=script_id, movie_id=movie_id,
            scene_id=scene_ids[block.scene_index] if block.scene_index is not None else None,
            character_id=character_ids.get(block.speaker_key), order=block.order, order_in_scene=order_in_scene,
            speaker=block.speaker, speaker_raw=block.speaker_raw, extensions=block.extensions, dual=block.dual,
            text=block.text, original_text=block.original_text, parentheticals=block.parentheticals,
            segments=[asdict(s) for s in block.segments], char_start=block.char_start, char_end=block.char_end,
            line_start=block.line_start, line_end=block.line_end, element_start=block.element_start,
            element_end=block.element_end, word_count=len(block.text.split()), confidence=block.confidence,
            uncertain=block.confidence < 0.7, warnings=block.warnings, script_version_hash=content_hash,
            parser_version=PARSER_VERSION, created_at=now,
        ))

    characters: list[CharacterDoc] = []
    for char in parsed.characters:
        characters.append(CharacterDoc(
            _id=character_ids[char.key], script_id=script_id, movie_id=movie_id, name=char.name, name_key=char.key,
            variants=char.variants, extensions=char.extensions, dialogue_count=char.dialogue_count,
            word_count=char.word_count, first_line=char.first_line,
            first_dialogue_id=dialogue_ids[char.first_dialogue_order] if char.dialogue_orders else None,
            scene_ids=[scene_ids[i] for i in char.scene_indices],
            dialogue_ids=[dialogue_ids[o] for o in char.dialogue_orders],
            script_version_hash=content_hash, parser_version=PARSER_VERSION, created_at=now,
        ))
    return script, scenes, dialogues, characters


def _bson_size(doc: ScriptDoc) -> int:
    """Size of a script document once BSON-encoded.

    Args:
        doc: Script document.

    Returns:
        Encoded size in bytes.
    """
    return len(bson.encode(doc.to_mongo()))


def _fit_document(script: ScriptDoc, processed_dir: Path) -> ScriptDoc:
    """Offload large arrays to ``data/processed`` if the document would exceed the BSON limit.

    Step 1 moves ``elements`` and ``formatting_spans`` to a gzipped JSON file and records it in
    ``elements_file``; step 2, if still too large, drops ``source_markup`` (the raw HTML archive keeps it).
    Each step adds a parse warning.

    Args:
        script: Script document to check.
        processed_dir: Folder under which ``<script_id>/<hash>-<parser>.elements.json.gz`` is written.

    Returns:
        The original document if it fits within ``MAX_DOC_BYTES``, otherwise a reduced copy.

    Raises:
        ValueError: The document is still too large after both steps.
    """
    if _bson_size(script) <= MAX_DOC_BYTES:
        return script
    payload = {"script_id": script.id, "content_hash": script.content_hash, "parser_version": script.parser_version,
               "elements": script.elements, "formatting_spans": script.formatting_spans}
    data = gzip.compress(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"), mtime=0)
    target = processed_dir / script.id / f"{(script.content_hash or 'none')[:16]}-{script.parser_version}.elements.json.gz"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    script = script.model_copy(update={
        "elements": None, "formatting_spans": None,
        "elements_file": {"path": target.as_posix(), "sha256": sha256_bytes(data), "bytes": len(data),
                          "element_count": len(script.elements or [])},
        "parse_warnings": script.parse_warnings + ["elements/formatting offloaded to data/processed (document size)"],
    })
    if _bson_size(script) > MAX_DOC_BYTES:
        script = script.model_copy(update={
            "source_markup": None,
            "parse_warnings": script.parse_warnings + ["source_markup omitted (document size); see raw HTML archive"],
        })
    if _bson_size(script) > MAX_DOC_BYTES:
        raise ValueError(f"script {script.id} text alone exceeds the document size limit")
    return script
