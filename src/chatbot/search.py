"""Passage retrieval: MongoDB text search over ``chat_chunks`` plus optional semantic re-ranking.

Chinese questions are rewritten into English search terms by the planner before they reach this module;
the re-ranker (a multilingual embedding model) then compares candidates with the original question.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from pymongo import ASCENDING, DESCENDING
from pymongo.database import Database

from src.chatbot.chunks import CHUNKS
from src.chatbot.llm import Embedder, LocalEmbedder

log = logging.getLogger("imsdb.chatbot.search")

RRF_K = 60
RERANK_CHARS = 500  # what the re-ranker reads of each candidate (~120 tokens; ~0.7 s each on this CPU)
_PHRASE_RE = re.compile(r'"([^"]{3,})"')
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
# Words that carry no search value in questions about screenplays.
QUESTION_STOPWORDS = {
    "what", "who", "whom", "which", "when", "where", "why", "how", "does", "did", "do", "is", "are", "was", "were",
    "the", "a", "an", "in", "on", "of", "to", "and", "or", "about", "scene", "scenes", "film", "movie", "script",
    "screenplay", "say", "says", "said", "tell", "me", "happen", "happens", "happened", "there", "any", "some",
    "character", "characters", "line", "lines", "part", "moment", "with", "that", "this", "it", "he", "she",
    "they", "his", "her", "their", "be", "been", "can", "could", "would", "should", "find", "show",
}


@dataclass
class Passage:
    """A screenplay passage (one ``chat_chunks`` document) retrieved for a question, with its scores.

    Attributes:
        chunk_id: The chunk's ``_id`` (``"<scene_id>:kNN"``).
        script_id: ``_id`` of the screenplay.
        film: Display title of the film.
        year: Release year, or None.
        scene_number: Scene ordinal in the screenplay (0 is the title page), or None.
        heading: Cleaned scene heading.
        characters: Speaking characters of the scene (upper case).
        text: The passage text.
        chunk_index: 0-based position of this chunk within its scene.
        chunk_count: Number of chunks the scene was split into.
        keyword_score: MongoDB ``textScore`` (0 for passages not found by text search).
        semantic_score: Cosine similarity to the question after re-ranking, or None if not re-ranked.
        score: Final fused rank score (higher is better).
        via: How the passage was found: ``"keyword"``, ``"opening"``, ``"ending"`` and/or ``"character"``.
    """

    chunk_id: str
    script_id: str
    film: str
    year: int | None
    scene_number: int | None
    heading: str
    characters: list[str]
    text: str
    chunk_index: int = 0
    chunk_count: int = 1
    keyword_score: float = 0.0
    semantic_score: float | None = None
    score: float = 0.0
    via: list[str] = field(default_factory=list)

    @classmethod
    def from_doc(cls, doc: dict[str, Any], via: str) -> "Passage":
        """Build a passage from a ``chat_chunks`` document.

        Args:
            doc: The chunk document (projected with ``_PROJECTION``; ``score`` holds the text score if any).
            via: Retrieval route recorded in ``via``.

        Returns:
            The new passage.
        """
        return cls(chunk_id=doc["_id"], script_id=doc["script_id"], film=doc.get("film", ""), year=doc.get("year"),
                   scene_number=doc.get("scene_number"), heading=doc.get("heading", ""),
                   characters=doc.get("characters") or [], text=doc.get("text", ""),
                   chunk_index=doc.get("chunk_index", 0), chunk_count=doc.get("chunk_count", 1),
                   keyword_score=float(doc.get("score", 0.0)), via=[via])

    def label(self) -> str:
        """Human-readable source label used in prompts and citations.

        Returns:
            E.g. ``"Titanic (1997), scene 12 (part 1 of 2): EXT. BOW - DAY"``; the title page is labelled
            ``"title page"`` without a heading.
        """
        film = f"{self.film} ({self.year})" if self.year else self.film
        scene = "title page" if self.scene_number == 0 else f"scene {self.scene_number}"
        part = f" (part {self.chunk_index + 1} of {self.chunk_count})" if self.chunk_count > 1 else ""
        return f"{film}, {scene}{part}" + (f": {self.heading}" if self.heading and self.scene_number else "")

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict (all dataclass fields).

        Returns:
            The passage's fields as a dict.
        """
        return asdict(self)


def text_query(query: str) -> str:
    """MongoDB $text search string: quoted phrases kept, question filler words removed.

    Args:
        query: Search query from the planner (English keywords, possibly with a quoted phrase).

    Returns:
        Quoted phrases followed by the remaining words that are not in ``QUESTION_STOPWORDS`` and longer than
        one character; "" when nothing is left.
    """
    phrases = [p.strip() for p in _PHRASE_RE.findall(query)]
    rest = _PHRASE_RE.sub(" ", query)
    words = [w for w in _TOKEN_RE.findall(rest) if w.lower() not in QUESTION_STOPWORDS and len(w) > 1]
    return " ".join([f'"{p}"' for p in phrases] + words).strip()


_PROJECTION = {"script_id": 1, "film": 1, "year": 1, "scene_number": 1, "heading": 1, "characters": 1, "text": 1,
               "chunk_index": 1, "chunk_count": 1}


def keyword_search(db: Database, query: str, *, script_ids: list[str] | None = None, limit: int = 30,
                   character: str | None = None) -> list[Passage]:
    """Full-text search of ``chat_chunks`` using the MongoDB text index, best text score first.

    If a quoted phrase produces no hits, the search is retried once with the quotes removed.

    Args:
        db: The MongoDB database.
        query: Search query; cleaned with ``text_query``.
        script_ids: Restrict the search to these screenplays, or None for the whole archive.
        limit: Maximum number of passages.
        character: Character name appended to the search terms, or None.

    Returns:
        Passages with ``via=["keyword"]`` and ``keyword_score`` set; [] when the query is empty.
    """
    search = text_query(query)
    if character:
        search = f"{search} {character}".strip()
    if not search:
        return []
    flt: dict[str, Any] = {"$text": {"$search": search}}
    if script_ids:
        flt["script_id"] = {"$in": script_ids}
    projection = {**_PROJECTION, "score": {"$meta": "textScore"}}
    docs = list(db[CHUNKS].find(flt, projection).sort([("score", {"$meta": "textScore"})]).limit(limit))
    if not docs and '"' in search:  # a quoted phrase that does not occur verbatim: retry as plain words
        flt["$text"]["$search"] = search.replace('"', " ")
        docs = list(db[CHUNKS].find(flt, projection).sort([("score", {"$meta": "textScore"})]).limit(limit))
    return [Passage.from_doc(d, "keyword") for d in docs]


def positional_passages(db: Database, script_id: str, position: str, *, limit: int = 4) -> list[Passage]:
    """Opening or closing passages of a screenplay (skipping the title page).

    Args:
        db: The MongoDB database.
        script_id: The screenplay's ``_id``.
        position: ``"opening"`` for the first passages; any other value gives the last ones.
        limit: Number of passages.

    Returns:
        The passages in screenplay order, with ``via`` set to ``position``.
    """
    order = ASCENDING if position == "opening" else DESCENDING
    flt = {"script_id": script_id, "scene_number": {"$gt": 0}}
    docs = list(db[CHUNKS].find(flt, _PROJECTION).sort([("scene_number", order), ("chunk_index", order)]).limit(limit))
    if order == DESCENDING:
        docs.reverse()
    return [Passage.from_doc(d, position) for d in docs]


def character_passages(db: Database, script_id: str, character: str, *, limit: int = 12) -> list[Passage]:
    """Passages from scenes where a character speaks, in screenplay order (fallback when keywords miss).

    Args:
        db: The MongoDB database.
        script_id: The screenplay's ``_id``.
        character: Character name; matched in upper case against the chunk's ``characters``.
        limit: Maximum number of passages.

    Returns:
        The first matching passages, with ``via=["character"]``.
    """
    docs = list(db[CHUNKS].find({"script_id": script_id, "characters": character.upper()}, _PROJECTION)
                .sort([("scene_number", ASCENDING), ("chunk_index", ASCENDING)]).limit(limit))
    return [Passage.from_doc(d, "character") for d in docs]


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors.

    Args:
        a: First vector.
        b: Second vector of the same length.

    Returns:
        The similarity in [-1, 1]; a zero vector gives 0 instead of dividing by zero.
    """
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
                                              or 1.0)


def rerank(embedder: Embedder, question: str, passages: list[Passage], *, top_k: int) -> list[Passage]:
    """Order candidates by reciprocal-rank fusion of keyword rank and semantic similarity to the question.

    Embeds the question (with the embedder's query instruction) and each passage's label plus its first
    ``RERANK_CHARS`` characters in one batch. Score = ``1/(RRF_K + keyword_rank) + 1.5/(RRF_K + semantic_rank)``,
    so semantic similarity weighs more. Sets ``semantic_score`` and ``score`` on the passages in place.

    Args:
        embedder: The embedding model.
        question: The user's original question (any language).
        passages: Candidates in keyword order (their position is taken as the keyword rank).
        top_k: Number of passages to keep.

    Returns:
        The ``top_k`` best passages, highest fused score first; [] when there are no candidates.
    """
    if not passages:
        return []
    texts = [LocalEmbedder.QUERY_INSTRUCTION + question] + [
        f"{p.label()}\n{p.text[:RERANK_CHARS]}" for p in passages]
    vectors = embedder.embed(texts)
    query_vec, doc_vecs = vectors[0], vectors[1:]
    for p, vec in zip(passages, doc_vecs):
        p.semantic_score = round(_cosine(query_vec, vec), 4)
    by_semantic = sorted(passages, key=lambda p: -(p.semantic_score or 0.0))
    semantic_rank = {p.chunk_id: i for i, p in enumerate(by_semantic)}
    for i, p in enumerate(passages):  # passages arrive in keyword order
        p.score = round(1 / (RRF_K + i) + 1.5 / (RRF_K + semantic_rank[p.chunk_id]), 6)
    return sorted(passages, key=lambda p: -p.score)[:top_k]


def dedupe(passages: list[Passage]) -> list[Passage]:
    """Remove repeated chunks, keeping the first occurrence.

    Args:
        passages: Passages, possibly from several retrieval routes.

    Returns:
        The passages with unique ``chunk_id``, in their original order.
    """
    seen: set[str] = set()
    out = []
    for p in passages:
        if p.chunk_id not in seen:
            seen.add(p.chunk_id)
            out.append(p)
    return out
