"""The chat pipeline: plan -> look up facts / retrieve passages -> answer (streamed), in English or Chinese."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

from pymongo.database import Database

from src.chatbot import facts as F
from src.chatbot import templates
from src.chatbot.catalog import Film, FilmCatalog
from src.chatbot.llm import Cancelled, LocalModels, Message
from src.chatbot.prompts import (ANSWER_SYSTEM, CHITCHAT_SYSTEM, INTENTS, LANGUAGE_RULES, PLAN_SCHEMA,
                                 planner_system)
from src.chatbot.search import (Passage, character_passages, dedupe, keyword_search, positional_passages,
                                rerank)
from src.chatbot.verify import unverified_quotes

log = logging.getLogger("imsdb.chatbot.engine")

# Event callback: emit(event_name, data). Event names: status, plan, sources, token, done.
Emit = Callable[[str, dict[str, Any]], None]
_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_FORCE_EN = re.compile(r"\b(in|answer in|reply in) english\b|用英文|英文回答|用英语|英语回答", re.I)
_FORCE_ZH = re.compile(r"\b(in|answer in|reply in) chinese\b|用中文|中文回答", re.I)
HISTORY_TURNS = 2
HISTORY_ANSWER_CHARS = 500


def detect_language(text: str) -> str:
    """'zh' if the question is written mainly in Chinese, else 'en'. Explicit requests win.

    "Answer in English"/"用英文" and "in Chinese"/"用中文" override the script count; otherwise Chinese needs at
    least two CJK characters and no fewer CJK characters than Latin words.

    Args:
        text: The user's question.

    Returns:
        ``"zh"`` or ``"en"``.
    """
    if _FORCE_EN.search(text):
        return "en"
    if _FORCE_ZH.search(text):
        return "zh"
    cjk = len(_CJK_RE.findall(text))
    latin_words = len(re.findall(r"[A-Za-z]+", text))
    return "zh" if cjk >= 2 and cjk >= latin_words else "en"


@dataclass
class Turn:
    """One finished question/answer exchange kept as conversation history.

    Attributes:
        question: The user's question.
        answer: The full answer text.
        films: Film names from the plan (as the planner wrote them, including carried-over ones).
    """

    question: str
    answer: str
    films: list[str]


@dataclass
class ChatSession:
    """A conversation (one browser tab or CLI run).

    Attributes:
        id: Session id (random hex unless the client supplied one).
        turns: Finished exchanges, oldest first.
        created: Creation time (Unix seconds).
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    turns: list[Turn] = field(default_factory=list)
    created: float = field(default_factory=time.time)

    def recent(self) -> list[Turn]:
        """The turns used as context for the next question.

        Returns:
            The last ``HISTORY_TURNS`` turns, oldest first.
        """
        return self.turns[-HISTORY_TURNS:]


class SessionStore:
    """In-memory sessions (one browser tab each), least-recently-used eviction."""

    def __init__(self, capacity: int = 64) -> None:
        """Create an empty store.

        Args:
            capacity: Maximum number of sessions kept; the least recently used are dropped beyond it.
        """
        self._sessions: OrderedDict[str, ChatSession] = OrderedDict()
        self._capacity = capacity
        self._lock = threading.Lock()

    def get(self, session_id: str | None) -> ChatSession:
        """Return a session by id, creating it if unknown (thread-safe).

        Args:
            session_id: Id sent by the client, or None for a new session.

        Returns:
            The existing session (marked most recently used), or a new one with the given id or a random id.
        """
        with self._lock:
            if session_id and session_id in self._sessions:
                self._sessions.move_to_end(session_id)
                return self._sessions[session_id]
            session = ChatSession(id=session_id or uuid.uuid4().hex)
            self._sessions[session.id] = session
            while len(self._sessions) > self._capacity:
                self._sessions.popitem(last=False)
            return session

    def reset(self, session_id: str) -> None:
        """Forget a session and its history (no error if it does not exist).

        Args:
            session_id: The session to remove.
        """
        with self._lock:
            self._sessions.pop(session_id, None)


@dataclass
class Plan:
    """The planner's interpretation of a question (fields of ``PLAN_SCHEMA``).

    Attributes:
        intent: One of ``INTENTS``.
        films: Up to 3 English film titles named in the question.
        character: Character name as written in the screenplay, or "".
        genre: Genre for ``list_films``, or "".
        writer: Writer for ``list_films``, or "".
        year: Release year, or 0.
        scene_position: ``"any"``, ``"opening"`` or ``"ending"``.
        search_query: English search terms for the text index, or "".
        carried_over: True when ``films`` was copied from the previous turn (a follow-up question).
    """

    intent: str = "scene_search"
    films: list[str] = field(default_factory=list)
    character: str = ""
    genre: str = ""
    writer: str = ""
    year: int = 0
    scene_position: str = "any"
    search_query: str = ""
    carried_over: bool = False

    @classmethod
    def from_json(cls, text: str) -> "Plan":
        """Parse the planner's JSON output, tolerating bad output.

        Args:
            text: The planner's reply.

        Returns:
            The plan; a default plan (``scene_search``, no films) if the JSON is invalid or not an object.
            Fields of the wrong type keep their defaults. An unknown intent becomes ``scene_search``; films are
            limited to 3 non-empty strings.
        """
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            log.warning("planner returned invalid JSON: %r", text[:200])
            return cls()
        if not isinstance(data, dict):
            log.warning("planner returned JSON that is not an object: %r", text[:200])
            return cls()
        defaults = cls()
        values: dict[str, Any] = {}
        for key in cls.__dataclass_fields__:
            if key == "carried_over" or key not in data:
                continue
            value, default = data[key], getattr(defaults, key)
            if isinstance(default, bool) or not isinstance(value, type(default)) or isinstance(value, bool):
                continue  # wrong type (the grammar should prevent this): keep the default
            values[key] = value
        plan = cls(**values)
        if plan.intent not in INTENTS:
            plan.intent = "scene_search"
        plan.films = [f for f in plan.films if isinstance(f, str) and f.strip()][:3]
        return plan


@dataclass
class ChatResult:
    """Outcome of one question.

    Attributes:
        answer: The final answer text.
        language: ``"en"`` or ``"zh"``.
        plan: The plan used.
        sources: Cited passages as built by ``ChatEngine._source`` (numbered from 1).
        timings: Seconds per stage: ``plan``, ``search``, ``answer``, ``total`` (present when run).
        templated: True when the answer came from a template, without the language model.
        unverified_quotes: Quotes in the answer not found in the passages or facts.
    """

    answer: str
    language: str
    plan: Plan
    sources: list[dict[str, Any]]
    timings: dict[str, float]
    templated: bool = False
    unverified_quotes: list[str] = field(default_factory=list)


def _fmt_facts_film(f: dict[str, Any]) -> str:
    """Format film facts as the plain-text FACTS block of the answer prompt.

    Args:
        f: Output of ``facts.film_facts``.

    Returns:
        One ``Label: value`` line per known fact (title, genres, writers, draft, release, size, top characters,
        note).
    """
    parts = [f"Film: {f['film']}" + (f" ({f['year']})" if f.get("year") else "")]
    for label, key in (("Genres", "genres"), ("Writers", "writers")):
        if f.get(key):
            parts.append(f"{label}: {', '.join(f[key])}")
    if f.get("script_date") or f.get("draft"):
        parts.append(f"Screenplay draft: {' / '.join(x for x in (f.get('script_date'), f.get('draft')) if x)}")
    if f.get("release_date"):
        parts.append(f"Release: {f['release_date']}")
    if f.get("scenes"):
        parts.append(f"Scenes: {f['scenes']}, dialogue blocks: {f.get('dialogue_blocks')}, "
                     f"speaking characters: {f.get('speaking_characters')}, pages: {f.get('pages')}")
    if f.get("top_characters"):
        parts.append("Characters with most lines: " + ", ".join(f"{c['name']} ({c['lines']})" for c in f["top_characters"]))
    if f.get("note"):
        parts.append(f"Note: {f['note']}")
    return "\n".join(parts)


def _fmt_facts_character(c: dict[str, Any]) -> str:
    """Format character facts as plain text for the FACTS block of the answer prompt.

    Args:
        c: Output of ``facts.character_facts``.

    Returns:
        Lines with the character's counts, first appearance and quoted first lines.
    """
    lines = [f"Character: {c['character']} in {c['film']}: {c['lines']} lines, {c['words']} words, "
             f"appears in {c['scenes']} scenes."]
    if c.get("first_appearance"):
        fa = c["first_appearance"]
        lines.append(f"First appearance: scene {fa['scene_number']} ({fa['heading']})")
    if c.get("first_lines"):
        lines.append("First lines: " + " | ".join(f'"{x["text"]}"' for x in c["first_lines"]))
    return "\n".join(lines)


class ChatEngine:
    """Answers questions: plans with the model, gathers facts and passages, and writes a streamed answer.

    Only one question is processed at a time (``_busy`` lock); others wait and are reported as ``queued``.
    """

    def __init__(self, db: Database, catalog: FilmCatalog, models: LocalModels, *, max_passages: int = 4,
                 rerank_candidates: int = 10, use_reranker: bool = True) -> None:
        """Set up the engine and build the planner system prompt from the catalogue's genres.

        Args:
            db: The MongoDB database.
            catalog: The film catalogue.
            models: The loaded chat model and optional embedder.
            max_passages: Maximum number of passages shown to the answer model.
            rerank_candidates: Number of keyword hits passed to the re-ranker (twice as many are fetched).
            use_reranker: Re-rank with the embedder; ignored when no embedder is loaded.
        """
        self.db = db
        self.catalog = catalog
        self.models = models
        self.max_passages = max_passages
        self.rerank_candidates = rerank_candidates
        self.use_reranker = use_reranker and models.embedder is not None
        genres = sorted({g for f in catalog.films for g in f.genres})
        self._planner_system = planner_system(genres)
        self._busy = threading.Lock()

    # ------------------------------------------------------------------ planning
    def plan(self, question: str, session: ChatSession, cancel: threading.Event | None = None) -> Plan:
        """Turn a question into a search plan with the "planner" context (JSON constrained by ``PLAN_SCHEMA``).

        The previous question is given to the model for context only. If a film-related question names no
        film, the previous turn's films are carried over (``carried_over=True``).

        Args:
            question: The user's question.
            session: The conversation, for the previous question and its films.
            cancel: Event that aborts generation when set, or None.

        Returns:
            The plan.

        Raises:
            Cancelled: ``cancel`` was set during planning.
        """
        context = ""
        recent = session.recent()
        if recent:
            last = recent[-1]
            context = f"Previous question (for context only): {last.question}\n\n"
        raw = self.models.chat.generate(
            [Message("system", self._planner_system), Message("user", f"{context}Q: {question}")],
            max_tokens=160, temperature=0.0, json_schema=PLAN_SCHEMA, cancel=cancel, role="planner")
        plan = Plan.from_json(raw)
        # Follow-ups ("and how does it end?") name no film: carry the previous one over here, in code,
        # rather than asking the model to (it then tends to repeat old films even when a new one is named).
        if not plan.films and plan.intent in {"film_details", "character", "scene_search"} and recent:
            plan.films = list(recent[-1].films)
            plan.carried_over = True
        return plan

    def warm_up(self) -> None:
        """Pre-compute the fixed prompt prefixes so the first real question is as fast as later ones.

        Generates one token on the "planner" context (planner system prompt) and on the "answer" context
        (English answer system prompt), filling their KV caches.
        """
        started = time.time()
        self.models.chat.generate([Message("system", self._planner_system), Message("user", "Q: hello")],
                                  max_tokens=1, temperature=0.0, role="planner")
        self.models.chat.generate([Message("system", ANSWER_SYSTEM["en"]), Message("user", "hello")],
                                  max_tokens=1, temperature=0.0)
        log.info("prompt caches warmed in %.1fs", time.time() - started)

    # ------------------------------------------------------------------ main entry
    def ask(self, question: str, session: ChatSession, emit: Emit | None = None,
            cancel: threading.Event | None = None) -> ChatResult:
        """Answer one question, reporting progress through ``emit``.

        Events emitted, in order: ``status`` ``{"stage": "queued"}`` (only if another question is running),
        ``status`` ``{"stage": "planning"}``, ``plan`` ``{"plan", "language"}``, then the events of the chosen
        path (``status`` searching/writing, ``sources``, ``token`` pieces), and finally ``done``
        ``{"answer", "timings", "templated", "unverified_quotes"}``. The turn is added to the session history.
        On cancel, ``Cancelled`` propagates: no ``done`` event is sent and nothing is added to the history.

        Args:
            question: The user's question (English or Chinese).
            session: The conversation it belongs to.
            emit: Callback ``emit(event, data)``, or None to discard events.
            cancel: Event that aborts the model calls when set, or None.

        Returns:
            The result, with ``timings`` filled in.

        Raises:
            Cancelled: ``cancel`` was set while waiting for a previous question or while the model was generating.
        """
        emit = emit or (lambda _e, _d: None)
        question = question.strip()
        timings: dict[str, float] = {}
        started = time.time()
        lang = detect_language(question)
        if not self._busy.acquire(blocking=False):
            emit("status", {"stage": "queued"})
            # Wait for the previous question, but give up as soon as this one is cancelled.
            while not self._busy.acquire(timeout=0.25):
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
        try:
            emit("status", {"stage": "planning"})
            t0 = time.time()
            plan = self.plan(question, session, cancel)
            timings["plan"] = round(time.time() - t0, 1)
            emit("plan", {"plan": plan.__dict__, "language": lang})
            result = self._dispatch(question, lang, plan, session, emit, cancel, timings)
        finally:
            self._busy.release()
        timings["total"] = round(time.time() - started, 1)
        result.timings = timings
        session.turns.append(Turn(question, result.answer, [f for f in plan.films]))
        emit("done", {"answer": result.answer, "timings": timings, "templated": result.templated,
                      "unverified_quotes": result.unverified_quotes})
        return result

    def _resolve(self, plan: Plan) -> tuple[list[Film], list[str], list[str]]:
        """Look up the plan's film names in the catalogue.

        Args:
            plan: The plan (``films`` and ``year`` are used).

        Returns:
            ``(films, missing, alternatives)``: the best match per name (no duplicates), the names with no
            match, and labels of up to 5 other matches per name.
        """
        films: list[Film] = []
        missing: list[str] = []
        alternatives: list[str] = []
        for name in plan.films:
            matches = self.catalog.resolve(name, year=plan.year or None)
            if not matches:
                missing.append(name)
                continue
            if matches[0] not in films:
                films.append(matches[0])
            alternatives.extend(m.label() for m in matches[1:6])
        return films, missing, alternatives

    def _dispatch(self, question: str, lang: str, plan: Plan, session: ChatSession, emit: Emit,
                  cancel: threading.Event | None, timings: dict[str, float]) -> ChatResult:
        """Produce the answer for a plan: a template, chit-chat, or a grounded answer from facts and passages.

        Archive stats, filtered film lists and full-script requests are answered from templates. Otherwise the
        films' facts (MongoDB) and retrieved passages are put in one user message (FACTS, EXCERPTS, language
        rule, question) after the last turns, and the "answer" context writes the answer, streamed as
        ``token`` events; ``status`` (searching/writing) and ``sources`` events are emitted on the way.
        Quotes in the answer are then checked against the passages and facts.

        Args:
            question: The user's question.
            lang: Answer language, ``"en"`` or ``"zh"``.
            plan: The plan.
            session: The conversation (recent turns are included in the prompt).
            emit: Event callback.
            cancel: Event that aborts generation when set, or None.
            timings: Dict that receives ``search`` and ``answer`` durations in seconds.

        Returns:
            The result (``timings`` is filled in later by ``ask``).

        Raises:
            Cancelled: ``cancel`` was set during re-ranking or generation.
        """
        if plan.intent == "archive_stats":
            return self._templated(lang, plan, templates.archive(F.corpus_overview(self.db, self.catalog), lang), emit)
        if plan.intent == "list_films" and (plan.genre or plan.writer or plan.year):
            listing = F.list_films(self.catalog, genre=plan.genre or None, writer=plan.writer or None,
                                   year=plan.year or None)
            return self._templated(lang, plan, templates.film_list(listing, lang), emit)
        films, missing, alternatives = self._resolve(plan)
        if plan.intent == "full_script":
            return self._templated(lang, plan, templates.full_script(films, lang), emit)
        if plan.intent == "chitchat" and not films:
            return self._chitchat(question, lang, plan, session, emit, cancel, timings)

        # ---- facts
        fact_blocks: list[str] = []
        for film in films:
            fact_blocks.append(_fmt_facts_film(F.film_facts(self.db, self.catalog, film)))
            if plan.character:
                cf = F.character_facts(self.db, film, plan.character)
                if cf:
                    fact_blocks.append(_fmt_facts_character(cf))
        if missing:
            fact_blocks.append("Not in the archive: " + ", ".join(missing) +
                               ". (The archive only contains films listed on IMSDb.)")
        if alternatives:
            fact_blocks.append("Other titles that also match: " + ", ".join(alternatives))

        # ---- passages
        passages: list[Passage] = []
        if plan.intent != "film_details" or plan.search_query:
            emit("status", {"stage": "searching"})
            t0 = time.time()
            passages = self._retrieve(question, plan, films, missing, cancel)
            timings["search"] = round(time.time() - t0, 1)
        sources = [self._source(i + 1, p) for i, p in enumerate(passages)]
        emit("sources", {"sources": sources})

        # ---- answer
        emit("status", {"stage": "writing"})
        t0 = time.time()
        messages = [Message("system", ANSWER_SYSTEM[lang])]
        for turn in session.recent():
            messages.append(Message("user", turn.question))
            messages.append(Message("assistant", turn.answer[:HISTORY_ANSWER_CHARS]))
        body = ["FACTS:\n" + ("\n\n".join(fact_blocks) if fact_blocks else "(none)")]
        if passages:
            body.append("EXCERPTS:\n" + "\n\n".join(f"[{i + 1}] {p.label()}\n{p.text}" for i, p in enumerate(passages)))
        elif plan.intent != "film_details":
            body.append("EXCERPTS: (no matching passages were found in the archive)")
        body.append(LANGUAGE_RULES[lang])
        body.append(f"QUESTION: {question}")
        messages.append(Message("user", "\n\n".join(body)))
        answer = self.models.chat.generate(messages, max_tokens=700, temperature=0.15,
                                           on_token=lambda s: emit("token", {"text": s}), cancel=cancel)
        timings["answer"] = round(time.time() - t0, 1)
        unverified = unverified_quotes(answer, passages, fact_blocks)
        if unverified:
            log.warning("answer contains quotes not found in its sources: %s", unverified)
        return ChatResult(answer=answer, language=lang, plan=plan, sources=sources, timings=timings,
                          unverified_quotes=unverified)

    def _retrieve(self, question: str, plan: Plan, films: list[Film], missing: list[str],
                  cancel: threading.Event | None) -> list[Passage]:
        """Find the passages to show the answer model.

        Opening/ending questions get 2 positional passages of the first film. Then a ``chat_chunks`` text
        search (planner query, or the question itself if it is English) restricted to the films' screenplays;
        if nothing is found and a character is named, that character's scenes are used. When there are more
        candidates than room, the first ``rerank_candidates`` are re-ranked with the embedder.

        Args:
            question: The user's question (used for the fallback query and for re-ranking).
            plan: The plan (``search_query``, ``character``, ``scene_position``).
            films: Resolved films.
            missing: Film names that were not found in the catalogue.
            cancel: Checked before re-ranking, or None.

        Returns:
            At most ``max_passages`` passages: positional ones first, then the best matches. [] when the films
            have no stored screenplay, or when the named films are all missing from the archive.

        Raises:
            Cancelled: ``cancel`` was set before re-ranking.
        """
        script_ids = [f.script_id for f in films if f.script_id]
        if films and not script_ids:
            return []  # the film is known but no screenplay text is stored
        if missing and not films:
            return []  # the named film is not in the archive; do not pass off other films' scenes
        fixed: list[Passage] = []
        if plan.scene_position in {"opening", "ending"} and script_ids:
            fixed = positional_passages(self.db, script_ids[0], plan.scene_position, limit=2)
        query = plan.search_query or (question if detect_language(question) == "en" else "")
        candidates = keyword_search(self.db, query, script_ids=script_ids or None,
                                    limit=self.rerank_candidates * 2, character=plan.character or None)
        if not candidates and plan.character and script_ids:
            candidates = character_passages(self.db, script_ids[0], plan.character)
        candidates = [c for c in dedupe(candidates) if c.chunk_id not in {p.chunk_id for p in fixed}]
        room = self.max_passages - len(fixed)
        if self.use_reranker and len(candidates) > room and self.models.embedder is not None:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            chosen = rerank(self.models.embedder, question, candidates[: self.rerank_candidates], top_k=room)
        else:
            chosen = candidates[:room]
        return fixed + chosen

    @staticmethod
    def _source(n: int, p: Passage) -> dict[str, Any]:
        """Describe a passage for the client's source list.

        Args:
            n: Citation number (1-based, as used in the prompt's ``[n]``).
            p: The passage.

        Returns:
            A dict with ``n``, ``label``, ``film``, ``year``, ``scene_number``, ``heading``, ``chunk_id`` and
            ``excerpt`` (first 400 characters).
        """
        return {"n": n, "label": p.label(), "film": p.film, "year": p.year, "scene_number": p.scene_number,
                "heading": p.heading, "chunk_id": p.chunk_id, "excerpt": p.text[:400]}

    def _chitchat(self, question: str, lang: str, plan: Plan, session: ChatSession, emit: Emit,
                  cancel: threading.Event | None, timings: dict[str, float]) -> ChatResult:
        """Answer a question that is not about a specific film with the chit-chat system prompt.

        Uses the "answer" context; emits empty ``sources``, ``status`` writing and streamed ``token`` events.

        Args:
            question: The user's question.
            lang: Answer language, ``"en"`` or ``"zh"``.
            plan: The plan (returned in the result).
            session: The conversation (recent turns are included in the prompt).
            emit: Event callback.
            cancel: Event that aborts generation when set, or None.
            timings: Dict that receives the ``answer`` duration in seconds.

        Returns:
            The result, without sources.

        Raises:
            Cancelled: ``cancel`` was set during generation.
        """
        emit("sources", {"sources": []})
        emit("status", {"stage": "writing"})
        t0 = time.time()
        messages = [Message("system", CHITCHAT_SYSTEM)]
        for turn in session.recent():
            messages += [Message("user", turn.question), Message("assistant", turn.answer[:HISTORY_ANSWER_CHARS])]
        messages.append(Message("user", f"{question}\n\n{LANGUAGE_RULES[lang]}"))
        answer = self.models.chat.generate(messages, max_tokens=300, temperature=0.5,
                                           on_token=lambda s: emit("token", {"text": s}), cancel=cancel)
        timings["answer"] = round(time.time() - t0, 1)
        return ChatResult(answer=answer, language=lang, plan=plan, sources=[], timings=timings)

    def _templated(self, lang: str, plan: Plan, text: str, emit: Emit) -> ChatResult:
        """Return a template answer, sent to the client as empty ``sources`` and a single ``token`` event.

        Args:
            lang: Answer language.
            plan: The plan (returned in the result).
            text: The finished answer text.
            emit: Event callback.

        Returns:
            The result with ``templated=True``.
        """
        emit("sources", {"sources": []})
        emit("token", {"text": text})
        return ChatResult(answer=text, language=lang, plan=plan, sources=[], timings={}, templated=True)

