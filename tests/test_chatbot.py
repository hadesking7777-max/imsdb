"""Offline chatbot tests. No real models are loaded: the chat model and embedder are deterministic fakes."""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import re
import sys
import threading
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

from src.chatbot import model_store
from src.chatbot.catalog import Film, FilmCatalog
from src.chatbot.chunks import CHUNKS, build_chunks, chunk_index_status, normalize_scene_text, split_chunks
from src.chatbot.engine import ChatEngine, ChatSession, Plan, detect_language
from src.chatbot.llm import Cancelled, LocalModels, Message, chatml
from src.chatbot.prompts import LANGUAGE_RULES
from src.chatbot.search import keyword_search, text_query
from src.chatbot.server import EngineHolder, make_server
from src.config import load_settings
from src.errors import ConfigError
from src.services.ingestion import IngestionService
from tests.conftest import FakeSite

LIGHTHOUSE = "The Lighthouse Keeper"


# ====================================================================== fakes
class FakeChat:
    """Returns queued plans for the planner role and a fixed answer (streamed in pieces) otherwise.

    Stands in for ``LocalChatModel``; every call is recorded in ``calls`` so tests can inspect the prompts.
    """

    def __init__(self, plans: list[dict[str, Any]] | None = None, answer: str = "Margaret polishes the lens [1].") -> None:
        """Store the planner replies and the answer text.

        Args:
            plans: Partial plan dicts returned (one per planner call, in order) after filling in default fields;
                when exhausted, the planner returns the defaults (a ``scene_search`` plan).
            answer: Text returned for every answer call.
        """
        self.plans = list(plans or [])
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    def generate(self, messages: list[Message], *, max_tokens: int, temperature: float = 0.3,
                 json_schema: dict[str, Any] | None = None, on_token: Callable[[str], None] | None = None,
                 cancel: threading.Event | None = None, role: str = "answer") -> str:
        """Mimic ``ChatModel.generate``: return a JSON plan for ``role="planner"``, otherwise stream the answer.

        The answer is sent to ``on_token`` in two halves, checking ``cancel`` before each half.

        Args:
            messages: The chat messages (recorded, not interpreted).
            max_tokens: Ignored.
            temperature: Ignored.
            json_schema: Only recorded as present/absent.
            on_token: Receives the streamed answer pieces.
            cancel: When set, the answer stream raises ``Cancelled``.
            role: ``"planner"`` or ``"answer"``.

        Returns:
            The plan as a JSON string, or the full answer text.

        Raises:
            Cancelled: If ``cancel`` is set while streaming the answer.
        """
        self.calls.append({"role": role, "messages": messages, "schema": json_schema is not None})
        if role == "planner":
            plan = self.plans.pop(0) if self.plans else {}
            full = {"intent": "scene_search", "films": [], "character": "", "genre": "", "writer": "", "year": 0,
                    "scene_position": "any", "search_query": "", **plan}
            return json.dumps(full)
        half = len(self.answer) // 2
        for piece in (self.answer[:half], self.answer[half:]):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            if on_token:
                on_token(piece)
        return self.answer

    def answer_prompt(self) -> str:
        """Return the last user message of the most recent answer call.

        Returns:
            The prompt text the engine sent to the answer model (excerpts, film facts, language rules).
        """
        return next(c for c in reversed(self.calls) if c["role"] == "answer")["messages"][-1].content


class FakeEmbedder:
    """Hashed bag of words: texts sharing words get similar vectors.

    Stands in for the embedding model used by the reranker.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts as L2-normalized 64-bucket word-count vectors (MD5 of each lowercase word picks the bucket).

        Args:
            texts: Texts to embed.

        Returns:
            One 64-float unit vector per text (all zeros for a text without letters).
        """
        out = []
        for text in texts:
            vec = [0.0] * 64
            for word in re.findall(r"[a-z]+", text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


# ====================================================================== fixtures
@pytest.fixture
def kb(app, site: FakeSite):
    """A knowledge base: the fake site discovered and crawled (five records), then chunked for chat search.

    Args:
        app: Ingestion context on a throwaway database.
        site: The fake IMSDb that ``app`` fetches from.

    Returns:
        The same ``app`` context, now with screenplays and chat chunks in its database.
    """
    service = IngestionService(app)
    service.load_gate(refresh_robots=False)
    service.discover()
    service.crawl(limit=5)
    build_chunks(app.db)
    return app


def _engine(kb, chat: FakeChat, *, embedder: bool = True) -> ChatEngine:
    """Build a ``ChatEngine`` over the knowledge base using fake models.

    Args:
        kb: Context whose database holds the crawled films and chunks.
        chat: The fake chat model that plans and answers.
        embedder: Include ``FakeEmbedder`` (enables reranking); ``False`` tests the no-reranker path.

    Returns:
        An engine limited to 3 passages from 6 rerank candidates.
    """
    models = LocalModels(chat=chat, embedder=FakeEmbedder() if embedder else None, info={"fake": True})
    return ChatEngine(kb.db, FilmCatalog.load(kb.db), models, max_passages=3, rerank_candidates=6)


# ====================================================================== pure functions
def test_scene_text_normalization_keeps_lines_and_fixes_cp1252_controls() -> None:
    """Scene text normalization keeps line breaks, collapses spaces and extra blank lines, and fixes cp1252 dashes.

    The C1 control character 0x96 (a cp1252 en dash decoded as Latin-1) becomes a real en dash.
    """
    raw = "1    INT. ROOM \x96 NIGHT          1\n\n\n\n          MARGARET\n     Steady now.   Steady."
    assert normalize_scene_text(raw) == "1 INT. ROOM \u2013 NIGHT 1\n\nMARGARET\nSteady now. Steady."


def test_split_chunks_respects_limit_and_keeps_every_word() -> None:
    """Every chunk fits the size limit and, ignoring whitespace, the chunks rebuild the original text."""
    text = "\n".join(f"Line {i} " + "word " * (i % 7) for i in range(200)) + "\n" + "x" * 900
    parts = split_chunks(text, limit=300)
    assert all(len(p) <= 300 for p in parts)
    squash = lambda s: re.sub(r"\s+", "", s)  # noqa: E731
    assert squash("".join(parts)) == squash(text)


@pytest.mark.parametrize(("text", "lang"), [
    ("How does Titanic end?", "en"),
    ("《泰坦尼克号》的结局是怎样的？", "zh"),
    ("Titanic 里 Jack 说了什么", "zh"),
    ("《泰坦尼克号》讲了什么？请用英文回答", "en"),
    ("Tell me about 教父 in Chinese", "zh"),
])
def test_language_detection(text: str, lang: str) -> None:
    """The answer language follows the question's script unless the user explicitly asks for another language.

    Args:
        text: The question.
        lang: Expected language code, ``"en"`` or ``"zh"``.
    """
    assert detect_language(text) == lang


def test_text_query_drops_filler_and_keeps_quoted_phrases() -> None:
    """The keyword query drops question filler words but keeps quoted phrases intact."""
    assert text_query('What does Jack say about "king of the world" in the scene?') == '"king of the world" Jack'


def test_plan_parsing_falls_back_safely() -> None:
    """Bad planner JSON or an unknown intent falls back to ``scene_search``; blank films are dropped, max 3 kept."""
    assert Plan.from_json("not json").intent == "scene_search"
    plan = Plan.from_json(json.dumps({"intent": "bogus", "films": ["A", "", "B", "C", "D"]}))
    assert plan.intent == "scene_search" and plan.films == ["A", "B", "C"]


def test_chatml_disables_thinking() -> None:
    """The ChatML prompt ends with an empty ``<think>`` block so the model answers without reasoning text."""
    prompt = chatml([Message("system", "S"), Message("user", "U")])
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_chat_host_must_be_local(tmp_path: Path) -> None:
    """Binding the chat server to a non-loopback address is a configuration error.

    Args:
        tmp_path: Folder for the (absent) ``.env`` file.
    """
    with pytest.raises(ConfigError):
        load_settings(env_file=tmp_path / "none.env", chat_host="0.0.0.0")


# ====================================================================== catalogue
def _film(title: str, **kw: Any) -> Film:
    """Build a catalogue ``Film`` for title-resolution tests.

    Args:
        title: The IMSDb title, e.g. ``"Lighthouse Keeper, The"``.
        **kw: Optional ``year``, ``genres``, ``status`` and ``script_id`` overrides.

    Returns:
        A ``Film`` with display title and key derived from ``title``.
    """
    from src.utils.text_processing import display_title, title_key
    shown = display_title(title)
    return Film(id=f"mv-{title_key(shown)}", title=title, display_title=shown, key=title_key(shown),
                year=kw.get("year"), genres=kw.get("genres", []), writers=[], status=kw.get("status", "script_collected"),
                script_id=kw.get("script_id", "sc-x"), detail_url="")


def test_title_resolution_variants() -> None:
    """Titles resolve through article moves, Chinese aliases, title brackets, fuzzy spelling and franchise prefixes."""
    cat = FilmCatalog([_film("Lighthouse Keeper, The"), _film("Godfather"), _film("Star Wars: A New Hope"),
                       _film("Star Wars: Return of the Jedi"), _film("Forrest Gump")],
                      aliases={"灯塔守护人": "The Lighthouse Keeper", "教父": "Godfather"})
    first = lambda name: [f.display_title for f in cat.resolve(name)]  # noqa: E731
    assert first("The Lighthouse Keeper") == [LIGHTHOUSE]
    assert first("lighthouse keeper, the") == [LIGHTHOUSE]
    assert first("《灯塔守护人》") == [LIGHTHOUSE]
    assert first("The Godfather") == ["Godfather"]  # IMSDb drops the article
    assert first("教父") == ["Godfather"]
    assert first("Forest Gump") == ["Forrest Gump"]  # fuzzy
    assert len(first("Star Wars")) == 2  # franchise prefix
    assert first("Nonexistent Picture") == []


# ====================================================================== index + search (MongoDB)
pytestmark_db = pytest.mark.mongodb


@pytestmark_db
def test_chunk_index_is_incremental(kb) -> None:
    """The chunk index covers every parsed script, and rebuilding it re-chunks nothing that is unchanged.

    Args:
        kb: Ingestion context whose database holds the five crawled fake films and their chat chunks.
    """
    status = chunk_index_status(kb.db)
    assert status["ready"] and status["complete"] and status["scripts_indexed"] == 3
    again = build_chunks(kb.db)
    assert again["rechunked"] == 0 and again["unchanged"] == 3
    doc = kb.db[CHUNKS].find_one({"film": LIGHTHOUSE, "scene_number": 1})
    assert doc and "lighthouse" in doc["text"].lower() and doc["chars"] <= 1200


@pytestmark_db
def test_keyword_search_can_be_limited_to_one_script(kb) -> None:
    """Keyword search restricted to script ids only returns passages from those scripts.

    Args:
        kb: Ingestion context whose database holds the five crawled fake films and their chat chunks.
    """
    script_id = FilmCatalog.load(kb.db).resolve(LIGHTHOUSE)[0].script_id
    hits = keyword_search(kb.db, "storm lighthouse lens", script_ids=[script_id], limit=5)
    assert hits and all(h.script_id == script_id for h in hits)
    assert keyword_search(kb.db, "storm lighthouse lens", script_ids=["sc-none"], limit=5) == []


# ====================================================================== engine (MongoDB + fakes)
@pytestmark_db
def test_scene_question_is_grounded_cited_and_streamed(kb) -> None:
    """A scene question streams sources before tokens, cites only the named film and records it in the session.

    Args:
        kb: Ingestion context whose database holds the five crawled fake films and their chat chunks.
    """
    chat = FakeChat([{"films": [LIGHTHOUSE], "character": "MARGARET", "search_query": "storm lighthouse lens"}])
    engine = _engine(kb, chat)
    events: list[tuple[str, dict[str, Any]]] = []
    session = ChatSession()
    result = engine.ask("What does Margaret do in the lamp room?", session, lambda e, d: events.append((e, d)))
    names = [e for e, _ in events]
    assert names[0] == "status" and "plan" in names and names[-1] == "done"
    assert names.index("sources") < names.index("token")
    assert "".join(d["text"] for e, d in events if e == "token") == result.answer
    assert result.sources and all(s["film"] == LIGHTHOUSE for s in result.sources)
    prompt = chat.answer_prompt()
    assert "EXCERPTS:\n[1]" in prompt and LANGUAGE_RULES["en"] in prompt
    assert "Character: MARGARET" in prompt and "Film: The Lighthouse Keeper" in prompt
    assert session.turns[-1].films == [LIGHTHOUSE]


@pytestmark_db
def test_chinese_question_gets_chinese_answer_instructions(kb) -> None:
    """A Chinese question gets the Chinese language rules, and the planner sees the original question.

    Args:
        kb: Ingestion context whose database holds the five crawled fake films and their chat chunks.
    """
    chat = FakeChat([{"films": [LIGHTHOUSE], "search_query": "storm lighthouse"}], answer="玛格丽特在擦拭透镜 [1]。")
    result = _engine(kb, chat).ask("《灯塔守护人》开头发生了什么？", ChatSession())
    assert result.language == "zh"
    assert LANGUAGE_RULES["zh"] in chat.answer_prompt()
    assert "《灯塔守护人》" in chat.calls[0]["messages"][-1].content  # the planner sees the original question


@pytestmark_db
def test_listing_and_archive_questions_need_no_answer_model(kb) -> None:
    """Film listings and archive statistics are templated answers; only the planner is called.

    Args:
        kb: Ingestion context whose database holds the five crawled fake films and their chat chunks.
    """
    chat = FakeChat([{"intent": "list_films", "genre": "Drama"}, {"intent": "archive_stats"}])
    engine = _engine(kb, chat)
    listing = engine.ask("列出所有剧情片", ChatSession())
    stats = engine.ask("What is in the archive?", ChatSession())
    assert listing.templated and "部" in listing.answer and LIGHTHOUSE in listing.answer
    assert stats.templated and "5 films" in stats.answer
    assert [c["role"] for c in chat.calls] == ["planner", "planner"]


@pytestmark_db
def test_full_screenplay_requests_are_declined(kb) -> None:
    """A request for a whole screenplay is declined with a templated answer pointing to imsdb.com.

    Args:
        kb: Ingestion context whose database holds the five crawled fake films and their chat chunks.
    """
    chat = FakeChat([{"intent": "full_script", "films": [LIGHTHOUSE]}])
    result = _engine(kb, chat).ask("Print the whole screenplay of The Lighthouse Keeper", ChatSession())
    assert result.templated and "can't output a full screenplay" in result.answer and "imsdb.com" in result.answer


@pytestmark_db
def test_unknown_film_is_reported_without_borrowing_other_scenes(kb) -> None:
    """A film not in the archive yields no sources and the prompt says so, instead of using other films' scenes.

    Args:
        kb: Ingestion context whose database holds the five crawled fake films and their chat chunks.
    """
    chat = FakeChat([{"films": ["Nonexistent Picture"], "search_query": "storm lighthouse"}])
    result = _engine(kb, chat).ask("What happens in Nonexistent Picture?", ChatSession())
    assert result.sources == []
    assert "Not in the archive: Nonexistent Picture" in chat.answer_prompt()


@pytestmark_db
def test_film_without_stored_screenplay_explains_why(kb) -> None:
    """A film whose IMSDb page has no screenplay yields no sources and the prompt explains why.

    Args:
        kb: Ingestion context whose database holds the five crawled fake films and their chat chunks.
    """
    chat = FakeChat([{"films": ["Night Shift at Dawn"], "search_query": "night shift"}])
    result = _engine(kb, chat).ask("How does Night Shift at Dawn end?", ChatSession())
    assert result.sources == []
    assert "does not offer this screenplay" in chat.answer_prompt()


@pytestmark_db
def test_follow_up_questions_carry_the_previous_film(kb) -> None:
    """A follow-up that names no film reuses the previous film and includes the earlier turn as history.

    Args:
        kb: Ingestion context whose database holds the five crawled fake films and their chat chunks.
    """
    # The planner names no film in the follow-up; the engine carries the previous one over.
    chat = FakeChat([{"films": [LIGHTHOUSE], "search_query": "storm"}, {"films": [], "search_query": "knock"}])
    engine = _engine(kb, chat)
    session = ChatSession()
    engine.ask("How does The Lighthouse Keeper open?", session)
    second = engine.ask("And who knocks?", session)
    assert second.plan.carried_over and second.plan.films == [LIGHTHOUSE]
    second_plan_input = [c for c in chat.calls if c["role"] == "planner"][1]["messages"][-1].content
    assert "Previous question (for context only): How does The Lighthouse Keeper open?" in second_plan_input
    answer_messages = [c for c in chat.calls if c["role"] == "answer"][1]["messages"]
    assert [m.role for m in answer_messages[:3]] == ["system", "user", "assistant"]  # history included
    assert "Film: The Lighthouse Keeper" in chat.answer_prompt()


@pytestmark_db
def test_naming_a_new_film_replaces_the_previous_one(kb) -> None:
    """A follow-up that names a different film does not carry the previous film over.

    Args:
        kb: Ingestion context whose database holds the five crawled fake films and their chat chunks.
    """
    chat = FakeChat([{"films": [LIGHTHOUSE], "search_query": "storm"},
                     {"intent": "film_details", "films": ["Paper Moons"]}])
    engine = _engine(kb, chat)
    session = ChatSession()
    engine.ask("How does The Lighthouse Keeper open?", session)
    second = engine.ask("Who wrote Paper Moons?", session)
    assert not second.plan.carried_over
    prompt = chat.answer_prompt()
    assert "Film: Paper Moons" in prompt and "Film: The Lighthouse Keeper" not in prompt


@pytestmark_db
def test_works_without_the_reranker(kb) -> None:
    """Without an embedding model the engine still answers from keyword-search passages.

    Args:
        kb: Ingestion context whose database holds the five crawled fake films and their chat chunks.
    """
    chat = FakeChat([{"films": [LIGHTHOUSE], "search_query": "storm lighthouse lens"}])
    result = _engine(kb, chat, embedder=False).ask("Storm at the lighthouse?", ChatSession())
    assert result.sources


# ====================================================================== web server
def _request(port: int, method: str, path: str, body: dict[str, Any] | None = None,
             headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
    """Send one HTTP request to the local test server and read the whole response.

    Args:
        port: Server port on 127.0.0.1.
        method: HTTP method, e.g. ``"GET"``.
        path: Request path including any query string.
        body: JSON body to send (sets ``Content-Type: application/json``), or ``None``.
        headers: Extra or overriding headers (the default ``Host`` is ``127.0.0.1:<port>``).

    Returns:
        ``(status, headers, body)`` of the response.
    """
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    data = json.dumps(body).encode() if body is not None else None
    hdrs = {"Host": f"127.0.0.1:{port}", **({"Content-Type": "application/json"} if data else {}), **(headers or {})}
    conn.request(method, path, body=data, headers=hdrs)
    resp = conn.getresponse()
    payload = resp.read()
    conn.close()
    return resp.status, dict(resp.getheaders()), payload


@pytestmark_db
def test_web_server_streams_answers_and_rejects_foreign_requests(kb) -> None:
    """The UI loads nothing remote, ``/api/chat`` streams SSE events; foreign Host/Origin or empty input is refused.

    Args:
        kb: Ingestion context whose database holds the five crawled fake films and their chat chunks.
    """
    chat = FakeChat([{"films": [LIGHTHOUSE], "search_query": "storm lighthouse"}])
    holder = EngineHolder(lambda: _engine(kb, chat))
    server = make_server("127.0.0.1", 0, holder)
    port = server.server_address[1]
    holder.start()
    holder.ready.wait(10)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, headers, body = _request(port, "GET", "/")
        assert status == 200 and b"ScriptBot" in body and "default-src 'none'" in headers["Content-Security-Policy"]
        assert not re.search(rb"""(src|href)=["']?(https?:)?//""", body)  # the page loads nothing remote
        status, _, body = _request(port, "GET", "/api/health")
        assert status == 200 and json.loads(body)["state"] == "ready"

        status, headers, body = _request(port, "POST", "/api/chat", {"question": "Storm?", "session_id": ""})
        assert status == 200 and headers["Content-Type"].startswith("text/event-stream")
        events = re.findall(r"event: (\w+)\ndata: (.*)\n\n", body.decode("utf-8"))
        names = [e for e, _ in events]
        assert names[0] == "session" and names[-1] == "done" and "sources" in names and "token" in names
        chunk_id = json.loads(dict(events)["sources"])["sources"][0]["chunk_id"]
        status, _, body = _request(port, "GET", "/api/passage?id=" + chunk_id)
        assert status == 200 and json.loads(body)["film"] == LIGHTHOUSE

        assert _request(port, "GET", "/", headers={"Host": "evil.example:80"})[0] == 403
        assert _request(port, "POST", "/api/chat", {"question": "x"},
                        headers={"Origin": "http://evil.example"})[0] == 403
        assert _request(port, "POST", "/api/chat", {"question": ""})[0] == 400
    finally:
        server.shutdown()
        server.server_close()


# ====================================================================== model download
def _spec(data: bytes) -> model_store.ModelSpec:
    """Describe a fake downloadable model whose size and SHA-256 match ``data``.

    Args:
        data: The model file's expected bytes.

    Returns:
        A ``ModelSpec`` for ``org/tiny`` at revision ``abc`` named ``tiny.gguf``.
    """
    return model_store.ModelSpec(key="tiny", role="chat", repo="org/tiny", revision="abc", filename="tiny.gguf",
                                 size=len(data), sha256=hashlib.sha256(data).hexdigest(), license="Apache-2.0")


def _range_transport(data: bytes, requests: list[httpx.Request]) -> httpx.MockTransport:
    """Build a fake download server that honours ``Range: bytes=N-`` requests.

    Args:
        data: The bytes to serve.
        requests: List that receives every request made (for assertions).

    Returns:
        An httpx ``MockTransport`` serving ``data`` (206 with the tail for range requests, else 200).
    """
    def handler(request: httpx.Request) -> httpx.Response:
        """Record the request and serve the full body or, for a range request, the bytes from the offset on.

        Args:
            request: The download request.

        Returns:
            A 200 response with all of ``data`` or a 206 response with ``data[start:]``.
        """
        requests.append(request)
        rng = request.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].rstrip("-"))
            return httpx.Response(206, content=data[start:])
        return httpx.Response(200, content=data)
    return httpx.MockTransport(handler)


def test_model_fetch_resumes_and_verifies(tmp_path: Path) -> None:
    """A partial ``.part`` download resumes via a Range request and is hash-verified; a second fetch downloads nothing.

    Args:
        tmp_path: pytest's per-test temporary folder (the models directory).
    """
    data = bytes(range(256)) * 400
    spec = _spec(data)
    (tmp_path / "tiny.gguf.part").write_bytes(data[:1000])  # an interrupted earlier download
    requests: list[httpx.Request] = []
    with httpx.Client(transport=_range_transport(data, requests)) as client:
        path = model_store.fetch_model(tmp_path, spec, client=client)
    assert path.read_bytes() == data and requests[0].headers["Range"] == "bytes=1000-"
    assert "org/tiny/resolve/abc/tiny.gguf" in str(requests[0].url)
    assert model_store.require_model(tmp_path, spec) == path
    with httpx.Client(transport=_range_transport(data, requests)) as client:
        assert model_store.fetch_model(tmp_path, spec, client=client) == path  # idempotent
    assert len(requests) == 1


def test_model_pieces_round_trip_without_network(tmp_path: Path) -> None:
    """A model split into pieces (as stored in the repository) is reassembled and verified on first use,
    and ``fetch_model`` then needs no download.

    Args:
        tmp_path: pytest's per-test temporary folder (the models directory).
    """
    data = bytes(range(256)) * 50
    spec = _spec(data)
    (tmp_path / "tiny.gguf").write_bytes(data)
    parts = model_store.split_model(tmp_path, spec, part_bytes=1000)
    assert len(parts) == 13 and all(p.stat().st_size <= 1000 for p in parts)
    (tmp_path / "tiny.gguf").unlink()  # a fresh clone has only the pieces
    assert model_store.require_model(tmp_path, spec).read_bytes() == data
    (tmp_path / "tiny.gguf").unlink()

    def no_network(request: httpx.Request) -> httpx.Response:
        """Fail the test if a download is attempted.

        Args:
            request: The unexpected request.

        Returns:
            Never returns.
        """
        raise AssertionError("must not download when pieces are present")

    with httpx.Client(transport=httpx.MockTransport(no_network)) as client:
        assert model_store.fetch_model(tmp_path, spec, client=client).read_bytes() == data


def test_corrupt_model_piece_is_rejected(tmp_path: Path) -> None:
    """Pieces whose joined content fails the SHA-256 check never produce a model file.

    Args:
        tmp_path: pytest's per-test temporary folder (the models directory).
    """
    data = b"model bytes " * 300
    spec = _spec(data)
    (tmp_path / "tiny.gguf").write_bytes(data)
    parts = model_store.split_model(tmp_path, spec, part_bytes=1000)
    (tmp_path / "tiny.gguf").unlink()
    parts[1].write_bytes(b"X" * parts[1].stat().st_size)
    with pytest.raises(model_store.ModelError, match="SHA-256"):
        model_store.require_model(tmp_path, spec)
    assert not (tmp_path / "tiny.gguf").exists() and not (tmp_path / "tiny.gguf.assembling").exists()


def test_plan_parsing_survives_wrong_shapes() -> None:
    """JSON that is not an object, or fields of the wrong type, fall back to defaults instead of crashing."""
    assert Plan.from_json("[1, 2]").intent == "scene_search"
    plan = Plan.from_json(json.dumps({"intent": "film_details", "films": "Alien", "year": "1979",
                                      "character": None, "search_query": 5}))
    assert plan.intent == "film_details" and plan.films == [] and plan.year == 0
    assert plan.character == "" and plan.search_query == ""


def test_queued_question_can_be_cancelled(kb) -> None:
    """A question waiting for a previous one stops as soon as it is cancelled (e.g. the browser disconnects).

    Args:
        kb: Context whose database holds the crawled films and chunks.
    """
    engine = _engine(kb, FakeChat())
    cancel = threading.Event()
    events: list[str] = []
    engine._busy.acquire()  # a previous question is still being answered
    try:
        timer = threading.Timer(0.3, cancel.set)
        timer.start()
        with pytest.raises(Cancelled):
            engine.ask("How does it end?", ChatSession(), emit=lambda e, d: events.append(d.get("stage", e)),
                       cancel=cancel)
        timer.join()
    finally:
        engine._busy.release()
    assert events == ["queued"]


def test_model_fetch_gives_up_when_no_data_arrives(tmp_path: Path) -> None:
    """A server that keeps ending the connection without data stops the download after a few attempts.

    Args:
        tmp_path: pytest's per-test temporary folder (the models directory).
    """
    spec = _spec(b"x" * 1000)
    (tmp_path / "tiny.gguf.part").write_bytes(b"x" * 10)
    requests: list[httpx.Request] = []

    def empty(request: httpx.Request) -> httpx.Response:
        """Answer every range request with an empty partial response.

        Args:
            request: The download request.

        Returns:
            An empty 206 response.
        """
        requests.append(request)
        return httpx.Response(206, content=b"")

    with httpx.Client(transport=httpx.MockTransport(empty)) as client:
        with pytest.raises(model_store.ModelError, match="stopped making progress"):
            model_store.fetch_model(tmp_path, spec, client=client)
    assert len(requests) == model_store.MAX_STALLED_ATTEMPTS
    assert (tmp_path / "tiny.gguf.part").exists()  # kept for a later resume


def test_model_fetch_rejects_corrupted_download(tmp_path: Path) -> None:
    """A download with the wrong SHA-256 raises ``ModelError`` and leaves no file or partial file behind.

    Args:
        tmp_path: pytest's per-test temporary folder (the models directory).
    """
    spec = _spec(b"good bytes" * 100)
    with httpx.Client(transport=_range_transport(b"evil bytes" * 100, [])) as client:
        with pytest.raises(model_store.ModelError, match="SHA-256"):
            model_store.fetch_model(tmp_path, spec, client=client)
    assert not (tmp_path / "tiny.gguf").exists() and not (tmp_path / "tiny.gguf.part").exists()
    with pytest.raises(model_store.ModelError, match="models fetch"):
        model_store.require_model(tmp_path, spec)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DLL version check")
def test_windows_dll_version_reader() -> None:
    """``file_version`` reads the version resource of a Windows system DLL (Windows only)."""
    from src.chatbot.runtime import file_version
    version = file_version(Path(r"C:\Windows\System32\kernel32.dll"))
    assert version is not None and version[0] >= 6


# ====================================================================== quote verification
def test_quotes_are_checked_against_the_passages() -> None:
    """Quotes of 3+ Latin-script words are extracted and matched by words (ignoring case/punctuation) to passages."""
    from src.chatbot.search import Passage
    from src.chatbot.verify import extract_quotes, unverified_quotes
    passage = Passage(chunk_id="c", script_id="s", film="F", year=None, scene_number=1, heading="", characters=[],
                      text="MARGARET\n   Forty years I kept that light.\n   Steady now. Steady.")
    answer = ('MARGARET says: "Forty years I kept that LIGHT" [1] and “Steady now, steady.” '
              'Later she says "I never liked the sea at all" [1]. 她说："你好" and "Ok".')
    assert extract_quotes(answer) == ["Forty years I kept that LIGHT", "Steady now, steady.",
                                      "I never liked the sea at all"]
    assert unverified_quotes(answer, [passage]) == ["I never liked the sea at all"]


@pytestmark_db
def test_engine_reports_invented_quotes(kb) -> None:
    """A quote in the answer that appears in no retrieved passage is reported as unverified.

    Args:
        kb: Ingestion context whose database holds the five crawled fake films and their chat chunks.
    """
    chat = FakeChat([{"films": [LIGHTHOUSE], "search_query": "storm lighthouse"}],
                    answer='Margaret says "I will burn the lighthouse down tonight" [1].')
    result = _engine(kb, chat).ask("What does Margaret threaten?", ChatSession())
    assert result.unverified_quotes == ["I will burn the lighthouse down tonight"]
