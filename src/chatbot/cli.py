"""CLI commands for the offline chatbot (registered in ``src.main``)."""

from __future__ import annotations

import argparse
import json
import sys
import time
import webbrowser
from typing import Any

from src.errors import ConfigError, ExitCode


def _utf8_console() -> None:
    """Switch stdout/stderr to UTF-8 (unencodable characters replaced) so Chinese text prints on Windows."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")


def _out(text: str = "", end: str = "\n") -> None:
    """Write text to stdout and flush immediately (for streamed tokens and progress lines).

    Args:
        text: Text to write.
        end: Appended after the text.
    """
    sys.stdout.write(text + end)
    sys.stdout.flush()


def _engine(rt: Any, args: argparse.Namespace) -> Any:
    """Build a ready ``ChatEngine``: check the chunk index, load the models and the film catalogue.

    Args:
        rt: Application runtime from ``src.main`` (settings, lazily connected ``mongo``, ``stop_event``).
        args: Parsed arguments; ``no_rerank`` (if present) skips loading the embedder and re-ranking.

    Returns:
        The chat engine.

    Raises:
        ConfigError: The ``chat_chunks`` index is not built, a model file is missing, or the local runtime
            cannot be used.
    """
    from src.chatbot.catalog import FilmCatalog
    from src.chatbot.chunks import chunk_index_status
    from src.chatbot.engine import ChatEngine
    from src.chatbot.llm import load_local_models
    from src.chatbot.model_store import ModelError
    from src.chatbot.runtime import RuntimeUnavailable

    db = rt.mongo.db
    status = chunk_index_status(db)
    if not status["ready"]:
        raise ConfigError("The chat search index has not been built yet. Run: python -m src.main chat-index")
    try:
        models = load_local_models(rt.settings, with_embedder=not getattr(args, "no_rerank", False))
    except (ModelError, RuntimeUnavailable) as exc:
        raise ConfigError(str(exc)) from exc
    return ChatEngine(db, FilmCatalog.load(db), models, use_reranker=not getattr(args, "no_rerank", False))


# ---------------------------------------------------------------------- models
def cmd_models(rt: Any, args: argparse.Namespace) -> int:
    """``models fetch|status``: download or check the configured chat and embedding model files.

    ``fetch`` downloads (or resumes) each file with progress output every 2 s and verifies its SHA-256; this
    is the only command that uses the network. ``status`` prints one line per model.

    Args:
        rt: Application runtime from ``src.main`` (settings, lazily connected ``mongo``, ``stop_event``).
        args: Parsed arguments: ``action`` (``"fetch"`` or ``"status"``) and ``verify`` (with status, also check
            SHA-256).

    Returns:
        Exit code: OK, or USAGE when ``status`` finds a missing or incomplete file.

    Raises:
        ConfigError: A download failed or a checksum did not match.
    """
    from src.chatbot.model_store import MODELS, ModelError, fetch_model, model_status

    _utf8_console()
    wanted = [MODELS[rt.settings.chat_model], MODELS[rt.settings.embed_model]]
    if args.action == "status":
        ok = True
        for spec in wanted:
            st = model_status(rt.settings.models_dir, spec, verify_hash=args.verify)
            good = st.get("size_ok") and st.get("sha256_ok", True)
            ok &= bool(good)
            _out(f"{spec.role:<10} {spec.filename:<36} {'OK' if good else 'MISSING/INCOMPLETE':<20} {spec.license}")
        return ExitCode.OK if ok else ExitCode.USAGE
    for spec in wanted:
        last = [0.0]

        def progress(have: int, total: int) -> None:
            """Print download progress on one line, at most every 2 seconds and at completion.

            Args:
                have: Bytes downloaded so far.
                total: Expected file size in bytes.
            """
            if time.time() - last[0] >= 2 or have == total:
                last[0] = time.time()
                _out(f"\r  {spec.filename}: {have / 1e9:5.2f} / {total / 1e9:.2f} GB", end="")

        _out(f"Fetching {spec.filename} ({spec.size / 1e9:.2f} GB, {spec.license}) from {spec.repo}")
        try:
            path = fetch_model(rt.settings.models_dir, spec, stop_event=rt.stop_event, progress=progress)
        except ModelError as exc:
            raise ConfigError(str(exc)) from exc
        _out(f"\n  verified (SHA-256) -> {path}")
    _out("All models present. The chatbot now works without internet access.")
    return ExitCode.OK


# ---------------------------------------------------------------------- device
def cmd_device(rt: Any, args: argparse.Namespace) -> int:
    """``device``: show whether the models would run on a GPU or the CPU, and why (runs the GPU probes).

    Args:
        rt: Application runtime from ``src.main`` (settings, lazily connected ``mongo``, ``stop_event``).
        args: Parsed arguments: ``json`` (print the decision as JSON instead of text).

    Returns:
        Exit code OK.

    Raises:
        ConfigError: The local runtime cannot be used at all.
    """
    from src.chatbot.llm import choose_device
    from src.chatbot.runtime import RuntimeUnavailable

    _utf8_console()
    try:
        choice = choose_device(rt.settings)
    except RuntimeUnavailable as exc:
        raise ConfigError(str(exc)) from exc
    if args.json:
        _out(json.dumps(choice.as_dict(), ensure_ascii=False, indent=2))
        return ExitCode.OK
    _out(f"Setting:  IMSDB_LLM_DEVICE={rt.settings.llm_device}  (auto = GPU if usable, else CPU)")
    _out(f"Decision: {choice.label()}")
    _out(f"Reason:   {choice.reason}")
    for attempt in choice.attempts:
        _out(f"  tested {attempt['backend']:<7} {'passed' if attempt['ok'] else 'failed':<7} "
             f"{attempt.get('seconds') or 0:>5.1f}s  {attempt['detail']}")
    for gpu in choice.gpus:
        _out(f"  GPU {gpu.name}: {gpu.description}, {gpu.free_mib:,} of {gpu.total_mib:,} MiB free")
    return ExitCode.OK


# ---------------------------------------------------------------------- index
def cmd_chat_index(rt: Any, args: argparse.Namespace) -> int:
    """``chat-index``: build or update the ``chat_chunks`` passage index, then print counts and index status.

    Ctrl+C stops cleanly between scripts (signal handlers set ``rt.stop_event``).

    Args:
        rt: Application runtime from ``src.main`` (settings, lazily connected ``mongo``, ``stop_event``).
        args: Parsed arguments: ``rebuild`` (drop and rebuild the whole index).

    Returns:
        Exit code: OK, or INTERRUPTED when stopped early.
    """
    from src.chatbot.chunks import build_chunks, chunk_index_status

    rt.install_signal_handlers()
    started = time.time()

    def progress(n: int, total: int, film: str) -> None:
        """Print a progress line every 100 re-chunked scripts and at the end.

        Args:
            n: Position of the script just processed (1-based, among all parsed scripts).
            total: Number of parsed scripts.
            film: Film title of the script (unused).
        """
        if n % 100 == 0 or n == total:
            _out(f"  {n}/{total} scripts ({time.time() - started:.0f}s)")

    counts = build_chunks(rt.mongo.db, rebuild=args.rebuild, stop_event=rt.stop_event, progress=progress)
    _out(json.dumps({**counts, **chunk_index_status(rt.mongo.db)}, indent=2))
    return ExitCode.INTERRUPTED if counts.get("interrupted") else ExitCode.OK


# ---------------------------------------------------------------------- ask / chat
def _print_result(result: Any, show_sources: bool = True) -> None:
    """Print the warnings, sources and timings that follow a streamed answer.

    Args:
        result: The engine's ``ChatResult``.
        show_sources: Also list the cited passages.
    """
    for quote in getattr(result, "unverified_quotes", []):
        _out(f'\n  WARNING: "{quote}" was not found word-for-word in the cited passages; it may be inaccurate.')
    if show_sources and result.sources:
        _out("\nSources:")
        for s in result.sources:
            _out(f"  [{s['n']}] {s['label']}")
    if result.timings:
        _out(f"({', '.join(f'{k} {v}s' for k, v in result.timings.items())})")


def cmd_ask(rt: Any, args: argparse.Namespace) -> int:
    """``ask``: answer one question, streaming the answer to the terminal (or printing JSON).

    Args:
        rt: Application runtime from ``src.main`` (settings, lazily connected ``mongo``, ``stop_event``).
        args: Parsed arguments: ``question``, ``json`` (print answer, language, plan, sources, timings and
            unverified quotes as JSON, without streaming) and ``no_rerank``.

    Returns:
        Exit code OK.

    Raises:
        ConfigError: The index or the models are not available.
    """
    from src.chatbot.engine import ChatSession

    _utf8_console()
    engine = _engine(rt, args)
    stream = not args.json
    result = engine.ask(args.question, ChatSession(),
                        (lambda e, d: _out(d["text"], end="") if e == "token" else None) if stream else None)
    if args.json:
        _out(json.dumps({"question": args.question, "answer": result.answer, "language": result.language,
                         "plan": result.plan.__dict__, "sources": result.sources, "timings": result.timings,
                         "unverified_quotes": result.unverified_quotes},
                        ensure_ascii=False, indent=2))
    else:
        _out()
        _print_result(result)
    return ExitCode.OK


STAGE_TEXT = {"planning": "understanding the question", "searching": "searching", "writing": "writing",
              "queued": "waiting"}


def cmd_chat(rt: Any, args: argparse.Namespace) -> int:
    """``chat``: interactive terminal chat with conversation history.

    Commands: ``/new`` starts a new conversation, ``/quit`` (or ``/exit``, ``/q``, EOF, Ctrl+C at the prompt)
    exits. Ctrl+C during an answer stops it and returns to the prompt.

    Args:
        rt: Application runtime from ``src.main`` (settings, lazily connected ``mongo``, ``stop_event``).
        args: Parsed arguments: ``no_rerank``.

    Returns:
        Exit code OK when the user quits.

    Raises:
        ConfigError: The index or the models are not available.
    """
    import threading

    from src.chatbot.engine import ChatSession
    from src.chatbot.llm import Cancelled

    _utf8_console()
    _out("Loading local models…")
    engine = _engine(rt, args)
    session = ChatSession()
    _out(f"ScriptBot ready ({len(engine.catalog.films)} films). Ask in English or Chinese. "
         "Commands: /new (new conversation), /quit. Ctrl+C stops an answer.\n")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            _out()
            return ExitCode.OK
        if not question:
            continue
        if question in {"/quit", "/exit", "/q"}:
            return ExitCode.OK
        if question == "/new":
            session = ChatSession()
            _out("(new conversation)\n")
            continue
        cancel = threading.Event()
        started = [False]

        def emit(event: str, data: dict[str, Any]) -> None:
            """Print engine events: a line per ``status`` stage and ``token`` text after a ``bot>`` prefix.

            Args:
                event: Engine event name; events other than ``status`` and ``token`` are ignored.
                data: Event data (``stage`` or ``text``).
            """
            if event == "status":
                _out(f"  … {STAGE_TEXT.get(data['stage'], data['stage'])}")
            elif event == "token":
                if not started[0]:
                    started[0] = True
                    _out("bot> ", end="")
                _out(data["text"], end="")

        try:
            result = engine.ask(question, session, emit, cancel)
            _out()
            _print_result(result)
            _out()
        except (KeyboardInterrupt, Cancelled):
            cancel.set()
            _out("\n(stopped)\n")


def cmd_serve(rt: Any, args: argparse.Namespace) -> int:
    """``serve``: run the local web chat until Ctrl+C.

    Connects to MongoDB and checks the chunk index first, binds the server on ``settings.chat_host``, then
    loads the models in a background thread while the page is already served.

    Args:
        rt: Application runtime from ``src.main`` (settings, lazily connected ``mongo``, ``stop_event``).
        args: Parsed arguments: ``port`` (overrides ``settings.chat_port``), ``open`` (open the page in the
            browser) and ``no_rerank``.

    Returns:
        Exit code OK after the server stops.

    Raises:
        ConfigError: The index is not built or the port cannot be bound.
    """
    from src.chatbot.server import EngineHolder, make_server

    _utf8_console()
    host, port = rt.settings.chat_host, args.port or rt.settings.chat_port
    db = rt.mongo.db  # connect (and fail fast) before starting the web server
    from src.chatbot.chunks import chunk_index_status
    if not chunk_index_status(db)["ready"]:
        raise ConfigError("The chat search index has not been built yet. Run: python -m src.main chat-index")
    holder = EngineHolder(lambda: _engine(rt, args))
    try:
        server = make_server(host, port, holder)
    except OSError as exc:
        raise ConfigError(f"Cannot listen on {host}:{port} ({exc}). Is ScriptBot already running? "
                          f"Use --port to choose another port.") from exc
    holder.start()
    url = f"http://{'localhost' if host in {'127.0.0.1', 'localhost'} else host}:{port}/"
    _out(f"ScriptBot is running at {url}  (this computer only; press Ctrl+C to stop)")
    _out("Models are loading in the background; the page shows when they are ready.")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        _out("\nStopping.")
    finally:
        server.server_close()
    return ExitCode.OK


def register(add: Any) -> None:
    """Register the chatbot subcommands and their options.

    Args:
        add: ``add(name, func, help_text)`` from ``src.main``; creates a subparser bound to ``func`` and returns
            it.
    """
    p = add("models", cmd_models, "download (once) or check the local chatbot model files")
    p.add_argument("action", choices=["fetch", "status"])
    p.add_argument("--verify", action="store_true", help="with status: also verify SHA-256 checksums")

    p = add("device", cmd_device, "show whether the chatbot will run on a GPU or the CPU, and why")
    p.add_argument("--json", action="store_true", help="print the decision as JSON")

    p = add("chat-index", cmd_chat_index, "build or update the chatbot's passage search index (no network)")
    p.add_argument("--rebuild", action="store_true", help="drop and rebuild the whole index")

    p = add("ask", cmd_ask, "ask the offline chatbot one question (English or Chinese)")
    p.add_argument("question")
    p.add_argument("--json", action="store_true", help="print the answer, plan, sources and timings as JSON")
    p.add_argument("--no-rerank", action="store_true", help="skip semantic re-ranking (faster, less precise)")

    p = add("chat", cmd_chat, "interactive chatbot in the terminal")
    p.add_argument("--no-rerank", action="store_true")

    p = add("serve", cmd_serve, "start the chatbot web page on this computer (http://localhost:8765)")
    p.add_argument("--port", type=int, help="port (default 8765, or IMSDB_CHAT_PORT)")
    p.add_argument("--open", action="store_true", help="open the page in the default browser")
    p.add_argument("--no-rerank", action="store_true")
