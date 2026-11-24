"""Local model wrappers (llama-cpp-python, GPU or CPU). Nothing here touches the network.

Two chat-model contexts share the same weights file: one for the short JSON planning step and one for
writing answers. Each keeps its own KV cache, so the fixed instruction prefix of each prompt stays cached
between questions instead of being re-read every time (re-reading costs ~1 s per 40 tokens on this CPU).

Where the models run is decided by ``src/chatbot/device.py`` (GPU by default, CPU fallback). If a model
cannot be loaded on the GPU (for example, not enough GPU memory), it is loaded again on the CPU.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

from src.chatbot.device import DeviceChoice, activate, cpu_choice, select_device
from src.chatbot.model_store import MODELS, ModelSpec, model_path, require_model
from src.chatbot.runtime import default_threads, import_llama_cpp

log = logging.getLogger("imsdb.chatbot.llm")

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
PLANNER_CONTEXT = 3072  # tokens; the planner prompt plus its JSON answer
EMBED_CONTEXT = 1024  # tokens; re-ranking passages are cut to RERANK_CHARS before embedding
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


class Cancelled(Exception):
    """Raised inside generation when the caller asked to stop (e.g. the browser disconnected)."""


@dataclass
class Message:
    """One chat message in a prompt.

    Attributes:
        role: ``"system"``, ``"user"`` or ``"assistant"``.
        content: The message text.
    """

    role: str  # system | user | assistant
    content: str


def chatml(messages: list[Message]) -> str:
    """Qwen3 chat format with thinking disabled (an empty think block, as the official template does).

    Args:
        messages: The conversation, in order.

    Returns:
        The ChatML prompt text, ending with an open assistant turn that starts after an empty think block.
    """
    parts = [f"{IM_START}{m.role}\n{m.content}{IM_END}\n" for m in messages]
    return "".join(parts) + f"{IM_START}assistant\n<think>\n\n</think>\n\n"


class ChatModel(Protocol):
    """Interface of a chat model used by the engine (``LocalChatModel``, or a fake in tests)."""

    def generate(self, messages: list[Message], *, max_tokens: int, temperature: float = 0.3,
                 json_schema: dict[str, Any] | None = None, on_token: Callable[[str], None] | None = None,
                 cancel: threading.Event | None = None, role: str = "answer") -> str:
        """Generate a reply to a conversation.

        Args:
            messages: The prompt messages.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            json_schema: JSON schema the output must follow, or None for free text.
            on_token: Called with each generated text piece as it arrives (for streaming), or None.
            cancel: Event that, when set, stops generation by raising ``Cancelled``.
            role: Which context to use: ``"planner"`` or ``"answer"``.

        Returns:
            The complete reply text.
        """
        ...


class Embedder(Protocol):
    """Interface of a text embedding model used by the re-ranker (``LocalEmbedder``, or a fake in tests)."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts as vectors.

        Args:
            texts: The texts to embed.

        Returns:
            One vector per text, in the same order.
        """
        ...


class LocalChatModel:
    """Qwen3-8B (GGUF) with separate planner/answer contexts.

    Each context has its own lock, so a planning call and an answer can run at the same time, but two calls on
    the same context are serialized.
    """

    def __init__(self, path: Path, *, threads: int, context: int, runtime_dir: Path | None,
                 planner_context: int = PLANNER_CONTEXT, gpu_layers: int = 0) -> None:
        """Load the model twice (answer and planner contexts), retrying on the CPU if the GPU load fails.

        Sets ``gpu_layers`` (layers actually on the GPU), ``gpu_error`` (why the GPU load failed, or None) and
        ``load_seconds``.

        Args:
            path: The GGUF model file.
            threads: CPU threads for generation and prompt processing.
            context: Context size in tokens of the answer context.
            runtime_dir: Folder with the app-local ``msvcp140.dll`` (see ``runtime.py``).
            planner_context: Context size in tokens of the planner context.
            gpu_layers: Model layers to offload to the GPU (0 = CPU only, -1 = all).

        Raises:
            RuntimeUnavailable: llama-cpp-python cannot be imported.
        """
        llama_cpp = import_llama_cpp(runtime_dir)
        self._llama_cpp = llama_cpp
        started = time.time()
        self.gpu_layers = gpu_layers
        self.gpu_error: str | None = None
        try:
            self._contexts = self._load_contexts(path, threads, context, planner_context, gpu_layers)
        except Exception as exc:  # noqa: BLE001 - e.g. out of GPU memory: fall back to the CPU
            if gpu_layers == 0:
                raise
            self.gpu_error = f"{exc.__class__.__name__}: {exc}"
            log.warning("loading the chat model on the GPU failed (%s); loading it on the CPU", self.gpu_error)
            self.gpu_layers = 0
            self._contexts = self._load_contexts(path, threads, context, planner_context, 0)
        self._locks = {name: threading.Lock() for name in self._contexts}
        self._grammars: dict[str, Any] = {}
        self.load_seconds = round(time.time() - started, 1)
        log.info("chat model loaded in %.1fs (%d threads, gpu_layers=%d)", self.load_seconds, threads,
                 self.gpu_layers)

    def _load_contexts(self, path: Path, threads: int, context: int, planner_context: int,
                       gpu_layers: int) -> dict[str, Any]:
        """Create the two ``llama_cpp.Llama`` instances over the same weights file.

        Args:
            path: The GGUF model file.
            threads: CPU threads.
            context: Answer context size in tokens.
            planner_context: Planner context size in tokens.
            gpu_layers: Layers to offload to the GPU.

        Returns:
            ``{"answer": Llama, "planner": Llama}``.
        """
        common = dict(n_threads=threads, n_threads_batch=threads, n_batch=512, verbose=False, flash_attn=True,
                      n_gpu_layers=gpu_layers)
        return {
            "answer": self._llama_cpp.Llama(str(path), n_ctx=context, **common),
            "planner": self._llama_cpp.Llama(str(path), n_ctx=planner_context, **common),
        }

    def count_tokens(self, text: str) -> int:
        """Count tokens with the model's tokenizer (special tokens such as ``<|im_start|>`` count as one).

        Args:
            text: Text to tokenize.

        Returns:
            The number of tokens, without a BOS token.
        """
        return len(self._contexts["answer"].tokenize(text.encode("utf-8"), add_bos=False, special=True))

    def _grammar(self, schema: dict[str, Any]) -> Any:
        """Return a GBNF grammar for a JSON schema, compiled once and cached by the schema's JSON text.

        Args:
            schema: JSON schema of the required output.

        Returns:
            The cached ``llama_cpp.LlamaGrammar``.
        """
        key = json.dumps(schema, sort_keys=True)
        if key not in self._grammars:
            self._grammars[key] = self._llama_cpp.LlamaGrammar.from_json_schema(key, verbose=False)
        return self._grammars[key]

    def generate(self, messages: list[Message], *, max_tokens: int, temperature: float = 0.3,
                 json_schema: dict[str, Any] | None = None, on_token: Callable[[str], None] | None = None,
                 cancel: threading.Event | None = None, role: str = "answer") -> str:
        """Generate a reply on one context, streaming pieces to ``on_token``.

        Sampling uses Qwen3's non-thinking settings (top_p 0.8, top_k 20) and stops at ``<|im_end|>``. Holds the
        context's lock for the whole generation. ``cancel`` is checked before each streamed piece, so it takes
        effect once prompt processing is over; the stream is closed in every case.

        Args:
            messages: The prompt messages (formatted with ``chatml``).
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            json_schema: JSON schema enforced with a grammar, or None for free text.
            on_token: Called with each non-empty generated piece, or None.
            cancel: Event that stops generation when set, or None.
            role: Context to use: ``"planner"`` or ``"answer"``.

        Returns:
            The full reply with any ``<think>...</think>`` block removed, stripped.

        Raises:
            Cancelled: ``cancel`` was set during generation.
        """
        llm = self._contexts[role]
        prompt = chatml(messages)
        kwargs: dict[str, Any] = dict(max_tokens=max_tokens, temperature=temperature, top_p=0.8, top_k=20,
                                      repeat_penalty=1.05, stop=[IM_END, "<|endoftext|>"], stream=True)
        if json_schema is not None:
            kwargs["grammar"] = self._grammar(json_schema)
        out: list[str] = []
        with self._locks[role]:
            stream: Iterator[dict[str, Any]] = llm.create_completion(prompt, **kwargs)  # type: ignore[assignment]
            try:
                for chunk in stream:
                    if cancel is not None and cancel.is_set():
                        raise Cancelled()
                    piece = chunk["choices"][0]["text"]
                    out.append(piece)
                    if on_token and piece:
                        on_token(piece)
            finally:
                close = getattr(stream, "close", None)
                if close:
                    close()
        return _THINK_RE.sub("", "".join(out)).strip()


class LocalEmbedder:
    """Qwen3-Embedding-0.6B (GGUF, last-token pooling, EOS appended by the tokenizer).

    Attributes:
        QUERY_INSTRUCTION: Prefix put before a question so the model embeds it as a retrieval query.
    """

    QUERY_INSTRUCTION = ("Instruct: Given a question about a film screenplay, retrieve screenplay passages "
                         "that answer it\nQuery: ")

    def __init__(self, path: Path, *, threads: int, runtime_dir: Path | None, context: int = EMBED_CONTEXT,
                 gpu_layers: int = 0) -> None:
        """Load the embedding model, retrying on the CPU if the GPU load fails.

        Args:
            path: The GGUF model file.
            threads: CPU threads.
            runtime_dir: Folder with the app-local ``msvcp140.dll``.
            context: Context (and batch) size in tokens; longer texts are truncated.
            gpu_layers: Layers to offload to the GPU (0 = CPU only, -1 = all).

        Raises:
            RuntimeUnavailable: llama-cpp-python cannot be imported.
        """
        llama_cpp = import_llama_cpp(runtime_dir)
        options = dict(embedding=True, n_ctx=context, n_batch=context, n_ubatch=context, n_threads=threads,
                       n_threads_batch=threads, flash_attn=True, verbose=False)
        self.gpu_layers = gpu_layers
        try:
            self._llm = llama_cpp.Llama(str(path), n_gpu_layers=gpu_layers, **options)
        except Exception as exc:  # noqa: BLE001 - fall back to the CPU
            if gpu_layers == 0:
                raise
            log.warning("loading the embedding model on the GPU failed (%s); loading it on the CPU", exc)
            self.gpu_layers = 0
            self._llm = llama_cpp.Llama(str(path), n_gpu_layers=0, **options)
        self._lock = threading.Lock()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in one batch (serialized by a lock).

        Args:
            texts: The texts; each is truncated to the context size.

        Returns:
            One L2-normalized vector per text, in the same order.
        """
        with self._lock:
            return [list(v) for v in self._llm.embed(texts, normalize=True, truncate=True)]


@dataclass
class LocalModels:
    """The loaded models and a description of how they were loaded.

    Attributes:
        chat: The chat model.
        embedder: The embedding model, or None when loaded without one.
        info: ``chat_model``, ``threads``, ``context``, ``chat_load_seconds``, ``embed_model`` (if loaded) and
            ``device`` (``DeviceChoice.as_dict()``).
    """

    chat: ChatModel
    embedder: Embedder | None
    info: dict[str, Any]


def choose_device(settings: Any, *, with_embedder: bool = True) -> DeviceChoice:
    """Decide GPU or CPU for the configured models, estimating the GPU memory they need.

    The memory estimate is the chat model's KV cache for both contexts plus, if wanted, the embedding model
    and its KV cache. The embedding model file, if complete, is used for the GPU self-test.

    Args:
        settings: Application settings (``chat_model``, ``embed_model``, ``models_dir``, ``llm_context`` and
            the device options read by ``select_device``).
        with_embedder: Whether the embedding model will be loaded too.

    Returns:
        The device decision from ``device.select_device``.
    """
    chat_spec: ModelSpec = MODELS[settings.chat_model]
    embed_spec: ModelSpec = MODELS[settings.embed_model]
    embed_file = model_path(settings.models_dir, embed_spec)
    test_model = embed_file if embed_file.exists() and embed_file.stat().st_size == embed_spec.size else None
    kv_cache = chat_spec.kv_bytes_per_token * (settings.llm_context + PLANNER_CONTEXT)
    embedder = embed_spec.size + embed_spec.kv_bytes_per_token * EMBED_CONTEXT if with_embedder else 0
    return select_device(settings, test_model=test_model, chat_bytes=chat_spec.size,
                         chat_layers=chat_spec.n_layers, extra_gpu_bytes=kv_cache + embedder)


def load_local_models(settings: Any, *, with_embedder: bool = True) -> LocalModels:
    """Choose the device, activate the matching llama-cpp build and load the chat (and embedding) model.

    If the chat model had to fall back to the CPU, the embedding model is put on the CPU too. Slow: loading
    takes seconds to minutes.

    Args:
        settings: Application settings (``chat_model``, ``embed_model``, ``models_dir``, ``llm_threads``,
            ``llm_context``, ``msvc_runtime_dir`` and device options).
        with_embedder: Also load the embedding model used for re-ranking.

    Returns:
        The loaded models with their load info.

    Raises:
        ModelError: A model file is missing or incomplete.
        RuntimeUnavailable: llama-cpp-python cannot be used.
    """
    chat_spec: ModelSpec = MODELS[settings.chat_model]
    threads = settings.llm_threads or default_threads()
    chat_path = require_model(settings.models_dir, chat_spec)
    choice = activate(choose_device(settings, with_embedder=with_embedder))
    log.info("compute device: %s (%s)", choice.label(), choice.reason)
    chat = LocalChatModel(chat_path, threads=threads, context=settings.llm_context,
                          runtime_dir=settings.msvc_runtime_dir, gpu_layers=choice.gpu_layers)
    if chat.gpu_error:
        choice = cpu_choice(f"the chat model could not be loaded on the GPU ({chat.gpu_error}); using the CPU",
                            choice.attempts)
    embedder = None
    info: dict[str, Any] = {"chat_model": chat_spec.filename, "threads": threads, "context": settings.llm_context,
                            "chat_load_seconds": chat.load_seconds}
    if with_embedder:
        embed_spec = MODELS[settings.embed_model]
        embedder = LocalEmbedder(require_model(settings.models_dir, embed_spec), threads=threads,
                                 runtime_dir=settings.msvc_runtime_dir, gpu_layers=-1 if choice.on_gpu else 0)
        info["embed_model"] = embed_spec.filename
    info["device"] = choice.as_dict()
    return LocalModels(chat=chat, embedder=embedder, info=info)
