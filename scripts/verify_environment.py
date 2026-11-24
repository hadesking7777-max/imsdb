"""Verify the local environment: Python, packages, directories, MongoDB, database tools.

Optional checks (virtual environment, mongod bind address, MongoDB Database Tools, offline chatbot) only warn.

Usage:  python scripts/verify_environment.py
Exit code 0 when every required check passes, 1 otherwise.
"""

from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REQUIRED_PACKAGES = {"httpx": "httpx", "bs4": "beautifulsoup4", "lxml": "lxml", "pydantic": "pydantic",
                     "pymongo": "pymongo", "dotenv": "python-dotenv", "pytest": "pytest"}


def check(label: str, ok: bool, detail: str = "", required: bool = True) -> bool:
    """Print one result line, ``[OK  ]``, ``[FAIL]`` (required) or ``[WARN]`` (optional), with a detail.

    Args:
        label: What was checked.
        ok: Whether the check passed.
        detail: Extra information printed after the label (version, path, hint).
        required: Whether a failure should fail the whole run.

    Returns:
        ``True`` if the check passed or is optional, ``False`` for a failed required check.
    """
    mark = "OK  " if ok else ("FAIL" if required else "WARN")
    print(f"[{mark}] {label:<38} {detail}")
    return ok or not required


def check_chatbot(settings) -> list[bool]:  # noqa: ANN001 - Settings, imported lazily
    """Optional checks for the offline chatbot (warnings only: the crawler works without it).

    Checks the llama.cpp runtime, the compute device, the downloaded models and the chat search index.

    Args:
        settings: Loaded project ``Settings``.

    Returns:
        One result per check from ``check`` (always ``True``, because every check is optional).
    """
    print("\nOffline chatbot (optional):")
    results: list[bool] = []
    try:
        from src.chatbot.runtime import import_llama_cpp, prepare_windows_runtime
        report = prepare_windows_runtime(settings.msvc_runtime_dir)
        if report.get("status"):
            results.append(check("C++ runtime for llama.cpp", True, report["status"], required=False))
        llama_cpp = import_llama_cpp(settings.msvc_runtime_dir)
        results.append(check("package llama-cpp-python", True, llama_cpp.__version__, required=False))
    except Exception as exc:  # noqa: BLE001 - report, never crash the checker
        results.append(check("package llama-cpp-python", False, f"{exc} (python scripts/setup.py --chatbot)",
                             required=False))
    try:
        from src.chatbot.llm import choose_device
        choice = choose_device(settings)
        results.append(check("compute device", True, f"{choice.label()} - {choice.reason}", required=False))
    except Exception as exc:  # noqa: BLE001 - e.g. IMSDB_LLM_DEVICE=gpu without a usable GPU
        results.append(check("compute device", False, str(exc), required=False))
    try:
        from src.chatbot.model_store import MODELS, model_status
        for key in (settings.chat_model, settings.embed_model):
            if key not in MODELS:
                results.append(check(f"model {key}", False, f"unknown model key; use one of {', '.join(MODELS)}",
                                     required=False))
                continue
            st = model_status(settings.models_dir, MODELS[key])
            results.append(check(f"model {MODELS[key].filename}", bool(st.get("size_ok")),
                                 "present" if st.get("size_ok") else "python -m src.main models fetch",
                                 required=False))
    except Exception as exc:  # noqa: BLE001 - report, never crash the checker
        results.append(check("chatbot models", False, str(exc), required=False))
    try:
        from src.chatbot.chunks import chunk_index_status
        from src.database.connection import Mongo
        with Mongo.connect(settings) as mongo:
            st = chunk_index_status(mongo.db)
        results.append(check("chat search index", st["complete"],
                             f"{st['chunks']:,} passages" if st["ready"] else "python -m src.main chat-index",
                             required=False))
    except Exception as exc:  # noqa: BLE001
        results.append(check("chat search index", False, str(exc).splitlines()[0], required=False))
    return results


def main() -> int:
    """Run every check in order and print a summary.

    Returns:
        0 if all required checks passed, 1 otherwise (also 1 at once if the configuration cannot be loaded).
    """
    results: list[bool] = []
    version = sys.version_info
    results.append(check("Python >= 3.12", version >= (3, 12), sys.version.split()[0]))
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    results.append(check("Virtual environment active", in_venv, sys.prefix, required=False))

    for module, dist in REQUIRED_PACKAGES.items():
        try:
            mod = importlib.import_module(module)
            results.append(check(f"package {dist}", True, getattr(mod, "__version__", "installed")))
        except ImportError:
            results.append(check(f"package {dist}", False, "pip install -r requirements.txt"))

    try:
        from src.config import load_settings
        from src.errors import IMSDbError
        settings = load_settings()
        results.append(check("configuration (.env)", True, str(ROOT / ".env") if (ROOT / ".env").exists()
                             else "defaults (no .env file)"))
    except Exception as exc:  # noqa: BLE001
        results.append(check("configuration (.env)", False, str(exc)))
        return 1

    for path in (settings.raw_dir, settings.export_dir, settings.backup_dir, settings.log_dir):
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write-test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            results.append(check(f"writable {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}", True))
        except OSError as exc:
            results.append(check(f"writable {path}", False, str(exc)))

    free = shutil.disk_usage(settings.data_dir if settings.data_dir.exists() else ROOT).free
    results.append(check("free disk space >= 2 GB", free >= 2 * 1024**3, f"{free / 1024**3:.1f} GB free"))

    try:
        from src.database.connection import Mongo, assert_local_uri
        assert_local_uri(settings.mongodb_uri)
        results.append(check("MongoDB URI is local", True, settings.mongodb_uri))
        with Mongo.connect(settings) as mongo:
            results.append(check("MongoDB reachable", True, f"server {mongo.server_version}"))
            cmdline = mongo.client.admin.command("getCmdLineOpts").get("parsed", {})
            bind = cmdline.get("net", {}).get("bindIp", "127.0.0.1 (default)")
            local_bind = all(h.strip() in {"127.0.0.1", "localhost", "::1"} for h in str(bind).split(",")
                             if "default" not in str(bind))
            results.append(check("mongod bound to localhost", local_bind or "default" in str(bind), str(bind),
                                 required=False))
    except IMSDbError as exc:
        results.append(check("MongoDB reachable", False, str(exc).splitlines()[0]))
    except Exception as exc:  # noqa: BLE001 - e.g. getCmdLineOpts not permitted for this user
        results.append(check("MongoDB details", False, str(exc), required=False))

    from src.services.backup import find_tool
    for tool in ("mongodump", "mongorestore"):
        found = find_tool(tool, settings)
        results.append(check(f"{tool} available", found is not None,
                             str(found) if found else "optional: set MONGODB_TOOLS_DIR (python fallback is used)",
                             required=False))

    results.extend(check_chatbot(settings))
    ok = all(results)
    print("\nEnvironment " + ("ready." if ok else "has problems (see FAIL lines)."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
