"""Loading the local inference runtime (llama-cpp-python) safely on this machine.

Windows note: llama-cpp-python's DLLs are built with a recent MSVC toolset. Older system copies of
``msvcp140.dll`` (before 14.40) crash such builds with an access violation inside ``llama_backend_init``.
Python does not ship ``msvcp140.dll``, so we pre-load a newer, Microsoft-signed copy from the project's
``runtime/win64`` folder (app-local deployment, nothing is installed system-wide). Once a DLL is loaded,
Windows resolves later imports of the same name to it.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger("imsdb.chatbot.runtime")

MIN_MSVCP_VERSION = (14, 40)
_lock = threading.Lock()
_prepared = False


class RuntimeUnavailable(RuntimeError):
    """The local inference runtime cannot be used (llama-cpp-python missing, C++ runtime too old, no device)."""

    pass


def file_version(path: Path) -> tuple[int, int, int, int] | None:
    """Return a Windows DLL's file version, or None if it cannot be read.

    Reads the ``VS_FIXEDFILEINFO`` block through the Win32 ``version.dll`` API.

    Args:
        path: Path to the DLL (or any PE file with a version resource).

    Returns:
        The file version as ``(major, minor, build, revision)``, or None when not on Windows, the file is
        missing or it has no readable version resource.
    """
    if sys.platform != "win32" or not path.exists():
        return None
    version = ctypes.WinDLL("version")
    size = version.GetFileVersionInfoSizeW(str(path), None)
    if not size:
        return None
    buf = ctypes.create_string_buffer(size)
    if not version.GetFileVersionInfoW(str(path), 0, size, buf):
        return None

    class VSFixedFileInfo(ctypes.Structure):
        """ctypes layout of the Win32 ``VS_FIXEDFILEINFO`` structure (13 unsigned 32-bit fields)."""

        _fields_ = [(name, ctypes.c_uint32) for name in (
            "signature", "struc_version", "file_version_ms", "file_version_ls", "product_version_ms",
            "product_version_ls", "file_flags_mask", "file_flags", "file_os", "file_type", "file_subtype",
            "file_date_ms", "file_date_ls")]

    ptr = ctypes.c_void_p()
    length = ctypes.c_uint()
    if not version.VerQueryValueW(buf, "\\", ctypes.byref(ptr), ctypes.byref(length)):
        return None
    info = ctypes.cast(ptr, ctypes.POINTER(VSFixedFileInfo)).contents
    return (info.file_version_ms >> 16, info.file_version_ms & 0xFFFF,
            info.file_version_ls >> 16, info.file_version_ls & 0xFFFF)


def prepare_windows_runtime(runtime_dir: Path | None) -> dict[str, Any]:
    """Pre-load a new-enough msvcp140.dll on Windows. No-op elsewhere. Safe to call repeatedly.

    The app-local copy is loaded only when the System32 copy is older than ``MIN_MSVCP_VERSION``. Guarded by a
    module lock; after the first success later calls just report ``status: "ready"``.

    Args:
        runtime_dir: Folder holding the app-local ``msvcp140.dll`` (``runtime/win64``), or None.

    Returns:
        A report dict: ``platform`` and, on Windows, ``system_msvcp140`` / ``local_msvcp140`` (dotted version
        strings or None) and ``status`` (what was decided).

    Raises:
        RuntimeUnavailable: Neither the system nor the app-local DLL is new enough.
    """
    global _prepared
    report: dict[str, Any] = {"platform": sys.platform}
    if sys.platform != "win32":
        return report
    with _lock:
        system_dll = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "msvcp140.dll"
        system_version = file_version(system_dll)
        report["system_msvcp140"] = ".".join(map(str, system_version)) if system_version else None
        local_dll = runtime_dir / "msvcp140.dll" if runtime_dir else None
        local_version = file_version(local_dll) if local_dll else None
        report["local_msvcp140"] = ".".join(map(str, local_version)) if local_version else None
        system_ok = system_version is not None and system_version[:2] >= MIN_MSVCP_VERSION
        if _prepared:
            report["status"] = "ready"
            return report
        if local_version is not None and local_version[:2] >= MIN_MSVCP_VERSION and not system_ok:
            ctypes.WinDLL(str(local_dll))
            report["status"] = "using app-local msvcp140.dll"
        elif system_ok:
            report["status"] = "system msvcp140.dll is new enough"
        else:
            raise RuntimeUnavailable(
                f"The Windows C++ runtime (msvcp140.dll {report['system_msvcp140']}) is too old for the local model "
                f"runtime (needs {MIN_MSVCP_VERSION[0]}.{MIN_MSVCP_VERSION[1]} or newer). Copy a newer, "
                f"Microsoft-signed msvcp140.dll into {runtime_dir} (see runtime/README.txt).")
        _prepared = True
        return report


def default_threads() -> int:
    """Physical-core estimate: inference is memory/FPU bound, hyper-threads rarely help.

    Returns:
        Half the logical CPU count when there are at least 4, else the logical count (minimum 1).
    """
    logical = os.cpu_count() or 2
    return max(1, logical // 2 if logical >= 4 else logical)


def import_llama_cpp(runtime_dir: Path | None) -> Any:
    """Prepare the Windows C++ runtime, then import llama-cpp-python lazily.

    Whichever ``llama_cpp`` build is first on ``sys.path`` is imported (see ``device.activate``).

    Args:
        runtime_dir: Folder with the app-local ``msvcp140.dll``, passed to ``prepare_windows_runtime``.

    Returns:
        The imported ``llama_cpp`` module.

    Raises:
        RuntimeUnavailable: llama-cpp-python is not installed, or the C++ runtime is too old.
    """
    prepare_windows_runtime(runtime_dir)
    try:
        import llama_cpp  # noqa: PLC0415 - imported lazily: heavy, and optional for the crawler
    except ImportError as exc:
        raise RuntimeUnavailable("llama-cpp-python is not installed. Run: python scripts/setup.py --chatbot") from exc
    return llama_cpp
