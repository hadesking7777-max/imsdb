"""Choosing where the models run: on a GPU by default, on the CPU when no GPU is usable.

How it works
------------
llama-cpp-python is installed twice, as two builds of the same package:

* the **CPU build** in the virtual environment (``.venv``) - always present, always works;
* one or more **GPU builds** in ``runtime/gpu/<backend>/`` (installed by ``python scripts/setup.py --chatbot``):
  ``vulkan`` (any NVIDIA, AMD or Intel GPU; needs only the normal graphics driver) and, optionally,
  ``cuda`` (NVIDIA only, usually faster).

Both builds are imported as ``llama_cpp``, so only one can be used per process, and it must be chosen
*before* the first import. :func:`select_device` decides:

1. ``IMSDB_LLM_DEVICE=cpu`` -> CPU build, no GPU check.
2. Otherwise each installed GPU build (order: ``IMSDB_GPU_BACKENDS``, default ``cuda,vulkan``) is tested
   in a **separate Python process** (:func:`probe_backend`): it imports the build, lists the devices
   llama.cpp can see and, if there is a GPU, runs a real test on it (loads the small embedding model fully
   on the GPU and embeds one sentence). A broken driver, a missing DLL or an unsupported CPU instruction
   can crash a process outright; doing this in a child process keeps the chatbot alive.
3. The first build that passes is used, with as many model layers on the GPU as fit in its free memory
   (:func:`plan_gpu_layers`). If none passes, the CPU build is used (``auto``) or an error is raised
   (``IMSDB_LLM_DEVICE=gpu``).

:func:`activate` then puts the chosen build first on ``sys.path``. If loading the model on the GPU still
fails (e.g. out of memory), ``LocalChatModel`` retries on the CPU (see ``src/chatbot/llm.py``).

Run ``python -m src.main device`` to see the decision and the reason for it.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.chatbot.runtime import RuntimeUnavailable

log = logging.getLogger("imsdb.chatbot.device")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROBE_TIMEOUT_SECONDS = 180
PROBE_MARKER = "IMSDB-PROBE-RESULT:"
GGML_DEVICE_TYPES = {0: "cpu", 1: "gpu", 2: "igpu", 3: "accel"}  # enum ggml_backend_dev_type
GPU_DEVICE_TYPES = {"gpu", "igpu"}
GPU_MEMORY_RESERVE = 1024**3  # left free for compute buffers, the driver and other programs
MiB = 1024**2


@dataclass(frozen=True)
class GpuInfo:
    """A GPU reported by llama.cpp.

    Attributes:
        name: Backend device name, e.g. ``"Vulkan0"`` or ``"CUDA0"``.
        description: Human-readable model, e.g. ``"NVIDIA GeForce RTX 4070"``.
        kind: ``"gpu"`` (discrete) or ``"igpu"`` (integrated).
        total_mib: Total device memory in MiB.
        free_mib: Free device memory in MiB when probed.
    """

    name: str
    description: str
    kind: str
    total_mib: int
    free_mib: int


@dataclass
class DeviceChoice:
    """Where the models will run, and why.

    Attributes:
        device: ``"gpu"`` or ``"cpu"``.
        backend: ``"cuda"``, ``"vulkan"`` or ``"cpu"``.
        runtime_path: Folder holding the chosen GPU build of ``llama_cpp`` (put first on ``sys.path``), or
            None for the CPU build installed in the virtual environment.
        gpu_layers: Value for llama.cpp's ``n_gpu_layers``: ``-1`` = every layer on the GPU, ``0`` = CPU only,
            ``n`` = the first n layers on the GPU and the rest on the CPU.
        gpus: The GPUs the chosen build can see (empty on CPU).
        reason: One sentence explaining the decision, shown by ``python -m src.main device`` and on the page.
        attempts: One entry per GPU build that was tested: ``{"backend", "ok", "detail"}``.
    """

    device: str
    backend: str
    runtime_path: Path | None
    gpu_layers: int
    gpus: list[GpuInfo] = field(default_factory=list)
    reason: str = ""
    attempts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def on_gpu(self) -> bool:
        """Whether any part of the model runs on a GPU.

        Returns:
            True when the device is ``"gpu"`` and at least one layer is offloaded.
        """
        return self.device == "gpu" and self.gpu_layers != 0

    def label(self) -> str:
        """Short description for logs and the web page.

        Returns:
            E.g. ``"GPU: NVIDIA GeForce RTX 4070 (Vulkan, all layers)"`` or ``"CPU"``.
        """
        if not self.on_gpu:
            return "CPU"
        names = ", ".join(g.description for g in self.gpus) or "GPU"
        layers = "all layers" if self.gpu_layers < 0 else f"{self.gpu_layers} layers"
        return f"GPU: {names} ({self.backend.upper() if self.backend == 'cuda' else self.backend.title()}, {layers})"

    def as_dict(self) -> dict[str, Any]:
        """Convert to plain JSON-serialisable data (for ``/api/health`` and ``--json`` output).

        Returns:
            The fields as a dict, with ``runtime_path`` as a string and a ``label`` entry added.
        """
        data = asdict(self)
        data["runtime_path"] = str(self.runtime_path) if self.runtime_path else None
        data["label"] = self.label()
        return data


def cpu_choice(reason: str, attempts: list[dict[str, Any]] | None = None) -> DeviceChoice:
    """Build the CPU fallback decision.

    Args:
        reason: Why the CPU is used (shown to the user).
        attempts: GPU builds that were tried, if any.

    Returns:
        A choice that uses the CPU build from the virtual environment.
    """
    return DeviceChoice(device="cpu", backend="cpu", runtime_path=None, gpu_layers=0, reason=reason,
                        attempts=list(attempts or []))


def backend_order(setting: str) -> list[str]:
    """Parse the ``gpu_backends`` setting.

    Args:
        setting: Comma-separated backend names, e.g. ``"cuda,vulkan"``.

    Returns:
        The names in order, lower-case, without blanks or duplicates.
    """
    names: list[str] = []
    for name in (part.strip().lower() for part in setting.split(",")):
        if name and name not in names:
            names.append(name)
    return names


def installed_backends(gpu_runtime_dir: Path, order: list[str]) -> list[tuple[str, Path]]:
    """Find the GPU builds of llama-cpp-python that are installed.

    Args:
        gpu_runtime_dir: The folder holding one sub-folder per backend (``runtime/gpu``).
        order: Backend names to look for, in order of preference.

    Returns:
        ``(backend, folder)`` pairs for each backend whose folder contains a ``llama_cpp`` package.
    """
    found: list[tuple[str, Path]] = []
    for name in order:
        folder = gpu_runtime_dir / name
        if (folder / "llama_cpp" / "__init__.py").exists():
            found.append((name, folder))
    return found


def plan_gpu_layers(requested: int, free_bytes: int, model_bytes: int, n_layers: int,
                    extra_bytes: int) -> int:
    """Decide how many layers of the chat model to put on the GPU.

    Args:
        requested: The ``llm_gpu_layers`` setting: ``-1`` = automatic, ``0`` or more = exactly that many.
        free_bytes: Free memory on the GPU.
        model_bytes: Size of the chat model's weights file.
        n_layers: Number of transformer layers in the chat model (0 if unknown).
        extra_bytes: Other GPU memory needed besides the weights: the KV caches and the embedding model.

    Returns:
        ``-1`` when the whole model fits, otherwise the number of layers that fit (``0`` = none).
        An explicit ``requested`` value is returned unchanged.
    """
    if requested >= 0:
        return requested
    usable = free_bytes - GPU_MEMORY_RESERVE - extra_bytes
    if usable >= model_bytes or n_layers <= 0:
        return -1 if usable > 0 else 0
    return max(0, min(n_layers, int(n_layers * usable / model_bytes)))


def probe_backend(runtime_path: Path, msvc_runtime_dir: Path | None, test_model: Path | None,
                  timeout: float = PROBE_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Test one GPU build of llama-cpp-python in a separate Python process.

    Args:
        runtime_path: Folder holding the GPU build's ``llama_cpp`` package.
        msvc_runtime_dir: Folder with the app-local ``msvcp140.dll`` (see ``runtime.py``), or None.
        test_model: A small GGUF model to run on the GPU as a real test (the embedding model), or None to
            only import the build and list devices.
        timeout: Seconds before the test is abandoned.

    Returns:
        A dict with ``ok`` (bool: the build imported and the test passed), ``gpus`` (list of GpuInfo dicts),
        ``supports_gpu_offload`` (bool), ``tested`` (bool: the model test ran), ``seconds`` and, on failure,
        ``error`` (str). Never raises.
    """
    cmd = [sys.executable, "-m", "src.chatbot.device", "--probe", str(runtime_path)]
    if msvc_runtime_dir:
        cmd += ["--msvc", str(msvc_runtime_dir)]
    if test_model:
        cmd += ["--test-model", str(test_model)]
    started = time.time()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"ok": False, "gpus": [], "error": f"GPU test did not finish within {timeout:.0f} s"}
    except OSError as exc:
        return {"ok": False, "gpus": [], "error": f"could not start the GPU test: {exc}"}
    result: dict[str, Any] | None = None
    for line in proc.stdout.splitlines():
        if line.startswith(PROBE_MARKER):
            try:
                result = json.loads(line[len(PROBE_MARKER):])
            except json.JSONDecodeError:
                result = None
    if result is None:
        code = proc.returncode & 0xFFFFFFFF
        hint = " (illegal instruction: this build needs a newer CPU)" if code == 0xC000001D else ""
        tail = (proc.stderr.strip().splitlines() or [""])[-1][:300]
        result = {"ok": False, "gpus": [], "error": f"GPU test process exited with code 0x{code:08X}{hint}. {tail}".strip()}
    result["seconds"] = round(time.time() - started, 1)
    return result


def list_ggml_devices(llama_cpp: Any) -> list[dict[str, Any]]:
    """List the compute devices a loaded llama-cpp-python build can use.

    Uses ggml's public device API (``ggml_backend_dev_*``) through ctypes, because llama-cpp-python does not
    wrap it.

    Args:
        llama_cpp: The imported ``llama_cpp`` module.

    Returns:
        One dict per device: ``name``, ``description``, ``kind`` (``cpu``/``gpu``/``igpu``/``accel``),
        ``total_mib`` and ``free_mib``.
    """
    lib = Path(llama_cpp.__file__).parent / "lib"
    ggml = ctypes.CDLL(str(lib / ("ggml.dll" if sys.platform == "win32" else "libggml.so")))
    base = ctypes.CDLL(str(lib / ("ggml-base.dll" if sys.platform == "win32" else "libggml-base.so")))
    ggml.ggml_backend_dev_count.restype = ctypes.c_size_t
    ggml.ggml_backend_dev_get.restype = ctypes.c_void_p
    ggml.ggml_backend_dev_get.argtypes = [ctypes.c_size_t]
    for fn in ("ggml_backend_dev_name", "ggml_backend_dev_description"):
        getattr(base, fn).restype = ctypes.c_char_p
        getattr(base, fn).argtypes = [ctypes.c_void_p]
    base.ggml_backend_dev_type.restype = ctypes.c_int
    base.ggml_backend_dev_type.argtypes = [ctypes.c_void_p]
    base.ggml_backend_dev_memory.restype = None
    base.ggml_backend_dev_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                             ctypes.POINTER(ctypes.c_size_t)]
    devices: list[dict[str, Any]] = []
    for i in range(ggml.ggml_backend_dev_count()):
        dev = ggml.ggml_backend_dev_get(i)
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        base.ggml_backend_dev_memory(dev, ctypes.byref(free), ctypes.byref(total))
        devices.append({
            "name": (base.ggml_backend_dev_name(dev) or b"").decode("utf-8", "replace"),
            "description": (base.ggml_backend_dev_description(dev) or b"").decode("utf-8", "replace").strip(),
            "kind": GGML_DEVICE_TYPES.get(base.ggml_backend_dev_type(dev), "unknown"),
            "total_mib": total.value // MiB,
            "free_mib": free.value // MiB,
        })
    return devices


def run_probe(runtime_path: Path, msvc_runtime_dir: Path | None, test_model: Path | None) -> dict[str, Any]:
    """The test itself; runs inside the child process started by :func:`probe_backend`.

    Args:
        runtime_path: Folder holding the GPU build's ``llama_cpp`` package.
        msvc_runtime_dir: Folder with the app-local ``msvcp140.dll``, or None.
        test_model: GGUF model to run fully on the GPU as a real test, or None.

    Returns:
        The result dict described in :func:`probe_backend` (without ``seconds``).
    """
    from src.chatbot.runtime import prepare_windows_runtime  # noqa: PLC0415 - child process only

    result: dict[str, Any] = {"ok": False, "gpus": [], "supports_gpu_offload": False, "tested": False}
    try:
        sys.path.insert(0, str(runtime_path))
        prepare_windows_runtime(msvc_runtime_dir)
        import llama_cpp  # noqa: PLC0415 - must be imported after sys.path is changed

        module_dir = Path(llama_cpp.__file__).resolve().parent
        if not module_dir.is_relative_to(runtime_path.resolve()):
            result["error"] = f"imported llama_cpp from {module_dir}, not from {runtime_path}"
            return result
        result["version"] = getattr(llama_cpp, "__version__", "?")
        result["supports_gpu_offload"] = bool(llama_cpp.llama_supports_gpu_offload())
        devices = list_ggml_devices(llama_cpp)
        result["devices"] = devices
        result["gpus"] = [d for d in devices if d["kind"] in GPU_DEVICE_TYPES]
        if not result["gpus"]:
            result["error"] = "no GPU found (only " + (", ".join(d["description"] for d in devices) or "nothing") + ")"
            return result
        if test_model is not None and test_model.exists():
            model = llama_cpp.Llama(str(test_model), embedding=True, n_ctx=256, n_gpu_layers=-1, verbose=False)
            vector = model.embed(["GPU self-test"], normalize=True)[0]
            if not vector or any(v != v for v in vector):  # NaN check
                result["error"] = "the GPU produced invalid numbers in the self-test"
                return result
            result["tested"] = True
        result["ok"] = True
    except Exception as exc:  # noqa: BLE001 - reported to the parent, which falls back to the CPU
        result["error"] = f"{exc.__class__.__name__}: {exc}"
    return result


def select_device(settings: Any, *, test_model: Path | None = None, chat_bytes: int = 0, chat_layers: int = 0,
                  extra_gpu_bytes: int = 0) -> DeviceChoice:
    """Decide whether the models run on a GPU or the CPU.

    Args:
        settings: The application ``Settings`` (uses ``llm_device``, ``llm_gpu_layers``, ``gpu_runtime_dir``,
            ``gpu_backends`` and ``msvc_runtime_dir``).
        test_model: Small model used for the GPU self-test (the embedding model), or None to skip the test.
        chat_bytes: Size of the chat model file, used to work out how many layers fit on the GPU.
        chat_layers: Number of layers in the chat model.
        extra_gpu_bytes: Other GPU memory needed (KV caches, embedding model).

    Returns:
        The decision. With ``llm_device="auto"`` this is the CPU when no GPU build passes its test.

    Raises:
        RuntimeUnavailable: With ``llm_device="gpu"`` when no GPU is usable.
    """
    mode = settings.llm_device
    if mode == "cpu":
        return cpu_choice("CPU selected in the settings (IMSDB_LLM_DEVICE=cpu)")
    attempts: list[dict[str, Any]] = []
    backends = installed_backends(settings.gpu_runtime_dir, backend_order(settings.gpu_backends))
    for backend, folder in backends:
        result = probe_backend(folder, settings.msvc_runtime_dir, test_model)
        detail = result.get("error") or ", ".join(g["description"] for g in result.get("gpus", []))
        attempts.append({"backend": backend, "ok": bool(result.get("ok")), "detail": detail,
                         "seconds": result.get("seconds")})
        log.info("GPU test (%s): %s - %s", backend, "passed" if result.get("ok") else "failed", detail)
        if not result.get("ok"):
            continue
        gpus = [GpuInfo(name=g["name"], description=g["description"], kind=g["kind"],
                        total_mib=int(g["total_mib"]), free_mib=int(g["free_mib"])) for g in result["gpus"]]
        free = max(g.free_mib for g in gpus) * MiB
        layers = plan_gpu_layers(settings.llm_gpu_layers, free, chat_bytes, chat_layers, extra_gpu_bytes)
        if layers == 0:
            attempts[-1]["detail"] += f" (only {free // MiB} MiB free: not enough for any layer)"
            continue
        part = "the whole model" if layers < 0 else f"{layers} of {chat_layers} layers"
        return DeviceChoice(device="gpu", backend=backend, runtime_path=folder, gpu_layers=layers, gpus=gpus,
                            reason=f"{backend} GPU build passed its self-test; {part} on the GPU",
                            attempts=attempts)
    if not backends:
        reason = (f"no GPU build of llama-cpp-python is installed in {settings.gpu_runtime_dir} "
                  f"(install one with: python scripts/setup.py --chatbot)")
    else:
        reason = "; ".join(f"{a['backend']}: {a['detail']}" for a in attempts)
    if mode == "gpu":
        raise RuntimeUnavailable(f"IMSDB_LLM_DEVICE=gpu, but no GPU is usable: {reason}")
    return cpu_choice(f"no usable GPU, using the CPU ({reason})", attempts)


def activate(choice: DeviceChoice) -> DeviceChoice:
    """Make the chosen llama-cpp-python build the one that ``import llama_cpp`` loads.

    Must run before ``llama_cpp`` is first imported in this process. If a different build was already
    imported (e.g. by an earlier call in the same process), the choice is corrected to match it.

    Args:
        choice: The decision from :func:`select_device`.

    Returns:
        The choice actually in effect (unchanged in the normal case).
    """
    loaded = sys.modules.get("llama_cpp")
    if choice.runtime_path is None:
        if loaded is not None and _module_in(loaded, None) is False:
            log.warning("a GPU build of llama_cpp is already loaded; running on the CPU through it")
        return choice
    if loaded is not None:
        if _module_in(loaded, choice.runtime_path):
            return choice
        log.warning("llama_cpp was already imported from %s; cannot switch to the %s GPU build in this process",
                    Path(loaded.__file__).parent, choice.backend)
        return cpu_choice("the CPU build was already loaded in this process", choice.attempts)
    path = str(choice.runtime_path)
    if path not in sys.path:
        sys.path.insert(0, path)
    return choice


def _module_in(module: Any, folder: Path | None) -> bool:
    """Tell whether an imported module comes from a given folder.

    Args:
        module: An imported module (``llama_cpp``).
        folder: The folder to compare with, or None to mean "the virtual environment, not a GPU folder".

    Returns:
        True if the module's file lies inside ``folder`` (or, for None, outside every ``runtime/gpu`` folder).
    """
    where = Path(module.__file__).resolve()
    if folder is None:
        return "runtime" not in where.parts or "gpu" not in where.parts
    return where.is_relative_to(folder.resolve())


def _probe_cli(argv: list[str] | None = None) -> int:
    """Entry point of the child process: ``python -m src.chatbot.device --probe <folder> ...``.

    Args:
        argv: Command-line arguments; None reads ``sys.argv``.

    Returns:
        Exit code 0. The result is printed on stdout as one line starting with ``PROBE_MARKER``.
    """
    parser = argparse.ArgumentParser(description="Test a GPU build of llama-cpp-python (internal).")
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--msvc", type=Path)
    parser.add_argument("--test-model", type=Path)
    args = parser.parse_args(argv)
    result = run_probe(args.probe, args.msvc, args.test_model)
    print(PROBE_MARKER + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(_probe_cli())
