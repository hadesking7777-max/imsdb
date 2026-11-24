"""Tests for choosing the compute device (GPU by default, CPU fallback): ``src/chatbot/device.py``."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from src.chatbot import device
from src.chatbot.device import DeviceChoice, plan_gpu_layers, select_device
from src.chatbot.runtime import RuntimeUnavailable
from src.config import load_settings

GIB = 1024**3


def make_backend(root: Path, name: str) -> Path:
    """Create a fake installed GPU build folder (just ``llama_cpp/__init__.py``).

    Args:
        root: The folder standing in for ``runtime/gpu``.
        name: Backend name, ``"vulkan"`` or ``"cuda"``.

    Returns:
        The backend's folder.
    """
    package = root / name / "llama_cpp"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    return root / name


def gpu_result(free_gib: float = 12.0) -> dict[str, Any]:
    """Build a successful probe result for one 16 GiB GPU.

    Args:
        free_gib: Free GPU memory to report, in GiB.

    Returns:
        A dict shaped like the result of ``device.probe_backend``.
    """
    return {"ok": True, "tested": True, "supports_gpu_offload": True, "seconds": 1.0,
            "gpus": [{"name": "Vulkan0", "description": "Test GPU", "kind": "gpu", "total_mib": 16384,
                      "free_mib": int(free_gib * 1024)}]}


@pytest.fixture
def gpu_dir(tmp_path: Path) -> Path:
    """An empty stand-in for ``runtime/gpu``.

    Args:
        tmp_path: pytest's per-test temporary folder.

    Returns:
        The folder (created).
    """
    folder = tmp_path / "gpu"
    folder.mkdir()
    return folder


def test_plan_gpu_layers_whole_model_fits() -> None:
    """With enough free memory every layer goes on the GPU (-1)."""
    assert plan_gpu_layers(-1, 16 * GIB, 5 * GIB, 36, 2 * GIB) == -1


def test_plan_gpu_layers_partial_and_none() -> None:
    """With too little memory only some layers (or none) are offloaded; explicit values are kept."""
    partial = plan_gpu_layers(-1, 5 * GIB, 5 * GIB, 36, 1 * GIB)
    assert 0 < partial < 36
    assert plan_gpu_layers(-1, 1 * GIB, 5 * GIB, 36, 1 * GIB) == 0
    assert plan_gpu_layers(10, 0, 5 * GIB, 36, 0) == 10


def test_cpu_setting_never_probes(gpu_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """IMSDB_LLM_DEVICE=cpu skips the GPU test entirely.

    Args:
        gpu_dir: Stand-in for ``runtime/gpu`` (with a backend installed).
        monkeypatch: Used to make any probe fail the test.
    """
    make_backend(gpu_dir, "vulkan")
    monkeypatch.setattr(device, "probe_backend", lambda *a, **k: pytest.fail("must not probe"))
    choice = select_device(load_settings(llm_device="cpu", gpu_runtime_dir=gpu_dir))
    assert choice.device == "cpu" and choice.gpu_layers == 0 and choice.runtime_path is None


def test_auto_without_gpu_build_falls_back_to_cpu(gpu_dir: Path) -> None:
    """With no GPU build installed, ``auto`` uses the CPU and says why.

    Args:
        gpu_dir: Empty stand-in for ``runtime/gpu``.
    """
    choice = select_device(load_settings(gpu_runtime_dir=gpu_dir))
    assert choice.device == "cpu"
    assert "no GPU build" in choice.reason


def test_auto_uses_gpu_when_probe_passes(gpu_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """GPU is the default: a passing probe selects the GPU build with all layers offloaded.

    Args:
        gpu_dir: Stand-in for ``runtime/gpu``.
        monkeypatch: Replaces the real probe.
    """
    folder = make_backend(gpu_dir, "vulkan")
    monkeypatch.setattr(device, "probe_backend", lambda *a, **k: gpu_result())
    choice = select_device(load_settings(gpu_runtime_dir=gpu_dir), chat_bytes=5 * GIB, chat_layers=36,
                           extra_gpu_bytes=2 * GIB)
    assert choice.device == "gpu" and choice.backend == "vulkan" and choice.runtime_path == folder
    assert choice.gpu_layers == -1 and choice.on_gpu
    assert "Test GPU" in choice.label()
    assert choice.as_dict()["runtime_path"] == str(folder)


def test_failed_backend_falls_through_to_next(gpu_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A crashing CUDA build is skipped and the Vulkan build is used.

    Args:
        gpu_dir: Stand-in for ``runtime/gpu``.
        monkeypatch: Replaces the real probe.
    """
    make_backend(gpu_dir, "cuda")
    make_backend(gpu_dir, "vulkan")

    def fake_probe(path: Path, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Fail for CUDA, succeed for Vulkan.

        Args:
            path: The backend folder being probed.
            *args: Ignored.
            **kwargs: Ignored.

        Returns:
            A probe result.
        """
        if path.name == "cuda":
            return {"ok": False, "gpus": [], "error": "exited with code 0xC0000135"}
        return gpu_result()

    monkeypatch.setattr(device, "probe_backend", fake_probe)
    choice = select_device(load_settings(gpu_runtime_dir=gpu_dir), chat_bytes=5 * GIB, chat_layers=36)
    assert choice.backend == "vulkan"
    assert [a["backend"] for a in choice.attempts] == ["cuda", "vulkan"]
    assert choice.attempts[0]["ok"] is False


def test_gpu_too_small_falls_back_to_cpu(gpu_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A GPU without room for even one layer is not used.

    Args:
        gpu_dir: Stand-in for ``runtime/gpu``.
        monkeypatch: Replaces the real probe.
    """
    make_backend(gpu_dir, "vulkan")
    monkeypatch.setattr(device, "probe_backend", lambda *a, **k: gpu_result(free_gib=1.0))
    choice = select_device(load_settings(gpu_runtime_dir=gpu_dir), chat_bytes=5 * GIB, chat_layers=36,
                           extra_gpu_bytes=1 * GIB)
    assert choice.device == "cpu"
    assert "not enough" in choice.reason


def test_gpu_setting_raises_without_gpu(gpu_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """IMSDB_LLM_DEVICE=gpu refuses to silently fall back to the CPU.

    Args:
        gpu_dir: Stand-in for ``runtime/gpu``.
        monkeypatch: Replaces the real probe.
    """
    make_backend(gpu_dir, "vulkan")
    monkeypatch.setattr(device, "probe_backend", lambda *a, **k: {"ok": False, "gpus": [], "error": "no GPU"})
    with pytest.raises(RuntimeUnavailable, match="no GPU is usable"):
        select_device(load_settings(llm_device="gpu", gpu_runtime_dir=gpu_dir))


def test_probe_reports_crash_of_child_process(tmp_path: Path) -> None:
    """A child process that dies without a result is reported as a failed probe, not an exception.

    Args:
        tmp_path: pytest's per-test temporary folder (used as a folder without llama_cpp).
    """
    result = device.probe_backend(tmp_path, None, None, timeout=60)
    assert result["ok"] is False
    assert result["error"]


def test_activate_puts_gpu_build_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Activating a GPU choice puts its folder first on ``sys.path`` before ``llama_cpp`` is imported.

    Args:
        tmp_path: pytest's per-test temporary folder.
        monkeypatch: Isolates ``sys.path`` and ``sys.modules``.
    """
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "llama_cpp", raising=False)
    choice = DeviceChoice(device="gpu", backend="vulkan", runtime_path=tmp_path, gpu_layers=-1)
    assert device.activate(choice) is choice
    assert sys.path[0] == str(tmp_path)


REAL_VULKAN = Path(__file__).resolve().parents[1] / "runtime" / "gpu" / "vulkan"


@pytest.mark.skipif(not (REAL_VULKAN / "llama_cpp").exists(), reason="Vulkan GPU build not installed")
def test_real_vulkan_probe_runs() -> None:
    """The installed Vulkan build imports in a child process and reports its devices.

    On a computer without a GPU it must report ``ok=False`` with a clear reason (then the CPU is used);
    with a GPU it must pass.
    """
    settings = load_settings()
    result = device.probe_backend(REAL_VULKAN, settings.msvc_runtime_dir, None)
    assert "error" in result or result["ok"]
    if not result["ok"]:
        assert "no GPU found" in result["error"] or "GPU" in result["error"]
