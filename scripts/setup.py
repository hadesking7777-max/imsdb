"""One-shot local setup: virtual environment, dependencies, .env, data directories.

Usage (from the project root, with Python 3.12+):
    python scripts/setup.py                               # create .venv and install requirements
    python scripts/setup.py --no-venv                     # install into the current interpreter
    python scripts/setup.py --chatbot                     # also install the offline chatbot's runtime:
                                                          #   CPU build into .venv and the Vulkan GPU build
                                                          #   into runtime/gpu/vulkan
    python scripts/setup.py --chatbot --gpu-backend cuda  # ...plus the CUDA build (NVIDIA, ~510 MB)
    python scripts/setup.py --chatbot --gpu-backend none  # CPU build only

Everything is a prebuilt wheel: no compiler, CUDA toolkit or other system software is installed.
A GPU build that fails to install is reported but does not stop the setup: the chatbot then runs on the CPU.
Does not install MongoDB (see README, "Installing MongoDB").
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GPU_REQUIREMENTS = {"vulkan": "requirements-gpu-vulkan.txt", "cuda": "requirements-gpu-cuda.txt"}


def venv_python(path: Path) -> Path:
    """Locate the Python interpreter inside a virtual environment.

    Args:
        path: The virtual environment's folder (e.g. ``.venv``).

    Returns:
        ``Scripts/python.exe`` on Windows, ``bin/python`` elsewhere (the file may not exist yet).
    """
    return path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(cmd: list[str]) -> None:
    """Print and run a command from the project root, stopping the setup if it fails.

    Args:
        cmd: The program and its arguments.

    Raises:
        subprocess.CalledProcessError: If the command exits with a non-zero code.
    """
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def install_gpu_runtime(python: Path, backend: str) -> bool:
    """Install one GPU build of llama-cpp-python into ``runtime/gpu/<backend>``.

    The GPU build is kept out of the virtual environment so the CPU build there stays intact as the
    fallback; ``src/chatbot/device.py`` chooses between them at start-up.

    Args:
        python: The interpreter whose pip is used.
        backend: ``"vulkan"`` or ``"cuda"`` (a key of ``GPU_REQUIREMENTS``).

    Returns:
        True when the build was installed, False when pip failed (the chatbot then uses the CPU).
    """
    target = ROOT / "runtime" / "gpu" / backend
    cmd = [str(python), "-m", "pip", "install", "--no-deps", "--require-hashes", "--upgrade",
           "--target", str(target), "-r", GPU_REQUIREMENTS[backend]]
    print("+", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"WARNING: the {backend} GPU build could not be installed; the chatbot will use the CPU. "
              f"You can retry later with: python scripts/setup.py --chatbot --gpu-backend {backend}")
        return False
    return True


def main() -> int:
    """Run the setup steps selected on the command line.

    Returns:
        Process exit code: 0 on success, 1 if Python is too old. A failing pip install of a required
        package raises ``CalledProcessError``.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-venv", action="store_true", help="use the current interpreter")
    parser.add_argument("--venv-dir", default=".venv")
    parser.add_argument("--chatbot", action="store_true",
                        help="also install the offline chatbot runtime (requirements-chatbot.txt + a GPU build)")
    parser.add_argument("--gpu-backend", choices=["vulkan", "cuda", "both", "none"], default="vulkan",
                        help="GPU build(s) to install with --chatbot (default: vulkan)")
    args = parser.parse_args()

    if sys.version_info < (3, 12):
        print(f"Python 3.12+ is required (found {sys.version.split()[0]}).", file=sys.stderr)
        return 1

    python = Path(sys.executable)
    if not args.no_venv:
        target = ROOT / args.venv_dir
        if not venv_python(target).exists():
            print(f"Creating virtual environment in {target}")
            venv.EnvBuilder(with_pip=True, upgrade_deps=True).create(target)
        python = venv_python(target)
    run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(python), "-m", "pip", "install", "-r", "requirements.txt"])
    gpu_installed: list[str] = []
    if args.chatbot:
        run([str(python), "-m", "pip", "install", "-r", "requirements-chatbot.txt"])
        backends = {"both": ["cuda", "vulkan"], "none": []}.get(args.gpu_backend, [args.gpu_backend])
        gpu_installed = [b for b in backends if install_gpu_runtime(python, b)]

    env_file = ROOT / ".env"
    if not env_file.exists():
        shutil.copyfile(ROOT / ".env.example", env_file)
        print("Created .env from .env.example - review it before crawling.")
    for sub in ("data/raw", "data/processed", "data/exports", "data/backups", "logs"):
        (ROOT / sub).mkdir(parents=True, exist_ok=True)

    print("\nNext steps:")
    activate = rf"{args.venv_dir}\Scripts\activate" if os.name == "nt" else f"source {args.venv_dir}/bin/activate"
    print(f"  1. Activate the environment:   {activate}")
    print("  2. Start MongoDB (see README)")
    print("  3. python scripts/verify_environment.py")
    print("  4. python scripts/init_database.py")
    print("  5. python -m src.main inspect-source")
    if args.chatbot:
        print("\nChatbot (after the database is populated):")
        print("  python -m src.main models fetch   # one-time download (~5.7 GB), then fully offline")
        print("  python -m src.main chat-index     # build the passage search index (~2 minutes)")
        print("  python -m src.main device         # shows whether a GPU will be used "
              f"(GPU builds installed: {', '.join(gpu_installed) or 'none'})")
        print("  python -m src.main serve --open   # open the chat page at http://localhost:8765")
    return 0


if __name__ == "__main__":
    sys.exit(main())
