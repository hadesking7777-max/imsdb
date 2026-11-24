"""The project's documentation rule, enforced: every module, class and function is documented.

See README.md, section 12, and ``scripts/check_docstrings.py`` for the convention.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def load_checker() -> ModuleType:
    """Import ``scripts/check_docstrings.py`` (the ``scripts`` folder is not a package).

    Returns:
        The loaded checker module.
    """
    spec = importlib.util.spec_from_file_location("check_docstrings", ROOT / "scripts" / "check_docstrings.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses need the module registered while it executes
    spec.loader.exec_module(module)
    return module


def test_every_function_is_documented() -> None:
    """src/, scripts/ and tests/ have no undocumented module, class, function, parameter or return value."""
    checker = load_checker()
    problems = checker.check_paths([ROOT / "src", ROOT / "scripts", ROOT / "tests"])
    assert not problems, "Undocumented code:\n" + "\n".join(map(str, problems[:50]))


def test_checker_detects_missing_sections(tmp_path: Path) -> None:
    """The checker reports a missing docstring, an unlisted parameter and a missing Returns section.

    Args:
        tmp_path: pytest's per-test temporary folder (holds the sample file).
    """
    sample = tmp_path / "sample.py"
    sample.write_text(
        '"""Module."""\n'
        "def a(x):\n    return x\n"
        'def b(x, y):\n    """Summary.\n\n    Args:\n        x: First.\n    """\n'
        'def c():\n    """Summary."""\n    return 1\n'
        'def d(x: int) -> int:\n    """Summary.\n\n    Args:\n        x: Value.\n\n    Returns:\n        Value.\n    """\n'
        "    return x\n",
        encoding="utf-8",
    )
    messages = [p.message for p in load_checker().check_file(sample)]
    assert any("'a' has no docstring" in m for m in messages)
    assert any("'b'" in m and "y" in m for m in messages)
    assert any("'c' returns a value" in m for m in messages)
    assert not any("'d'" in m for m in messages)
