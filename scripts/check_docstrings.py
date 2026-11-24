"""Check that every module, class and function in the project is documented.

The project convention (see README.md, "Coding conventions") is Google-style docstrings:

    def resolve(name: str, year: int | None = None) -> Film | None:
        \"\"\"Find a film in the catalogue by title.

        Args:
            name: Title as typed by the user, in any case.
            year: Release year used to break ties, or None.

        Returns:
            The best matching film, or None when nothing matches.
        \"\"\"

Rules enforced here (run ``python scripts/check_docstrings.py``; ``tests/test_docstrings.py`` runs it too):

* every module, class, function and method (including nested functions) has a docstring;
* every parameter except ``self``/``cls`` is listed under ``Args:`` as ``name:`` (``*args``/``**kwargs``
  may be written with or without their stars);
* a function that returns a value has a ``Returns:`` section, and a generator has ``Yields:``.
  "Returns a value" means a return annotation other than ``None``/``NoReturn``, or, when there is no
  annotation, a ``return <value>`` statement in the function's own body. ``__init__`` never needs one.

Exit code 0 when everything is documented, 1 otherwise (each problem is printed as ``path:line: message``).
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = ("src", "scripts", "tests")
SKIP_DIRS = {".venv", "__pycache__", "runtime", "data", "logs", "backups", "exports"}
_SECTION_RE = re.compile(r"^\s*(Args|Arguments|Parameters|Returns|Yields|Raises|Attributes|Examples?|Notes?):\s*$")


@dataclass(frozen=True)
class Problem:
    """One documentation problem found by the checker.

    Attributes:
        path: File the problem is in, relative to the project root.
        line: 1-based line number of the module, class or function.
        message: What is missing.
    """

    path: str
    line: int
    message: str

    def __str__(self) -> str:
        """Format the problem as ``path:line: message``.

        Returns:
            The problem on one line, in the format editors recognise as a link.
        """
        return f"{self.path}:{self.line}: {self.message}"


def iter_python_files(paths: list[Path]) -> list[Path]:
    """List the Python files to check under the given files or folders.

    Args:
        paths: Files or directories. Directories are searched recursively, skipping virtual
            environments, caches and data folders (``SKIP_DIRS``).

    Returns:
        The ``.py`` files found, sorted by path.
    """
    files: set[Path] = set()
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            files.add(path)
        elif path.is_dir():
            for file in path.rglob("*.py"):
                if not SKIP_DIRS.intersection(file.relative_to(path).parts):
                    files.add(file)
    return sorted(files)


def section_names(doc: str, section: str) -> set[str]:
    """Collect the names listed in one section of a Google-style docstring.

    Args:
        doc: The docstring text.
        section: Section heading without the colon, e.g. ``"Args"``. ``"Arguments"`` and
            ``"Parameters"`` are accepted as synonyms of ``"Args"``.

    Returns:
        The names that start an entry (``name:`` or ``name (type):``) in that section, with
        leading ``*`` removed. Empty when the section is absent.
    """
    synonyms = {"Args": {"Args", "Arguments", "Parameters"}}.get(section, {section})
    names: set[str] = set()
    inside = False
    entry_indent: int | None = None
    for line in doc.splitlines():
        heading = _SECTION_RE.match(line)
        if heading:
            inside = heading.group(1) in synonyms
            entry_indent = None
            continue
        if not inside or not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if entry_indent is None:
            entry_indent = indent
        if indent < entry_indent:
            inside = False
            continue
        if indent == entry_indent:
            match = re.match(r"\s*(\*{0,2}[A-Za-z_][A-Za-z0-9_]*)\s*(\([^)]*\))?\s*:", line)
            if match:
                names.add(match.group(1).lstrip("*"))
    return names


def has_section(doc: str, section: str) -> bool:
    """Tell whether a docstring contains a given section heading.

    Args:
        doc: The docstring text.
        section: Section heading without the colon, e.g. ``"Returns"``.

    Returns:
        True when a line consisting of ``<section>:`` is present.
    """
    return any(m and m.group(1) == section for m in map(_SECTION_RE.match, doc.splitlines()))


def own_body_nodes(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.AST]:
    """List the AST nodes that belong to a function itself, excluding nested functions and classes.

    Args:
        func: The function definition.

    Returns:
        Every node inside the function's body that is not inside a nested ``def``, ``lambda`` or ``class``.
    """
    nodes: list[ast.AST] = []
    stack: list[ast.AST] = list(func.body)
    while stack:
        node = stack.pop()
        nodes.append(node)
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                stack.append(child)
    return nodes


def returns_value(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Decide whether a function needs a ``Returns:`` section.

    Args:
        func: The function definition.

    Returns:
        True if the return annotation is something other than ``None``/``NoReturn``/``Never``, or, when
        there is no annotation, if the function's own body contains ``return <value>``.
    """
    if func.name == "__init__":
        return False
    if func.returns is not None:
        text = ast.unparse(func.returns).replace("typing.", "")
        if text in {"None", "NoReturn", "Never"}:
            return False
        if is_generator(func) and text.startswith(("Iterator", "Generator", "Iterable", "AsyncIterator")):
            return False
        return True
    return any(isinstance(n, ast.Return) and n.value is not None and not
               (isinstance(n.value, ast.Constant) and n.value.value is None) for n in own_body_nodes(func))


def is_generator(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Tell whether a function is a generator.

    Args:
        func: The function definition.

    Returns:
        True when the function's own body contains ``yield`` or ``yield from``.
    """
    return any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in own_body_nodes(func))


def parameter_names(func: ast.FunctionDef | ast.AsyncFunctionDef, in_class: bool) -> list[str]:
    """List the parameters that must be documented.

    Args:
        func: The function definition.
        in_class: True for methods; their first parameter (``self``/``cls``) is then skipped unless the
            method is a ``@staticmethod``.

    Returns:
        Parameter names in signature order, without stars.
    """
    args = func.args
    names = [a.arg for a in (*args.posonlyargs, *args.args)]
    static = any(isinstance(d, ast.Name) and d.id == "staticmethod" for d in func.decorator_list)
    if in_class and not static and names and names[0] in {"self", "cls", "mcs"}:
        names = names[1:]
    if args.vararg:
        names.append(args.vararg.arg)
    names += [a.arg for a in args.kwonlyargs]
    if args.kwarg:
        names.append(args.kwarg.arg)
    return [n for n in names if n != "_"]


def check_function(func: ast.FunctionDef | ast.AsyncFunctionDef, in_class: bool, rel: str) -> list[Problem]:
    """Check one function's docstring against the project convention.

    Args:
        func: The function definition.
        in_class: True when the function is a method defined directly in a class body.
        rel: The file path to report, relative to the project root.

    Returns:
        The problems found (empty when the docstring is complete).
    """
    doc = ast.get_docstring(func)
    where = f"function '{func.name}'"
    if not doc:
        return [Problem(rel, func.lineno, f"{where} has no docstring")]
    problems: list[Problem] = []
    params = parameter_names(func, in_class)
    if params:
        documented = section_names(doc, "Args")
        missing = [p for p in params if p not in documented]
        if missing:
            problems.append(Problem(rel, func.lineno, f"{where}: parameters not listed under 'Args:': {', '.join(missing)}"))
    if is_generator(func):
        if not has_section(doc, "Yields"):
            problems.append(Problem(rel, func.lineno, f"{where} is a generator but has no 'Yields:' section"))
    elif returns_value(func) and not has_section(doc, "Returns"):
        problems.append(Problem(rel, func.lineno, f"{where} returns a value but has no 'Returns:' section"))
    return problems


def check_file(path: Path) -> list[Problem]:
    """Check every module, class and function docstring in one file.

    Args:
        path: The Python file.

    Returns:
        The problems found in the file.
    """
    rel = path.resolve().relative_to(ROOT).as_posix() if path.resolve().is_relative_to(ROOT) else str(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    problems: list[Problem] = []
    if not ast.get_docstring(tree):
        problems.append(Problem(rel, 1, "module has no docstring"))

    def visit(node: ast.AST, in_class: bool) -> None:
        """Walk the tree, checking each class and function.

        Args:
            node: The node whose children are visited.
            in_class: True when ``node`` is a class body (its functions are methods).
        """
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                if not ast.get_docstring(child):
                    problems.append(Problem(rel, child.lineno, f"class '{child.name}' has no docstring"))
                visit(child, True)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                problems.extend(check_function(child, in_class, rel))
                visit(child, False)
            else:
                visit(child, in_class)

    visit(tree, False)
    return problems


def check_paths(paths: list[Path]) -> list[Problem]:
    """Check all Python files under the given paths.

    Args:
        paths: Files or directories to check.

    Returns:
        All problems found, in file order.
    """
    problems: list[Problem] = []
    for file in iter_python_files(paths):
        problems.extend(check_file(file))
    return problems


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Command-line arguments (paths to check, ``--summary``); None reads ``sys.argv``.

    Returns:
        Process exit code: 0 when everything is documented, 1 when problems were found.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help=f"files or folders (default: {' '.join(DEFAULT_PATHS)})")
    parser.add_argument("--summary", action="store_true", help="print problem counts per file only")
    args = parser.parse_args(argv)
    paths = [Path(p) for p in args.paths] or [ROOT / p for p in DEFAULT_PATHS]
    problems = check_paths(paths)
    if args.summary:
        counts: dict[str, int] = {}
        for p in problems:
            counts[p.path] = counts.get(p.path, 0) + 1
        for path, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"{count:4d}  {path}")
    else:
        for problem in problems:
            print(problem)
    print(f"{len(problems)} documentation problem(s) in {len(iter_python_files(paths))} file(s).")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
