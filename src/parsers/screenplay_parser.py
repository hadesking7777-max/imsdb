"""Screenplay extraction (HTML -> exact text) and structural parsing (text -> elements).

Design rules
------------
* The extracted ``text`` is the canonical representation. Every structural element,
  scene and dialogue block stores ``char_start``/``char_end`` offsets into it, so the
  original wording, ordering, line breaks and indentation are always recoverable.
* Normalizations applied to the text are minimal and listed in ``normalizations``.
* Classification is heuristic. Each element has a ``confidence`` and ``basis``;
  elements below :data:`UNCERTAIN_THRESHOLD` are flagged ``uncertain`` rather than
  silently treated as correct.
* HTML traversal is iterative, so deeply nested, malformed markup cannot overflow the stack.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from bs4 import BeautifulSoup, CData, Comment, Declaration, Doctype, NavigableString, ProcessingInstruction, Tag

from src.parsers import PARSER_VERSION
from src.parsers.dialogue_parser import (
    SHOT_RE,
    CharacterSummary,
    DialogueBlock,
    aggregate_characters,
    build_dialogue_blocks,
    looks_like_cue_text,
)
from src.parsers.scene_parser import Scene, is_scene_heading, segment_scenes
from src.utils.text_processing import collapse_whitespace, encoding_problems

UNCERTAIN_THRESHOLD = 0.7

SKIP_TAGS = {"script", "style", "ins", "noscript", "iframe", "form", "button", "select",
             "textarea", "input", "head", "object", "embed", "svg"}
BLOCK_TAGS = {"p", "div", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "center",
              "table", "ul", "ol", "dd", "dt", "hr", "section", "article"}
STYLE_TAGS = {"b": "bold", "strong": "bold", "i": "italic", "em": "italic", "u": "underline"}
PRE_TAGS = {"pre", "xmp", "listing", "plaintext"}
_COLLAPSE_RE = re.compile(r"[ \t\r\n\f]+")

# A line that is only a page number: "12", "12.", "(12)", "45A".
PAGE_RE = re.compile(r"^\(?\d{1,3}[A-Z]{0,2}\.?\)?$")
# Page-break furniture: "(MORE)", "CONTINUED:", "CONTINUED: (2)", "(CONT'D)", "12 CONTINUED:", "CONTINUED: 12".
CONTINUATION_RE = re.compile(
    r"^(?:\(\s*MORE\s*\)|\(?\s*CONTINUED\s*\)?:?(?:\s*\(\d+\))?|CONTINUED:\s*\(\d+\)|\(CONT'?D\)|"
    r"\d{1,3}[A-Z]?\.?\s+CONTINUED:?(?:\s*\(\d+\))?|CONTINUED:?\s+\d{1,3}[A-Z]?\.?)$",
    re.IGNORECASE,
)
# Known transitions (CUT TO:, SMASH CUT TO:, FADE OUT., DISSOLVE TO:, THE END ...) or any short
# upper-case line ending in " TO:" (second alternative). Case-sensitive.
TRANSITION_RE = re.compile(
    r"^(?:(?:(?:SMASH|MATCH|JUMP|HARD|QUICK|FLASH|TIME|STRAIGHT|SHOCK)\s+)?CUT(?:\s+BACK)?(?:\s+TO)?"
    r"|FADE\s+(?:IN|OUT|UP|TO(?:\s+(?:BLACK|WHITE))?)|(?:SLOW\s+|QUICK\s+|LAP\s+)?DISSOLVE(?:\s+TO)?"
    r"|WIPE(?:\s+TO)?|IRIS\s+(?:IN|OUT)|FADE\s+OUT|FADE\s+TO\s+BLACK|BACK\s+TO\s+BLACK"
    r"|THE\s+END|END\s+OF\s+(?:FILM|SCRIPT|PICTURE|SCREENPLAY)|END\s+CREDITS)\s*[:.!]?$"
    r"|^[A-Z][A-Z .'\-]{0,30}\s+TO:$",
)
# Structural markers at line start: ACT ONE / ACT II / END OF ACT, PROLOGUE, TEASER, PART 2, CHAPTER X ...
ACT_RE = re.compile(
    r"^(?:ACT\s+(?:ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|[IVX]{1,4}|\d{1,2})\b|END\s+OF\s+ACT\b"
    r"|(?:FIRST|SECOND|THIRD|FOURTH|FINAL)\s+ACT\b|PROLOGUE\b|EPILOGUE\b|TEASER\b|COLD\s+OPEN\b"
    r"|INTERMISSION\b|PART\s+(?:ONE|TWO|THREE|FOUR|FIVE|[IVX]{1,4}|\d{1,2})\b|SEQUENCE\s+[A-Z0-9]+\b"
    r"|BOOK\s+(?:ONE|TWO|THREE|[IVX]{1,4}|\d{1,2})\b|CHAPTER\s+(?:[A-Z]+|\d{1,3})\b)",
)
# Flush-left "NAME: speech" or "NAME (V.O.): speech" on a single line (play/transcript style).
INLINE_CUE_RE = re.compile(r"^(?P<cue>[A-Z][A-Z0-9 .'\-]{0,30}[A-Z.](?:\s*\([^)]{1,25}\))?):\s+(?P<speech>\S.*)$")
# Weak front-matter hints (credits, draft notes, copyright, any year); only applied before the first scene.
TITLE_PAGE_RE = re.compile(
    r"\b(?:written\s+by|screenplay\s+by|story\s+by|teleplay\s+by|based\s+on|draft|revised|revision|"
    r"copyright|all\s+rights\s+reserved|final|shooting\s+script|by)\b|©|\(c\)|\b(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
_PAREN_CONTENT_RE = re.compile(r"\([^)]*\)")
# Unambiguous credit phrases used to recognise a title + "written by" pair mis-read as cue + dialogue.
STRONG_TITLE_RE = re.compile(
    r"\b(?:written\s+by|screenplay\s+by|story\s+by|teleplay\s+by|based\s+on|adapted\s+from|revised\s+draft|"
    r"first\s+draft|final\s+draft|shooting\s+script)\b",
    re.IGNORECASE,
)


# ====================================================================== extraction
@dataclass(slots=True)
class ScriptLine:
    """One line of the extracted script text.

    Attributes:
        index: 0-based line number.
        text: The line exactly as in the text, without the trailing newline.
        start: Character offset of the line start in the full text.
        end: Character offset just past the last character of the line (the newline is excluded).
        indent: Leading spaces after expanding tabs to 8 columns.
        bold: True when at least half of the line's non-padding characters are inside bold spans.
    """

    index: int
    text: str
    start: int
    end: int
    indent: int
    bold: bool

    @property
    def stripped(self) -> str:
        """The line without surrounding whitespace.

        Returns:
            ``text.strip()``.
        """
        return self.text.strip()

    @property
    def blank(self) -> bool:
        """Whether the line holds only whitespace.

        Returns:
            True for an empty or whitespace-only line.
        """
        return not self.text.strip()

    @property
    def content_start(self) -> int:
        """Absolute offset of the first non-whitespace character.

        Returns:
            ``start`` plus the length of the leading whitespace (equals ``end`` for a blank line).
        """
        return self.start + (len(self.text) - len(self.text.lstrip()))

    @property
    def content_end(self) -> int:
        """Absolute offset just past the last non-whitespace character.

        Returns:
            ``start`` plus the length of the right-stripped line.
        """
        return self.start + len(self.text.rstrip())


@dataclass(slots=True)
class FormattingSpan:
    """A run of the extracted text that was inside a formatting tag.

    Attributes:
        style: ``"bold"``, ``"italic"`` or ``"underline"``.
        start: Start offset in the extracted text.
        end: End offset (exclusive). Adjacent runs of the same style are merged.
    """

    style: str
    start: int
    end: int


@dataclass(slots=True)
class ExtractedScript:
    """The screenplay text extracted from a script page, with provenance and quality data.

    Attributes:
        text: Canonical screenplay text; all element offsets point into it.
        lines: ``text`` split on ``\\n``, with offsets, indentation and boldness.
        markup: HTML of the node(s) the text was extracted from.
        method: How the script was located: ``pre``, ``multi_pre``, ``scrtext_html``, ``largest_pre`` or
            ``body_fallback``.
        formatting_spans: Bold/italic/underline runs as offsets into ``text``.
        footer_metadata: Title, writers and genres from the IMSDb footer table inside the script cell.
        page_title: The page ``<title>``, or None.
        warnings: Notes about fallbacks and removed elements.
        normalizations: Every change applied to the text relative to the source.
        visible_page_chars: Non-whitespace visible characters on the whole page.
        extraction_ratio: Non-whitespace characters in ``text`` divided by ``visible_page_chars``, or None
            when the page has no visible text.
        removed_tags: Number of skipped elements per tag name (plus ``footer_table``).
    """

    text: str
    lines: list[ScriptLine]
    markup: str
    method: str
    formatting_spans: list[FormattingSpan]
    footer_metadata: dict[str, Any]
    page_title: str | None
    warnings: list[str]
    normalizations: list[str]
    visible_page_chars: int
    extraction_ratio: float | None
    removed_tags: dict[str, int]

    @property
    def found(self) -> bool:
        """Whether any screenplay text was extracted.

        Returns:
            True when ``text`` contains a non-whitespace character.
        """
        return bool(self.text.strip())


def _visible_chars(soup: BeautifulSoup) -> int:
    """Count the non-whitespace visible characters of a whole page, the baseline for the extraction ratio.

    Args:
        soup: The parsed page.

    Returns:
        Non-whitespace characters in text nodes, excluding comments/doctypes and text inside
        ``SKIP_TAGS`` or ``<title>``.
    """
    total = 0
    for string in soup.find_all(string=True):
        if isinstance(string, (Comment, Doctype, Declaration, ProcessingInstruction)):
            continue
        parent = string.parent
        if parent is not None and parent.name in SKIP_TAGS | {"title"}:
            continue
        total += sum(1 for ch in str(string) if not ch.isspace())
    return total


def _is_footer_table(tag: Tag) -> bool:
    """Recognise IMSDb's metadata footer table that sits inside the script cell after the script.

    Args:
        tag: Any tag.

    Returns:
        True for a ``<table>`` without a ``<pre>`` that has a ``<b>Writers</b>`` or ``<b>Genres</b>`` label.
    """
    if tag.name != "table" or tag.find("pre"):
        return False
    labels = {collapse_whitespace(b.get_text(" ")).rstrip(":").lower() for b in tag.find_all("b")}
    return bool(labels & {"writers", "genres"})


def _extract_footer(container: Tag) -> dict[str, Any]:
    """Read title, writers and genres from the first footer table in the script container.

    Args:
        container: The script cell (``td.scrtext``).

    Returns:
        ``{"title"?, "writers", "genres"}`` (``title`` only when the table has an ``<h1>``); empty dict
        when there is no footer table.
    """
    footer: dict[str, Any] = {}
    for table in container.find_all("table"):
        if not _is_footer_table(table):
            continue
        h1 = table.find("h1")
        if h1:
            footer["title"] = collapse_whitespace(h1.get_text(" "))
        footer["writers"] = [collapse_whitespace(a.get_text(" ")) for a in table.find_all("a", href=True)
                             if "writer.php" in a["href"]]
        footer["genres"] = [collapse_whitespace(a.get_text(" ")) for a in table.find_all("a", href=True)
                            if "/genre/" in a["href"]]
        break
    return footer


def _flatten(roots: list[Tag], preformatted: bool, removed: Counter[str]) -> list[tuple[str, frozenset[str]]]:
    """Iteratively flatten HTML into ``(text, styles)`` segments.

    Uses an explicit stack (no recursion). Inside preformatted content text is kept verbatim; outside
    it, whitespace runs collapse to one space, leading spaces at a line start are dropped, ``<br>``
    becomes a newline and block tags (``BLOCK_TAGS``) start and end on their own line. ``SKIP_TAGS``
    and footer tables are dropped and counted. Comments, doctypes and CDATA are ignored.

    Args:
        roots: Container elements to flatten, in order; a newline is inserted between them.
        preformatted: Treat all text as preformatted even outside ``<pre>`` (raw newlines are breaks).
        removed: Counter updated in place with the number of skipped elements per tag name.

    Returns:
        Text segments in document order, each with the set of active styles (``bold``, ``italic``,
        ``underline``). Concatenating the texts gives the script text (before CR/nbsp normalisation).
    """
    out: list[tuple[str, frozenset[str]]] = []
    at_line_start = True

    def emit(text: str, styles: frozenset[str], pre: bool) -> None:
        """Append a text segment, collapsing whitespace unless preformatted.

        Args:
            text: Raw text node content.
            styles: Styles active at this node.
            pre: Whether the node is inside preformatted content.
        """
        nonlocal at_line_start
        if not text:
            return
        if not pre:
            text = _COLLAPSE_RE.sub(" ", text)
            if at_line_start:
                text = text.lstrip(" ")
                if not text:
                    return
        out.append((text, styles))
        at_line_start = text.endswith("\n")

    def newline(styles: frozenset[str]) -> None:
        """Append a line break and mark the output as being at a line start.

        Args:
            styles: Styles active where the break occurs.
        """
        nonlocal at_line_start
        out.append(("\n", styles))
        at_line_start = True

    for index, root in enumerate(roots):
        if index:
            newline(frozenset())
        root_pre = preformatted or (root.name in PRE_TAGS)
        # Stack frames: (children iterator, active styles, preformatted?, emit newline when the frame closes?)
        stack: list[tuple[Any, frozenset[str], bool, bool]] = [(iter(root.children), frozenset(), root_pre, False)]
        while stack:
            iterator, styles, pre, block_after = stack[-1]
            child = next(iterator, None)
            if child is None:
                stack.pop()
                if block_after:
                    newline(styles)
                continue
            if isinstance(child, (Comment, Doctype, Declaration, ProcessingInstruction, CData)):
                continue
            if isinstance(child, NavigableString):
                emit(str(child), styles, pre)
                continue
            if not isinstance(child, Tag):
                continue
            name = (child.name or "").lower()
            if name in SKIP_TAGS:
                removed[name] += 1
                continue
            if not pre and _is_footer_table(child):
                removed["footer_table"] += 1
                continue
            if name == "br":
                newline(styles)
                continue
            child_styles = styles | {STYLE_TAGS[name]} if name in STYLE_TAGS else styles
            child_pre = pre or name in PRE_TAGS
            block = (not pre) and name in BLOCK_TAGS
            if block and not at_line_start:
                newline(styles)
            stack.append((iter(child.children), child_styles, child_pre, block))
    return out


def extract_screenplay(html: str) -> ExtractedScript:
    """Locate the screenplay inside an IMSDb script page and extract it losslessly.

    Source preference: the top-level ``<pre>`` block(s) inside ``td.scrtext``; else the cell's HTML with
    lines rebuilt from ``<br>``/block tags (or raw newlines when the cell has no ``<br>`` but over 20
    newlines); else the largest ``<pre>`` on the page; else the whole ``<body>``. Ads, scripts and the
    metadata footer table are removed. Inside ``<pre>`` only CRLF/CR line endings and non-breaking spaces
    are normalised; HTML-mode extraction also collapses whitespace. Every change is recorded in
    ``normalizations``; formatting is kept as offset spans.

    Args:
        html: Decoded HTML of a script page.

    Returns:
        The extracted script. For a page without any script text, ``text`` is empty (``found`` is
        False) and a ``"no screenplay text extracted"`` warning is added.
    """
    soup = BeautifulSoup(html, "lxml")
    warnings: list[str] = []
    removed: Counter[str] = Counter()
    page_title = collapse_whitespace(soup.title.get_text(" ")) if soup.title else None
    visible = _visible_chars(soup)
    footer: dict[str, Any] = {}

    container = soup.find("td", class_="scrtext")
    preformatted = False
    if container is not None:
        footer = _extract_footer(container)
        pres = [p for p in container.find_all("pre") if p.find_parent("pre") is None]
        if len(pres) == 1:
            roots, method = pres, "pre"
        elif len(pres) > 1:
            roots, method = pres, "multi_pre"
            warnings.append(f"{len(pres)} <pre> blocks found in script container; concatenated in order")
        else:
            roots, method = [container], "scrtext_html"
            text_probe = container.get_text()
            if not container.find("br") and text_probe.count("\n") > 20:
                preformatted = True
                warnings.append("script container has no <pre>/<br>; treated raw newlines as line breaks")
            else:
                warnings.append("script container has no <pre>; lines reconstructed from HTML breaks")
    else:
        pres = [p for p in soup.find_all("pre") if p.find_parent("pre") is None]
        if pres:
            largest = max(pres, key=lambda p: len(p.get_text()))
            roots, method = [largest], "largest_pre"
            warnings.append("script container (td.scrtext) not found; used the largest <pre> block")
        else:
            roots, method = [soup.body or soup], "body_fallback"
            warnings.append("no script container or <pre> found; used whole page body (low confidence)")

    segments = _flatten(roots, preformatted, removed)
    parts: list[str] = []
    spans: list[FormattingSpan] = []
    pos = 0
    saw_cr = saw_nbsp = False
    for text, styles in segments:
        if "\r" in text:
            saw_cr = True
            text = text.replace("\r\n", "\n").replace("\r", "\n")
        if "\xa0" in text:
            saw_nbsp = True
            text = text.replace("\xa0", " ")
        if not text:
            continue
        end = pos + len(text)
        for style in sorted(styles):
            last = next((s for s in reversed(spans) if s.style == style), None)
            if last is not None and last.end == pos:
                last.end = end
            else:
                spans.append(FormattingSpan(style, pos, end))
        parts.append(text)
        pos = end
    full_text = "".join(parts)

    normalizations = ["HTML entities decoded", "markup tags removed (formatting kept in formatting_spans/markup)"]
    if saw_cr:
        normalizations.append("CRLF/CR line endings converted to LF")
    if saw_nbsp:
        normalizations.append("non-breaking spaces converted to spaces")
    if method not in {"pre", "multi_pre"} and not preformatted:
        normalizations.append("HTML whitespace collapsed outside <pre>; <br> and block tags became line breaks")
    if removed:
        warnings.append("removed non-content elements: " + ", ".join(f"{k}={v}" for k, v in sorted(removed.items())))

    # One byte per character, 1 where the character is bold; a line is bold when >= half its content is.
    bold_mask = bytearray(len(full_text))
    for span in spans:
        if span.style == "bold":
            bold_mask[span.start:span.end] = b"\x01" * (span.end - span.start)

    lines: list[ScriptLine] = []
    start = 0
    for idx, raw_line in enumerate(full_text.split("\n")):
        end = start + len(raw_line)
        expanded = raw_line.expandtabs(8)
        indent = len(expanded) - len(expanded.lstrip(" "))
        stripped_len = len(raw_line.strip())
        bold = False
        if stripped_len:
            cs = start + (len(raw_line) - len(raw_line.lstrip()))
            ce = start + len(raw_line.rstrip())
            bold = sum(bold_mask[cs:ce]) * 2 >= (ce - cs)
        lines.append(ScriptLine(idx, raw_line, start, end, indent, bold))
        start = end + 1

    nonspace = sum(1 for ch in full_text if not ch.isspace())
    ratio = round(nonspace / visible, 4) if visible else None
    markup = "".join(str(root) for root in roots) if method != "body_fallback" else str(roots[0])
    if not nonspace:
        warnings.append("no screenplay text extracted")
    return ExtractedScript(
        text=full_text, lines=lines, markup=markup, method=method, formatting_spans=spans,
        footer_metadata=footer, page_title=page_title, warnings=warnings,
        normalizations=normalizations, visible_page_chars=visible, extraction_ratio=ratio,
        removed_tags=dict(removed),
    )


# ====================================================================== classification
@dataclass(slots=True)
class Element:
    """A classified structural unit of the screenplay: one or more consecutive lines of one type.

    Attributes:
        index: 0-based position in the element list.
        type: ``scene_heading``, ``action``, ``character``, ``dialogue``, ``parenthetical``,
            ``transition``, ``shot``, ``act_marker``, ``page_marker``, ``continuation`` or ``title_page``.
        line_start: First line index.
        line_end: Last line index (inclusive).
        char_start: Start offset in the script text.
        char_end: End offset (exclusive) in the script text.
        text: Exact source slice ``text[char_start:char_end]``, including indentation of later lines.
        content: ``text`` with whitespace collapsed.
        confidence: Classification confidence in [0, 1], rounded to 3 decimals.
        basis: Sorted, distinct reasons for the classification (e.g. ``cue-indent``, ``int-ext``).
        bold: True when every line of the element is bold.
        indent: Indentation of the first line.
        in_dialogue: True for dialogue/parenthetical/continuation elements that belong to a speech block.
        cue_text: For inline ``NAME: speech`` cues, the name part without the colon; otherwise None.
        scene_index: Index of the containing scene, set by :func:`parse_screenplay`.
        notes: Human-readable remarks, e.g. possible dual dialogue.
    """

    index: int
    type: str
    line_start: int
    line_end: int
    char_start: int
    char_end: int
    text: str
    content: str
    confidence: float
    basis: list[str]
    bold: bool
    indent: int
    in_dialogue: bool = False
    cue_text: str | None = None
    scene_index: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def uncertain(self) -> bool:
        """Whether the classification is too weak to trust.

        Returns:
            True when ``confidence`` is below :data:`UNCERTAIN_THRESHOLD`.
        """
        return self.confidence < UNCERTAIN_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        """Convert the element to a plain dict for storage.

        Returns:
            All fields plus the computed ``uncertain`` flag.
        """
        data = asdict(self)
        data["uncertain"] = self.uncertain
        return data


@dataclass(slots=True)
class LayoutProfile:
    """Indentation and bold statistics of a script, used as the reference for classification.

    Attributes:
        mode: ``"indented"`` when a consistent dialogue column was found, else ``"flush"``.
        action_indent: Typical indentation of action paragraphs.
        dialogue_indent: Typical indentation of dialogue lines, or None in flush mode.
        cue_indent: Typical indentation of character cues, or None when unknown.
        bold_ratio: Share of non-blank lines that are bold (4 decimals).
        bold_meaningful: True when 1%-60% of lines are bold, i.e. bold can distinguish elements.
    """

    mode: str  # "indented" | "flush"
    action_indent: int
    dialogue_indent: int | None
    cue_indent: int | None
    bold_ratio: float
    bold_meaningful: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert the profile to a plain dict for storage.

        Returns:
            All fields as a dict.
        """
        return asdict(self)


@dataclass(slots=True)
class _Label:
    """Classification of a single line, before consecutive lines are merged into elements.

    Attributes:
        type: Element type for the line (see :class:`Element`), or ``"inline_cue"`` for ``NAME: speech``.
        confidence: Confidence in [0, 1].
        basis: Reasons for the classification.
        in_dialogue: Whether the line belongs to a speech block.
        joins_previous: Continuation of a multi-line parenthetical.
        split_at: Absolute char offset where speech starts on the same line (inline cue, or text after
            a closing parenthesis), else None.
        notes: Remarks copied to the resulting element.
    """

    type: str
    confidence: float
    basis: list[str]
    in_dialogue: bool = False
    joins_previous: bool = False  # continuation of a multi-line parenthetical
    split_at: int | None = None  # inline "NAME: speech" split offset (absolute char)
    notes: list[str] = field(default_factory=list)


def _has_lower(text: str) -> bool:
    """Check for lower-case letters outside parentheses (prose rather than a caps line).

    Args:
        text: A line of text.

    Returns:
        True when any character outside ``(...)`` groups is lower-case.
    """
    return any(c.islower() for c in _PAREN_CONTENT_RE.sub("", text))


def _is_upper_text(text: str) -> bool:
    """Check that text outside parentheses is all caps and contains at least one letter.

    Args:
        text: A line of text.

    Returns:
        True for upper-case lines such as ``"CUT TO:"`` or ``"JOHN (V.O.)"``; False for lines without
        letters outside parentheses.
    """
    outside = _PAREN_CONTENT_RE.sub("", text)
    return any(c.isalpha() for c in outside) and not any(c.islower() for c in outside)


def _mode_with_support(values: list[int], tolerance: int = 2) -> tuple[int | None, int]:
    """Find the most common indentation, counting near values (within ``tolerance``) as support.

    Ties are broken in favour of the smaller value.

    Args:
        values: Indentation values.
        tolerance: Maximum distance for a value to support a candidate.

    Returns:
        ``(value, support)`` where support is the number of values within ``tolerance`` of it;
        ``(None, 0)`` for an empty list.
    """
    if not values:
        return None, 0
    counts = Counter(values)
    best = max(counts, key=lambda v: (sum(c for k, c in counts.items() if abs(k - v) <= tolerance), -v))
    support = sum(c for k, c in counts.items() if abs(k - best) <= tolerance)
    return best, support


def profile_layout(lines: list[ScriptLine]) -> LayoutProfile:
    """Infer the script's column layout from line indentation.

    Action indent: the dominant indent of long (>= 45 chars) mixed-case lines, else the smallest
    mixed-case indent, else the smallest indent. Dialogue indent: the dominant indent of mixed-case
    lines at least 5 columns right of action, accepted only with support >= max(5, 3% of mixed-case
    lines). Cue indent: the dominant indent of cue-shaped lines right of the dialogue column.
    Parenthesised lines are ignored when measuring mixed-case lines.

    Args:
        lines: The extracted script lines.

    Returns:
        The profile; ``mode`` is ``"indented"`` only when a dialogue column was found. No non-blank
        lines give a flush profile with zero indents and ``bold_meaningful=False``.
    """
    nonblank = [ln for ln in lines if not ln.blank]
    if not nonblank:
        return LayoutProfile("flush", 0, None, None, 0.0, False)
    mixed = [ln for ln in nonblank if _has_lower(ln.stripped) and not ln.stripped.startswith("(")]
    long_mixed = [ln.indent for ln in mixed if len(ln.stripped) >= 45]
    if long_mixed:
        action_indent = _mode_with_support(long_mixed)[0] or 0
    elif mixed:
        action_indent = min(ln.indent for ln in mixed)
    else:
        action_indent = min(ln.indent for ln in nonblank)

    dialogue_indent: int | None = None
    candidates = [ln.indent for ln in mixed if ln.indent >= action_indent + 5]
    value, support = _mode_with_support(candidates)
    if value is not None and support >= max(5, int(0.03 * len(mixed))):
        dialogue_indent = value

    cue_indent: int | None = None
    if dialogue_indent is not None:
        cues = [ln.indent for ln in nonblank
                if ln.indent > dialogue_indent + 1 and looks_like_cue_text(ln.stripped)]
        cue_indent, _ = _mode_with_support(cues)

    bold_ratio = sum(1 for ln in nonblank if ln.bold) / len(nonblank)
    return LayoutProfile(
        mode="indented" if dialogue_indent is not None else "flush",
        action_indent=action_indent,
        dialogue_indent=dialogue_indent,
        cue_indent=cue_indent,
        bold_ratio=round(bold_ratio, 4),
        bold_meaningful=0.01 <= bold_ratio <= 0.6,
    )


class _Classifier:
    """Line-by-line heuristic classifier driven by a :class:`LayoutProfile`."""

    def __init__(self, lines: list[ScriptLine], profile: LayoutProfile) -> None:
        """Store the lines to classify and the layout they are judged against.

        Args:
            lines: The extracted script lines.
            profile: Layout inferred by :func:`profile_layout`.
        """
        self.lines = lines
        self.p = profile
        self.n = len(lines)

    # -------------------------------------------------------------- helpers
    def _next_nonblank(self, i: int) -> int | None:
        """Find the next non-blank line after line ``i``.

        Args:
            i: Current line index.

        Returns:
            Index of the next non-blank line, or None when only blank lines follow.
        """
        j = i + 1
        while j < self.n and self.lines[j].blank:
            j += 1
        return j if j < self.n else None

    def _blank_around(self, i: int) -> tuple[bool, bool]:
        """Check whether line ``i`` is isolated by blank lines.

        Args:
            i: Line index.

        Returns:
            ``(blank_before, blank_after)``; the start and end of the text count as blank.
        """
        before = i == 0 or self.lines[i - 1].blank
        after = i + 1 >= self.n or self.lines[i + 1].blank
        return before, after

    def _cue(self, i: int) -> tuple[float, list[str]] | None:
        """Score line ``i`` as a character cue using its shape, its indent and the line that follows.

        The line must pass :func:`looks_like_cue_text` and be followed directly by its speech (one blank
        line in between is tolerated in indented mode, at a penalty); a following scene heading or
        transition rules it out. Indented mode: base 0.5, +0.25 at the cue column (or +0.15 right of
        the dialogue column), +0.2 when the next line is at the dialogue column or a parenthetical
        (else -0.2); a cue at the action margin is accepted only when proper dialogue follows (0.6), and
        lines 15+ columns right of the cue column are rejected. Flush mode: requires mixed-case or
        parenthetical text next (+0.2), +0.05 if indented. Bold adds 0.05 when bold is meaningful.

        Args:
            i: Line index.

        Returns:
            ``(confidence, basis)`` capped at 0.98, or None when the line is not a cue or scores below 0.55.
        """
        line = self.lines[i]
        s = line.stripped
        if not looks_like_cue_text(s):
            return None
        j = i + 1
        gap = False
        if j >= self.n:
            return None
        if self.lines[j].blank:
            if self.p.mode != "indented" or j + 1 >= self.n or self.lines[j + 1].blank:
                return None
            j += 1
            gap = True
        nxt = self.lines[j]
        ns = nxt.stripped
        if is_scene_heading(ns)[0] or TRANSITION_RE.match(ns):
            return None
        nxt_paren = ns.startswith("(")
        score, basis = 0.5, ["caps"]
        if self.p.mode == "indented":
            di = self.p.dialogue_indent or 0
            if line.indent <= self.p.action_indent + 2:
                if abs(nxt.indent - di) <= 3 and (_has_lower(ns) or nxt_paren):
                    score, basis = 0.6, ["caps", "action-margin-cue"]
                else:
                    return None
            elif self.p.cue_indent is not None and abs(line.indent - self.p.cue_indent) <= 4:
                score += 0.25
                basis.append("cue-indent")
            elif line.indent > di:
                score += 0.15
                basis.append("right-of-dialogue")
            if self.p.cue_indent is not None and line.indent >= self.p.cue_indent + 15:
                return None  # far right: transition or page furniture
            if abs(nxt.indent - di) <= 4 or nxt_paren:
                score += 0.2
                basis.append("dialogue-follows")
            else:
                score -= 0.2
        else:
            if not (_has_lower(ns) or nxt_paren):
                return None
            score += 0.2
            basis.append("dialogue-follows")
            if line.indent >= self.p.action_indent + 5:
                score += 0.05
                basis.append("indented")
        if self.p.bold_meaningful and line.bold:
            score += 0.05
            basis.append("bold")
        if gap:
            score -= 0.15
            basis.append("blank-after-cue")
        if score < 0.55:
            return None  # too weak: leave it to the shot/action rules (and flag as uncertain there)
        return round(min(score, 0.98), 3), basis

    @staticmethod
    def _split_trailing_speech(line: ScriptLine, lab: _Label) -> None:
        """``(aside) I'm on hold..`` - speech after the closing parenthesis on the same line.

        Sets ``lab.split_at`` to the absolute offset of the first non-space character after the first
        ``)`` when letters follow it, and adds a note. Does nothing when the line has no ``)``.

        Args:
            line: A parenthetical line.
            lab: Its label, modified in place.
        """
        close = line.text.find(")")
        if close < 0:
            return
        rest = line.text[close + 1:]
        if any(c.isalpha() for c in rest):
            lab.split_at = line.start + close + 1 + (len(rest) - len(rest.lstrip()))
            lab.notes.append("speech follows the parenthetical on the same line")

    def _is_transition(self, line: ScriptLine) -> bool:
        """Decide whether a line is a transition such as ``CUT TO:``.

        Args:
            line: The line to test.

        Returns:
            True for an all-caps line of at most 40 characters that matches ``TRANSITION_RE``, or (in
            indented mode) ends with ``:`` and sits at least 10 columns right of the cue column
            (column 40 when the cue column is unknown).
        """
        s = line.stripped
        if not _is_upper_text(s) or len(s) > 40:
            return False
        if TRANSITION_RE.match(s):
            return True
        right = (self.p.cue_indent or 30) + 10
        return self.p.mode == "indented" and line.indent >= right and s.endswith(":")

    # -------------------------------------------------------------- main
    def label(self) -> list[_Label | None]:
        """Classify every line, tracking whether we are inside a speech block.

        Order of checks per line: continuation marker, page number (far right, bold or isolated), then,
        inside a speech block, parentheticals (including multi-line ones) and dialogue lines. A speech
        block ends at a blank line, a scene heading/transition, a new inline ``NAME:`` line (flush
        mode), a return to the action margin, or a new cue at the cue column. Everything else is
        classified by :meth:`_classify_free`; a ``character``/``inline_cue`` result opens a speech block.

        Returns:
            One label per line, in line order; None for blank lines.
        """
        labels: list[_Label | None] = [None] * self.n
        in_dialogue = False
        paren_open = False
        for i, line in enumerate(self.lines):
            if line.blank:
                in_dialogue = False
                paren_open = False
                continue
            s = line.stripped
            before_blank, after_blank = self._blank_around(i)
            lab: _Label | None = None
            extra_notes: list[str] = []
            bold = ["bold"] if line.bold and self.p.bold_meaningful else []

            if CONTINUATION_RE.match(s):
                lab = _Label("continuation", 0.9, ["continued-marker"], in_dialogue=in_dialogue)
            elif PAGE_RE.match(s) and (line.indent >= self.p.action_indent + 15 or line.bold
                                       or (before_blank and after_blank)):
                lab = _Label("page_marker", 0.85, ["page-number"])
            elif in_dialogue:
                if paren_open:
                    lab = _Label("parenthetical", 0.9, ["parenthetical-continues"], True, joins_previous=True)
                    paren_open = ")" not in s
                    self._split_trailing_speech(line, lab)
                elif s.startswith("("):
                    lab = _Label("parenthetical", 0.92, ["parenthesized"], True)
                    paren_open = ")" not in s
                    self._split_trailing_speech(line, lab)
                elif is_scene_heading(s)[0] or self._is_transition(line):
                    in_dialogue = False
                elif self.p.mode == "flush" and (m := INLINE_CUE_RE.match(s)) and looks_like_cue_text(m.group("cue")):
                    in_dialogue = False  # next "NAME: speech" line starts a new block
                elif (self.p.mode == "indented" and self.p.dialogue_indent is not None
                      and self.p.dialogue_indent - self.p.action_indent >= 6
                      and line.indent <= self.p.action_indent + 2):
                    in_dialogue = False  # action resumed without a blank line
                    extra_notes.append("dialogue block ended without blank line")
                elif self.p.mode == "indented" and self.p.cue_indent is not None \
                        and abs(line.indent - self.p.cue_indent) <= 4 and self._cue(i):
                    in_dialogue = False  # a new cue directly after dialogue
                else:
                    conf = 0.9 if self.p.mode == "indented" else 0.78
                    basis = ["follows-cue"]
                    if self.p.mode == "indented" and self.p.dialogue_indent is not None:
                        if abs(line.indent - self.p.dialogue_indent) <= 4:
                            basis.append("dialogue-indent")
                        else:
                            conf -= 0.15
                            basis.append("off-indent")
                    lab = _Label("dialogue", conf, basis, True)
                    if re.search(r"\S {8,}\S", s):
                        lab.notes.append("possible dual dialogue (two text columns)")
                        lab.confidence = min(lab.confidence, 0.6)

            if lab is None:
                lab = self._classify_free(i, line, before_blank, after_blank, bold)
                lab.notes.extend(extra_notes)
                if lab.type in {"character", "inline_cue"}:
                    in_dialogue = True
                    paren_open = False
            labels[i] = lab
        return labels

    def _classify_free(self, i: int, line: ScriptLine, before_blank: bool, after_blank: bool,
                       bold: list[str]) -> _Label:
        """Classify a line that is not inside a speech block.

        Checks in priority order: scene heading, transition, act marker, character cue, inline
        ``NAME: speech`` cue (flush mode only), upper-case shot/sub-heading lines, and finally action,
        whose confidence depends on how well the indent matches the action column (a line at the
        dialogue column without a cue gets 0.5).

        Args:
            i: Line index.
            line: The line itself.
            before_blank: Whether the previous line is blank (or this is the first line).
            after_blank: Whether the next line is blank (or this is the last line).
            bold: ``["bold"]`` when the line is bold and bold is meaningful, else ``[]``; appended to basis.

        Returns:
            The line's label; never None (action is the fallback).
        """
        s = line.stripped
        ok, conf = is_scene_heading(s)
        if ok:
            return _Label("scene_heading", min(0.99, conf + (0.02 if bold else 0)), ["int-ext"] + bold)
        if self._is_transition(line):
            return _Label("transition", 0.9, ["transition-pattern"] + bold)
        if _is_upper_text(s) and len(s) <= 40 and ACT_RE.match(s):
            return _Label("act_marker", 0.9, ["act-pattern"] + bold)
        cue = self._cue(i)
        if cue:
            return _Label("character", cue[0], cue[1])
        if self.p.mode == "flush":
            match = INLINE_CUE_RE.match(s)
            if match and _has_lower(match.group("speech")) and looks_like_cue_text(match.group("cue")):
                split = line.content_start + match.start("speech")
                return _Label("inline_cue", 0.65, ["inline-name-colon"], split_at=split)
        if _is_upper_text(s) and len(s) <= 60:
            if SHOT_RE.match(s):
                return _Label("shot", 0.8, ["shot-pattern"] + bold)
            if bold and before_blank:
                return _Label("shot", 0.65, ["bold-subheading"] + bold,
                              notes=["upper-case bold line treated as secondary heading"])
            if before_blank and after_blank and len(s.split()) <= 6:
                return _Label("shot", 0.55, ["standalone-caps"],
                              notes=["isolated upper-case line; could be a cue, shot or title"])
        conf, basis = 0.85, ["default"]
        if self.p.mode == "indented":
            if abs(line.indent - self.p.action_indent) <= 3:
                basis = ["action-indent"]
            elif self.p.dialogue_indent is not None and abs(line.indent - self.p.dialogue_indent) <= 3:
                conf, basis = 0.5, ["dialogue-indent-without-cue"]
            else:
                conf, basis = 0.65, ["unusual-indent"]
        else:
            conf = 0.8
        return _Label("action", conf, basis)


# Types whose consecutive lines are merged into a single multi-line element.
_MERGEABLE = {"action", "dialogue", "title_page"}


def _make_element(index: int, kind: str, lines: list[ScriptLine], cs: int, ce: int, full: str,
                  conf: float, basis: list[str], in_dialogue: bool, notes: list[str]) -> Element:
    """Build an :class:`Element` covering ``full[cs:ce]``.

    Args:
        index: Position of the new element in the element list.
        kind: Element type.
        lines: The source lines covered (first and last give the line range, first gives the indent).
        cs: Start char offset.
        ce: End char offset (exclusive).
        full: The full script text.
        conf: Confidence (rounded to 3 decimals).
        basis: Classification reasons (deduplicated and sorted).
        in_dialogue: Whether the element belongs to a speech block.
        notes: Remarks (deduplicated, order kept).

    Returns:
        The new element; ``bold`` is True only when every covered line is bold.
    """
    text = full[cs:ce]
    return Element(
        index=index, type=kind, line_start=lines[0].index, line_end=lines[-1].index,
        char_start=cs, char_end=ce, text=text, content=collapse_whitespace(text), confidence=round(conf, 3),
        basis=sorted(set(basis)), bold=all(ln.bold for ln in lines), indent=lines[0].indent,
        in_dialogue=in_dialogue, notes=list(dict.fromkeys(notes)),
    )


def build_elements(lines: list[ScriptLine], labels: list[_Label | None], full: str) -> list[Element]:
    """Merge labelled lines into elements.

    Contiguous lines (no blank line between) of the same mergeable type and dialogue state become one
    element, as do the lines of a multi-line parenthetical; all other lines become their own element.
    An inline ``NAME: speech`` line is split into a ``character`` element and a ``dialogue`` element,
    and a parenthetical followed by speech on the same line is split at the closing ``)``. The element
    confidence is the minimum over its lines.

    Args:
        lines: The extracted script lines.
        labels: One label per line from :meth:`_Classifier.label` (None for blank lines).
        full: The full script text the line offsets refer to.

    Returns:
        Elements in document order with ``index`` equal to their list position; empty when every line
        is blank.
    """
    elements: list[Element] = []
    group: list[ScriptLine] = []
    group_label: _Label | None = None
    group_start: int | None = None
    conf = 1.0
    basis: list[str] = []
    notes: list[str] = []

    def flush() -> None:
        """Emit the current group of lines as one element (if any) and reset the group state."""
        nonlocal group, group_label, group_start, conf, basis, notes
        if group and group_label is not None:
            cs = group_start if group_start is not None else group[0].content_start
            elements.append(_make_element(len(elements), group_label.type, group, cs, group[-1].content_end,
                                          full, conf, basis, group_label.in_dialogue, notes))
        group, group_label, group_start, conf, basis, notes = [], None, None, 1.0, [], []

    prev_index = -2
    for line, lab in zip(lines, labels):
        if lab is None:
            continue
        contiguous = line.index == prev_index + 1
        prev_index = line.index
        if lab.type == "inline_cue":
            flush()
            assert lab.split_at is not None
            cue_end = line.content_start + len(line.stripped[: lab.split_at - line.content_start].rstrip())
            el = _make_element(len(elements), "character", [line], line.content_start, cue_end, full,
                               lab.confidence, lab.basis, False, lab.notes)
            el.cue_text = el.content.rstrip(":").strip()
            elements.append(el)
            group, group_label, group_start = [line], _Label("dialogue", lab.confidence, lab.basis, True), lab.split_at
            conf, basis, notes = lab.confidence, list(lab.basis), []
            continue
        same = (group_label is not None and contiguous
                and ((lab.type == group_label.type and lab.type in _MERGEABLE
                      and lab.in_dialogue == group_label.in_dialogue)
                     or (lab.joins_previous and group_label.type == "parenthetical")))
        if not same:
            flush()
            group_label = lab
        group.append(line)
        conf = min(conf, lab.confidence)
        basis.extend(lab.basis)
        notes.extend(lab.notes)
        if lab.type == "parenthetical" and lab.split_at is not None:
            # Close the parenthetical at ")" and start a dialogue element with the trailing speech.
            close_end = line.start + line.text.find(")") + 1
            cs = group_start if group_start is not None else group[0].content_start
            elements.append(_make_element(len(elements), "parenthetical", group, cs, close_end, full, conf,
                                          basis, True, notes))
            group, group_label, group_start = [line], _Label("dialogue", lab.confidence, ["same-line-speech"], True), \
                lab.split_at
            conf, basis, notes = lab.confidence, ["same-line-speech"], []
    flush()
    return elements


def _mark_title_page(elements: list[Element], profile: LayoutProfile) -> None:
    """Re-label front matter (before the first heading / FADE IN / act) as ``title_page``.

    Modifies the elements in place. First, a ``character`` followed within three elements by a strong
    credit phrase (``written by``, ``screenplay by``...) is re-labelled together with its follow-up
    lines. Then action/shot elements in the front region become ``title_page`` when centred (8+ columns
    right of the action column), when they match ``TITLE_PAGE_RE``, or when the region has no
    dialogue at all (character elements too in that case). Nothing changes when there is no boundary
    element or it is the first element.

    Args:
        elements: Elements from :func:`build_elements`.
        profile: Layout profile, for the action column.
    """
    boundary = next((e.index for e in elements
                     if e.type in {"scene_heading", "act_marker"}
                     or (e.type == "transition" and "FADE" in e.content.upper())), None)
    if boundary is None or boundary == 0:
        return
    region = elements[:boundary]
    # A centred title followed by "written by ..." looks like cue + dialogue; relabel such pairs.
    for pos, el in enumerate(region):
        if el.type != "character":
            continue
        follow: list[Element] = []
        for e in region[pos + 1:pos + 4]:
            if e.type == "character":
                break
            follow.append(e)
        if follow and STRONG_TITLE_RE.search(" ".join(e.content for e in follow)):
            for e in [el, *[f for f in follow if f.type in {"dialogue", "parenthetical", "action", "shot"}]]:
                e.type = "title_page"
                e.in_dialogue = False
                e.confidence = min(e.confidence, 0.7)
                e.basis = sorted(set(e.basis) | {"front-matter", "credit-line"})
    has_dialogue = any(e.type == "dialogue" for e in region)
    for el in region:
        if el.type not in {"action", "shot", "character"}:
            continue
        if el.type == "character" and has_dialogue:
            continue
        centered = el.indent >= profile.action_indent + 8
        if centered or TITLE_PAGE_RE.search(el.content) or not has_dialogue:
            el.type = "title_page"
            el.confidence = min(el.confidence, 0.75) if centered or TITLE_PAGE_RE.search(el.content) else 0.6
            el.basis = sorted(set(el.basis) | {"front-matter"})


# ====================================================================== orchestration
@dataclass(slots=True)
class ParsedScreenplay:
    """The full structural parse of a screenplay.

    Attributes:
        parser_version: :data:`src.parsers.PARSER_VERSION` used to produce the parse.
        layout: The inferred layout profile.
        elements: All classified elements in document order.
        scenes: Scenes (plus an optional prelude), each with characters and dialogue counts filled in.
        dialogues: Dialogue blocks with their ``scene_index`` set.
        characters: Per-character summaries, by first appearance.
        stats: Counts (lines, words, elements by type, scenes, dialogue blocks, uncertain ratio, page
            markers...) and ``encoding_problems``.
        warnings: Human-readable problems (segmentation, high uncertainty).
        quality_flags: Machine-readable flags such as ``no_scene_headings``, ``no_dialogue``,
            ``high_uncertainty``, ``very_short_text``, ``fallback_extraction``, ``encoding_suspect``.
    """

    parser_version: str
    layout: LayoutProfile
    elements: list[Element]
    scenes: list[Scene]
    dialogues: list[DialogueBlock]
    characters: list[CharacterSummary]
    stats: dict[str, Any]
    warnings: list[str]
    quality_flags: list[str]


def parse_screenplay(extracted: ExtractedScript) -> ParsedScreenplay:
    """Parse extracted screenplay text into elements, scenes, dialogue blocks and characters.

    Pipeline: layout profiling, line classification, element building, front-matter re-labelling,
    scene segmentation, dialogue assembly and character aggregation, then statistics and quality flags.
    A cue without speech that is directly followed by an action paragraph gets both flagged for review
    (the action's confidence drops to 0.5) instead of being re-classified.

    Args:
        extracted: Output of :func:`extract_screenplay`.

    Returns:
        The parse. Empty text yields no elements or scenes, a ``"no elements to segment"`` warning and
        flags such as ``no_scene_headings``, ``no_dialogue`` and ``very_short_text``.
    """
    lines = extracted.lines
    profile = profile_layout(lines)
    labels = _Classifier(lines, profile).label()
    elements = build_elements(lines, labels, extracted.text)
    _mark_title_page(elements, profile)

    scenes, scene_warnings = segment_scenes(elements, len(extracted.text))
    for scene_idx, scene in enumerate(scenes):
        for el in elements[scene.element_start:scene.element_end + 1]:
            el.scene_index = scene_idx

    dialogues = build_dialogue_blocks(elements, extracted.text)
    for block in dialogues:
        block.scene_index = elements[block.cue_element].scene_index
        if block.text or block.element_end + 1 >= len(elements):
            continue
        # A cue with no speech followed (after at most one blank line) by an action paragraph:
        # often speech mis-indented in the source. Flag it for review instead of guessing.
        nxt = elements[block.element_end + 1]
        if nxt.type == "action" and nxt.line_start - elements[block.element_end].line_end <= 2:
            nxt.confidence = min(nxt.confidence, 0.5)
            nxt.notes.append(f"possibly dialogue of preceding cue {block.speaker} (source layout ambiguous)")
            block.warnings.append(f"element {nxt.index} may be this cue's speech (mis-indented in source)")
    for scene_idx, scene in enumerate(scenes):
        speakers: list[str] = []
        count = 0
        for block in dialogues:
            if block.scene_index == scene_idx:
                count += 1
                if block.speaker and block.speaker not in speakers:
                    speakers.append(block.speaker)
        scene.characters = speakers
        scene.dialogue_count = count
    characters = aggregate_characters(dialogues)

    counts = Counter(e.type for e in elements)
    uncertain = sum(1 for e in elements if e.uncertain)
    page_numbers = [e.content.strip("().") for e in elements if e.type == "page_marker"]
    stats = {
        "lines": len(lines),
        "nonblank_lines": sum(1 for ln in lines if not ln.blank),
        "characters_text": len(extracted.text),
        "words": len(extracted.text.split()),
        "elements": len(elements),
        "element_counts": dict(sorted(counts.items())),
        "scenes": sum(1 for s in scenes if s.kind == "scene"),
        "has_prelude": any(s.kind == "prelude" for s in scenes),
        "dialogue_blocks": len(dialogues),
        "characters": len(characters),
        "parentheticals": counts.get("parenthetical", 0),
        "transitions": counts.get("transition", 0),
        "act_markers": counts.get("act_marker", 0),
        "uncertain_elements": uncertain,
        "uncertain_ratio": round(uncertain / len(elements), 4) if elements else 0.0,
        "page_markers": len(page_numbers),
        "last_page": page_numbers[-1] if page_numbers else None,
        "empty_dialogue_blocks": sum(1 for b in dialogues if not b.text),
    }

    warnings = list(scene_warnings)
    flags: list[str] = []
    if stats["scenes"] == 0:
        flags.append("no_scene_headings")
    if not dialogues:
        flags.append("no_dialogue")
    if stats["uncertain_ratio"] > 0.25:
        flags.append("high_uncertainty")
        warnings.append(f"{uncertain} of {len(elements)} elements have low classification confidence")
    if len(extracted.text.strip()) < 2000:
        flags.append("very_short_text")
    if extracted.extraction_ratio is not None and extracted.visible_page_chars > 5000 \
            and extracted.extraction_ratio < 0.5:
        flags.append("low_extraction_ratio")
    if extracted.method in {"largest_pre", "body_fallback"}:
        flags.append("fallback_extraction")
    problems = encoding_problems(extracted.text)
    if problems["replacement_chars"] or problems["mojibake_sequences"] > 3 or problems["control_chars"]:
        flags.append("encoding_suspect")
    if stats["empty_dialogue_blocks"]:
        flags.append("empty_dialogue_blocks")
    if any("dual dialogue" in n for e in elements for n in e.notes) or any(b.dual for b in dialogues):
        flags.append("possible_dual_dialogue")
    stats["encoding_problems"] = problems

    return ParsedScreenplay(
        parser_version=PARSER_VERSION, layout=profile, elements=elements, scenes=scenes,
        dialogues=dialogues, characters=characters, stats=stats, warnings=warnings, quality_flags=flags,
    )
