"""Scene heading recognition and scene segmentation."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from src.utils.text_processing import collapse_whitespace

if TYPE_CHECKING:  # pragma: no cover
    from src.parsers.screenplay_parser import Element

# Optional leading scene number ("12", "12A", "12."), then the INT/EXT prefix.
HEADING_RE = re.compile(
    r"""^(?P<num>\d{1,4}[A-Z]{0,2}[.)]?\s+)?
        (?P<prefix>INT\.?\s*/\s*EXT|EXT\.?\s*/\s*INT|INT\.?\s*-\s*EXT|I\s*/\s*E|E\s*/\s*I|
           INTERIOR|EXTERIOR|INT|EXT|EST)
        (?=[\s.:/\-–—]|$)""",
    re.VERBOSE | re.IGNORECASE,
)
TIME_OF_DAY = {
    "DAY", "NIGHT", "MORNING", "EVENING", "AFTERNOON", "DUSK", "DAWN", "SUNSET", "SUNRISE",
    "NOON", "MIDNIGHT", "LATER", "MOMENTS LATER", "CONTINUOUS", "SAME", "SAME TIME", "LATE NIGHT",
    "EARLY MORNING", "MAGIC HOUR", "TWILIGHT", "PRE-DAWN", "FLASHBACK", "PRESENT", "NIGHT (LATER)",
    "DAY (LATER)", "LATE AFTERNOON", "LATE DAY", "NEXT DAY", "NEXT MORNING", "DAYS LATER",
}
# Heading part separator: a spaced "-"/"--"/en/em dash, or an unspaced "--"/en/em dash. A bare hyphen
# inside a word ("SUB-BASEMENT") does not split.
_SEP_RE = re.compile(r"\s+(?:-{1,2}|–|—)\s+|\s*(?:–|—|--)\s*")
# Scene number repeated at the end of a numbered heading ("... - DAY 12A").
_TRAILING_NUM_RE = re.compile(r"\s+\d{1,4}[A-Z]{0,2}\.?$")


@dataclass(slots=True)
class HeadingParts:
    """A scene heading split into its components, e.g. ``"12 INT. KITCHEN - NIGHT"``.

    Attributes:
        raw: The heading with whitespace collapsed.
        scene_number: Printed scene number without trailing ``.``/``)`` (e.g. ``"12A"``), or None.
        int_ext: Normalised prefix: ``INT``, ``EXT``, ``INT/EXT``, ``EXT/INT`` or ``EST``; None when the
            text is not a heading.
        location: Location parts joined with ``" - "`` (e.g. ``"HOUSE - KITCHEN"``), or None.
        time_of_day: Trailing time part such as ``"NIGHT"`` or ``"MOMENTS LATER"``, or None.
        extra: Reserved for additional parts; always empty as currently produced by :func:`parse_heading`.
    """

    raw: str
    scene_number: str | None
    int_ext: str | None
    location: str | None
    time_of_day: str | None
    extra: list[str] = field(default_factory=list)


def is_scene_heading(text: str) -> tuple[bool, float]:
    """Decide whether a line is a scene heading (slugline). Upper-case headings score higher.

    The line must start with an optional scene number and an INT/EXT-style prefix (``HEADING_RE``)
    followed by a separator or end of line, and be at most 160 characters. The share of upper-case
    letters then decides: >= 0.8 gives confidence 0.97, >= 0.5 gives 0.75, less is rejected. A bare
    prefix with nothing after it (``"INT."``) counts only when > 90% upper case, at confidence 0.6.

    Args:
        text: One line of screenplay text (surrounding whitespace is ignored).

    Returns:
        ``(is_heading, confidence)``; ``(False, 0.0)`` for empty, too long or non-matching text.
        For a bare prefix the confidence is 0.6 even when ``is_heading`` is False.
    """
    stripped = text.strip()
    if not stripped or len(stripped) > 160:
        return False, 0.0
    match = HEADING_RE.match(stripped)
    if not match:
        return False, 0.0
    letters = [c for c in stripped if c.isalpha()]
    upper_ratio = sum(c.isupper() for c in letters) / max(1, len(letters))
    # Prose such as "Int. design matters" must not count: require a separator after the prefix.
    rest = stripped[match.end():]
    if not rest.strip(" .:/-–—"):
        return (upper_ratio > 0.9, 0.6)
    if upper_ratio >= 0.8:
        return True, 0.97
    if upper_ratio >= 0.5:
        return True, 0.75
    return False, 0.0


def parse_heading(text: str) -> HeadingParts:
    """Split a scene heading into scene number, INT/EXT, location and time of day.

    Prefix variants (``INTERIOR``, ``I/E``, ``INT.-EXT``...) are normalised. The rest is split on dash
    separators; the last part becomes ``time_of_day`` when it looks like a time (``DAY``, ``LATER``, a
    four-digit year...). A lone part is taken as the time only if it is an exact ``TIME_OF_DAY`` entry,
    otherwise it is the location. A scene number repeated at the end is dropped.

    Args:
        text: The heading line, e.g. ``"12 INT. HOUSE - KITCHEN - NIGHT 12"``.

    Returns:
        The parsed parts. When ``text`` does not match ``HEADING_RE`` only ``raw`` is set and all other
        fields are None.
    """
    raw = collapse_whitespace(text)
    match = HEADING_RE.match(raw)
    if not match:
        return HeadingParts(raw=raw, scene_number=None, int_ext=None, location=None, time_of_day=None)
    number = (match.group("num") or "").strip().rstrip(".)") or None
    prefix = re.sub(r"\s+", "", match.group("prefix").upper()).rstrip(".")
    int_ext = {
        "INTERIOR": "INT", "EXTERIOR": "EXT", "I/E": "INT/EXT", "E/I": "EXT/INT",
        "INT-EXT": "INT/EXT", "INT.-EXT": "INT/EXT", "INT./EXT": "INT/EXT", "EXT./INT": "EXT/INT",
    }.get(prefix, prefix)
    rest = raw[match.end():].lstrip(" .:/-–—")
    if number and _TRAILING_NUM_RE.search(rest):
        rest = _TRAILING_NUM_RE.sub("", rest)  # "12 INT. HOUSE - DAY 12"
    parts = [p.strip(" .,") for p in _SEP_RE.split(rest) if p and p.strip(" .,")]
    time_of_day: str | None = None
    if len(parts) >= 2 and _looks_like_time(parts[-1]):
        time_of_day = parts.pop()
    elif len(parts) == 1 and _looks_like_time(parts[0]) and parts[0].upper() in TIME_OF_DAY:
        time_of_day = parts.pop()
    location = " - ".join(parts) or None
    return HeadingParts(raw=raw, scene_number=number, int_ext=int_ext, location=location,
                        time_of_day=time_of_day)


_TIME_WORDS = {"DAY", "NIGHT", "MORNING", "EVENING", "AFTERNOON", "DUSK", "DAWN", "LATER",
               "CONTINUOUS", "SAME", "SUNSET", "SUNRISE", "NOON", "MIDNIGHT", "TWILIGHT", "MOMENTS"}


def _looks_like_time(part: str) -> bool:
    """Guess whether a heading part is a time of day rather than a location.

    Args:
        part: One dash-separated part of a heading, e.g. ``"NIGHT (LATER)"`` or ``"(1985)"``.

    Returns:
        True when the part (ignoring surrounding parentheses) is a ``TIME_OF_DAY`` entry, contains any
        ``_TIME_WORDS`` word, or is a four-digit year optionally in parentheses.
    """
    upper = part.upper().strip("() ")
    if upper in TIME_OF_DAY:
        return True
    words = set(re.findall(r"[A-Z]+", upper))
    return bool(words & _TIME_WORDS) or bool(re.fullmatch(r"\(?\d{4}\)?", part.strip()))


@dataclass(slots=True)
class Scene:
    """A contiguous run of elements: one scene, the prelude before the first heading, or the whole script.

    Attributes:
        scene_number: 1-based number among real scenes; 0 for the prelude or an unsegmented script.
        kind: ``"prelude"``, ``"scene"`` or ``"unsegmented"`` (script without any headings).
        heading: The scene heading text, or None for prelude/unsegmented.
        heading_parts: :class:`HeadingParts` as a dict, or None without a heading.
        element_start: Index of the first element.
        element_end: Index of the last element (inclusive).
        line_start: First source line of the scene.
        line_end: Last source line of the scene.
        char_start: Character offset of the scene start in the script text.
        char_end: Character offset of the scene end, clamped to the text length.
        confidence: Confidence of the heading element; 0.5 without a heading.
        act: Act in force at the scene start, or the first act marker inside it; None if none.
        page_start: Page number where the scene starts, when the script has page markers.
        page_end: Page number in force at the scene end.
        description: Action paragraphs of the scene joined by blank lines.
        characters: Speaking characters (filled in later by the screenplay parser).
        shots: Shot directions (``CLOSE ON ...``) in order.
        transitions: Transitions (``CUT TO:``) in order.
        element_counts: Number of elements of each type, including the heading.
        dialogue_count: Number of dialogue blocks (filled in later by the screenplay parser).
        warnings: Problems found, e.g. a heading with no content.
    """

    scene_number: int  # 0 = prelude (content before the first heading)
    kind: str  # "prelude" | "scene" | "unsegmented"
    heading: str | None
    heading_parts: dict[str, Any] | None
    element_start: int
    element_end: int  # inclusive
    line_start: int
    line_end: int
    char_start: int
    char_end: int
    confidence: float
    act: str | None = None
    page_start: str | None = None
    page_end: str | None = None
    description: str = ""
    characters: list[str] = field(default_factory=list)
    shots: list[str] = field(default_factory=list)
    transitions: list[str] = field(default_factory=list)
    element_counts: dict[str, int] = field(default_factory=dict)
    dialogue_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the scene to a plain dict for storage.

        Returns:
            All fields as a (recursively converted) dict.
        """
        return asdict(self)


def segment_scenes(elements: list["Element"], text_length: int) -> tuple[list[Scene], list[str]]:
    """Group elements into scenes. Every element belongs to exactly one scene.

    A new scene starts at every ``scene_heading`` element; elements before the first heading form a
    ``prelude`` scene, and a script with no headings becomes one ``unsegmented`` scene. Act markers
    and page markers are tracked across scenes (text before the first page marker is page "1").

    Args:
        elements: Classified elements in document order; ``element.index`` must equal its list position.
        text_length: Length of the script text, used to clamp ``char_end``.

    Returns:
        ``(scenes, warnings)``. For an empty element list: ``([], ["no elements to segment"])``.
    """
    warnings: list[str] = []
    if not elements:
        return [], ["no elements to segment"]
    heading_idx = [e.index for e in elements if e.type == "scene_heading"]
    starts: list[tuple[int, str]] = []
    if not heading_idx:
        warnings.append("no scene headings detected; script stored as a single unsegmented scene")
        starts = [(0, "unsegmented")]
    else:
        if heading_idx[0] > 0:
            starts.append((0, "prelude"))
        starts.extend((idx, "scene") for idx in heading_idx)

    scenes: list[Scene] = []
    act: str | None = None
    # Page markers usually print the number at the top of the *next* page, so text before
    # the first marker is page 1 (only assumed when the script has page markers at all).
    page: str | None = "1" if any(e.type == "page_marker" for e in elements) else None
    number = 0
    for pos, (start, kind) in enumerate(starts):
        end = (starts[pos + 1][0] - 1) if pos + 1 < len(starts) else len(elements) - 1
        chunk = elements[start:end + 1]
        heading_el = chunk[0] if kind == "scene" else None
        if kind == "scene":
            number += 1
        counts: dict[str, int] = {}
        description_parts: list[str] = []
        shots: list[str] = []
        transitions: list[str] = []
        page_start = page
        scene_act = act
        for el in chunk:
            counts[el.type] = counts.get(el.type, 0) + 1
            if el.type == "action":
                description_parts.append(el.content)
            elif el.type == "shot":
                shots.append(el.content)
            elif el.type == "transition":
                transitions.append(el.content)
            elif el.type == "act_marker":
                ends = el.content.upper().startswith("END OF")
                if scene_act is None and not ends:
                    scene_act = el.content
                act = None if ends else el.content
            elif el.type == "page_marker":
                page = el.content.strip("().")
                page_start = page_start or page
        scene = Scene(
            scene_number=number if kind == "scene" else 0,
            kind=kind,
            heading=heading_el.content if heading_el else None,
            heading_parts=asdict(parse_heading(heading_el.content)) if heading_el else None,
            element_start=start,
            element_end=end,
            line_start=chunk[0].line_start,
            line_end=chunk[-1].line_end,
            char_start=chunk[0].char_start,
            char_end=min(chunk[-1].char_end, text_length),
            confidence=heading_el.confidence if heading_el else 0.5,
            act=scene_act,
            page_start=page_start,
            page_end=page,
            description="\n\n".join(description_parts),
            shots=shots,
            transitions=transitions,
            element_counts=counts,
        )
        if kind == "scene" and len(chunk) == 1:
            scene.warnings.append("scene heading with no content")
        scenes.append(scene)
    return scenes, warnings
