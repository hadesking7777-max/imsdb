"""Character cue parsing, dialogue block assembly and character aggregation."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from src.utils.text_processing import collapse_whitespace, slugify

if TYPE_CHECKING:  # pragma: no cover
    from src.parsers.screenplay_parser import Element

_PAREN_RE = re.compile(r"\(([^)]*)\)")
# Unparenthesised cue extension at the end of a name: "BOB CONT'D", "ANNA V.O.", "MAN O.S.".
_TRAILING_EXT_RE = re.compile(
    r"\s+(CONT'?D\.?|CONT\.|CONTINUING|CONTD|V\.\s?O\.?|O\.\s?S\.?|O\.\s?C\.?|VO|OS|OC)$", re.IGNORECASE
)
# Name ending in an abbreviation whose final dot must be kept: "MR.", "J.D.", "DR. LEE JR.".
_ABBREVIATION_TAIL_RE = re.compile(r"(?:^|\s)(?:[A-Z]{1,3}\.)+$")
# Words that look like cues (short, upper-case) but are camera/structure directions.
NON_CUE_WORDS = {
    "CLOSE ON", "CLOSE UP", "CLOSE-UP", "CLOSEUP", "ANGLE ON", "NEW ANGLE", "ANOTHER ANGLE",
    "REVERSE ANGLE", "WIDE SHOT", "WIDE ON", "INSERT", "POV", "BACK TO SCENE", "AERIAL SHOT",
    "TRACKING SHOT", "MONTAGE", "END MONTAGE", "SERIES OF SHOTS", "FLASHBACK", "END FLASHBACK",
    "BACK TO PRESENT", "LATER", "MOMENTS LATER", "CONTINUOUS", "SAME", "INTERCUT", "SUPER",
    "TITLE", "TITLE CARD", "CREDITS", "OPENING CREDITS", "END CREDITS", "BLACK", "BLACKNESS",
    "OVER BLACK", "SILENCE", "THE END", "END", "FREEZE FRAME", "SLOW MOTION", "SPLIT SCREEN",
    "CONTINUED", "MORE", "OMITTED", "PRE-LAP", "TIME CUT", "BEGIN MONTAGE", "MONTAGE ENDS",
    "BEGIN FLASHBACK", "FLASHBACK ENDS", "DREAM SEQUENCE", "END DREAM SEQUENCE",
}
# Camera / editing directions at the start of a line (CLOSE ON, ANGLE ON, JOHN'S POV, SUPER:, ...).
SHOT_RE = re.compile(
    r"^(?:CLOSE(?:\s|-)?(?:ON|UP|SHOT)|EXTREME CLOSE|ANGLE ON|NEW ANGLE|ANOTHER ANGLE|REVERSE(?: ANGLE)?|"
    r"WIDE(?: SHOT| ON| ANGLE)?|ON THE\b|ON\s+[A-Z]|INSERT|(?:.+'S\s+)?P\.?O\.?V\.?|BACK TO(?: SCENE| PRESENT)?|"
    r"AERIAL|TRACKING|MONTAGE|SERIES OF SHOTS|END (?:MONTAGE|FLASHBACK|SERIES)|FLASHBACK|SUPER(?:IMPOSE)?:?|"
    r"TITLE(?: CARD)?:?|CARD:|(?:OPENING |END )?CREDITS|LATER|MOMENTS LATER|CONTINUOUS|INTERCUT|"
    r"SPLIT SCREEN|FREEZE FRAME|SLOW MOTION|OVER BLACK|BLACK SCREEN|PRE-LAP|DREAM SEQUENCE|"
    r"UNDERWATER|OMITTED|SERIES OF|SAME TIME|MEANWHILE)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class Cue:
    """A parsed character cue (the speaker line above a dialogue block).

    Attributes:
        raw: The cue as written, stripped.
        name: Normalised upper-case speaker name without extensions, e.g. ``"ANCHORWOMAN"``.
        key: Slug of ``name`` used to group variants of the same character.
        extensions: Parenthesised and trailing extensions in order found, e.g. ``["ON TV", "CONT'D"]``.
        dual: True when the cue ended with ``^`` (dual-dialogue marker).
    """

    raw: str
    name: str
    key: str
    extensions: list[str] = field(default_factory=list)
    dual: bool = False


def parse_cue(raw: str) -> Cue:
    """Split a character cue into name and extensions: ``"ANCHORWOMAN (ON TV)"`` -> ``ANCHORWOMAN``, ``["ON TV"]``.

    Curly apostrophes become ``'``; a trailing ``^`` sets ``dual``; a trailing ``:`` is dropped. Every
    ``(...)`` group becomes an extension, then unparenthesised trailing ``CONT'D``/``V.O.``/``O.S.``/
    ``O.C.`` forms are peeled off (upper-cased). A final ``.`` is removed unless the name ends in a short
    abbreviation (``MR.``).

    Args:
        raw: The cue line text.

    Returns:
        The parsed cue. ``name`` and ``key`` are ``""`` and ``"untitled"`` respectively when nothing is
        left after removing extensions.
    """
    text = collapse_whitespace(raw.replace("’", "'").replace("‘", "'"))
    dual = False
    if text.endswith("^"):
        dual = True
        text = text.rstrip("^ ").strip()
    text = text.rstrip(":").strip()
    extensions = [collapse_whitespace(e) for e in _PAREN_RE.findall(text) if e.strip()]
    name = _PAREN_RE.sub(" ", text)
    name = name.replace("(", " ").replace(")", " ")
    while True:
        match = _TRAILING_EXT_RE.search(name)
        if not match:
            break
        extensions.append(match.group(1).upper())
        name = name[: match.start()]
    name = collapse_whitespace(name).strip(" -,;:*")
    if name.endswith(".") and not _ABBREVIATION_TAIL_RE.search(name):
        name = name.rstrip(".").strip()
    name = name.upper()
    return Cue(raw=raw.strip(), name=name, key=slugify(name, max_length=120), extensions=extensions, dual=dual)


def looks_like_cue_text(text: str) -> bool:
    """Check whether text has the shape of a character cue. Shape test only (layout is checked by the caller).

    Outside any parentheses (and ignoring a trailing ``^`` or ``:``) the text must contain letters and
    no lower-case letters, have 1-6 words, not end in ``! ? , ; ... --``, and not be a known
    camera/structure direction (``NON_CUE_WORDS``, ``SHOT_RE``). Whole text at most 50 characters.

    Args:
        text: A candidate line.

    Returns:
        True when the line could be a cue, otherwise False.
    """
    stripped = text.strip()
    if not stripped or len(stripped) > 50:
        return False
    outside = _PAREN_RE.sub(" ", stripped).strip().rstrip("^").rstrip(":").strip()
    if not outside or not any(c.isalpha() for c in outside):
        return False
    if any(c.islower() for c in outside):
        return False
    words = outside.split()
    if not 1 <= len(words) <= 6:
        return False
    if outside.endswith(("!", "?", ",", ";")) or outside.endswith("...") or outside.endswith("--"):
        return False
    if outside.upper().strip(".") in NON_CUE_WORDS:
        return False
    return not SHOT_RE.match(outside)


@dataclass(slots=True)
class DialogueSegment:
    """One element inside a dialogue block.

    Attributes:
        type: ``"dialogue"``, ``"parenthetical"`` or ``"continuation"``.
        text: The element's content.
        element_index: Index of the source element.
    """

    type: str  # "dialogue" | "parenthetical" | "continuation"
    text: str
    element_index: int


@dataclass(slots=True)
class DialogueBlock:
    """A character cue plus the dialogue, parenthetical and continuation elements that follow it.

    Attributes:
        order: 0-based position of the block in the script.
        speaker: Normalised speaker name (see :class:`Cue`).
        speaker_key: Slug of the speaker name.
        speaker_raw: The cue as written.
        extensions: Cue extensions such as ``"V.O."``.
        dual: Whether the cue carried the dual-dialogue marker.
        cue_element: Index of the cue element.
        element_start: Index of the first element (the cue).
        element_end: Index of the last element of the block (the cue itself if nothing follows).
        line_start: First source line.
        line_end: Last source line.
        char_start: Start character offset in the script text.
        char_end: End character offset in the script text.
        text: Spoken words only, whitespace-normalized.
        original_text: Exact source slice incl. cue, parentheticals, indentation.
        parentheticals: Parenthetical texts in order.
        segments: All elements after the cue, in order.
        confidence: Lowest confidence among the cue and its segments, rounded to 3 decimals.
        scene_index: Index of the containing scene, assigned later; None until then.
        warnings: Problems such as an empty block or a dual-dialogue marker.
    """

    order: int
    speaker: str
    speaker_key: str
    speaker_raw: str
    extensions: list[str]
    dual: bool
    cue_element: int
    element_start: int
    element_end: int
    line_start: int
    line_end: int
    char_start: int
    char_end: int
    text: str  # spoken words only, whitespace-normalized
    original_text: str  # exact source slice incl. cue, parentheticals, indentation
    parentheticals: list[str]
    segments: list[DialogueSegment]
    confidence: float
    scene_index: int | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the block to a plain dict for storage.

        Returns:
            All fields as a dict; segments become nested dicts.
        """
        return asdict(self)


def build_dialogue_blocks(elements: list["Element"], text: str) -> list[DialogueBlock]:
    """Assemble dialogue blocks from classified elements.

    Each ``character`` element starts a block that absorbs the immediately following ``dialogue``,
    ``parenthetical`` and ``continuation`` elements flagged ``in_dialogue``; any other element ends it.

    Args:
        elements: Classified elements in document order.
        text: The full script text the element offsets refer to, used for ``original_text``.

    Returns:
        The blocks in script order; empty when there are no character cues. Blocks without spoken
        text are kept with a warning.
    """
    blocks: list[DialogueBlock] = []
    i = 0
    n = len(elements)
    while i < n:
        el = elements[i]
        if el.type != "character":
            i += 1
            continue
        cue = parse_cue(el.cue_text or el.content)
        j = i + 1
        segments: list[DialogueSegment] = []
        while j < n and elements[j].type in {"dialogue", "parenthetical", "continuation"} \
                and elements[j].in_dialogue:
            seg_el = elements[j]
            segments.append(DialogueSegment(seg_el.type, seg_el.content, seg_el.index))
            j += 1
        last = elements[j - 1]
        spoken = [s.text for s in segments if s.type == "dialogue"]
        parentheticals = [s.text for s in segments if s.type == "parenthetical"]
        block = DialogueBlock(
            order=len(blocks),
            speaker=cue.name,
            speaker_key=cue.key,
            speaker_raw=cue.raw,
            extensions=cue.extensions,
            dual=cue.dual,
            cue_element=el.index,
            element_start=el.index,
            element_end=last.index,
            line_start=el.line_start,
            line_end=last.line_end,
            char_start=el.char_start,
            char_end=last.char_end,
            text=collapse_whitespace(" ".join(spoken)),
            original_text=text[el.char_start:last.char_end],
            parentheticals=parentheticals,
            segments=segments,
            confidence=round(min([el.confidence] + [elements[k].confidence for k in range(i + 1, j)]), 3),
        )
        if not spoken:
            block.warnings.append("empty dialogue block (cue without spoken text)")
        if not cue.name:
            block.warnings.append("could not normalize speaker name")
        if cue.dual:
            block.warnings.append("dual-dialogue marker")
        blocks.append(block)
        i = j
    return blocks


@dataclass(slots=True)
class CharacterSummary:
    """Per-character statistics aggregated over all dialogue blocks.

    Attributes:
        name: Speaker name as first seen.
        key: Speaker key shared by all variants.
        variants: Distinct raw cue spellings, in order of appearance.
        extensions: Distinct cue extensions used, in order of appearance.
        dialogue_count: Number of dialogue blocks.
        word_count: Total spoken words (whitespace-separated).
        first_dialogue_order: ``order`` of the character's first block.
        first_line: Source line of the first block.
        scene_indices: Distinct scene indices the character speaks in, in order of appearance.
        dialogue_orders: ``order`` of every block, ascending.
    """

    name: str
    key: str
    variants: list[str]
    extensions: list[str]
    dialogue_count: int
    word_count: int
    first_dialogue_order: int
    first_line: int
    scene_indices: list[int]
    dialogue_orders: list[int]

    def to_dict(self) -> dict[str, Any]:
        """Convert the summary to a plain dict for storage.

        Returns:
            All fields as a dict.
        """
        return asdict(self)


def aggregate_characters(blocks: list[DialogueBlock]) -> list[CharacterSummary]:
    """Aggregate dialogue blocks into one summary per speaker key.

    Blocks whose speaker name or key is empty are skipped. Scene indices are collected only for
    blocks whose ``scene_index`` has been set.

    Args:
        blocks: Dialogue blocks in script order.

    Returns:
        One summary per character, sorted by first appearance; empty when there are no named speakers.
    """
    by_key: dict[str, CharacterSummary] = {}
    for block in blocks:
        if not block.speaker_key or not block.speaker:
            continue
        summary = by_key.get(block.speaker_key)
        if summary is None:
            summary = CharacterSummary(
                name=block.speaker, key=block.speaker_key, variants=[], extensions=[],
                dialogue_count=0, word_count=0, first_dialogue_order=block.order,
                first_line=block.line_start, scene_indices=[], dialogue_orders=[],
            )
            by_key[block.speaker_key] = summary
        if block.speaker_raw not in summary.variants:
            summary.variants.append(block.speaker_raw)
        for ext in block.extensions:
            if ext not in summary.extensions:
                summary.extensions.append(ext)
        summary.dialogue_count += 1
        summary.word_count += len(block.text.split())
        summary.dialogue_orders.append(block.order)
        if block.scene_index is not None and block.scene_index not in summary.scene_indices:
            summary.scene_indices.append(block.scene_index)
    return sorted(by_key.values(), key=lambda c: c.first_dialogue_order)
