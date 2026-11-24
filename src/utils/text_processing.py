"""Text decoding and normalization helpers.

Decoding never loses bytes silently: when no strict decoding succeeds, Latin-1 is used
(which maps every byte) and the result is flagged so validation can report it.
"""

from __future__ import annotations

import codecs
import re
import unicodedata
from dataclasses import dataclass, field

# charset="..." inside a <meta> tag (both <meta charset=...> and http-equiv content="...; charset=...").
_META_CHARSET_RE = re.compile(
    rb"""<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9_\-:.]+)""", re.IGNORECASE
)
# Canonical codec names of charset labels that are decoded as Windows-1252 first (see decode_html_bytes).
LATIN1_LABELS = frozenset({"iso8859-1", "ascii"})
# Typical UTF-8-read-as-cp1252 sequences ("â€™", "Ã©", ...).
_MOJIBAKE_RE = re.compile(r"(?:â€[™œ\x9d“”˜¦]|Ã[©¨ª«±³¶¼¡¢£¤§­®¯]|Â[ ­°·»«])")
_WS_RE = re.compile(r"[ \t\f\v]+")
_SLUG_RE = re.compile(r"[^a-z0-9]+")
# IMSDb lists titles with the leading article moved to the end: "Dark Knight, The", "Haine, La".
_ARTICLE_SUFFIX_RE = re.compile(r"^(?P<body>.+),\s*(?P<article>The|A|An|Les|La|Le|L'|El|Il|Das|Der|Die)$", re.IGNORECASE)


@dataclass(slots=True)
class DecodedText:
    """Result of decoding an HTML body, with enough provenance to audit the choice of codec.

    Attributes:
        text: The decoded text.
        encoding: Python codec name actually used (e.g. ``"utf-8"``, ``"cp1252"``, ``"latin-1"``).
        strict: True when a strict decode succeeded; False for the lossless Latin-1 fallback.
        declared_http: Charset from the HTTP ``Content-Type`` header, as written, or None.
        declared_meta: Charset from the first ``<meta>`` tag in the first 4 KiB, as written, or None.
        warnings: Human-readable notes about mismatches or fallbacks.
    """

    text: str
    encoding: str
    strict: bool
    declared_http: str | None = None
    declared_meta: str | None = None
    warnings: list[str] = field(default_factory=list)


def _canonical_codec(name: str | None) -> str | None:
    """Resolve a declared charset label to Python's canonical codec name.

    Args:
        name: Charset label such as ``"ISO-8859-1"`` or ``'"utf-8"'`` (surrounding quotes and spaces
            are stripped), or None.

    Returns:
        The canonical codec name (e.g. ``"iso8859-1"``, ``"utf-8"``), or None when the label is empty
        or unknown to Python.
    """
    if not name:
        return None
    try:
        return codecs.lookup(name.strip().strip("\"'")).name
    except LookupError:
        return None


def charset_from_content_type(content_type: str | None) -> str | None:
    """Extract the ``charset=`` parameter from an HTTP ``Content-Type`` header.

    Args:
        content_type: Header value such as ``"text/html; charset=UTF-8"``, or None.

    Returns:
        The charset label exactly as written (not normalised), or None when absent.
    """
    if not content_type:
        return None
    match = re.search(r"charset\s*=\s*[\"']?([A-Za-z0-9_\-:.]+)", content_type, re.IGNORECASE)
    return match.group(1) if match else None


def decode_html_bytes(body: bytes, content_type: str | None = None) -> DecodedText:
    """Decode an HTML response body.

    Order: BOM, UTF-8 (strict), HTTP-declared charset (strict), ``<meta>`` charset
    (strict), cp1252 (strict), Latin-1 (lossless fallback). IMSDb declares UTF-8 in HTTP
    but ISO-8859-1 in ``<meta>`` on some pages, so both declarations are recorded.
    Only the first 4 KiB are searched for the ``<meta>`` charset.

    Args:
        body: Raw response bytes.
        content_type: The HTTP ``Content-Type`` header, or None.

    Returns:
        The decoded text with the codec used, both declared charsets and any warnings (e.g. HTTP said
        UTF-8 but the body is not valid UTF-8). ``strict`` is False only for the Latin-1 fallback.
    """
    http_cs = charset_from_content_type(content_type)
    meta_match = _META_CHARSET_RE.search(body[:4096])
    meta_cs = meta_match.group(1).decode("ascii", "ignore") if meta_match else None
    warnings: list[str] = []

    for bom, codec in ((codecs.BOM_UTF8, "utf-8-sig"), (codecs.BOM_UTF16_LE, "utf-16"),
                       (codecs.BOM_UTF16_BE, "utf-16")):
        if body.startswith(bom):
            try:
                return DecodedText(body.decode(codec), codec, True, http_cs, meta_cs)
            except UnicodeDecodeError:
                warnings.append(f"BOM indicated {codec} but strict decoding failed")

    candidates: list[str] = []
    for name in ("utf-8", _canonical_codec(http_cs), _canonical_codec(meta_cs), "cp1252"):
        # Like browsers (WHATWG Encoding Standard), read a Latin-1/ASCII label as Windows-1252: such pages
        # use bytes 0x80-0x9F for curly quotes and dashes, which Latin-1 would turn into control characters.
        # Strict Latin-1 stays as the next candidate for the five bytes Windows-1252 leaves undefined.
        expanded = ["cp1252", name] if name in LATIN1_LABELS else [name]
        for codec in expanded:
            if codec and codec not in candidates:
                candidates.append(codec)

    for codec in candidates:
        try:
            text = body.decode(codec)
        except (UnicodeDecodeError, LookupError):
            continue
        if codec != "utf-8" and http_cs and _canonical_codec(http_cs) == "utf-8":
            warnings.append(f"HTTP declared utf-8 but body is not valid UTF-8; decoded as {codec}")
        return DecodedText(text, codec, True, http_cs, meta_cs, warnings)

    warnings.append("no strict decoding succeeded; used latin-1 fallback (lossless byte mapping)")
    return DecodedText(body.decode("latin-1"), "latin-1", False, http_cs, meta_cs, warnings)


def normalize_newlines(text: str) -> str:
    """Convert Windows (``\\r\\n``) and old Mac (``\\r``) line endings to ``\\n``.

    Args:
        text: Text with any mix of line endings.

    Returns:
        The text with only ``\\n`` line breaks.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_space(text: str) -> str:
    """Collapse runs of horizontal whitespace and strip; keeps newlines.

    Args:
        text: Text whose lines are already separated by ``\\n``.

    Returns:
        The text with each line's spaces/tabs collapsed to single spaces and stripped, and leading or
        trailing blank lines removed. Interior blank lines are kept (as empty lines).
    """
    return "\n".join(_WS_RE.sub(" ", line).strip() for line in text.split("\n")).strip()


def collapse_whitespace(text: str) -> str:
    """Collapse all whitespace, including newlines, into single spaces.

    Args:
        text: Any text.

    Returns:
        The text on one line with single spaces between words and no leading/trailing space.
    """
    return " ".join(text.split())


def slugify(text: str, max_length: int = 80) -> str:
    """Build a lowercase ASCII URL/filename slug.

    Accents are folded via NFKD (``"Amélie"`` -> ``"amelie"``); characters without an ASCII form are
    dropped; every other run of non-alphanumerics becomes one hyphen.

    Args:
        text: Text to slugify.
        max_length: Maximum slug length; the slug is cut there and any trailing hyphen removed.

    Returns:
        The slug, or ``"untitled"`` when nothing alphanumeric remains.
    """
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = _SLUG_RE.sub("-", folded.lower()).strip("-")
    return (slug[:max_length].rstrip("-")) or "untitled"


def display_title(title: str) -> str:
    """Turn IMSDb's ``"Dark Knight, The"`` into ``"The Dark Knight"``.

    Recognises English, French, Spanish, Italian and German articles (case-insensitive). An elided
    article such as ``L'`` is joined without a space (``"Avventura, L'"`` -> ``"L'Avventura"``).

    Args:
        title: Title as listed on IMSDb.

    Returns:
        The title in natural reading order, stripped; unchanged (but stripped) when it has no
        trailing article.
    """
    match = _ARTICLE_SUFFIX_RE.match(title.strip())
    if not match:
        return title.strip()
    article = match.group("article")
    sep = "" if article.endswith("'") else " "
    return f"{article}{sep}{match.group('body').strip()}"


def title_key(title: str) -> str:
    """Build the normalized key used to detect duplicate movie titles.

    Args:
        title: Title in either IMSDb (``"Matrix, The"``) or natural (``"The Matrix"``) order.

    Returns:
        The slug of the natural-order title (up to 200 characters), so both forms map to the same key.
    """
    return slugify(display_title(title), max_length=200)


def mojibake_score(text: str) -> int:
    """Count suspicious UTF-8/cp1252 mojibake sequences.

    Args:
        text: Decoded text to inspect.

    Returns:
        Number of matches of ``_MOJIBAKE_RE`` (e.g. ``"â€™"``, ``"Ã©"``); 0 for clean text.
    """
    return len(_MOJIBAKE_RE.findall(text))


def control_char_count(text: str) -> int:
    """Count control characters other than newline, tab and carriage return.

    Args:
        text: Decoded text to inspect.

    Returns:
        Number of Unicode category ``Cc`` characters (including C1 controls U+0080-U+009F, which
        appear when cp1252 bytes are decoded as Latin-1), excluding ``\\n``, ``\\t`` and ``\\r``.
    """
    return sum(1 for ch in text if unicodedata.category(ch) == "Cc" and ch not in "\n\t\r")


def encoding_problems(text: str) -> dict[str, int]:
    """Summarise signs of bad decoding in a text, for validation reports.

    Args:
        text: Decoded text to inspect.

    Returns:
        A dict with ``replacement_chars`` (count of U+FFFD), ``mojibake_sequences`` (see
        :func:`mojibake_score`) and ``control_chars`` (see :func:`control_char_count`).
    """
    return {
        "replacement_chars": text.count("�"),
        "mojibake_sequences": mojibake_score(text),
        "control_chars": control_char_count(text),
    }
