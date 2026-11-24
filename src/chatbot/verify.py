"""Post-answer check: every English quotation in an answer must occur in the passages it was given.

Small local models occasionally invent plausible-sounding lines. Quotes are checked word for word
(case, punctuation and line breaks ignored); anything not found is reported to the user as unverified.
"""

from __future__ import annotations

import re

from src.chatbot.search import Passage

# "…", “…”, and 「…」 quotations that contain Latin letters (Chinese translations are not checked).
_QUOTE_RE = re.compile(r'"([^"\n]{2,400})"|“([^”\n]{2,400})”|「([^」\n]{2,400})」')
_WORD_RE = re.compile(r"[a-z0-9]+")
MIN_WORDS = 3


def _words(text: str) -> list[str]:
    """Normalize text to a list of lowercase alphanumeric words for verbatim comparison.

    Apostrophes (straight and curly) are dropped so "don't" and "don’t" both become "dont".

    Args:
        text: Any text (an answer quote, a passage or a fact string).

    Returns:
        The words in order, lowercased, without punctuation.
    """
    return _WORD_RE.findall(text.lower().replace("’", "'").replace("'", ""))


def extract_quotes(answer: str) -> list[str]:
    """Find the quotations in an answer that need checking.

    Matches "...", “...” and 「...」 quotes on a single line (2-400 characters); keeps only those containing
    Latin letters and at least ``MIN_WORDS`` words, so Chinese translations and short phrases are skipped.

    Args:
        answer: The model's answer text.

    Returns:
        The stripped quotation texts in the order they appear (duplicates kept).
    """
    quotes = []
    for match in _QUOTE_RE.finditer(answer):
        quote = next(g for g in match.groups() if g is not None).strip()
        if re.search(r"[A-Za-z]", quote) and len(_words(quote)) >= MIN_WORDS:
            quotes.append(quote)
    return quotes


def unverified_quotes(answer: str, passages: list[Passage], extra_texts: list[str] | None = None) -> list[str]:
    """Quotes (3+ words) that do not appear verbatim in any passage or fact text.

    All texts are normalized with ``_words`` and joined into one space-padded haystack, so a quote may span
    passage boundaries; the match is on whole words.

    Args:
        answer: The model's answer text.
        passages: The screenplay passages the answer was generated from.
        extra_texts: Further texts the answer may quote (e.g. character lines from the facts), or None.

    Returns:
        The unverified quotes, de-duplicated, in order of appearance; [] when every quote was found.
    """
    haystack = " " + " ".join(" ".join(_words(t)) for t in [p.text for p in passages] + (extra_texts or [])) + " "
    missing = []
    for quote in extract_quotes(answer):
        needle = " " + " ".join(_words(quote)) + " "
        if needle not in haystack and quote not in missing:
            missing.append(quote)
    return missing
