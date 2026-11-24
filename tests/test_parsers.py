"""Parser tests (pure functions, local fixtures only)."""

from __future__ import annotations

import pytest

from src.parsers.dialogue_parser import looks_like_cue_text, parse_cue
from src.parsers.movie_parser import parse_draft, parse_listing_page, parse_movie_detail
from src.parsers.scene_parser import is_scene_heading, parse_heading
from src.parsers.screenplay_parser import extract_screenplay, parse_screenplay
from src.utils.text_processing import decode_html_bytes, display_title, encoding_problems, title_key
from tests.conftest import BASE_URL, fixture_text


# ====================================================================== listing pages
def test_listing_entries_metadata_and_dedup() -> None:
    """Listing entries come out in order and de-duplicated, with detail URL, writers and parsed draft info."""
    page = parse_listing_page(fixture_text("all_scripts.html"), BASE_URL("/all-scripts.html"))
    titles = [e.title for e in page.entries]
    assert titles == ["Ember Road", "Lighthouse Keeper, The", "Night Shift at Dawn", "Paper Moons", "Quiet Orchard"]
    assert any("duplicate listing entry" in w for w in page.warnings)
    lighthouse = page.entries[1]
    assert lighthouse.detail_url == "https://imsdb.com/Movie%20Scripts/Lighthouse%20Keeper,%20The%20Script.html"
    assert lighthouse.writers == ["Ada Fenwick", "Tom Reyes"]
    assert (lighthouse.draft_text, lighthouse.draft_date, lighthouse.draft_label) == \
        ("2021-03 Revised draft", "2021-03", "Revised draft")
    assert page.entries[2].draft_date is None and page.entries[2].draft_label == "Draft"


def test_listing_separates_sidebar_and_navigation() -> None:
    """Sidebar links are not listing entries; alphabetical and genre navigation links are collected separately."""
    page = parse_listing_page(fixture_text("all_scripts.html"), BASE_URL("/all-scripts.html"))
    assert len(page.entries) == 5  # sidebar "Top rated" link is not a listing entry
    assert BASE_URL("/alphabetical/L") in page.listing_links
    assert BASE_URL("/genre/Drama") in page.listing_links


def test_home_page_navigation_only_yields_crawlable_listings() -> None:
    """The home page yields only the crawlable listing links (alphabetical, genre, all-scripts)."""
    page = parse_listing_page(fixture_text("home.html"), BASE_URL("/"))
    assert set(page.listing_links) == {BASE_URL("/alphabetical/L"), BASE_URL("/alphabetical/P"),
                                       BASE_URL("/genre/Drama"), BASE_URL("/all-scripts.html")}


@pytest.mark.parametrize("text,expected", [
    ("(1994-06 Draft)", ("1994-06 Draft", "1994-06", "Draft")),
    ("(Undated Final draft)", ("Undated Final draft", None, "Final draft")),
    ("(2018 Draft)", ("2018 Draft", "2018", "Draft")),
    ("no parens", (None, None, None)),
    (None, (None, None, None)),
])
def test_parse_draft(text: str | None, expected: tuple) -> None:
    """Draft text in parentheses splits into ``(text, date, label)``; anything else gives all ``None``.

    Args:
        text: The raw draft annotation from a listing entry.
        expected: Expected ``(draft_text, draft_date, draft_label)``.
    """
    assert parse_draft(text) == expected


# ====================================================================== detail pages
def test_movie_detail_full() -> None:
    """A complete detail page yields every field; unknown labels are kept and directors/IMDb ids are not invented."""
    url = BASE_URL("/Movie Scripts/Lighthouse Keeper, The Script.html")
    detail = parse_movie_detail(fixture_text("detail_lighthouse.html"), url)
    assert detail.title == "Lighthouse Keeper, The"
    assert [w.name for w in detail.writers] == ["Ada Fenwick", "Tom Reyes"]
    assert detail.writers[0].url == "https://imsdb.com/writer.php?w=Ada%20Fenwick"
    assert detail.genres == ["Drama", "Mystery"]
    assert detail.script_date == "March 2021"
    assert detail.movie_release_date == "October 2022"
    assert detail.release_year == 2022
    assert detail.imsdb_opinion == "A quiet, salt-stained character piece."
    assert detail.imsdb_rating == 8.0
    assert detail.average_user_rating == 9.0
    assert detail.script_url == "https://imsdb.com/scripts/Lighthouse-Keeper,-The.html"
    assert detail.script_availability == "html"
    assert detail.user_comment_count == 2
    assert detail.extra_fields["Country"]["text"] == "Ireland"  # unknown label preserved
    assert detail.poster_url and detail.poster_url.startswith("https://imsdb.com/posters/")
    assert detail.directors == [] and detail.imdb_ids == []  # never invented
    assert detail.warnings == []


def test_movie_detail_not_available_ratings() -> None:
    """"Not available" ratings become ``None`` with the original text kept."""
    detail = parse_movie_detail(fixture_text("detail_paper_moons.html"), BASE_URL("/Movie Scripts/Paper Moons Script.html"))
    assert detail.imsdb_opinion is None
    assert detail.imsdb_rating is None and detail.imsdb_rating_text == "Not available"
    assert detail.release_year == 2020


def test_movie_detail_without_script_link() -> None:
    """A detail page without a script link is ``missing`` with a warning."""
    detail = parse_movie_detail(fixture_text("detail_night_shift.html"),
                                BASE_URL("/Movie Scripts/Night Shift at Dawn Script.html"))
    assert detail.script_url is None
    assert detail.script_availability == "missing"
    assert any("not available" in w for w in detail.warnings)


def test_movie_detail_pdf_script() -> None:
    """A PDF script link gives availability ``pdf``; the script date is not used as a release year."""
    detail = parse_movie_detail(fixture_text("detail_ember_road.html"), BASE_URL("/Movie Scripts/Ember Road Script.html"))
    assert detail.script_availability == "pdf"
    assert detail.script_url == "https://imsdb.com/scripts/Ember-Road.pdf"
    assert detail.release_year is None  # script date is not a release date


def test_movie_detail_malformed_html_is_tolerated() -> None:
    """Malformed detail HTML still yields title and script URL, with a warning about the missing table."""
    detail = parse_movie_detail(fixture_text("detail_malformed.html"), BASE_URL("/Movie Scripts/Broken Page Script.html"))
    assert detail.title == "Broken Page"
    assert detail.script_url == "https://imsdb.com/scripts/Broken-Page.html"
    assert any("script-details table not found" in w for w in detail.warnings)


# ====================================================================== titles / encoding
@pytest.mark.parametrize("raw,display", [("Dark Knight, The", "The Dark Knight"), ("Few Good Men, A", "A Few Good Men"),
                                         ("12 Monkeys", "12 Monkeys")])
def test_display_title(raw: str, display: str) -> None:
    """Trailing articles move to the front for display, and both forms share one title key.

    Args:
        raw: IMSDb title, e.g. ``"Dark Knight, The"``.
        display: Expected display title.
    """
    assert display_title(raw) == display
    assert title_key(raw) == title_key(display)


def test_decoding_falls_back_when_http_charset_is_wrong() -> None:
    """Bytes that are not valid UTF-8 despite the HTTP header decode as cp1252 with a warning."""
    html = '<html><head><meta charset="windows-1252"></head><body><pre>It’s café time.</pre></body></html>'
    decoded = decode_html_bytes(html.encode("cp1252"), "text/html; charset=UTF-8")
    assert decoded.encoding == "cp1252" and decoded.strict
    assert "It’s café" in decoded.text
    assert any("not valid UTF-8" in w for w in decoded.warnings)


def test_latin1_label_is_read_as_windows_1252() -> None:
    """IMSDb's ``<meta charset="iso-8859-1">`` pages use Windows-1252 punctuation: decode it like browsers do,
    instead of turning curly quotes and dashes into C1 control characters; fall back to strict Latin-1
    for bytes Windows-1252 does not define."""
    body = '<meta charset="iso-8859-1"><pre>It’s “fine” – ok… café</pre>'.encode("cp1252")
    decoded = decode_html_bytes(body, "text/html; charset=UTF-8")
    assert decoded.encoding == "cp1252" and decoded.strict
    assert "It’s “fine” – ok… café" in decoded.text
    assert encoding_problems(decoded.text)["control_chars"] == 0
    undefined = decode_html_bytes(b'<meta charset="iso-8859-1">caf\xe9 \x81', None)
    assert undefined.encoding == "iso8859-1" and undefined.text.endswith("café \x81")


def test_decoding_never_loses_bytes() -> None:
    """Decoding arbitrary bytes round-trips exactly (or uses cp1252, which cannot map every byte)."""
    body = bytes(range(256))
    decoded = decode_html_bytes(body, None)
    assert decoded.text.encode(decoded.encoding if decoded.encoding != "latin-1" else "latin-1") == body \
        or decoded.encoding == "cp1252"


def test_encoding_problem_detection() -> None:
    """Mojibake sequences and replacement characters are counted; clean text has no problems."""
    assert encoding_problems("Itâ€™s broken")["mojibake_sequences"] == 1
    assert encoding_problems("bad � char")["replacement_chars"] == 1
    assert encoding_problems("clean text") == {"replacement_chars": 0, "mojibake_sequences": 0, "control_chars": 0}


# ====================================================================== screenplay extraction
@pytest.fixture(scope="module")
def lighthouse():
    """Extract and parse the Lighthouse Keeper screenplay fixture once per module.

    Returns:
        ``(extracted, parsed)``: the ``extract_screenplay`` result and the ``parse_screenplay`` result.
    """
    extracted = extract_screenplay(fixture_text("script_lighthouse.html"))
    return extracted, parse_screenplay(extracted)


def test_extraction_preserves_text_and_formatting(lighthouse) -> None:
    """Extraction keeps the ``<pre>`` text, indentation, bold spans and markup, and reads footer metadata separately.

    Args:
        lighthouse: ``(extracted, parsed)`` for the Lighthouse Keeper screenplay fixture.
    """
    extracted, _ = lighthouse
    assert extracted.method == "pre"
    assert "should be ignored" not in extracted.text  # <script> in <head> is not screenplay text
    assert "Ada Fenwick & Tom Reyes" in extracted.text  # entities decoded
    assert "                         MARGARET (V.O.)" in extracted.text  # indentation preserved
    assert extracted.footer_metadata == {"title": "Lighthouse Keeper, The", "writers": ["Ada Fenwick", "Tom Reyes"],
                                         "genres": ["Drama", "Mystery"]}
    assert "User Comments" not in extracted.text  # footer table excluded from screenplay text
    bold = [s for s in extracted.formatting_spans if s.style == "bold"]
    assert bold and all(extracted.text[s.start:s.end].strip() for s in bold)
    assert "<pre>" in extracted.markup and "<b>" in extracted.markup  # original formatting kept
    assert extracted.extraction_ratio and extracted.extraction_ratio > 0.8


def test_element_types_and_order(lighthouse) -> None:
    """Elements get the expected types and counts, consecutive indexes and increasing character offsets.

    Args:
        lighthouse: ``(extracted, parsed)`` for the Lighthouse Keeper screenplay fixture.
    """
    _, parsed = lighthouse
    types = [e.type for e in parsed.elements]
    assert types[:6] == ["title_page"] * 4 + ["transition", "act_marker"]
    assert types.count("scene_heading") == 4
    assert types.count("character") == 8
    assert types.count("transition") == 5  # FADE IN, CUT TO, DISSOLVE TO, FADE OUT, THE END
    assert types.count("act_marker") == 2
    assert "page_marker" in types and "continuation" in types and "shot" in types
    assert [e.index for e in parsed.elements] == list(range(len(parsed.elements)))
    assert all(a.char_start < b.char_start for a, b in zip(parsed.elements, parsed.elements[1:]))


def test_offsets_reproduce_original_text(lighthouse) -> None:
    """Element, dialogue and scene offsets slice exactly their text out of the extracted screenplay.

    Args:
        lighthouse: ``(extracted, parsed)`` for the Lighthouse Keeper screenplay fixture.
    """
    extracted, parsed = lighthouse
    for el in parsed.elements:
        assert extracted.text[el.char_start:el.char_end] == el.text
    for block in parsed.dialogues:
        assert extracted.text[block.char_start:block.char_end] == block.original_text
    for scene in parsed.scenes:
        assert extracted.text[scene.char_start:scene.char_end].strip()


def test_dialogue_identification(lighthouse) -> None:
    """Dialogue blocks get speaker, extensions, multi-line parentheticals and text; an empty block is flagged.

    Args:
        lighthouse: ``(extracted, parsed)`` for the Lighthouse Keeper screenplay fixture.
    """
    _, parsed = lighthouse
    first = parsed.dialogues[0]
    assert (first.speaker, first.extensions) == ("MARGARET", ["V.O."])
    assert first.text == "Forty years I kept that light. Forty years, and not one ship lost."
    multi = parsed.dialogues[2]
    assert multi.speaker == "MARGARET" and multi.extensions == ["CONT'D"]
    assert multi.parentheticals == ["(calling down, irritated, wiping her hands)"]  # multi-line parenthetical
    assert multi.text == "Nobody climbs these stairs at midnight unless they're lost."
    empty = parsed.dialogues[-1]
    assert empty.speaker == "DANIEL" and empty.text == "" and empty.warnings
    assert "empty_dialogue_blocks" in parsed.quality_flags
    assert {c.name: c.dialogue_count for c in parsed.characters} == {"MARGARET": 4, "DANIEL": 4}


def test_scene_segmentation(lighthouse) -> None:
    """Scenes get numbers, heading parts, acts, pages, characters and shots, and cover every element exactly once.

    Args:
        lighthouse: ``(extracted, parsed)`` for the Lighthouse Keeper screenplay fixture.
    """
    _, parsed = lighthouse
    kinds = [(s.scene_number, s.kind) for s in parsed.scenes]
    assert kinds == [(0, "prelude"), (1, "scene"), (2, "scene"), (3, "scene"), (4, "scene")]
    s1 = parsed.scenes[1]
    assert s1.heading_parts["scene_number"] == "1"
    assert (s1.heading_parts["int_ext"], s1.heading_parts["location"], s1.heading_parts["time_of_day"]) == \
        ("EXT", "NORTHERN CLIFFS", "NIGHT")
    assert s1.act == "ACT ONE"
    assert parsed.scenes[4].act == "ACT ONE"  # "END OF ACT ONE" closes the act inside scene 4
    assert parsed.scenes[3].page_start == "1" and parsed.scenes[3].page_end == "2"
    assert parsed.scenes[2].characters == ["MARGARET", "DANIEL"]
    assert "CLOSE ON" in parsed.scenes[2].shots
    assert "storm batters" in s1.description
    # every element belongs to exactly one scene
    covered = [i for s in parsed.scenes for i in range(s.element_start, s.element_end + 1)]
    assert covered == list(range(len(parsed.elements)))


def test_uncertain_classifications_are_flagged(lighthouse) -> None:
    """Title-page elements are marked uncertain and the stats count every uncertain element.

    Args:
        lighthouse: ``(extracted, parsed)`` for the Lighthouse Keeper screenplay fixture.
    """
    _, parsed = lighthouse
    title_elements = [e for e in parsed.elements if e.type == "title_page"]
    assert title_elements and all(e.uncertain for e in title_elements)
    assert parsed.stats["uncertain_elements"] == sum(1 for e in parsed.elements if e.uncertain)


def test_flush_layout_and_inline_cues() -> None:
    """Flush-left scripts are detected, and inline ``NAME: speech`` cues are parsed with low confidence."""
    parsed = parse_screenplay(extract_screenplay(fixture_text("script_flush.html")))
    assert parsed.layout.mode == "flush"
    speakers = [(d.speaker, d.text) for d in parsed.dialogues]
    assert speakers[0] == ("JUNO", "You're late again.")
    assert speakers[1] == ("OSCAR", "The bus was later.") and parsed.dialogues[1].parentheticals == ["(shrugging)"]
    assert ("OSCAR", "I could quit tomorrow, you know.") in speakers  # inline "NAME: speech"
    assert ("JUNO", "You say that every single night.") in speakers
    inline = [d for d in parsed.dialogues if d.text.startswith("I could quit")][0]
    assert inline.confidence < 0.7  # inline cues are marked uncertain
    assert parsed.stats["scenes"] == 3


def test_br_based_script_without_pre() -> None:
    """A script laid out with ``<br>`` instead of ``<pre>`` is extracted from the scrtext cell, with a warning."""
    extracted = extract_screenplay(fixture_text("script_br.html"))
    parsed = parse_screenplay(extracted)
    assert extracted.method == "scrtext_html"
    assert any("no <pre>" in w for w in extracted.warnings)
    assert [d.speaker for d in parsed.dialogues] == ["ELSIE", "THEO", "ELSIE"]
    assert parsed.dialogues[0].text == "These won't last the week. Not with this frost coming."
    assert parsed.stats["scenes"] == 2 and not parsed.stats["has_prelude"]


def test_malformed_page_uses_fallback_and_is_flagged() -> None:
    """A page without the expected structure falls back to the largest ``<pre>`` and is flagged."""
    extracted = extract_screenplay(fixture_text("script_malformed.html"))
    parsed = parse_screenplay(extracted)
    assert extracted.method == "largest_pre"
    assert "fallback_extraction" in parsed.quality_flags
    assert parsed.dialogues and parsed.dialogues[0].speaker == "MILES"


def test_empty_script_is_reported_not_crashing() -> None:
    """An empty script page yields quality flags (no scenes, no dialogue) instead of an exception."""
    extracted = extract_screenplay(fixture_text("script_empty.html"))
    parsed = parse_screenplay(extracted)
    assert not extracted.found
    assert {"no_scene_headings", "no_dialogue"} <= set(parsed.quality_flags)


def test_deeply_nested_markup_does_not_overflow_stack() -> None:
    """Thousands of nested tags are handled without recursion errors."""
    html = "<html><body><td class='scrtext'><pre>" + "<b><i>" * 3000 + "INT. HOUSE - DAY\n\nText.\n</pre></td>"
    extracted = extract_screenplay(html)
    assert "INT. HOUSE - DAY" in extracted.text


def test_crlf_and_nbsp_normalization_is_recorded() -> None:
    """CRLF line endings and non-breaking spaces are normalized and the normalization is recorded."""
    html = "<td class='scrtext'><pre>INT. ROOM - DAY\r\n\r\nA chair.\r\n</pre></td>"
    extracted = extract_screenplay(html)
    assert "\r" not in extracted.text and " " not in extracted.text
    assert any("non-breaking" in n for n in extracted.normalizations)


# ====================================================================== cues / headings
@pytest.mark.parametrize("raw,name,ext", [
    ("JOHN (V.O.)", "JOHN", ["V.O."]),
    ("MARY (CONT'D)", "MARY", ["CONT'D"]),
    ("MARY (CONT’D)", "MARY", ["CONT'D"]),
    ("DR. SMITH (O.S.) (CONT'D)", "DR. SMITH", ["O.S.", "CONT'D"]),
    ("OLD MAN CONT'D", "OLD MAN", ["CONT'D"]),
    ("ANNA:", "ANNA", []),
    ("BOB ^", "BOB", []),
])
def test_parse_cue(raw: str, name: str, ext: list[str]) -> None:
    """Character cues split into name and normalized extensions (V.O., O.S., CONT'D incl. curly apostrophe).

    Args:
        raw: The cue line.
        name: Expected character name.
        ext: Expected extensions.
    """
    cue = parse_cue(raw)
    assert cue.name == name and cue.extensions == ext


@pytest.mark.parametrize("text,expected", [
    ("JOHN", True), ("MRS. DALLOWAY", True), ("CLOSE ON", False), ("CUT TO:", True),  # transitions filtered elsewhere
    ("THE END", False), ("Hello there", False), ("WHAT?!", False), ("A VERY LONG LINE OF SHOUTING TEXT HERE", False),
])
def test_cue_shape(text: str, expected: bool) -> None:
    """Only short upper-case lines without !/? look like cues (transitions are filtered elsewhere).

    Args:
        text: The candidate line.
        expected: Whether it should look like a cue.
    """
    assert looks_like_cue_text(text) is expected


@pytest.mark.parametrize("text,ok", [
    ("INT. HOUSE - DAY", True), ("EXT. BEACH -- NIGHT", True), ("INT./EXT. CAR - MOVING - DAY", True),
    ("12A INT. LAB - NIGHT 12A", True), ("I/E VAN - DAY", True), ("Interior design matters", False),
    ("INTERCUT WITH:", False), ("EXTRA! EXTRA!", False),
])
def test_scene_heading_detection(text: str, ok: bool) -> None:
    """INT./EXT. variants, numbered and I/E headings are scene headings; look-alike words are not.

    Args:
        text: The candidate line.
        ok: Whether it should be detected as a heading.
    """
    assert is_scene_heading(text)[0] is ok


def test_parse_heading_components() -> None:
    """A heading splits into scene number, INT/EXT, location and time of day (``None`` when absent)."""
    parts = parse_heading("12A INT./EXT. CAR - MOVING - DAY 12A")
    assert (parts.scene_number, parts.int_ext, parts.location, parts.time_of_day) == ("12A", "INT/EXT", "CAR - MOVING", "DAY")
    parts = parse_heading("EXT. HOUSE - FRONT YARD")
    assert parts.location == "HOUSE - FRONT YARD" and parts.time_of_day is None


# ====================================================================== regressions found in the live sample run
_LIVE_SHAPES = """<td class="scrtext"><pre>
                    TEN THINGS ABOUT A LIGHTHOUSE
                    written by Ada Fenwick

INT. LAMP ROOM - NIGHT

Margaret polishes the lens. She checks the clock, then the window.

                         MARGARET
          (aside) Steady now, steady.
          Hold the light.

                         DANIEL
                    (to Margaret)

Is the light still burning after all these years of storms?

                         MARGARET
          It always burns.

                         DANIEL
          Then I can rest.
</pre></td>"""


def test_speech_on_the_parenthetical_line_is_split_off() -> None:
    """Speech on the same line as a parenthetical becomes dialogue text, and offsets still match."""
    extracted = extract_screenplay(_LIVE_SHAPES)
    parsed = parse_screenplay(extracted)
    block = next(d for d in parsed.dialogues if d.parentheticals == ["(aside)"])
    assert block.text == "Steady now, steady. Hold the light."
    for el in parsed.elements:
        assert extracted.text[el.char_start:el.char_end] == el.text


@pytest.mark.parametrize("gap", ["", "          \n"])
def test_credit_line_after_title_is_front_matter_not_dialogue(gap: str) -> None:
    """A "written by" line after the title is title-page text, not dialogue under a title "cue".

    Args:
        gap: Text inserted after the title line (nothing, or a whitespace-only line as on the live site).
    """
    html = _LIVE_SHAPES.replace("LIGHTHOUSE\n", "LIGHTHOUSE\n" + gap, 1)  # with/without blank line (live case)
    parsed = parse_screenplay(extract_screenplay(html))
    assert [e.type for e in parsed.elements[:2]] == ["title_page", "title_page"]
    assert all(d.speaker != "TEN THINGS ABOUT A LIGHTHOUSE" for d in parsed.dialogues)


def test_misindented_speech_after_cue_is_flagged_not_guessed() -> None:
    """Unindented speech after a cue stays action but is flagged as possible dialogue; the empty block warns."""
    parsed = parse_screenplay(extract_screenplay(_LIVE_SHAPES))
    empty = next(d for d in parsed.dialogues if d.speaker == "DANIEL" and not d.text)
    follow = parsed.elements[empty.element_end + 1]
    assert follow.type == "action" and follow.uncertain  # kept as action, but flagged
    assert any("possibly dialogue of preceding cue DANIEL" in n for n in follow.notes)
    assert empty.warnings
