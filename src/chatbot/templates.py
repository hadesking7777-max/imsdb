"""Answers produced without the language model (instant, exact): lists, archive totals, refusals."""

from __future__ import annotations

from typing import Any

from src.chatbot.catalog import Film


def archive(overview: dict[str, Any], lang: str) -> str:
    """Describe the archive's size and largest genres (answer to ``archive_stats``).

    Args:
        overview: Output of ``facts.corpus_overview`` (``films_listed``, ``screenplays_stored``, ``scenes``,
            ``dialogue_blocks``, ``genres`` sorted by count).
        lang: ``"zh"`` for Chinese; anything else gives English.

    Returns:
        The answer text, listing the top 8 genres and what the user can ask.
    """
    top = list(overview["genres"].items())[:8]
    if lang == "zh":
        genres = "、".join(f"{g}（{n}部）" for g, n in top)
        return (f"本地剧本库收录了 IMSDb 上列出的 {overview['films_listed']:,} 部电影，其中 "
                f"{overview['screenplays_stored']:,} 部存有完整剧本文本，共 {overview['scenes']:,} 个场景、"
                f"{overview['dialogue_blocks']:,} 段对白。数量最多的类型：{genres}。\n\n"
                "你可以问我某部电影的编剧、年份和主要角色，查找场景或台词，了解某个角色说了什么，"
                "或按类型、编剧、年份列出电影。")
    genres = ", ".join(f"{g} ({n})" for g, n in top)
    return (f"The local archive holds the {overview['films_listed']:,} films listed on IMSDb; "
            f"{overview['screenplays_stored']:,} of them have the full screenplay text, with "
            f"{overview['scenes']:,} scenes and {overview['dialogue_blocks']:,} dialogue blocks in total. "
            f"Largest genres: {genres}.\n\n"
            "You can ask about a film's writers, year and main characters, find scenes or quotes, ask what a "
            "character says, or list films by genre, writer or year.")


def film_list(listing: dict[str, Any], lang: str) -> str:
    """Format a filtered film list (answer to ``list_films``).

    Args:
        listing: Output of ``facts.list_films`` (``filters``, ``total``, ``films`` labels, ``truncated``).
        lang: ``"zh"`` for Chinese; anything else gives English.

    Returns:
        A bulleted list with the filters and total, a truncation note when needed, or a "no films match"
        sentence.
    """
    filters = listing["filters"]
    films = listing["films"]
    if lang == "zh":
        names = {"genre": "类型", "writer": "编剧", "year": "年份"}
        desc = "，".join(f"{names[k]}：{v}" for k, v in filters.items())
        if not films:
            return f"剧本库中没有符合条件的电影（{desc}）。"
        more = f"\n\n（只显示前 {len(films)} 部，共 {listing['total']} 部。）" if listing["truncated"] else ""
        return f"符合条件（{desc}）的电影共 {listing['total']} 部：\n\n" + "\n".join(f"- {f}" for f in films) + more
    desc = ", ".join(f"{k}: {v}" for k, v in filters.items())
    if not films:
        return f"No films in the archive match ({desc})."
    more = f"\n\n(Showing the first {len(films)} of {listing['total']}.)" if listing["truncated"] else ""
    return f"{listing['total']} films match ({desc}):\n\n" + "\n".join(f"- {f}" for f in films) + more


def full_script(films: list[Film], lang: str) -> str:
    """Politely refuse to output a whole screenplay (answer to ``full_script``).

    Args:
        films: Resolved films; the first one's IMSDb page is linked, if any.
        lang: ``"zh"`` for Chinese; anything else gives English.

    Returns:
        The refusal text, with the IMSDb link and suggestions of what can be asked instead.
    """
    film = films[0] if films else None
    if lang == "zh":
        where = f"《{film.display_title}》的剧本可在 IMSDb 阅读：{film.detail_url}\n\n" if film else ""
        return ("抱歉，我不能输出完整剧本。这些剧本的版权属于各自的制片公司和编剧，本地剧本库仅供学习和研究使用。\n\n"
                f"{where}我可以做的：概述某个场景、查找特定台词（简短引用）、介绍角色，或说明电影开头和结尾发生了什么。")
    where = f"You can read {film.display_title} on IMSDb: {film.detail_url}\n\n" if film else ""
    return ("Sorry, I can't output a full screenplay. The scripts belong to their studios and writers, and this "
            "local archive is for study and research use.\n\n"
            f"{where}What I can do: summarize a scene, find a specific line (as a short quote), describe a "
            "character, or tell you how the film opens or ends.")
