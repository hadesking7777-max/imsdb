"""Time helpers. All timestamps are timezone-aware UTC."""

from __future__ import annotations

from datetime import UTC, datetime


def utcnow() -> datetime:
    """Return the current UTC time truncated to milliseconds (MongoDB's BSON date precision).

    Truncating up front means a value written to MongoDB and read back compares equal to the original.

    Returns:
        A timezone-aware ``datetime`` in UTC whose microseconds are a multiple of 1000.
    """
    now = datetime.now(UTC)
    return now.replace(microsecond=(now.microsecond // 1000) * 1000)


def as_utc(value: datetime | None) -> datetime | None:
    """Coerce a datetime to timezone-aware UTC.

    Naive datetimes (as returned by non-tz-aware clients such as PyMongo by default) are assumed to
    already be in UTC and only get ``tzinfo`` attached; aware datetimes are converted.

    Args:
        value: The datetime to normalise, or None.

    Returns:
        The same instant as an aware UTC datetime, or None when ``value`` is None.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def stamp(value: datetime | None = None) -> str:
    """Format a filesystem-safe compact UTC timestamp, e.g. ``20261117T110054Z``.

    Args:
        value: The moment to format; defaults to :func:`utcnow`. Aware datetimes are converted to UTC;
            naive ones are taken to be UTC already (see :func:`as_utc`).

    Returns:
        The timestamp as ``YYYYMMDDTHHMMSSZ`` (second precision).
    """
    return as_utc(value or utcnow()).strftime("%Y%m%dT%H%M%SZ")
