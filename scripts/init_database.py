"""Create the local database, collections, validators, indexes and the default permission record.

Usage:  python scripts/init_database.py
Safe to run repeatedly (idempotent).
Exit codes: 0 when the database is ready, otherwise the project error's code (2 configuration error,
5 MongoDB unreachable; see ``src/errors.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_settings  # noqa: E402
from src.database.connection import Mongo  # noqa: E402
from src.database.indexes import ensure_schema  # noqa: E402
from src.database.repositories import PermissionRepository  # noqa: E402
from src.errors import IMSDbError  # noqa: E402
from src.services.permissions import default_imsdb_permission  # noqa: E402


def main() -> int:
    """Create the data folders, the schema (collections, validators, indexes) and the default permission record.

    Returns:
        0 on success, otherwise the exit code of the project error raised (e.g. 5 when MongoDB is unreachable).
    """
    try:
        settings = load_settings()
        settings.ensure_directories()
        with Mongo.connect(settings) as mongo:
            created = ensure_schema(mongo.db)
            seeded = PermissionRepository(mongo.db).seed(default_imsdb_permission())
            print(f"Database '{mongo.db.name}' on MongoDB {mongo.server_version} is ready.")
            for name, indexes in created.items():
                print(f"  {name:<20} {len(indexes)} index(es)")
            print("Default source permission record " + ("created (level: sample)." if seeded else "already present."))
        return 0
    except IMSDbError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return int(exc.exit_code)


if __name__ == "__main__":
    sys.exit(main())
