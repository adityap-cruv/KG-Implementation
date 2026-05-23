"""Per-brand state persistence — MongoDB-backed, one document per brand.

Document shape (in collection `brand_states`):

    {
      "_id": ObjectId(...),
      "folder": "genuin",                  # unique
      "base_summary": "...",
      "base_files": ["...", ...],
      "ranked_files": [{name, summary, reason}, ...],
      "errors": [...],
      "created_at": <UTC datetime>,        # set on first save
      "updated_at": <UTC datetime>,        # refreshed on every save
    }
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection

from app.config import settings


@lru_cache(maxsize=1)
def _client() -> MongoClient:
    return MongoClient(
        settings.MONGODB_URI,
        serverSelectionTimeoutMS=settings.MONGODB_TIMEOUT_MS,
        appname="brand-summarizer",
    )


def _collection() -> Collection:
    coll = _client()[settings.MONGODB_DATABASE][settings.MONGODB_COLLECTION]
    # idempotent — creates only on first call per Mongo's index machinery
    coll.create_index("folder", unique=True, name="folder_unique")
    return coll


def _strip_mongo_fields(doc: dict) -> dict:
    """Remove fields that shouldn't flow back into the pipeline state."""
    return {k: v for k, v in doc.items() if k != "_id"}


def _maybe_migrate_legacy(folder: str) -> dict | None:
    """If a legacy `state/<folder>.json` exists, migrate it into Mongo once.
    The legacy file is renamed to `<folder>.json.migrated` so this only runs once.
    """
    legacy_path = settings.BASE_DIR / "state" / f"{folder}.json"
    if not legacy_path.exists():
        return None
    payload = json.loads(legacy_path.read_text(encoding="utf-8"))
    save_state(folder, payload)
    legacy_path.rename(legacy_path.with_suffix(".json.migrated"))
    return _strip_mongo_fields(_collection().find_one({"folder": folder}) or {})


def load_state(folder: str) -> dict | None:
    """Return the persisted state for a folder, or None if nothing exists."""
    doc = _collection().find_one({"folder": folder})
    if doc is not None:
        return _strip_mongo_fields(doc)
    # First-time access for this folder — try a one-shot legacy migration.
    return _maybe_migrate_legacy(folder)


def save_state(folder: str, payload: dict) -> None:
    """Upsert the brand state document. `folder` is the unique key."""
    coll = _collection()
    now = datetime.now(timezone.utc)
    # Preserve created_at across updates.
    existing = coll.find_one({"folder": folder}, projection={"created_at": True})
    created_at = existing["created_at"] if existing and "created_at" in existing else now

    doc: dict[str, Any] = {k: v for k, v in payload.items() if k != "_id"}
    doc["folder"] = folder
    doc["created_at"] = created_at
    doc["updated_at"] = now
    coll.replace_one({"folder": folder}, doc, upsert=True)


def delete_state(folder: str) -> bool:
    """Delete a brand's state document. Returns True if anything was deleted."""
    result = _collection().delete_one({"folder": folder})
    return result.deleted_count > 0


def storage_description() -> str:
    """Human-readable string describing where state is stored, for logs/UI."""
    return f"MongoDB → db={settings.MONGODB_DATABASE!r}, collection={settings.MONGODB_COLLECTION!r}"
