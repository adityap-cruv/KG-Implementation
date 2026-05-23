import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def _env_defaults(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setenv("MONGODB_URI", "mongodb://test-host:27017")
    yield


@pytest.fixture
def state_store(monkeypatch):
    """In-memory replacement for MongoDB-backed state.

    Patches `app.state.load_state` and `app.state.save_state` with a dict-backed
    implementation. The dict itself is returned so tests can pre-seed or assert.
    nodes.py imports via `from app import state as state_module` so patching the
    module attributes here is enough — no need to also patch import sites.
    """
    store: dict[str, dict] = {}

    def fake_load(folder: str):
        doc = store.get(folder)
        return dict(doc) if doc is not None else None

    def fake_save(folder: str, payload: dict):
        store[folder] = {k: v for k, v in payload.items() if k != "_id"} | {"folder": folder}

    from app import state as state_module

    monkeypatch.setattr(state_module, "load_state", fake_load)
    monkeypatch.setattr(state_module, "save_state", fake_save)
    return store


@pytest.fixture
def fixture_folder(tmp_path, monkeypatch, state_store):
    """Disposable brand folder + in-memory state store.

    Sets BASE_DIR to a tmp dir and creates `acme/` with three small files.
    The `state_store` fixture is pulled in so any state I/O during the test
    goes through the in-memory replacement (never touches real Mongo).
    """
    base = tmp_path
    brand = base / "acme"
    brand.mkdir()
    (brand / "company.md").write_text("# Acme Corp\nWe make rockets.\n")
    (brand / "products.md").write_text("# Products\nFlagship rocket A1.\n")
    (brand / "privacy.md").write_text("# Privacy\nLegal text.\n")

    from app import config as config_module

    monkeypatch.setattr(config_module.settings, "BASE_DIR", base)
    return brand
