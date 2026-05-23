import os
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
    yield


@pytest.fixture
def fixture_folder(tmp_path, monkeypatch):
    """Create a small brand folder under a temporary BASE_DIR."""
    base = tmp_path
    brand = base / "acme"
    brand.mkdir()
    (brand / "company.md").write_text("# Acme Corp\nWe make rockets.\n")
    (brand / "products.md").write_text("# Products\nFlagship rocket A1.\n")
    (brand / "privacy.md").write_text("# Privacy\nLegal text.\n")

    from app import config as config_module

    monkeypatch.setattr(config_module.settings, "BASE_DIR", base)
    return brand
