import json
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app


def _build_mock_llm(summary_text: str = "Acme builds rockets."):
    selection_payload = {
        "selected": [{"name": "company.md", "reason": "core brand"}],
        "skipped": [{"name": "privacy.md", "reason": "legal"}],
    }
    selection_response = MagicMock()
    selection_response.content = json.dumps(selection_payload)

    summary_resp = MagicMock()
    summary_resp.content = summary_text

    bound = MagicMock()
    bound.invoke.return_value = selection_response

    mock_llm = MagicMock()
    mock_llm.bind.return_value = bound
    mock_llm.invoke.return_value = summary_resp
    return mock_llm


def test_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_summarize_happy_path(fixture_folder):
    from app.graph import builder

    builder.get_compiled_graph.cache_clear()

    with patch("app.graph.nodes.get_llm", return_value=_build_mock_llm()):
        client = TestClient(app)
        response = client.post("/summarize", json={"folder": "acme"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["folder"] == "acme"
    assert len(body["all_files"]) == 3
    assert body["selected_files"] == [
        {"name": "company.md", "reason": "core brand"}
    ]
    assert body["skipped_files"] == [
        {"name": "privacy.md", "reason": "legal"}
    ]
    assert "rockets" in body["summary"]
    assert body["errors"] == []


def test_summarize_invalid_folder_name():
    client = TestClient(app)
    response = client.post("/summarize", json={"folder": "../etc"})
    assert response.status_code == 422  # pydantic validation


def test_summarize_missing_folder(fixture_folder):
    from app.graph import builder

    builder.get_compiled_graph.cache_clear()

    with patch("app.graph.nodes.get_llm", return_value=_build_mock_llm()):
        client = TestClient(app)
        response = client.post("/summarize", json={"folder": "nonexistent"})
    assert response.status_code == 404
