from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app


def _response(text: str):
    response = MagicMock()
    response.content = text
    return response


def _build_mock_llm():
    async def fake_ainvoke(messages):
        user_content = messages[1].content
        if "company.md" in user_content:
            return _response("Acme is a rocket company.")
        if "products.md" in user_content:
            return _response("Acme sells the A1 rocket.")
        if "privacy.md" in user_content:
            return _response("Privacy policy summary.")
        raise AssertionError("unexpected file in prompt")

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=fake_ainvoke)
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
    assert {s["name"] for s in body["file_summaries"]} == {
        "company.md",
        "products.md",
        "privacy.md",
    }
    assert body["errors"] == []


def test_summarize_invalid_folder_name():
    client = TestClient(app)
    response = client.post("/summarize", json={"folder": "../etc"})
    assert response.status_code == 422


def test_summarize_missing_folder(fixture_folder):
    from app.graph import builder

    builder.get_compiled_graph.cache_clear()

    with patch("app.graph.nodes.get_llm", return_value=_build_mock_llm()):
        client = TestClient(app)
        response = client.post("/summarize", json={"folder": "nonexistent"})
    assert response.status_code == 404
