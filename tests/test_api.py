import json
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app


def _response(text: str):
    r = MagicMock()
    r.content = text
    return r


def _build_mock_llm():
    bound = MagicMock()

    def bound_invoke(messages):
        sys_text = messages[0].content
        if "ORDERED list" in sys_text:
            return _response(
                json.dumps(
                    {
                        "order": [
                            {"name": "company.md", "reason": "core"},
                            {"name": "products.md", "reason": "product"},
                            {"name": "privacy.md", "reason": "legal"},
                        ]
                    }
                )
            )
        if "FILENAMES of every file" in sys_text:
            return _response(
                json.dumps(
                    {"selected": ["company.md"], "reason": "foundational"}
                )
            )
        raise AssertionError(f"unexpected: {sys_text[:80]}")

    bound.invoke.side_effect = bound_invoke

    async def fake_ainvoke(messages):
        user_content = messages[1].content
        if "company.md" in user_content:
            return _response("Acme is a rocket company.")
        if "products.md" in user_content:
            return _response("Acme sells A1.")
        if "privacy.md" in user_content:
            return _response("Privacy summary.")
        raise AssertionError("unknown summarize prompt")

    mock_llm = MagicMock()
    mock_llm.bind.return_value = bound
    mock_llm.invoke.return_value = _response("Acme is a rocket company.")
    mock_llm.ainvoke = AsyncMock(side_effect=fake_ainvoke)
    return mock_llm


def _clear_graph_caches():
    from app.graph import builder

    builder.get_onboarding_graph.cache_clear()
    builder.get_update_graph.cache_clear()


def test_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_summarize_onboarding_mode(fixture_folder):
    _clear_graph_caches()
    with patch("app.graph.nodes.get_llm", return_value=_build_mock_llm()):
        client = TestClient(app)
        response = client.post("/summarize", json={"folder": "acme"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["folder"] == "acme"
    assert body["mode"] == "onboard"
    assert body["base_summary"].startswith("Acme is a rocket company")
    assert body["base_files"] == ["company.md"]
    assert [r["name"] for r in body["ranked_files"]] == [
        "company.md",
        "products.md",
        "privacy.md",
    ]
    assert len(body["newly_added"]) == 3


def test_summarize_update_mode(fixture_folder):
    from app import state as state_module

    state_module.save_state(
        "acme",
        {
            "folder": "acme",
            "base_summary": "Existing.",
            "base_files": ["company.md"],
            "ranked_files": [
                {"name": "company.md", "summary": "old", "reason": "r"},
                {"name": "products.md", "summary": "old", "reason": "r"},
                {"name": "privacy.md", "summary": "old", "reason": "r"},
            ],
            "errors": [],
        },
    )
    (fixture_folder / "newpage.md").write_text("New content.\n")

    _clear_graph_caches()

    # Override the ranking payload to only include the new file.
    def _build_update_llm():
        bound = MagicMock()

        def bound_invoke(messages):
            sys_text = messages[0].content
            if "ORDERED list" in sys_text:
                return _response(
                    json.dumps(
                        {"order": [{"name": "newpage.md", "reason": "fresh"}]}
                    )
                )
            raise AssertionError("unexpected on update")

        bound.invoke.side_effect = bound_invoke

        async def fake_ainvoke(messages):
            return _response("Newpage summary.")

        mock_llm = MagicMock()
        mock_llm.bind.return_value = bound
        mock_llm.invoke = MagicMock()  # should never be called in update mode
        mock_llm.ainvoke = AsyncMock(side_effect=fake_ainvoke)
        return mock_llm

    with patch("app.graph.nodes.get_llm", return_value=_build_update_llm()):
        client = TestClient(app)
        response = client.post("/summarize", json={"folder": "acme"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode"] == "update"
    assert body["base_summary"] == "Existing."
    assert [r["name"] for r in body["newly_added"]] == ["newpage.md"]
    assert [r["name"] for r in body["ranked_files"]] == [
        "company.md",
        "products.md",
        "privacy.md",
        "newpage.md",
    ]


def test_summarize_invalid_folder_name():
    client = TestClient(app)
    response = client.post("/summarize", json={"folder": "../etc"})
    assert response.status_code == 422
