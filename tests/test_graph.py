from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.graph.builder import build_graph


def _response(text: str):
    response = MagicMock()
    response.content = text
    return response


@pytest.mark.asyncio
async def test_graph_end_to_end_with_mocked_llm(fixture_folder):
    async def fake_ainvoke(messages):
        user_content = messages[1].content
        if "company.md" in user_content:
            return _response("Acme is a rocket company.")
        if "products.md" in user_content:
            return _response("Acme sells the A1 rocket.")
        if "privacy.md" in user_content:
            return _response("Acme privacy policy summary.")
        raise AssertionError(f"unexpected file in prompt: {user_content[:80]}")

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=fake_ainvoke)

    with patch("app.graph.nodes.get_llm", return_value=mock_llm):
        graph = build_graph()
        state = await graph.ainvoke({"folder": "acme"})

    assert [f["name"] for f in state["all_files"]] == [
        "company.md",
        "privacy.md",
        "products.md",
    ]
    assert [s["name"] for s in state["file_summaries"]] == [
        "company.md",
        "privacy.md",
        "products.md",
    ]
    assert state["file_summaries"][0]["summary"] == "Acme is a rocket company."
    assert state["errors"] == []
