import json
from unittest.mock import MagicMock, patch

from app.graph.builder import build_graph


def test_graph_end_to_end_with_mocked_llm(fixture_folder):
    selection_payload = {
        "selected": [
            {"name": "company.md", "reason": "core brand info"},
            {"name": "products.md", "reason": "product catalog"},
        ],
        "skipped": [{"name": "privacy.md", "reason": "legal boilerplate"}],
    }
    selection_response = MagicMock()
    selection_response.content = json.dumps(selection_payload)

    summary_resp = MagicMock()
    summary_resp.content = "Acme builds rockets, including the A1."

    bound = MagicMock()
    bound.invoke.return_value = selection_response

    mock_llm = MagicMock()
    mock_llm.bind.return_value = bound
    mock_llm.invoke.return_value = summary_resp

    with patch("app.graph.nodes.get_llm", return_value=mock_llm):
        graph = build_graph()
        state = graph.invoke({"folder": "acme"})

    assert [f["name"] for f in state["all_files"]] == [
        "company.md",
        "privacy.md",
        "products.md",
    ]
    assert [f["name"] for f in state["selected_files"]] == [
        "company.md",
        "products.md",
    ]
    assert [f["name"] for f in state["skipped_files"]] == ["privacy.md"]
    assert "rockets" in state["summary"]
    assert state["errors"] == []
