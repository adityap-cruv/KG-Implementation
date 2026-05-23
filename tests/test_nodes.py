import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from app.graph.nodes import (
    list_files_node,
    read_files_node,
    select_files_node,
    summarize_node,
)


def _selection_llm(payload: dict) -> MagicMock:
    """Build a mock LLM that returns a JSON string from `.bind(...).invoke(...)`."""
    response = MagicMock()
    response.content = json.dumps(payload)
    bound = MagicMock()
    bound.invoke.return_value = response
    llm = MagicMock()
    llm.bind.return_value = bound
    return llm


def test_list_files_node_returns_sorted_files(fixture_folder):
    result = list_files_node({"folder": "acme"})
    names = [f["name"] for f in result["all_files"]]
    assert names == ["company.md", "privacy.md", "products.md"]
    assert all(f["size_bytes"] > 0 for f in result["all_files"])
    assert result["folder_path"].endswith("acme")


def test_list_files_node_rejects_path_traversal(fixture_folder):
    with pytest.raises(HTTPException) as excinfo:
        list_files_node({"folder": "../etc"})
    assert excinfo.value.status_code in (400, 404)


def test_list_files_node_missing_folder(fixture_folder):
    with pytest.raises(HTTPException) as excinfo:
        list_files_node({"folder": "nonexistent"})
    assert excinfo.value.status_code == 404


def test_list_files_node_empty_folder(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    from app import config as config_module

    monkeypatch.setattr(config_module.settings, "BASE_DIR", tmp_path)
    with pytest.raises(HTTPException) as excinfo:
        list_files_node({"folder": "empty"})
    assert excinfo.value.status_code == 404


def test_select_files_node_filters_unknown_files():
    mock_llm = _selection_llm(
        {
            "selected": [
                {"name": "company.md", "reason": "brand info"},
                {"name": "hallucinated.md", "reason": "not real"},
            ],
            "skipped": [{"name": "privacy.md", "reason": "legal"}],
        }
    )

    with patch("app.graph.nodes.get_llm", return_value=mock_llm):
        result = select_files_node(
            {
                "folder": "acme",
                "all_files": [
                    {"name": "company.md", "size_bytes": 10},
                    {"name": "privacy.md", "size_bytes": 5},
                ],
            }
        )

    assert [f["name"] for f in result["selected_files"]] == ["company.md"]
    assert [f["name"] for f in result["skipped_files"]] == ["privacy.md"]


def test_select_files_node_handles_markdown_fenced_json():
    response = MagicMock()
    response.content = '```json\n{"selected": [{"name": "company.md", "reason": "x"}], "skipped": []}\n```'
    bound = MagicMock()
    bound.invoke.return_value = response
    mock_llm = MagicMock()
    mock_llm.bind.return_value = bound

    with patch("app.graph.nodes.get_llm", return_value=mock_llm):
        result = select_files_node(
            {
                "folder": "acme",
                "all_files": [{"name": "company.md", "size_bytes": 10}],
            }
        )
    assert result["selected_files"] == [{"name": "company.md", "reason": "x"}]


def test_select_files_node_raises_when_empty_selection():
    mock_llm = _selection_llm({"selected": [], "skipped": []})

    with patch("app.graph.nodes.get_llm", return_value=mock_llm):
        with pytest.raises(HTTPException) as excinfo:
            select_files_node(
                {
                    "folder": "acme",
                    "all_files": [{"name": "company.md", "size_bytes": 10}],
                }
            )
    assert excinfo.value.status_code == 422


def test_select_files_node_raises_on_invalid_json():
    response = MagicMock()
    response.content = "this is not json"
    bound = MagicMock()
    bound.invoke.return_value = response
    mock_llm = MagicMock()
    mock_llm.bind.return_value = bound

    with patch("app.graph.nodes.get_llm", return_value=mock_llm):
        with pytest.raises(HTTPException) as excinfo:
            select_files_node(
                {
                    "folder": "acme",
                    "all_files": [{"name": "company.md", "size_bytes": 10}],
                }
            )
    assert excinfo.value.status_code == 502


def test_read_files_node_reads_selected(fixture_folder):
    state = {
        "folder_path": str(fixture_folder),
        "selected_files": [
            {"name": "company.md", "reason": "x"},
            {"name": "products.md", "reason": "y"},
        ],
        "errors": [],
    }
    result = read_files_node(state)
    assert "Acme Corp" in result["file_contents"]["company.md"]
    assert "Flagship rocket" in result["file_contents"]["products.md"]
    assert result["errors"] == []


def test_read_files_node_records_missing_file(fixture_folder):
    state = {
        "folder_path": str(fixture_folder),
        "selected_files": [{"name": "ghost.md", "reason": "x"}],
        "errors": [],
    }
    result = read_files_node(state)
    assert result["file_contents"] == {}
    assert any("ghost.md" in e for e in result["errors"])


def test_summarize_node_invokes_llm_with_documents():
    mock_response = MagicMock()
    mock_response.content = "Acme makes rockets."
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    with patch("app.graph.nodes.get_llm", return_value=mock_llm):
        result = summarize_node(
            {
                "folder": "acme",
                "selected_files": [{"name": "company.md", "reason": "x"}],
                "file_contents": {"company.md": "We make rockets."},
            }
        )

    assert result["summary"] == "Acme makes rockets."
    sent_messages = mock_llm.invoke.call_args[0][0]
    assert any("company.md" in m.content for m in sent_messages)
    assert any("We make rockets." in m.content for m in sent_messages)
