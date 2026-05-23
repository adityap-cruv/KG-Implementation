from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.graph.nodes import (
    list_files_node,
    read_files_node,
    summarize_each_node,
)


def _make_async_response(text: str):
    response = MagicMock()
    response.content = text
    return response


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


def test_read_files_node_reads_every_file(fixture_folder):
    state = {
        "folder_path": str(fixture_folder),
        "all_files": [
            {"name": "company.md", "size_bytes": 1},
            {"name": "products.md", "size_bytes": 1},
            {"name": "privacy.md", "size_bytes": 1},
        ],
        "errors": [],
    }
    result = read_files_node(state)
    assert set(result["file_contents"].keys()) == {
        "company.md",
        "products.md",
        "privacy.md",
    }
    assert "Acme Corp" in result["file_contents"]["company.md"]
    assert result["errors"] == []


def test_read_files_node_records_missing_file(fixture_folder):
    state = {
        "folder_path": str(fixture_folder),
        "all_files": [{"name": "ghost.md", "size_bytes": 1}],
        "errors": [],
    }
    result = read_files_node(state)
    assert result["file_contents"] == {}
    assert any("ghost.md" in e for e in result["errors"])


@pytest.mark.asyncio
async def test_summarize_each_node_runs_one_call_per_file():
    summaries = {
        "company.md": "Summary of Acme company page.",
        "products.md": "Summary of Acme products.",
    }

    async def fake_ainvoke(messages):
        # Find which file we are summarizing from the user message.
        user_content = messages[1].content
        for name, text in summaries.items():
            if name in user_content:
                return _make_async_response(text)
        raise AssertionError("unknown file in prompt")

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=fake_ainvoke)

    with patch("app.graph.nodes.get_llm", return_value=mock_llm):
        result = await summarize_each_node(
            {
                "all_files": [
                    {"name": "company.md", "size_bytes": 1},
                    {"name": "products.md", "size_bytes": 1},
                ],
                "file_contents": {
                    "company.md": "Acme makes rockets.",
                    "products.md": "Flagship rocket A1.",
                },
                "errors": [],
            }
        )

    assert [s["name"] for s in result["file_summaries"]] == [
        "company.md",
        "products.md",
    ]
    assert result["file_summaries"][0]["summary"] == summaries["company.md"]
    assert result["errors"] == []
    assert mock_llm.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_summarize_each_node_records_per_file_errors():
    async def fake_ainvoke(messages):
        user_content = messages[1].content
        if "company.md" in user_content:
            return _make_async_response("Company summary.")
        raise RuntimeError("upstream blew up")

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=fake_ainvoke)

    with patch("app.graph.nodes.get_llm", return_value=mock_llm):
        result = await summarize_each_node(
            {
                "all_files": [
                    {"name": "company.md", "size_bytes": 1},
                    {"name": "products.md", "size_bytes": 1},
                ],
                "file_contents": {
                    "company.md": "Acme makes rockets.",
                    "products.md": "Flagship rocket A1.",
                },
                "errors": [],
            }
        )

    assert [s["name"] for s in result["file_summaries"]] == ["company.md"]
    assert any("products.md" in e for e in result["errors"])
