import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.graph.nodes import (
    build_base_summary_node,
    detect_new_files_node,
    list_files_node,
    rank_files_node,
    read_all_files_for_summarization_node,
    read_new_files_node,
    save_state_node,
    summarize_each_node,
)


# ---- helpers ---------------------------------------------------------------


def _bound_llm_returning(payload_or_text):
    """Build a mock LLM whose .bind(...).invoke(...) returns the given payload."""
    response = MagicMock()
    response.content = (
        payload_or_text
        if isinstance(payload_or_text, str)
        else json.dumps(payload_or_text)
    )
    bound = MagicMock()
    bound.invoke.return_value = response
    llm = MagicMock()
    llm.bind.return_value = bound
    return llm


def _sync_llm_returning(text: str):
    response = MagicMock()
    response.content = text
    llm = MagicMock()
    llm.invoke.return_value = response
    return llm


# ---- list_files ------------------------------------------------------------


def test_list_files_node_returns_sorted_files(fixture_folder):
    result = list_files_node({"folder": "acme"})
    names = [f["name"] for f in result["all_files"]]
    assert names == ["company.md", "privacy.md", "products.md"]


def test_list_files_node_rejects_path_traversal(fixture_folder):
    with pytest.raises(HTTPException) as excinfo:
        list_files_node({"folder": "../etc"})
    assert excinfo.value.status_code in (400, 404)


def test_list_files_node_missing_folder(fixture_folder):
    with pytest.raises(HTTPException) as excinfo:
        list_files_node({"folder": "nonexistent"})
    assert excinfo.value.status_code == 404


# ---- build_base_summary ----------------------------------------------------


def test_build_base_summary_node_uses_all_files_from_state_contents():
    mock_llm = _sync_llm_returning("Acme makes rockets and sells the A1.")
    with patch("app.graph.nodes.get_llm", return_value=mock_llm):
        result = build_base_summary_node(
            {
                "folder": "acme",
                "all_files": [
                    {"name": "company.md", "size_bytes": 10},
                    {"name": "products.md", "size_bytes": 10},
                    {"name": "privacy.md", "size_bytes": 10},
                ],
                "target_file_contents": {
                    "company.md": "# Acme Corp\nWe make rockets.",
                    "products.md": "# Products\nFlagship rocket A1.",
                    "privacy.md": "# Privacy\nLegal text.",
                },
            }
        )
    assert result["base_summary"] == "Acme makes rockets and sells the A1."
    # base_files records that ALL files contributed (in folder order).
    assert result["base_files"] == ["company.md", "products.md", "privacy.md"]

    # Every file's content made it into the prompt.
    user_content = mock_llm.invoke.call_args[0][0][1].content
    assert "Acme Corp" in user_content
    assert "Flagship rocket A1" in user_content
    assert "Legal text" in user_content
    # And no per-filename headers leaked into the prompt.
    assert "## File: company.md" not in user_content


def test_build_base_summary_node_raises_when_no_contents():
    with pytest.raises(HTTPException) as excinfo:
        build_base_summary_node({"folder": "acme", "all_files": [], "target_file_contents": {}})
    assert excinfo.value.status_code == 500


# ---- read_all_files / read_new_files --------------------------------------


def test_read_all_files_for_summarization_node(fixture_folder):
    result = read_all_files_for_summarization_node(
        {
            "folder_path": str(fixture_folder),
            "all_files": [
                {"name": "company.md", "size_bytes": 1},
                {"name": "products.md", "size_bytes": 1},
                {"name": "privacy.md", "size_bytes": 1},
            ],
        }
    )
    assert set(result["target_file_contents"].keys()) == {
        "company.md",
        "products.md",
        "privacy.md",
    }


def test_read_new_files_node_reads_only_listed_names(fixture_folder):
    result = read_new_files_node(
        {
            "folder_path": str(fixture_folder),
            "new_file_names": ["products.md"],
        }
    )
    assert list(result["target_file_contents"].keys()) == ["products.md"]


# ---- detect_new_files ------------------------------------------------------


def test_detect_new_files_node_finds_difference():
    existing = {
        "base_summary": "An existing brand.",
        "base_files": ["company.md"],
        "ranked_files": [
            {"name": "company.md", "summary": "...", "reason": "..."},
            {"name": "products.md", "summary": "...", "reason": "..."},
        ],
    }
    state = {
        "folder": "acme",
        "existing_state": existing,
        "all_files": [
            {"name": "company.md", "size_bytes": 1},
            {"name": "products.md", "size_bytes": 1},
            {"name": "new-page.md", "size_bytes": 1},
        ],
    }
    result = detect_new_files_node(state)
    assert result["new_file_names"] == ["new-page.md"]
    assert result["base_summary"] == "An existing brand."
    assert result["base_files"] == ["company.md"]


def test_detect_new_files_node_returns_empty_when_no_changes():
    existing = {
        "base_summary": "x",
        "base_files": [],
        "ranked_files": [
            {"name": "company.md", "summary": "s", "reason": "r"},
        ],
    }
    result = detect_new_files_node(
        {
            "folder": "acme",
            "existing_state": existing,
            "all_files": [{"name": "company.md", "size_bytes": 1}],
        }
    )
    assert result["new_file_names"] == []


# ---- summarize_each --------------------------------------------------------


@pytest.mark.asyncio
async def test_summarize_each_node_uses_base_summary_in_prompt():
    summaries = {
        "company.md": "Company summary.",
        "products.md": "Product summary.",
    }

    async def fake_ainvoke(messages):
        user_content = messages[1].content
        assert "BRAND CONTEXT" in user_content
        assert "Brand: Acme makes rockets." in user_content
        for name, text in summaries.items():
            if name in user_content:
                resp = MagicMock()
                resp.content = text
                return resp
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
                "target_file_contents": {
                    "company.md": "Acme makes rockets.",
                    "products.md": "Flagship rocket A1.",
                },
                "base_summary": "Brand: Acme makes rockets.",
            }
        )

    assert [s["name"] for s in result["new_file_summaries"]] == [
        "company.md",
        "products.md",
    ]


@pytest.mark.asyncio
async def test_summarize_each_node_short_circuits_when_no_contents():
    mock_llm = MagicMock()
    with patch("app.graph.nodes.get_llm", return_value=mock_llm):
        result = await summarize_each_node({"target_file_contents": {}})
    assert result["new_file_summaries"] == []


# ---- rank_files ------------------------------------------------------------


def test_rank_files_node_orders_files_by_array_position():
    payload = {
        "order": [
            {"name": "products.md", "reason": "product detail"},
            {"name": "company.md", "reason": "core brand"},
        ]
    }
    mock_llm = _bound_llm_returning(payload)
    with patch("app.graph.nodes.get_llm", return_value=mock_llm):
        result = rank_files_node(
            {
                "base_summary": "x",
                "new_file_summaries": [
                    {"name": "company.md", "summary": "a"},
                    {"name": "products.md", "summary": "b"},
                ],
            }
        )
    names = [r["name"] for r in result["new_ranked_files"]]
    assert names == ["products.md", "company.md"]
    assert result["new_ranked_files"][0]["summary"] == "b"


def test_rank_files_node_rejects_unknown_filename():
    payload = {"order": [{"name": "ghost.md", "reason": "x"}]}
    mock_llm = _bound_llm_returning(payload)
    with patch("app.graph.nodes.get_llm", return_value=mock_llm):
        with pytest.raises(HTTPException) as excinfo:
            rank_files_node(
                {
                    "base_summary": "x",
                    "new_file_summaries": [{"name": "company.md", "summary": "a"}],
                }
            )
    assert excinfo.value.status_code == 502


def test_rank_files_node_short_circuits_with_empty_summaries():
    # No mock needed — early-return before any LLM call.
    result = rank_files_node({"base_summary": "x", "new_file_summaries": []})
    assert result["new_ranked_files"] == []


# ---- save_state ------------------------------------------------------------


def test_save_state_node_writes_full_merged_list_on_update(fixture_folder, state_store):
    existing = {
        "base_summary": "x",
        "base_files": ["company.md"],
        "ranked_files": [
            {"name": "company.md", "summary": "c", "reason": "r1"},
        ],
    }
    state = {
        "folder": "acme",
        "base_summary": "x",
        "base_files": ["company.md"],
        "existing_state": existing,
        "new_ranked_files": [
            {"name": "new.md", "summary": "n", "reason": "r2"},
        ],
        "errors": [],
    }
    result = save_state_node(state)
    assert [r["name"] for r in result["ranked_files"]] == ["company.md", "new.md"]

    saved = state_store["acme"]
    assert [r["name"] for r in saved["ranked_files"]] == ["company.md", "new.md"]
    assert saved["base_summary"] == "x"
    assert saved["folder"] == "acme"
