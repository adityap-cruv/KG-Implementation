import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.graph.builder import build_onboarding_graph, build_update_graph


def _response(text: str):
    r = MagicMock()
    r.content = text
    return r


def _make_llm(json_payloads_by_keyword: dict, sync_text: str, per_file_summaries: dict):
    """Build an LLM mock that handles all three call patterns:
    - sync .invoke() returning plain text → base-summary path
    - .bind().invoke() returning JSON → base-file-selection + ranking paths
    - async .ainvoke() per-file → summarize_each path
    """
    bound = MagicMock()

    def bound_invoke(messages):
        sys_text = messages[0].content
        if "knowledge graph" in sys_text and "ORDERED list" in sys_text:
            return _response(json.dumps(json_payloads_by_keyword["ranking"]))
        if "FILENAMES of every file" in sys_text:
            return _response(json.dumps(json_payloads_by_keyword["base_select"]))
        raise AssertionError(f"unexpected bound call. system head: {sys_text[:80]}")

    bound.invoke.side_effect = bound_invoke

    async def fake_ainvoke(messages):
        user_content = messages[1].content
        for name, text in per_file_summaries.items():
            if name in user_content:
                return _response(text)
        raise AssertionError("unknown file in summarize prompt")

    llm = MagicMock()
    llm.bind.return_value = bound
    llm.invoke.return_value = _response(sync_text)  # base summary path
    llm.ainvoke = AsyncMock(side_effect=fake_ainvoke)
    return llm


@pytest.mark.asyncio
async def test_onboarding_graph_end_to_end(fixture_folder, state_store):
    mock_llm = _make_llm(
        json_payloads_by_keyword={
            "base_select": {
                "selected": ["company.md", "products.md"],
                "reason": "core brand pages",
            },
            "ranking": {
                "order": [
                    {"name": "company.md", "reason": "core brand"},
                    {"name": "products.md", "reason": "product detail"},
                    {"name": "privacy.md", "reason": "legal last"},
                ]
            },
        },
        sync_text="Acme is a rocket company headquartered in Earth.",
        per_file_summaries={
            "company.md": "Acme company summary.",
            "products.md": "Acme products summary.",
            "privacy.md": "Privacy policy summary.",
        },
    )

    with patch("app.graph.nodes.get_llm", return_value=mock_llm):
        graph = build_onboarding_graph()
        state = await graph.ainvoke({"folder": "acme"})

    assert state["base_summary"].startswith("Acme is a rocket company")
    assert state["base_files"] == ["company.md", "products.md"]
    assert [r["name"] for r in state["ranked_files"]] == [
        "company.md",
        "products.md",
        "privacy.md",
    ]
    assert state["ranked_files"][0]["summary"] == "Acme company summary."

    # State was persisted (via the in-memory fixture stand-in for Mongo).
    saved = state_store["acme"]
    assert len(saved["ranked_files"]) == 3
    assert saved["folder"] == "acme"


@pytest.mark.asyncio
async def test_update_graph_processes_only_new_files(fixture_folder):
    # Pre-seed an existing state as if onboarding already happened.
    from app import state as state_module

    state_module.save_state(
        "acme",
        {
            "folder": "acme",
            "base_summary": "Acme is a rocket company.",
            "base_files": ["company.md"],
            "ranked_files": [
                {"name": "company.md", "summary": "old c", "reason": "r"},
                {"name": "products.md", "summary": "old p", "reason": "r"},
                {"name": "privacy.md", "summary": "old priv", "reason": "r"},
            ],
            "errors": [],
        },
    )

    # Add a brand-new file to the folder.
    (fixture_folder / "newpage.md").write_text("# New\nA newly added page.\n")

    mock_llm = _make_llm(
        json_payloads_by_keyword={
            "base_select": {"selected": [], "reason": "unused on update"},
            "ranking": {
                "order": [{"name": "newpage.md", "reason": "fresh content"}]
            },
        },
        sync_text="unused on update",
        per_file_summaries={"newpage.md": "Newpage summary."},
    )

    with patch("app.graph.nodes.get_llm", return_value=mock_llm):
        graph = build_update_graph()
        state = await graph.ainvoke({"folder": "acme"})

    # Only one new ranked entry, but the merged list keeps the existing three.
    assert [r["name"] for r in state["new_ranked_files"]] == ["newpage.md"]
    assert [r["name"] for r in state["ranked_files"]] == [
        "company.md",
        "products.md",
        "privacy.md",
        "newpage.md",
    ]
    # Base summary was reused, not regenerated — sync invoke should not have been called.
    mock_llm.invoke.assert_not_called()


@pytest.mark.asyncio
async def test_update_graph_with_no_new_files_is_a_no_op(fixture_folder):
    from app import state as state_module

    state_module.save_state(
        "acme",
        {
            "folder": "acme",
            "base_summary": "Acme.",
            "base_files": [],
            "ranked_files": [
                {"name": "company.md", "summary": "x", "reason": "r"},
                {"name": "products.md", "summary": "x", "reason": "r"},
                {"name": "privacy.md", "summary": "x", "reason": "r"},
            ],
            "errors": [],
        },
    )

    mock_llm = _make_llm(
        json_payloads_by_keyword={
            "base_select": {"selected": [], "reason": ""},
            "ranking": {"order": []},
        },
        sync_text="unused",
        per_file_summaries={},
    )

    with patch("app.graph.nodes.get_llm", return_value=mock_llm):
        graph = build_update_graph()
        state = await graph.ainvoke({"folder": "acme"})

    assert state["new_ranked_files"] == []
    assert [r["name"] for r in state["ranked_files"]] == [
        "company.md",
        "products.md",
        "privacy.md",
    ]
    # No per-file LLM calls when nothing is new.
    mock_llm.ainvoke.assert_not_awaited()
