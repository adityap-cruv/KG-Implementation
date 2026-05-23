import asyncio
import json
import re
from pathlib import Path

from fastapi import HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from app.config import settings
from app.llm import get_llm
from app.prompts import (
    BASE_SUMMARY_SYSTEM_PROMPT,
    BASE_SUMMARY_USER_TEMPLATE,
    FILE_SUMMARY_SYSTEM_PROMPT,
    FILE_SUMMARY_USER_TEMPLATE,
    RANKING_SYSTEM_PROMPT,
    RANKING_USER_TEMPLATE,
)
from app import state as state_module
from app.schemas import IngestOrder, PipelineState

_MAX_CONCURRENT_SUMMARIES = 5
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_folder(folder: str) -> Path:
    base = settings.BASE_DIR.resolve()
    candidate = (base / folder).resolve()
    if base != candidate and base not in candidate.parents:
        raise HTTPException(
            status_code=400,
            detail=f"Folder '{folder}' resolves outside BASE_DIR",
        )
    if not candidate.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Folder '{folder}' does not exist",
        )
    return candidate


def _parse_json_response(raw: str, model_cls, kind: str):
    """Parse a JSON response, tolerating ```json fences. Raises HTTP 502 on failure."""
    text = raw.strip()
    match = _JSON_BLOCK_RE.search(text)
    if match:
        text = match.group(1)
    elif text.startswith("```"):
        text = text.strip("`").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM returned non-JSON {kind}: {exc}; raw head={raw[:200]!r}",
        ) from exc
    try:
        return model_cls.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"{kind} JSON did not match schema: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Common nodes
# ---------------------------------------------------------------------------


def list_files_node(state: PipelineState) -> dict:
    folder_path = _resolve_folder(state["folder"])
    md_files = sorted(folder_path.glob("*.md"))
    if not md_files:
        raise HTTPException(
            status_code=404,
            detail=f"Folder '{state['folder']}' contains no .md files",
        )
    return {
        "folder_path": str(folder_path),
        "all_files": [
            {"name": p.name, "size_bytes": p.stat().st_size} for p in md_files
        ],
        "errors": state.get("errors", []),
    }


def load_state_node(state: PipelineState) -> dict:
    existing = state_module.load_state(state["folder"])
    return {"existing_state": existing}


# ---------------------------------------------------------------------------
# Onboarding-only nodes
# ---------------------------------------------------------------------------


def build_base_summary_node(state: PipelineState) -> dict:
    """Synthesize the brand-level summary from EVERY file in the folder.

    Reads from `state.target_file_contents` (populated by read_all_files_for_summarization_node).
    Source documents are joined with anonymous dividers (no filenames) so the
    LLM can't structure its output by source — the goal is one unified brand
    narrative, not a stitched-together set of per-file paragraphs.
    """
    contents = state.get("target_file_contents", {})
    if not contents:
        raise HTTPException(
            status_code=500,
            detail="No file contents available to build the base summary",
        )
    # Preserve folder order so the model sees foundational pages early in context.
    ordered_names = [f["name"] for f in state["all_files"] if f["name"] in contents]
    documents = "\n\n---\n\n".join(contents[name].strip() for name in ordered_names)

    messages = [
        SystemMessage(content=BASE_SUMMARY_SYSTEM_PROMPT),
        HumanMessage(
            content=BASE_SUMMARY_USER_TEMPLATE.format(
                folder=state["folder"],
                documents=documents,
            )
        ),
    ]
    response = get_llm().invoke(messages)
    text = response.content if isinstance(response.content, str) else str(response.content)
    return {
        "base_summary": text.strip(),
        # Record which files contributed to the base summary (= all files at onboarding time).
        "base_files": ordered_names,
    }


def read_all_files_for_summarization_node(state: PipelineState) -> dict:
    """Onboarding: read EVERY file in the folder so summarize_each has it ready."""
    folder_path = Path(state["folder_path"])
    contents: dict[str, str] = {}
    errors = list(state.get("errors", []))
    for entry in state["all_files"]:
        name = entry["name"]
        try:
            contents[name] = (folder_path / name).read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"Failed to read {name}: {exc}")
    return {"target_file_contents": contents, "errors": errors}


# ---------------------------------------------------------------------------
# Update-only nodes
# ---------------------------------------------------------------------------


def detect_new_files_node(state: PipelineState) -> dict:
    """Diff current folder against the saved state; identify net-new filenames."""
    existing = state.get("existing_state")
    if not existing:
        raise HTTPException(
            status_code=500,
            detail="Update mode invoked but no existing state was loaded",
        )
    known_names = {entry["name"] for entry in existing.get("ranked_files", [])}
    current_names = {entry["name"] for entry in state["all_files"]}
    new_names = sorted(current_names - known_names)

    # Reuse the saved base summary and base files — do not regenerate.
    return {
        "base_summary": existing["base_summary"],
        "base_files": existing.get("base_files", []),
        "new_file_names": new_names,
    }


def read_new_files_node(state: PipelineState) -> dict:
    """Update: read ONLY the new files."""
    folder_path = Path(state["folder_path"])
    contents: dict[str, str] = {}
    errors = list(state.get("errors", []))
    for name in state.get("new_file_names", []):
        try:
            contents[name] = (folder_path / name).read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"Failed to read {name}: {exc}")
    return {"target_file_contents": contents, "errors": errors}


# ---------------------------------------------------------------------------
# Common after-this-point nodes (per-file summarize + ranking + save)
# ---------------------------------------------------------------------------


async def _summarize_one(
    llm,
    semaphore: asyncio.Semaphore,
    name: str,
    content: str,
    base_summary: str,
) -> tuple[str, str | None, str | None]:
    async with semaphore:
        messages = [
            SystemMessage(content=FILE_SUMMARY_SYSTEM_PROMPT),
            HumanMessage(
                content=FILE_SUMMARY_USER_TEMPLATE.format(
                    base_summary=base_summary,
                    filename=name,
                    content=content,
                )
            ),
        ]
        try:
            response = await llm.ainvoke(messages)
            text = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
            return name, text.strip(), None
        except Exception as exc:
            return name, None, f"summarize {name}: {exc}"


async def summarize_each_node(state: PipelineState) -> dict:
    """Run one LLM call per file in `target_file_contents`, parallel, grounded in base summary."""
    contents = state.get("target_file_contents", {})
    if not contents:
        # Nothing to summarize this run (e.g., update mode with zero new files).
        return {"new_file_summaries": []}

    llm = get_llm()
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_SUMMARIES)
    base_summary = state["base_summary"]

    tasks = [
        _summarize_one(llm, semaphore, name, content, base_summary)
        for name, content in contents.items()
    ]
    results = await asyncio.gather(*tasks)

    errors = list(state.get("errors", []))
    by_name = {name: (summary, err) for name, summary, err in results}
    # Preserve folder order so the ranking input is stable.
    ordered_names = [f["name"] for f in state["all_files"] if f["name"] in by_name]
    summaries: list[dict] = []
    for name in ordered_names:
        summary, err = by_name[name]
        if err:
            errors.append(err)
        if summary:
            summaries.append({"name": name, "summary": summary})

    return {"new_file_summaries": summaries, "errors": errors}


def rank_files_node(state: PipelineState) -> dict:
    """One LLM call that orders the files-summarized-this-run for KG ingestion."""
    summaries = state.get("new_file_summaries", [])
    if not summaries:
        # Update mode with no new files: nothing to rank.
        return {"new_ranked_files": []}

    summaries_block = "\n\n".join(
        f"### {entry['name']}\n{entry['summary']}" for entry in summaries
    )
    messages = [
        SystemMessage(content=RANKING_SYSTEM_PROMPT),
        HumanMessage(
            content=RANKING_USER_TEMPLATE.format(
                base_summary=state["base_summary"],
                summaries_block=summaries_block,
            )
        ),
    ]
    llm = get_llm().bind(response_format={"type": "json_object"})
    response = llm.invoke(messages)
    raw = response.content if isinstance(response.content, str) else str(response.content)
    decision = _parse_json_response(raw, IngestOrder, "ranking")

    expected_names = {entry["name"] for entry in summaries}
    seen_names: set[str] = set()
    for entry in decision.order:
        if entry.name not in expected_names:
            raise HTTPException(
                status_code=502,
                detail=f"Ranking referenced unknown file: {entry.name!r}",
            )
        if entry.name in seen_names:
            raise HTTPException(
                status_code=502,
                detail=f"Ranking duplicated file: {entry.name!r}",
            )
        seen_names.add(entry.name)
    missing = expected_names - seen_names
    if missing:
        raise HTTPException(
            status_code=502,
            detail=f"Ranking missing files: {sorted(missing)}",
        )

    summary_by_name = {entry["name"]: entry["summary"] for entry in summaries}
    new_ranked = [
        {
            "name": entry.name,
            "summary": summary_by_name[entry.name],
            "reason": entry.reason,
        }
        for entry in decision.order
    ]
    return {"new_ranked_files": new_ranked}


def save_state_node(state: PipelineState) -> dict:
    """Persist the brand state. In onboarding the full list IS the new ranked.
    In update, the new ranked is appended to the existing list."""
    existing = state.get("existing_state") or {}
    existing_ranked = list(existing.get("ranked_files", []))
    new_ranked = state.get("new_ranked_files", [])

    merged_ranked = existing_ranked + new_ranked

    payload = {
        "folder": state["folder"],
        "base_summary": state["base_summary"],
        "base_files": state.get("base_files", []),
        "ranked_files": merged_ranked,
        "errors": state.get("errors", []),
    }
    state_module.save_state(state["folder"], payload)
    return {"ranked_files": merged_ranked}
