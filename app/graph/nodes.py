import json
import re
from pathlib import Path

from fastapi import HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from app.config import settings
from app.llm import get_llm
from app.prompts import (
    SELECTION_SYSTEM_PROMPT,
    SELECTION_USER_TEMPLATE,
    SUMMARIZATION_SYSTEM_PROMPT,
    SUMMARIZATION_USER_TEMPLATE,
)
from app.schemas import PipelineState, SelectionResult


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_selection(raw: str) -> SelectionResult:
    """Parse the LLM's selection response, tolerating markdown code fences."""
    text = raw.strip()
    match = _JSON_BLOCK_RE.search(text)
    if match:
        text = match.group(1)
    elif text.startswith("```"):
        # fenced block without `json` hint
        text = text.strip("`").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM returned non-JSON selection: {exc}; raw head={raw[:200]!r}",
        ) from exc
    try:
        return SelectionResult.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Selection JSON did not match schema: {exc}",
        ) from exc


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
        "errors": [],
    }


def select_files_node(state: PipelineState) -> dict:
    file_list = "\n".join(
        f"- {f['name']} ({f['size_bytes']} bytes)" for f in state["all_files"]
    )
    messages = [
        SystemMessage(content=SELECTION_SYSTEM_PROMPT),
        HumanMessage(
            content=SELECTION_USER_TEMPLATE.format(
                folder=state["folder"],
                file_list=file_list,
            )
        ),
    ]
    # json_mode is more reliable than function_calling for OpenRouter-routed
    # reasoning models — no tool-call wrapper for the model to navigate.
    llm = get_llm().bind(response_format={"type": "json_object"})
    response = llm.invoke(messages)
    raw = response.content if isinstance(response.content, str) else str(response.content)
    result = _parse_selection(raw)

    known = {f["name"] for f in state["all_files"]}
    selected = [s.model_dump() for s in result.selected if s.name in known]
    skipped = [s.model_dump() for s in result.skipped if s.name in known]

    if not selected:
        raise HTTPException(
            status_code=422,
            detail="LLM selected no files for summarization",
        )
    return {"selected_files": selected, "skipped_files": skipped}


def read_files_node(state: PipelineState) -> dict:
    folder_path = Path(state["folder_path"])
    contents: dict[str, str] = {}
    errors: list[str] = list(state.get("errors", []))
    for entry in state["selected_files"]:
        name = entry["name"]
        path = folder_path / name
        try:
            contents[name] = path.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"Failed to read {name}: {exc}")
    return {"file_contents": contents, "errors": errors}


def summarize_node(state: PipelineState) -> dict:
    documents = "\n\n".join(
        f"## File: {entry['name']}\n\n{state['file_contents'].get(entry['name'], '')}"
        for entry in state["selected_files"]
        if entry["name"] in state["file_contents"]
    )
    messages = [
        SystemMessage(content=SUMMARIZATION_SYSTEM_PROMPT),
        HumanMessage(
            content=SUMMARIZATION_USER_TEMPLATE.format(
                folder=state["folder"],
                documents=documents,
            )
        ),
    ]
    response = get_llm().invoke(messages)
    summary = response.content if isinstance(response.content, str) else str(response.content)
    return {"summary": summary}
