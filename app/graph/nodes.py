import json
import re
from pathlib import Path

from fastapi import HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from app.config import settings
from app.llm import get_llm
from app.prompts import (
    FILE_SCORING_SYSTEM_PROMPT,
    FILE_SCORING_USER_TEMPLATE,
    SELECTION_SYSTEM_PROMPT,
    SELECTION_USER_TEMPLATE,
    SUMMARIZATION_SYSTEM_PROMPT,
    SUMMARIZATION_USER_TEMPLATE,
)
from app.ranking import rank_scored_files
from app.schemas import FileScore, PipelineState, SelectionResult


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_SCORE_KEYS = (
    "authority_score",
    "foundational_value",
    "currentness_score",
    "entity_relationship_clarity",
    "content_specificity",
    "document_structure_quality",
    "conflict_risk",
    "duplicate_penalty",
    "noise_penalty",
    "ambiguity_penalty",
)
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


def _parse_file_score(raw: str, name: str, size_bytes: int) -> FileScore:
    """Parse one file-scoring LLM response, tolerating markdown code fences."""
    text = raw.strip()
    match = _JSON_BLOCK_RE.search(text)
    if match:
        text = match.group(1)
    elif text.startswith("```"):
        text = text.strip("`").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        payload = _parse_partial_file_score(text, name, exc)

    payload["name"] = name
    payload["size_bytes"] = size_bytes
    try:
        return FileScore.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Score JSON for {name} did not match schema: {exc}",
        ) from exc


def _parse_partial_file_score(text: str, name: str, exc: json.JSONDecodeError) -> dict:
    """Recover score fields when the LLM returns malformed JSON text."""
    payload: dict[str, int | str] = {}
    for key in _SCORE_KEYS:
        match = re.search(rf'"{key}"\s*:\s*(\d+)', text)
        if not match:
            raise HTTPException(
                status_code=502,
                detail=f"LLM returned unusable score for {name}: missing {key}; JSON error={exc}; raw head={text[:200]!r}",
            ) from exc
        value = int(match.group(1))
        payload[key] = max(0, min(100, value))

    reason_match = re.search(r'"reason"\s*:\s*"(.+?)"\s*[,}]', text, re.DOTALL)
    if reason_match:
        reason = reason_match.group(1).replace("\n", " ").strip()
    else:
        reason = f"Recovered numeric scores from malformed LLM JSON for {name}."
    payload["reason"] = reason[:500]
    return payload


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


def _resolve_file_in_folder(folder_path: Path, filename: str) -> Path:
    candidate = (folder_path / filename).resolve()
    if folder_path.resolve() != candidate and folder_path.resolve() not in candidate.parents:
        raise HTTPException(
            status_code=400,
            detail=f"File '{filename}' resolves outside requested folder",
        )
    if not candidate.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"File '{filename}' does not exist in requested folder",
        )
    return candidate


def _resolve_summary_file(folder_path: Path, filename: str) -> Path:
    folder_candidate = (folder_path / filename).resolve()
    if folder_candidate.is_file():
        return _resolve_file_in_folder(folder_path, filename)

    base_candidate = (settings.BASE_DIR.resolve() / filename).resolve()
    if settings.BASE_DIR.resolve() != base_candidate and settings.BASE_DIR.resolve() not in base_candidate.parents:
        raise HTTPException(
            status_code=400,
            detail=f"Summary file '{filename}' resolves outside BASE_DIR",
        )
    if not base_candidate.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Summary file '{filename}' does not exist in requested folder or BASE_DIR",
        )
    return base_candidate


def _extract_company_summary(summary_path: Path) -> str:
    try:
        text = summary_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read summary file '{summary_path.name}': {exc}",
        ) from exc

    if summary_path.suffix.lower() != ".json":
        return text

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Summary JSON file '{summary_path.name}' is invalid: {exc}",
        ) from exc

    summary = payload.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise HTTPException(
            status_code=400,
            detail=f"Summary JSON file '{summary_path.name}' must contain a non-empty 'summary' field",
        )
    return summary


def _extract_markdown_headings(text: str, limit: int = 20) -> list[str]:
    headings: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            if heading:
                headings.append(heading)
        if len(headings) >= limit:
            break
    return headings


def _build_preview(text: str, max_chars: int = 5000) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)[:max_chars]


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


def read_company_summary_node(state: PipelineState) -> dict:
    folder_path = Path(state["folder_path"])
    summary_path = _resolve_summary_file(folder_path, state["summary_file"])
    return {"company_summary": _extract_company_summary(summary_path)}


def build_file_previews_node(state: PipelineState) -> dict:
    folder_path = Path(state["folder_path"])
    previews: list[dict] = []
    errors: list[str] = list(state.get("errors", []))

    for entry in state["all_files"]:
        name = entry["name"]
        if name == state.get("summary_file"):
            continue

        path = folder_path / name
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"Failed to preview {name}: {exc}")
            continue

        previews.append(
            {
                "name": name,
                "size_bytes": entry["size_bytes"],
                "headings": _extract_markdown_headings(text),
                "preview": _build_preview(text),
            }
        )

    return {"file_previews": previews, "errors": errors}


def score_files_node(state: PipelineState) -> dict:
    scored_files: list[dict] = []
    llm = get_llm().bind(response_format={"type": "json_object"})

    for preview in state["file_previews"]:
        headings = "\n".join(f"- {heading}" for heading in preview["headings"])
        if not headings:
            headings = "(none detected)"

        messages = [
            SystemMessage(content=FILE_SCORING_SYSTEM_PROMPT),
            HumanMessage(
                content=FILE_SCORING_USER_TEMPLATE.format(
                    company_summary=state["company_summary"],
                    name=preview["name"],
                    size_bytes=preview["size_bytes"],
                    headings=headings,
                    preview=preview["preview"],
                )
            ),
        ]
        response = llm.invoke(messages)
        raw = response.content if isinstance(response.content, str) else str(response.content)
        scored_files.append(
            _parse_file_score(
                raw=raw,
                name=preview["name"],
                size_bytes=preview["size_bytes"],
            ).model_dump()
        )

    return {"scored_files": scored_files}


def rank_scored_files_node(state: PipelineState) -> dict:
    return {"scored_files": rank_scored_files(state["scored_files"])}


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
