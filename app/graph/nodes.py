import asyncio
from pathlib import Path

from fastapi import HTTPException
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.llm import get_llm
from app.prompts import FILE_SUMMARY_SYSTEM_PROMPT, FILE_SUMMARY_USER_TEMPLATE
from app.schemas import PipelineState

_MAX_CONCURRENT_SUMMARIES = 5


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


def read_files_node(state: PipelineState) -> dict:
    """Read EVERY file in the folder — no selection step anymore."""
    folder_path = Path(state["folder_path"])
    contents: dict[str, str] = {}
    errors: list[str] = list(state.get("errors", []))
    for entry in state["all_files"]:
        name = entry["name"]
        path = folder_path / name
        try:
            contents[name] = path.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"Failed to read {name}: {exc}")
    return {"file_contents": contents, "errors": errors}


async def _summarize_one(
    llm,
    semaphore: asyncio.Semaphore,
    name: str,
    content: str,
) -> tuple[str, str | None, str | None]:
    """Summarize one file. Returns (name, summary_or_None, error_or_None)."""
    async with semaphore:
        messages = [
            SystemMessage(content=FILE_SUMMARY_SYSTEM_PROMPT),
            HumanMessage(
                content=FILE_SUMMARY_USER_TEMPLATE.format(
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
    """One LLM call per file, run in parallel under a concurrency cap."""
    llm = get_llm()
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_SUMMARIES)
    contents = state["file_contents"]
    tasks = [
        _summarize_one(llm, semaphore, name, content)
        for name, content in contents.items()
    ]
    results = await asyncio.gather(*tasks)

    summaries: list[dict] = []
    errors: list[str] = list(state.get("errors", []))
    # Preserve folder order (the order of all_files), not gather completion order.
    by_name = {name: (summary, err) for name, summary, err in results}
    for entry in state["all_files"]:
        name = entry["name"]
        if name not in by_name:
            continue
        summary, err = by_name[name]
        if err:
            errors.append(err)
        if summary:
            summaries.append({"name": name, "summary": summary})

    return {"file_summaries": summaries, "errors": errors}
