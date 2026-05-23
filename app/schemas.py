from typing import TypedDict

from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    folder: str = Field(
        ...,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Folder name under BASE_DIR containing markdown files",
        examples=["genuin"],
    )


class FileInfo(BaseModel):
    name: str
    size_bytes: int


# ---- LLM structured-output schemas ----------------------------------------


class IngestEntry(BaseModel):
    """One slot in the ranking — order in the list is the rank."""

    name: str
    reason: str


class IngestOrder(BaseModel):
    """Ordered list: position [0] is ingested first, [N-1] last."""

    order: list[IngestEntry]


# ---- File-level output shape used in state + responses --------------------


class RankedFile(BaseModel):
    name: str
    summary: str
    reason: str  # why this file sits where it sits in the ingestion order


# ---- Full state + API response --------------------------------------------


class BrandState(BaseModel):
    """The full persisted state per brand. Written to state/<folder>.json."""

    folder: str
    base_summary: str
    base_files: list[str]
    ranked_files: list[RankedFile]
    errors: list[str] = Field(default_factory=list)


class SummarizeResponse(BaseModel):
    folder: str
    mode: str  # "onboard" or "update"
    base_summary: str
    base_files: list[str]
    all_files: list[FileInfo]
    ranked_files: list[RankedFile]       # full ordered list after this run
    newly_added: list[RankedFile]        # only what was processed THIS run
    errors: list[str]


# ---- Internal LangGraph state ---------------------------------------------


class PipelineState(TypedDict, total=False):
    folder: str
    folder_path: str
    all_files: list[dict]              # everything currently in the folder
    existing_state: dict | None         # loaded BrandState, or None on onboarding

    # Onboarding fills these; update reuses from existing_state:
    base_files: list[str]
    base_summary: str

    # Update-mode-only: filenames detected as new this run.
    new_file_names: list[str]

    # Files being processed this run (all on onboard, just new on update):
    target_file_contents: dict[str, str]

    # Per-file summaries produced this run:
    new_file_summaries: list[dict]      # [{name, summary}, ...]

    # Final outputs produced this run:
    new_ranked_files: list[dict]        # ordered [{name, summary, reason}, ...]
    ranked_files: list[dict]            # full merged ordered list (existing + new)

    errors: list[str]
