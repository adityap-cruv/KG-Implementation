from typing import TypedDict

from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    folder: str = Field(
        ...,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Folder name under BASE_DIR containing markdown files",
        examples=["genuin"],
    )


class ScoreFilesRequest(BaseModel):
    folder: str = Field(
        ...,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Folder name under BASE_DIR containing markdown files",
        examples=["genuin"],
    )
    summary_file: str = Field(
        ...,
        pattern=r"^[a-zA-Z0-9_.-]+$",
        description="Company abstract summary file inside the folder",
        examples=["summary.md"],
    )


class FileInfo(BaseModel):
    name: str
    size_bytes: int


class SelectedFile(BaseModel):
    name: str
    reason: str


class SelectionResult(BaseModel):
    """Structured-output schema for the file-selection LLM call."""

    selected: list[SelectedFile] = Field(
        default_factory=list,
        description="Files chosen as most relevant for a brand summary",
    )
    skipped: list[SelectedFile] = Field(
        default_factory=list,
        description="Files explicitly excluded with reasoning",
    )


class SummarizeResponse(BaseModel):
    folder: str
    all_files: list[FileInfo]
    selected_files: list[SelectedFile]
    skipped_files: list[SelectedFile]
    summary: str
    errors: list[str]


class FilePreview(BaseModel):
    name: str
    size_bytes: int
    headings: list[str]
    preview: str


class FileScore(BaseModel):
    name: str
    size_bytes: int
    rank: int | None = None
    weighted_score: float | None = None
    authority_score: int = Field(..., ge=0, le=100)
    foundational_value: int = Field(..., ge=0, le=100)
    currentness_score: int = Field(..., ge=0, le=100)
    entity_relationship_clarity: int = Field(..., ge=0, le=100)
    content_specificity: int = Field(..., ge=0, le=100)
    document_structure_quality: int = Field(..., ge=0, le=100)
    conflict_risk: int = Field(..., ge=0, le=100)
    duplicate_penalty: int = Field(..., ge=0, le=100)
    noise_penalty: int = Field(..., ge=0, le=100)
    ambiguity_penalty: int = Field(..., ge=0, le=100)
    reason: str


class ScoreFilesResponse(BaseModel):
    folder: str
    summary_file: str
    all_files: list[FileInfo]
    scored_files: list[FileScore]
    errors: list[str]


class PipelineState(TypedDict, total=False):
    folder: str
    summary_file: str
    folder_path: str
    all_files: list[dict]
    company_summary: str
    file_previews: list[dict]
    scored_files: list[dict]
    selected_files: list[dict]
    skipped_files: list[dict]
    file_contents: dict[str, str]
    summary: str
    errors: list[str]
