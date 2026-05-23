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


class PipelineState(TypedDict, total=False):
    folder: str
    folder_path: str
    all_files: list[dict]
    selected_files: list[dict]
    skipped_files: list[dict]
    file_contents: dict[str, str]
    summary: str
    errors: list[str]
