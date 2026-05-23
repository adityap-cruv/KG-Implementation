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


class FileSummary(BaseModel):
    name: str
    summary: str


class SummarizeResponse(BaseModel):
    folder: str
    all_files: list[FileInfo]
    file_summaries: list[FileSummary]
    errors: list[str]


class PipelineState(TypedDict, total=False):
    folder: str
    folder_path: str
    all_files: list[dict]
    file_contents: dict[str, str]
    file_summaries: list[dict]
    errors: list[str]
