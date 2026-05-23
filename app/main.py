import logging

from fastapi import FastAPI, HTTPException

from app.graph.builder import get_compiled_graph, get_compiled_scoring_graph
from app.schemas import (
    ScoreFilesRequest,
    ScoreFilesResponse,
    SummarizeRequest,
    SummarizeResponse,
)

logger = logging.getLogger("brand_summarizer")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Brand Summarizer",
    description="LangGraph pipeline that selects and summarizes brand documents.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest) -> SummarizeResponse:
    graph = get_compiled_graph()
    try:
        final_state = graph.invoke({"folder": request.folder})
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("LLM pipeline failed")
        raise HTTPException(status_code=502, detail=f"LLM pipeline failed: {exc}") from exc

    return SummarizeResponse(
        folder=request.folder,
        all_files=final_state.get("all_files", []),
        selected_files=final_state.get("selected_files", []),
        skipped_files=final_state.get("skipped_files", []),
        summary=final_state.get("summary", ""),
        errors=final_state.get("errors", []),
    )


@app.post("/score-files", response_model=ScoreFilesResponse)
def score_files(request: ScoreFilesRequest) -> ScoreFilesResponse:
    graph = get_compiled_scoring_graph()
    try:
        final_state = graph.invoke(
            {
                "folder": request.folder,
                "summary_file": request.summary_file,
            }
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("LLM scoring pipeline failed")
        raise HTTPException(
            status_code=502,
            detail=f"LLM scoring pipeline failed: {exc}",
        ) from exc

    return ScoreFilesResponse(
        folder=request.folder,
        summary_file=request.summary_file,
        all_files=final_state.get("all_files", []),
        scored_files=final_state.get("scored_files", []),
        errors=final_state.get("errors", []),
    )
