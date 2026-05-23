import logging

from fastapi import FastAPI, HTTPException

from app import state as state_module
from app.graph.builder import get_onboarding_graph, get_update_graph
from app.schemas import SummarizeRequest, SummarizeResponse

logger = logging.getLogger("brand_summarizer")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Brand Summarizer",
    description="Two-mode LangGraph pipeline: onboard a new brand (build base "
    "summary + summarize + rank every file), or update an existing brand "
    "(summarize + rank only newly-added files using the saved base summary).",
    version="0.4.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest) -> SummarizeResponse:
    existing = state_module.load_state(request.folder)
    mode = "update" if existing is not None else "onboard"
    graph = get_update_graph() if existing else get_onboarding_graph()

    try:
        final_state = await graph.ainvoke({"folder": request.folder})
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("LLM pipeline failed")
        raise HTTPException(status_code=502, detail=f"LLM pipeline failed: {exc}") from exc

    return SummarizeResponse(
        folder=request.folder,
        mode=mode,
        base_summary=final_state.get("base_summary", ""),
        base_files=final_state.get("base_files", []),
        all_files=final_state.get("all_files", []),
        ranked_files=final_state.get("ranked_files", []),
        newly_added=final_state.get("new_ranked_files", []),
        errors=final_state.get("errors", []),
    )
