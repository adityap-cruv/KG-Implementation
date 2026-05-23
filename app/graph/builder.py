from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    build_base_summary_node,
    detect_new_files_node,
    list_files_node,
    load_state_node,
    rank_files_node,
    read_all_files_for_summarization_node,
    read_new_files_node,
    save_state_node,
    summarize_each_node,
)
from app.schemas import PipelineState


def build_onboarding_graph():
    """First-time pipeline for a brand: build base summary from ALL files,
    then summarize + rank every file."""
    g = StateGraph(PipelineState)
    g.add_node("list_files", list_files_node)
    g.add_node("read_all_files", read_all_files_for_summarization_node)
    g.add_node("build_base_summary", build_base_summary_node)
    g.add_node("summarize_each", summarize_each_node)
    g.add_node("rank_files", rank_files_node)
    g.add_node("save_state", save_state_node)

    g.add_edge(START, "list_files")
    g.add_edge("list_files", "read_all_files")
    g.add_edge("read_all_files", "build_base_summary")
    g.add_edge("build_base_summary", "summarize_each")
    g.add_edge("summarize_each", "rank_files")
    g.add_edge("rank_files", "save_state")
    g.add_edge("save_state", END)

    return g.compile()


def build_update_graph():
    """Incremental pipeline: reuse base summary, summarize + rank only NEW files."""
    g = StateGraph(PipelineState)
    g.add_node("load_state", load_state_node)
    g.add_node("list_files", list_files_node)
    g.add_node("detect_new_files", detect_new_files_node)
    g.add_node("read_new_files", read_new_files_node)
    g.add_node("summarize_each", summarize_each_node)
    g.add_node("rank_files", rank_files_node)
    g.add_node("save_state", save_state_node)

    g.add_edge(START, "load_state")
    g.add_edge("load_state", "list_files")
    g.add_edge("list_files", "detect_new_files")
    g.add_edge("detect_new_files", "read_new_files")
    g.add_edge("read_new_files", "summarize_each")
    g.add_edge("summarize_each", "rank_files")
    g.add_edge("rank_files", "save_state")
    g.add_edge("save_state", END)

    return g.compile()


@lru_cache(maxsize=1)
def get_onboarding_graph():
    return build_onboarding_graph()


@lru_cache(maxsize=1)
def get_update_graph():
    return build_update_graph()
