from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    build_file_previews_node,
    list_files_node,
    read_files_node,
    read_company_summary_node,
    rank_scored_files_node,
    select_files_node,
    score_files_node,
    summarize_node,
)
from app.schemas import PipelineState


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("list_files", list_files_node)
    graph.add_node("select_files", select_files_node)
    graph.add_node("read_files", read_files_node)
    graph.add_node("summarize", summarize_node)

    graph.add_edge(START, "list_files")
    graph.add_edge("list_files", "select_files")
    graph.add_edge("select_files", "read_files")
    graph.add_edge("read_files", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()


def build_scoring_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("list_files", list_files_node)
    graph.add_node("read_company_summary", read_company_summary_node)
    graph.add_node("build_file_previews", build_file_previews_node)
    graph.add_node("score_files", score_files_node)
    graph.add_node("rank_scored_files", rank_scored_files_node)

    graph.add_edge(START, "list_files")
    graph.add_edge("list_files", "read_company_summary")
    graph.add_edge("read_company_summary", "build_file_previews")
    graph.add_edge("build_file_previews", "score_files")
    graph.add_edge("score_files", "rank_scored_files")
    graph.add_edge("rank_scored_files", END)

    return graph.compile()


@lru_cache(maxsize=1)
def get_compiled_graph():
    return build_graph()


@lru_cache(maxsize=1)
def get_compiled_scoring_graph():
    return build_scoring_graph()
