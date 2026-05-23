from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    list_files_node,
    read_files_node,
    select_files_node,
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


@lru_cache(maxsize=1)
def get_compiled_graph():
    return build_graph()
