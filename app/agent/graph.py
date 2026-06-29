from langgraph.graph import END, StateGraph

from app.agent.nodes import AgentNodes
from app.agent.state import AgentState
from app.indexer import IndexManager
from app.reranker import BaseReranker


def route_after_grading(state: AgentState) -> str:
    if state.get("retrieval_passed"):
        return "rerank_documents"
    if state.get("need_rewrite"):
        return "rewrite_query"
    return "fallback_response"


def route_after_answer_grading(state: AgentState) -> str:
    if state.get("answer_supported"):
        return END
    return "fallback_response"


def build_agentic_graph(
    index_manager: IndexManager,
    reranker: BaseReranker | None = None,
    top_k: int = 5,
):
    nodes = AgentNodes(index_manager=index_manager, reranker=reranker, top_k=top_k)
    graph = StateGraph(AgentState)

    graph.add_node("classify_question", nodes.classify_question)
    graph.add_node("retrieve_documents", nodes.retrieve_documents)
    graph.add_node("grade_documents", nodes.grade_documents)
    graph.add_node("rewrite_query", nodes.rewrite_query)
    graph.add_node("rerank_documents", nodes.rerank_documents)
    graph.add_node("generate_answer", nodes.generate_answer)
    graph.add_node("grade_answer", nodes.grade_answer)
    graph.add_node("fallback_response", nodes.fallback_response)

    graph.set_entry_point("classify_question")
    graph.add_edge("classify_question", "retrieve_documents")
    graph.add_edge("retrieve_documents", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "rerank_documents": "rerank_documents",
            "rewrite_query": "rewrite_query",
            "fallback_response": "fallback_response",
        },
    )
    graph.add_edge("rewrite_query", "retrieve_documents")
    graph.add_edge("rerank_documents", "generate_answer")
    graph.add_edge("generate_answer", "grade_answer")
    graph.add_conditional_edges(
        "grade_answer",
        route_after_answer_grading,
        {
            END: END,
            "fallback_response": "fallback_response",
        },
    )
    graph.add_edge("fallback_response", END)
    return graph.compile()


def run_agentic_rag(
    question: str,
    index_manager: IndexManager,
    reranker: BaseReranker | None = None,
    top_k: int = 5,
) -> AgentState:
    graph = build_agentic_graph(index_manager=index_manager, reranker=reranker, top_k=top_k)
    initial_state: AgentState = {
        "question": question,
        "rewritten_question": None,
        "route_trace": [],
        "rewrite_count": 0,
        "retrieved_chunk_ids": [],
        "reranked_chunk_ids": [],
        "citations": [],
    }
    return graph.invoke(initial_state)
