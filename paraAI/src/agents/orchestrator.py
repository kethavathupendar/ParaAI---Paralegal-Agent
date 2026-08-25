"""
LangGraph orchestrator.

This is the heart of "multiple agents orchestrated into a model-agnostic
agentic workflow". Instead of one agent deciding ad hoc what to call next
(implicit, hard to debug), we define an explicit graph:

                        ┌───────────┐
                        │  router   │   inspects GraphState.kind
                        └─────┬─────┘
             ┌────────────────┼────────────────┐
             │                │                 │
      kind==DOCUMENT    kind==HEARING     kind==QUESTION
             │                │                 │
   ┌─────────▼───────┐ ┌──────▼───────────┐     │
   │ document_analyzer│ │ hearing_transcript│     │
   └─────────┬────────┘ └──────┬───────────┘     │
             │                 │                 │
             └────────┬────────┴─────────┬───────┘
                       │                  │
                ┌──────▼──────┐    ┌──────▼───────┐
                │  retrieve    │    │legal_research │  (runs when a question
                │  (RAG lookup)│    │(Indian Kanoon)│   is present, regardless
                └──────┬───────┘    └──────┬────────┘   of kind)
                       │                   │
                       └─────────┬─────────┘
                                 │
                          ┌──────▼──────┐
                          │ summarizer   │  always runs last
                          └──────┬──────┘
                                 │
                                END

Every node is a plain async function operating on `GraphState` — this is
what "model-agnostic" means in practice: nodes call `get_chat_model()` from
config.py, so switching MODEL_PROVIDER in `.env` changes every agent at
once without touching this graph definition.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agents.document_analyzer_agent import run_document_analyzer
from src.agents.hearing_transcript_agent import run_hearing_transcript_agent
from src.agents.legal_research_agent import run_legal_research_agent
from src.agents.summarizer_agent import run_summarizer
from src.models.schemas import GraphState, RequestKind
from src.rag.vector_store import VectorStore


# --- Nodes -------------------------------------------------------------

async def node_document_analyzer(state: GraphState) -> dict:
    if state.kind != RequestKind.DOCUMENT or not state.raw_input_path:
        return {}
    analysis = await run_document_analyzer(case_id=state.case_id, doc_id=_doc_id(state))
    return {"document_analysis": analysis}


async def node_hearing_transcript(state: GraphState) -> dict:
    if state.kind != RequestKind.HEARING or not state.raw_input_path:
        return {}
    minutes = await run_hearing_transcript_agent(case_id=state.case_id, doc_id=_doc_id(state))
    return {"hearing_minutes": minutes}


async def node_retrieve(state: GraphState) -> dict:
    """Pulls RAG context relevant to the question, scoped to this case only."""
    if not state.question:
        return {}
    store = VectorStore.get()
    results = store.query(state.question, case_id=state.case_id, top_k=6)
    return {"retrieved_context": results}


async def node_legal_research(state: GraphState) -> dict:
    if not state.question:
        return {}
    result = await run_legal_research_agent(state.question)
    return {"research": result}


async def node_summarizer(state: GraphState) -> dict:
    brief = await run_summarizer(state)
    return {"final_brief": brief}


def _doc_id(state: GraphState) -> str:
    # raw_input_path is set by the API layer to the doc_id-suffixed filename;
    # callers pass doc_id explicitly via state when known. Falls back to the
    # path stem if not.
    from pathlib import Path

    return Path(state.raw_input_path).stem


# --- Routing -------------------------------------------------------------

def route_after_ingestion(state: GraphState) -> str:
    """After the kind-specific extraction node, decide whether research/retrieval is needed."""
    if state.question:
        return "fan_out_research"
    return "summarizer"


def route_entry(state: GraphState) -> str:
    if state.kind == RequestKind.DOCUMENT:
        return "document_analyzer"
    if state.kind == RequestKind.HEARING:
        return "hearing_transcript"
    return "fan_out_research"  # pure QUESTION kind — skip straight to retrieval/research


# --- Graph construction ----------------------------------------------------

def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("document_analyzer", node_document_analyzer)
    graph.add_node("hearing_transcript", node_hearing_transcript)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("legal_research", node_legal_research)
    graph.add_node("summarizer", node_summarizer)

    graph.set_conditional_entry_point(
        route_entry,
        {
            "document_analyzer": "document_analyzer",
            "hearing_transcript": "hearing_transcript",
            "fan_out_research": "retrieve",
        },
    )

    graph.add_conditional_edges(
        "document_analyzer",
        route_after_ingestion,
        {"fan_out_research": "retrieve", "summarizer": "summarizer"},
    )
    graph.add_conditional_edges(
        "hearing_transcript",
        route_after_ingestion,
        {"fan_out_research": "retrieve", "summarizer": "summarizer"},
    )

    # retrieve (case-file RAG) and legal_research (Indian Kanoon) both feed
    # the summarizer; retrieve triggers legal_research so both complete
    # before summarization.
    graph.add_edge("retrieve", "legal_research")
    graph.add_edge("legal_research", "summarizer")

    graph.add_edge("summarizer", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
