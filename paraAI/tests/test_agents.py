"""
Lightweight tests that don't require live API keys — they check that the
graph compiles, routing logic picks the right entry node, and schemas
validate. Full agent behavior (LLM calls, Indian Kanoon calls) is exercised
via integration tests that require .env credentials and network access, and
are marked accordingly so `pytest -m "not integration"` skips them in CI
sandboxes without egress.
"""
import pytest

from src.agents.orchestrator import build_graph, route_entry
from src.models.schemas import GraphState, RequestKind


def test_graph_compiles():
    graph = build_graph()
    assert graph is not None


@pytest.mark.parametrize(
    "kind,expected",
    [
        (RequestKind.DOCUMENT, "document_analyzer"),
        (RequestKind.HEARING, "hearing_transcript"),
        (RequestKind.QUESTION, "fan_out_research"),
    ],
)
def test_route_entry(kind, expected):
    state = GraphState(case_id="CASE-TEST", kind=kind, question="q" if kind == RequestKind.QUESTION else None)
    assert route_entry(state) == expected


def test_graph_state_defaults():
    state = GraphState(case_id="CASE-TEST", kind=RequestKind.QUESTION, question="What is the limitation period?")
    assert state.retrieved_context == []
    assert state.document_analysis is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ask_end_to_end():
    """Requires ANTHROPIC_API_KEY/OPENAI_API_KEY and IK_API_TOKEN set in .env."""
    from src.agents.orchestrator import get_graph

    graph = get_graph()
    state = GraphState(
        case_id="CASE-DEMO",
        kind=RequestKind.QUESTION,
        question="What is the limitation period for filing a civil suit in India?",
    )
    result = await graph.ainvoke(state)
    final_brief = result.get("final_brief") if isinstance(result, dict) else result.final_brief
    assert final_brief is not None
    assert final_brief.summary
