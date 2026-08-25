"""
Summarizer Agent — the terminal node in the LangGraph.

Merges whatever the upstream agents produced (document analysis, hearing
minutes, legal research, raw RAG context) plus the user's original question
into one coherent, structured `FinalBrief`. This is deliberately the only
agent that talks to the end user in prose; every other agent produces
structured data.
"""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_chat_model
from src.models.schemas import Citation, FinalBrief, GraphState

SYSTEM_PROMPT = """You are a senior paralegal drafting an internal case brief
for the attorney of record. You are given: the user's question, any
extracted document analysis, any hearing minutes, any legal research
findings, and raw retrieved context snippets. Synthesize ALL of this into a
clear, well-organized brief.

Respond with ONLY a JSON object:

{
  "summary": "the main narrative answer/brief, several paragraphs, plain professional English",
  "key_issues": ["bullet list of the key legal issues in play"],
  "citation_doc_ids": ["doc_id or ik_doc_id values you actually relied on, from the provided context only"]
}

Rules:
- Only state facts that are supported by the provided context. If the
  context doesn't answer part of the question, say so explicitly rather
  than filling the gap with general knowledge.
- Never state a predicted case outcome or give legal advice — describe the
  facts, issues, and relevant precedent, and flag anything the attorney
  should weigh in on.
- Keep citation_doc_ids to only the ids you actually used."""


async def run_summarizer(state: GraphState) -> FinalBrief:
    context_blob = _build_context_blob(state)

    llm = get_chat_model()
    response = await llm.ainvoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"User question / task: {state.question or '(document/hearing ingestion — summarize what was processed)'}\n\n"
                    f"{context_blob}"
                )
            ),
        ]
    )

    parsed = _safe_json(response.content)
    cited_ids = set(parsed.get("citation_doc_ids", []))

    citations: list[Citation] = []
    if state.document_analysis and state.document_analysis.doc_id in cited_ids:
        citations.append(Citation(doc_id=state.document_analysis.doc_id, source="case_file"))
    if state.hearing_minutes and state.hearing_minutes.doc_id in cited_ids:
        citations.append(Citation(doc_id=state.hearing_minutes.doc_id, source="hearing_transcript"))
    if state.research:
        for p in state.research.precedents:
            if p.ik_doc_id in cited_ids:
                citations.append(
                    Citation(
                        doc_id=p.ik_doc_id,
                        source="indian_kanoon",
                        title=p.title,
                        snippet=p.relevance_snippet,
                        url=p.url,
                    )
                )
    for r in state.retrieved_context:
        if r.get("doc_id") in cited_ids:
            citations.append(
                Citation(
                    doc_id=r["doc_id"],
                    source="case_file",
                    snippet=r.get("text", "")[:200],
                )
            )

    return FinalBrief(
        case_id=state.case_id,
        summary=parsed.get("summary", response.content),
        key_issues=parsed.get("key_issues", []),
        supporting_precedent=state.research.precedents if state.research else [],
        citations=citations,
    )


def _build_context_blob(state: GraphState) -> str:
    parts = []
    if state.document_analysis:
        parts.append(
            "DOCUMENT ANALYSIS:\n" + state.document_analysis.model_dump_json(indent=2)
        )
    if state.hearing_minutes:
        parts.append("HEARING MINUTES:\n" + state.hearing_minutes.model_dump_json(indent=2))
    if state.research:
        parts.append("LEGAL RESEARCH:\n" + state.research.model_dump_json(indent=2))
    if state.retrieved_context:
        snippets = "\n---\n".join(
            f"[{r.get('doc_id')}] {r.get('text', '')[:600]}" for r in state.retrieved_context
        )
        parts.append("RETRIEVED CASE-FILE CONTEXT:\n" + snippets)
    return "\n\n".join(parts) if parts else "(no additional context available)"


def _safe_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}
