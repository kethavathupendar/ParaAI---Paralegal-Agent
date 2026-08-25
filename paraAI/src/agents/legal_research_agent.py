"""
Legal Research Agent.

Given a question (or statutes/issues surfaced by the Document Analyzer),
queries Indian Kanoon for relevant judgments, pulls short relevance
fragments, and asks the LLM to rank/explain relevance — rather than just
dumping raw search results, which is what makes this an "agent" rather than
a search box.
"""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_chat_model
from src.integrations.indian_kanoon_client import IndianKanoonClient
from src.models.schemas import LegalResearchResult, PrecedentResult

SYSTEM_PROMPT = """You are a legal research assistant. You are given a legal
question and a list of candidate precedents (title, court, date, and a short
snippet) retrieved from Indian Kanoon. Select and order the precedents that
are actually relevant to the question, and write 1-2 sentences per selected
precedent explaining why it's relevant. Respond with ONLY a JSON object:

{
  "selected_ik_doc_ids": ["...", "..."],
  "notes": {"<ik_doc_id>": "why this is relevant to the question"},
  "summary": "2-4 sentence overview of what the case law says on this question"
}

If none of the candidates are actually relevant, return empty lists/objects
and say so plainly in "summary". Never invent a citation that wasn't in the
candidate list."""


async def run_legal_research_agent(question: str, top_n_candidates: int = 8) -> LegalResearchResult:
    client = IndianKanoonClient()
    try:
        candidates = await client.search(question, page_num=0)
    except Exception as e:  # noqa: BLE001 — degrade gracefully, don't fail the whole graph
        return LegalResearchResult(
            query=question,
            precedents=[],
            notes=f"Indian Kanoon lookup unavailable: {e}",
        )

    candidates = candidates[:top_n_candidates]
    if not candidates:
        return LegalResearchResult(query=question, precedents=[], notes="No candidates found.")

    candidate_json = [c.model_dump() for c in candidates]
    llm = get_chat_model()
    response = await llm.ainvoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Question: {question}\n\n"
                    f"Candidates:\n{json.dumps(candidate_json, indent=2)}"
                )
            ),
        ]
    )

    parsed = _safe_json(response.content)
    selected_ids = set(parsed.get("selected_ik_doc_ids", []))
    notes_by_id: dict[str, str] = parsed.get("notes", {})

    selected: list[PrecedentResult] = []
    for c in candidates:
        if c.ik_doc_id in selected_ids:
            enriched = c.model_copy()
            note = notes_by_id.get(c.ik_doc_id)
            if note:
                enriched.relevance_snippet = f"{enriched.relevance_snippet} — {note}"
            selected.append(enriched)

    return LegalResearchResult(
        query=question,
        precedents=selected,
        notes=parsed.get("summary", ""),
    )


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
