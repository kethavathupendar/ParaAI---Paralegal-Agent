"""
Document Analyzer Agent.

Given a newly ingested filing (petition, affidavit, order, notice), extracts
parties, dates, statutes cited, and a clause-level summary into
`DocumentAnalysis`. Runs the retrieved chunks for that doc through the LLM
with a strict JSON-output instruction, then validates against the Pydantic
schema so downstream agents get a reliable shape instead of free text.
"""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_chat_model
from src.models.schemas import DocumentAnalysis, ExtractedEntities
from src.rag.vector_store import VectorStore

SYSTEM_PROMPT = """You are a meticulous legal document analyst working for a paralegal team.
Given excerpts from a legal filing, extract structured information.
Respond with ONLY a JSON object matching this schema, no prose, no markdown fences:

{
  "parties": ["..."],
  "dates": ["..."],
  "statutes_cited": ["..."],
  "court": "... or null",
  "case_number": "... or null",
  "clause_summary": ["one bullet per major clause/section, plain English"],
  "raw_notes": "any ambiguity or missing info the reviewing paralegal should double check"
}

Be conservative: if something isn't clearly stated in the text, leave it out
rather than guessing. Do not provide legal advice or conclusions about
outcomes — only extraction and summarization."""


async def run_document_analyzer(case_id: str, doc_id: str) -> DocumentAnalysis:
    store = VectorStore.get()
    # Pull every chunk belonging to this doc (case-scoped) for a full read,
    # rather than a similarity query, since we want the whole document.
    chunks = [r for r in _all_chunks_for_doc(store, case_id, doc_id)]
    if not chunks:
        return DocumentAnalysis(
            doc_id=doc_id,
            entities=ExtractedEntities(),
            raw_notes="No indexed content found for this document.",
        )

    text = "\n\n".join(c["text"] for c in chunks)
    llm = get_chat_model()
    response = await llm.ainvoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Document excerpts:\n\n{text[:12000]}"),
        ]
    )

    parsed = _safe_json(response.content)
    entities = ExtractedEntities(
        parties=parsed.get("parties", []),
        dates=parsed.get("dates", []),
        statutes_cited=parsed.get("statutes_cited", []),
        court=parsed.get("court"),
        case_number=parsed.get("case_number"),
    )
    return DocumentAnalysis(
        doc_id=doc_id,
        entities=entities,
        clause_summary=parsed.get("clause_summary", []),
        raw_notes=parsed.get("raw_notes", ""),
    )


def _all_chunks_for_doc(store: VectorStore, case_id: str, doc_id: str) -> list[dict]:
    return [
        record
        for record in store._id_map  # internal access is fine within the agents layer
        if record["case_id"] == case_id and record["doc_id"] == doc_id
    ]


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
