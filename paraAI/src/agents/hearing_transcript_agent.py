"""
Hearing Transcript Agent.

Consumes the Whisper-transcribed (or pre-existing text) hearing chunks
already indexed by the RAG pipeline for a given doc_id, and produces
structured minutes: issues argued, each counsel's oral submissions, and the
court's oral observations. This is the agent that turns "40 minutes of
audio" into something a paralegal can skim in 30 seconds.
"""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_chat_model
from src.models.schemas import HearingMinutes
from src.rag.vector_store import VectorStore

SYSTEM_PROMPT = """You are a court reporter's assistant. Given a raw hearing
transcript (possibly with minor transcription errors from speech-to-text),
produce structured minutes. Respond with ONLY a JSON object, no prose, no
markdown fences:

{
  "issues_argued": ["..."],
  "submissions": {"Counsel/Party name": ["point 1", "point 2"], "...": ["..."]},
  "court_observations": ["oral remarks or observations made by the judge/bench"],
}

If speaker identity is unclear from the transcript, use labels like
"Counsel for Petitioner" / "Counsel for Respondent" / "Court" based on
context rather than guessing a name. Do not editorialize or predict the
outcome — record only what was said."""


async def run_hearing_transcript_agent(case_id: str, doc_id: str) -> HearingMinutes:
    store = VectorStore.get()
    chunks = sorted(
        (r for r in store._id_map if r["case_id"] == case_id and r["doc_id"] == doc_id),
        key=lambda r: r["metadata"].get("start_s", 0),
    )
    if not chunks:
        return HearingMinutes(doc_id=doc_id, raw_transcript_excerpt="No transcript indexed.")

    transcript = "\n".join(c["text"] for c in chunks)
    llm = get_chat_model()
    response = await llm.ainvoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Hearing transcript:\n\n{transcript[:14000]}"),
        ]
    )

    parsed = _safe_json(response.content)
    return HearingMinutes(
        doc_id=doc_id,
        issues_argued=parsed.get("issues_argued", []),
        submissions=parsed.get("submissions", {}),
        court_observations=parsed.get("court_observations", []),
        raw_transcript_excerpt=transcript[:500],
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
