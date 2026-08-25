"""
FastAPI gateway for paraAI.

Endpoints:
  POST /ingest        upload a document/hearing file for a case, index it now
  POST /ask            ask a question about a case; runs the LangGraph
  GET  /health          liveness check
  WS   /ws/stream/{case_id}   streaming variant of /ask for chat-style UIs

Kept thin on purpose: this layer does I/O and delegates all reasoning to
the orchestrator graph.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, Form
from fastapi.middleware.cors import CORSMiddleware

from src.agents.orchestrator import get_graph
from src.config import get_settings
from src.models.schemas import AskRequest, FinalBrief, GraphState, IngestResponse, RequestKind
from src.rag.pathway_pipeline import ingest_file_sync
from src.utils.logging_config import configure_logging

configure_logging()

app = FastAPI(title="paraAI — Agentic Paralegal Assistant", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    case_id: str = Form(...),
    kind: str = Form("document"),  # "document" | "hearing"
):
    settings = get_settings()

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb}MB limit")

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = ingest_file_sync(tmp_path, case_id=case_id)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Run the appropriate extraction agent immediately so the doc is
    # analyzed (not just embedded) right after upload.
    graph = get_graph()
    state = GraphState(
        case_id=case_id,
        kind=RequestKind.HEARING if kind == "hearing" else RequestKind.DOCUMENT,
        raw_input_path=f"{case_id}__{result['doc_id']}__{file.filename}",
        modality=result["modality"],
    )
    await graph.ainvoke(state)

    return IngestResponse(
        case_id=case_id,
        doc_id=result["doc_id"],
        modality=result["modality"],
        chunks_indexed=result["chunks_indexed"],
        message="Ingested and analyzed successfully.",
    )


@app.post("/ask", response_model=FinalBrief)
async def ask(req: AskRequest):
    graph = get_graph()
    state = GraphState(case_id=req.case_id, kind=RequestKind.QUESTION, question=req.question)
    result_state = await graph.ainvoke(state)

    final_brief = result_state.get("final_brief") if isinstance(result_state, dict) else result_state.final_brief
    if final_brief is None:
        raise HTTPException(500, "Graph completed without producing a brief.")
    return final_brief


@app.websocket("/ws/stream/{case_id}")
async def ws_stream(websocket: WebSocket, case_id: str):
    """
    Streaming chat interface: client sends {"question": "..."} messages,
    server runs the graph and streams back node-by-node progress followed
    by the final brief. Useful for a chat UI that wants to show "researching
    precedent..." rather than a blank spinner.
    """
    await websocket.accept()
    graph = get_graph()
    try:
        while True:
            payload = await websocket.receive_json()
            question = payload.get("question", "")
            state = GraphState(case_id=case_id, kind=RequestKind.QUESTION, question=question)

            async for event in graph.astream(state):
                for node_name, node_output in event.items():
                    await websocket.send_json({"type": "progress", "node": node_name})

            final_state = await graph.ainvoke(state)
            final_brief = (
                final_state.get("final_brief")
                if isinstance(final_state, dict)
                else final_state.final_brief
            )
            await websocket.send_json(
                {"type": "final", "brief": final_brief.model_dump() if final_brief else None}
            )
    except WebSocketDisconnect:
        pass
