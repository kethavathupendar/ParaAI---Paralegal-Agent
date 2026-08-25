# paraAI — Agentic Paralegal Assistant

paraAI autonomously analyzes and summarizes legal documents and court hearing
transcripts. It is built as a graph of cooperating agents (LangGraph +
LangChain) sitting on top of a **streaming, multimodal RAG pipeline** (Pathway)
and **Indian Kanoon's** case-law API, so it can answer questions that require
both the documents a user uploads *and* live retrieval of relevant precedent.

```
                         ┌────────────────────────────┐
                         │        FastAPI Gateway       │
                         │   /ingest  /ask  /ws/stream   │
                         └──────────────┬───────────────┘
                                        │
                              ┌─────────▼──────────┐
                              │   LangGraph Router   │
                              │  (orchestrator.py)   │
                              └───┬───────┬───────┬──┘
                     ┌────────────┘       │       └─────────────┐
             ┌───────▼───────┐   ┌────────▼────────┐   ┌────────▼────────┐
             │  Document      │   │   Summarizer     │   │  Legal Research  │
             │  Analyzer      │   │   Agent          │   │  Agent (IK API)  │
             │  Agent         │   │                  │   │                  │
             └───────┬────────┘   └────────┬─────────┘   └────────┬─────────┘
                     │                     │                     │
                     └──────────┬──────────┴──────────┬──────────┘
                                │                     │
                       ┌────────▼────────┐   ┌────────▼─────────┐
                       │ Hearing Transcript│   │  Pathway RAG      │
                       │ Agent (audio/video│   │  Pipeline         │
                       │ → text → summary) │   │ (live vector index)│
                       └───────────────────┘   └────────────────────┘
```

## Why these pieces

| Concern | Tool | Reason |
|---|---|---|
| Multi-agent orchestration | **LangGraph** | Explicit, inspectable state machine instead of an implicit agent loop — each legal task (extract → summarize → cross-check precedent) is a node with defined edges and can be retried/branched. |
| Model access | **LangChain** `Runnable` interfaces | Keeps the graph model-agnostic — swap Claude / GPT-4 / a local model per-agent via `config.py` without touching graph logic. |
| Retrieval | **Pathway** | Documents and hearing transcripts arrive continuously (new filings, new hearing days). Pathway's incremental dataflow re-embeds and re-indexes only what changed, instead of a batch re-index. |
| Case law / statutes | **Indian Kanoon API** | Programmatic, structured access to judgments, citations and headnotes instead of scraping. |

## Repository layout

```
paraAI/
├── src/
│   ├── main.py                     FastAPI app: /ingest, /ask, /ws/stream
│   ├── config.py                   Env-driven settings, model registry
│   ├── agents/
│   │   ├── orchestrator.py         LangGraph StateGraph wiring all agents
│   │   ├── document_analyzer_agent.py   Clause/entity/date extraction
│   │   ├── summarizer_agent.py          Structured legal summarization
│   │   ├── legal_research_agent.py      Indian Kanoon retrieval + citation check
│   │   └── hearing_transcript_agent.py  Audio/video hearing → structured minutes
│   ├── rag/
│   │   ├── pathway_pipeline.py     Streaming ingestion + incremental vector index
│   │   ├── vector_store.py         Embedding + similarity search wrapper
│   │   └── multimodal_loader.py    PDF / DOCX / audio / video → normalized chunks
│   ├── integrations/
│   │   └── indian_kanoon_client.py Thin typed client for Indian Kanoon API
│   ├── models/
│   │   └── schemas.py              Pydantic request/response/state models
│   └── utils/
│       └── logging_config.py
├── frontend/
│   └── index.html                  Minimal chat + upload UI (no build step)
├── tests/
│   └── test_agents.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Agent responsibilities

1. **Document Analyzer Agent** — takes freshly ingested filings (petitions,
   affidavits, orders) and extracts parties, dates, statutes cited, and
   clause-level structure into a typed JSON schema.
2. **Hearing Transcript Agent** — takes an audio/video hearing recording (or
   an existing transcript), diarizes/cleans it, and produces structured
   minutes: issues argued, oral submissions per counsel, court's oral
   observations.
3. **Legal Research Agent** — given a legal question or a statute/clause
   found by the Document Analyzer, queries the **Indian Kanoon client** for
   relevant judgments, pulls headnotes, and checks whether cited case law is
   still good law (not overruled).
4. **Summarizer Agent** — the terminal node. Merges outputs from the other
   three agents plus retrieved RAG context into a single structured brief
   (facts, issues, arguments, precedent, recommendation-flagging — **not**
   legal advice).

The **orchestrator** is a `langgraph.graph.StateGraph` with conditional
routing: a router node inspects the incoming request (`document`, `hearing`,
`question`) and only invokes the agents needed, then always funnels into the
Summarizer node before returning.

## RAG pipeline (Pathway)

`rag/pathway_pipeline.py` defines a Pathway dataflow that:

1. Watches an input directory / object-store prefix for new files
   (`pw.io.fs.read` in streaming mode, or `pw.io.s3` in production).
2. Runs each new/changed file through `multimodal_loader.py` to normalize
   PDFs, DOCX, images (OCR), and audio/video (transcription) into text
   chunks with metadata (`doc_id`, `page`, `modality`, `timestamp`).
3. Embeds chunks incrementally and upserts them into the vector index
   (`vector_store.py`), so a hearing that finishes at 4:05 PM is queryable
   at 4:06 PM without a full re-index.
4. Exposes a `query()` method the agents call for top-k retrieval, filtered
   by `doc_id`/`case_id` so cross-case leakage can't happen.

## Indian Kanoon integration

`integrations/indian_kanoon_client.py` wraps the three endpoints paraAI uses:

- `search(query, doctypes, pagenum)` — full-text search over judgments/acts.
- `get_doc(doc_id)` — full judgment text + metadata (court, bench, date, citations).
- `get_doc_fragment(doc_id, query)` — highlighted fragment matching a query,
  used for quick relevance checks before pulling the full document.

All calls are authenticated with an API token (`IK_API_TOKEN` in `.env`) and
wrapped with retry/backoff since paralegal workflows are latency-tolerant but
not failure-tolerant.

## Running it

```bash
git clone <this-repo> && cd paraAI
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY / OPENAI_API_KEY, IK_API_TOKEN
uvicorn src.main:app --reload --port 8000
```

Open `frontend/index.html` directly in a browser (it talks to
`http://localhost:8000`), or `POST` to the API directly:

```bash
# Ingest a filing
curl -F "file=@petition.pdf" -F "case_id=CASE-001" http://localhost:8000/ingest

# Ask a question (routes through the LangGraph orchestrator)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"case_id":"CASE-001","question":"Summarize the key issues and cite any relevant precedent on limitation."}'
```

Or with Docker:

```bash
docker compose up --build
```

## Notes on scope / honesty

This scaffold is a real, runnable architecture: the LangGraph wiring, the
FastAPI routes, the Pathway pipeline shape, and the Indian Kanoon client are
all functional Python. Two things you must supply to run it end-to-end in
production:

- API credentials (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`, `IK_API_TOKEN`).
- A production vector DB connection if you don't want the default in-memory
  FAISS store (swap in `rag/vector_store.py`; Qdrant/Pinecone/pgvector are
  all drop-in since it's a thin wrapper).

paraAI does not give legal advice; the Summarizer Agent's output is always
framed as an informational brief for a supervising paralegal/attorney to
review, and citations are always traceable back to the Indian Kanoon
document ID they came from.
