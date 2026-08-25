"""
Streaming ingestion using Pathway.

Why Pathway instead of a batch "re-embed everything on a cron job" script:
court case folders grow continuously (new filings, a hearing recording
landing right after the session ends) and paralegals need the answer to
"what did the judge say about limitation today" to be queryable within
seconds of the hearing ending, not after a nightly batch job.

Pathway gives us an incremental dataflow: `pw.io.fs.read(..., mode="streaming")`
watches `settings.watch_dir` for new/changed files, and only the delta is
pushed through parsing -> chunking -> embedding -> index upsert. Existing
rows are untouched.

This module exposes:
  - build_pipeline()  constructs the Pathway graph (call once at startup)
  - run_forever()     blocks, running the Pathway runtime (call in a
                       background thread/process, NOT in the request path)
  - ingest_file_sync() a synchronous helper the FastAPI /ingest route can
                       call directly for the "upload now, index now" path,
                       so users aren't stuck waiting on filesystem-watch
                       latency for their own upload.
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pathway as pw

from loguru import logger

from src.config import get_settings
from src.rag.multimodal_loader import load_any
from src.rag.vector_store import VectorStore


class IncomingFileSchema(pw.Schema):
    path: str
    modified_at: float


def build_pipeline() -> pw.Table:
    """
    Defines the Pathway dataflow graph. Filenames are expected to encode the
    case id as a prefix, e.g. `data/incoming/CASE-001__petition.pdf`, so the
    watched-folder ingestion path can route each file to the right case
    without a separate control channel. (The interactive /ingest endpoint
    passes case_id explicitly instead — see ingest_file_sync.)
    """
    settings = get_settings()

    files = pw.io.fs.read(
        settings.watch_dir,
        format="binary",
        mode="streaming",
        with_metadata=True,
    )

    def _process(key, row, time, is_addition):
        if not is_addition:
            return
        path = row["path"]
        case_id = _case_id_from_filename(path)
        try:
            chunks = load_any(path)
            n = VectorStore.get().upsert(chunks, case_id=case_id)
            logger.info(f"[pathway] indexed {n} chunks from {path} for case {case_id}")
        except Exception as e:  # noqa: BLE001 - streaming pipeline must not crash on one bad file
            logger.error(f"[pathway] failed to process {path}: {e}")

    pw.io.subscribe(files, on_change=_process)
    return files


def _case_id_from_filename(path: str) -> str:
    name = Path(path).name
    if "__" in name:
        return name.split("__", 1)[0]
    return "UNASSIGNED"


def run_forever() -> None:
    """Entry point for a background worker process: `python -m src.rag.pathway_pipeline`."""
    build_pipeline()
    logger.info("Pathway streaming pipeline started, watching for new case files...")
    pw.run()


def ingest_file_sync(uploaded_path: str, case_id: str, doc_id: str | None = None) -> dict:
    """
    Synchronous ingestion path used by the FastAPI /ingest endpoint for an
    interactive upload — we don't want a user's upload to wait on the
    filesystem watcher's poll interval. We still copy the file into
    watch_dir (with the case-id-prefixed name) so it's also picked up by
    the streaming pipeline for consistency / reprocessing if the pipeline
    restarts.
    """
    settings = get_settings()
    doc_id = doc_id or str(uuid.uuid4())

    dest_dir = Path(settings.watch_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_name = f"{case_id}__{doc_id}__{Path(uploaded_path).name}"
    dest_path = dest_dir / dest_name
    shutil.copy(uploaded_path, dest_path)

    chunks = load_any(str(dest_path), doc_id=doc_id)
    n_indexed = VectorStore.get().upsert(chunks, case_id=case_id)

    modality = chunks[0].modality if chunks else "unknown"
    return {"doc_id": doc_id, "modality": modality, "chunks_indexed": n_indexed}


if __name__ == "__main__":
    run_forever()
