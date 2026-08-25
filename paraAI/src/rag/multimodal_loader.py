"""
Normalizes heterogeneous inputs (PDF filings, DOCX orders, scanned image
exhibits, audio/video hearing recordings) into a common list of text chunks
with metadata, so the rest of the pipeline never has to think about file
format again.

Each loader function returns list[Chunk]; `load_any` dispatches by extension.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config import get_settings


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    modality: str                      # "pdf" | "docx" | "image" | "audio" | "video" | "text"
    metadata: dict[str, Any] = field(default_factory=dict)


def _splitter() -> RecursiveCharacterTextSplitter:
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def _chunk_text(text: str, doc_id: str, modality: str, extra_meta: dict[str, Any]) -> list[Chunk]:
    pieces = _splitter().split_text(text)
    return [
        Chunk(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            text=piece,
            modality=modality,
            metadata={**extra_meta, "chunk_index": i},
        )
        for i, piece in enumerate(pieces)
    ]


def load_pdf(path: str, doc_id: str) -> list[Chunk]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    all_chunks: list[Chunk] = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        all_chunks.extend(
            _chunk_text(text, doc_id, "pdf", {"page": page_num + 1, "source_path": path})
        )
    return all_chunks


def load_docx(path: str, doc_id: str) -> list[Chunk]:
    import docx

    document = docx.Document(path)
    text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    return _chunk_text(text, doc_id, "docx", {"source_path": path})


def load_image(path: str, doc_id: str) -> list[Chunk]:
    """OCR for scanned exhibits / photographed documents."""
    import pytesseract
    from PIL import Image

    text = pytesseract.image_to_string(Image.open(path))
    return _chunk_text(text, doc_id, "image", {"source_path": path})


def load_audio_or_video(path: str, doc_id: str) -> list[Chunk]:
    """
    Transcribes a hearing recording. Uses Whisper locally so no audio ever
    leaves the environment (relevant for privileged court proceedings).
    Video files are demuxed to audio first via ffmpeg.
    """
    import whisper

    audio_path = path
    if Path(path).suffix.lower() in {".mp4", ".mov", ".mkv"}:
        audio_path = _extract_audio(path)

    model = whisper.load_model("base")
    result = model.transcribe(audio_path, verbose=False)

    all_chunks: list[Chunk] = []
    for segment in result.get("segments", []):
        all_chunks.append(
            Chunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                text=segment["text"].strip(),
                modality="hearing_audio",
                metadata={
                    "start_s": segment["start"],
                    "end_s": segment["end"],
                    "source_path": path,
                },
            )
        )
    return all_chunks


def _extract_audio(video_path: str) -> str:
    import ffmpeg

    out_path = str(Path(video_path).with_suffix(".wav"))
    (
        ffmpeg.input(video_path)
        .output(out_path, ac=1, ar=16000)
        .overwrite_output()
        .run(quiet=True)
    )
    return out_path


def load_plain_text(path: str, doc_id: str) -> list[Chunk]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return _chunk_text(text, doc_id, "text", {"source_path": path})


_LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".png": load_image,
    ".jpg": load_image,
    ".jpeg": load_image,
    ".mp3": load_audio_or_video,
    ".wav": load_audio_or_video,
    ".m4a": load_audio_or_video,
    ".mp4": load_audio_or_video,
    ".mov": load_audio_or_video,
    ".mkv": load_audio_or_video,
    ".txt": load_plain_text,
}


def load_any(path: str, doc_id: str | None = None) -> list[Chunk]:
    doc_id = doc_id or str(uuid.uuid4())
    ext = Path(path).suffix.lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader(path, doc_id)
