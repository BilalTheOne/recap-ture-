"""Parse Teams transcripts (VTT, TXT, JSON, DOCX) into timestamped segments."""

import json
import re
from pathlib import Path

import webvtt


def parse_transcript(path: str) -> list[dict]:
    """Return transcript lines as [{"start", "end", "text"}, ...] (seconds)."""
    suffix = Path(path).suffix.lower()
    if suffix == ".vtt":
        return _parse_vtt(path)
    if suffix == ".json":
        return _parse_json(path)
    if suffix == ".txt":
        return _parse_txt(path)
    if suffix == ".docx":
        return _parse_docx(path)
    raise ValueError(f"Unsupported transcript format: {suffix}")


def _parse_vtt(path: str) -> list[dict]:
    segments = []
    for caption in webvtt.read(path):
        text = re.sub(r"<v[^>]*>|</v>", "", caption.text).strip()
        segments.append(
            {
                "start": caption.start_in_seconds,
                "end": caption.end_in_seconds,
                "text": text,
            }
        )
    return segments


def _parse_json(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [
        {"start": entry["start"], "end": entry["end"], "text": entry["text"]}
        for entry in data
    ]


_TXT_LINE_RE = re.compile(
    r"\[?(?P<start>\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]?\s*-->?\s*"
    r"\[?(?P<end>\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]?\s*(?P<text>.+)"
)


def _to_seconds(timestamp: str) -> float:
    hours, minutes, seconds = timestamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _parse_txt(path: str) -> list[dict]:
    segments = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            match = _TXT_LINE_RE.match(line.strip())
            if not match:
                continue
            segments.append(
                {
                    "start": _to_seconds(match["start"]),
                    "end": _to_seconds(match["end"]),
                    "text": match["text"].strip(),
                }
            )
    return segments


def _parse_docx(path: str) -> list[dict]:
    from docx import Document

    document = Document(path)
    text = "\n".join(p.text for p in document.paragraphs if p.text.strip())

    tmp_path = Path(path).with_suffix(".txt")
    tmp_path.write_text(text, encoding="utf-8")
    try:
        return _parse_txt(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)
