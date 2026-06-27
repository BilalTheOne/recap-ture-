"""Core speaker-attribution pipeline steps, shared between the CLI (main.py)
and the web dashboard (webapp/app.py).

This module only contains the processing steps themselves — no interactive
prompts (input()/print()) or CLI argument handling, so it can be driven by
either a terminal session or HTTP requests.
"""

import subprocess
from pathlib import Path

from diarization.clustering import (
    DEFAULT_DISTANCE_THRESHOLD,
    cluster_speakers,
    merge_consecutive_segments,
)
from diarization.embeddings import extract_embeddings
from diarization.overlap import detect_overlaps, split_segments_by_overlap
from diarization.vad import detect_speech_segments
from export.to_docx import export_docx
from export.to_json import export_json
from export.to_markdown import export_markdown
from export.to_vtt import export_vtt
from transcript.align import assign_speakers
from transcript.parser import parse_transcript
from transcript.speaker_map import apply_speaker_map
from transcript.transcribe import transcribe_audio

from biometrics.identify import DEFAULT_THRESHOLD, identify_clusters

__all__ = [
    "DEFAULT_DISTANCE_THRESHOLD",
    "DEFAULT_THRESHOLD",
    "convert_to_wav",
    "build_speaker_timeline",
    "identify_speakers",
    "apply_cluster_renames",
    "get_transcript_lines",
    "finalize_and_export",
]


def convert_to_wav(recording_path: str, wav_path: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", recording_path, "-ar", "16000", "-ac", "1", wav_path],
        check=True,
    )


def build_speaker_timeline(
    wav_path: str,
    n_speakers: int | None,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> list[dict]:
    speech_segments = detect_speech_segments(wav_path)
    overlaps = detect_overlaps(wav_path)
    clean_segments, overlapping_segments = split_segments_by_overlap(
        speech_segments, overlaps
    )

    embeddings = extract_embeddings(wav_path, clean_segments)
    clustered = cluster_speakers(clean_segments, embeddings, n_speakers, distance_threshold)

    multiple = [
        {"start": s["start"], "end": s["end"], "speaker": "Speaker_multiple"}
        for s in overlapping_segments
    ]

    timeline = sorted(clustered + multiple, key=lambda s: s["start"])
    return merge_consecutive_segments(timeline)


def identify_speakers(
    wav_path: str,
    speaker_timeline: list[dict],
    voiceprints_dir: str,
    identify_threshold: float = DEFAULT_THRESHOLD,
) -> tuple[list[dict], dict[str, dict]]:
    """Match clusters against enrolled voiceprints.

    Returns (relabeled_timeline, cluster_info) — see
    biometrics.identify.identify_clusters for the cluster_info shape.
    """
    return identify_clusters(wav_path, speaker_timeline, voiceprints_dir, identify_threshold)


def apply_cluster_renames(speaker_timeline: list[dict], renames: dict[str, str]) -> list[dict]:
    """Apply a {cluster_label: name} mapping and re-merge consecutive segments."""
    renamed = [
        {**segment, "speaker": renames.get(segment["speaker"], segment["speaker"])}
        for segment in speaker_timeline
    ]
    return merge_consecutive_segments(renamed)


def get_transcript_lines(
    wav_path: str, transcript_path: str | None, whisper_model: str = "base"
) -> list[dict]:
    if transcript_path:
        return parse_transcript(transcript_path)
    return transcribe_audio(wav_path, whisper_model)


def finalize_and_export(
    transcript_lines: list[dict],
    speaker_timeline: list[dict],
    speaker_map: dict[str, str] | None,
    out_dir: str,
) -> list[dict]:
    labeled_lines = assign_speakers(transcript_lines, speaker_timeline)
    final_lines = apply_speaker_map(labeled_lines, speaker_map or {})

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    export_json(final_lines, str(out_path / "transcript.json"))
    export_markdown(final_lines, str(out_path / "transcript.md"))
    export_docx(final_lines, str(out_path / "transcript.docx"))
    export_vtt(final_lines, str(out_path / "transcript.vtt"))

    return final_lines
