"""Teams meeting speaker attribution pipeline.

Usage:
    python main.py <recording> <transcript> --speakers N [--speaker-map FILE] --out OUTPUT_DIR
"""

import argparse
import subprocess
from pathlib import Path

from diarization.clustering import cluster_speakers, merge_consecutive_segments
from diarization.embeddings import extract_embeddings
from diarization.overlap import detect_overlaps, split_segments_by_overlap
from diarization.vad import detect_speech_segments
from export.to_docx import export_docx
from export.to_json import export_json
from export.to_markdown import export_markdown
from export.to_vtt import export_vtt
from transcript.align import assign_speakers
from transcript.parser import parse_transcript
from transcript.speaker_map import apply_speaker_map, load_speaker_map


def convert_to_wav(recording_path: str, wav_path: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", recording_path, "-ar", "16000", "-ac", "1", wav_path],
        check=True,
    )


def build_speaker_timeline(wav_path: str, n_speakers: int) -> list[dict]:
    speech_segments = detect_speech_segments(wav_path)
    overlaps = detect_overlaps(wav_path)
    clean_segments, overlapping_segments = split_segments_by_overlap(
        speech_segments, overlaps
    )

    embeddings = extract_embeddings(wav_path, clean_segments)
    clustered = cluster_speakers(clean_segments, embeddings, n_speakers)

    multiple = [
        {"start": s["start"], "end": s["end"], "speaker": "Speaker_multiple"}
        for s in overlapping_segments
    ]

    timeline = sorted(clustered + multiple, key=lambda s: s["start"])
    return merge_consecutive_segments(timeline)


def run_pipeline(
    recording_path: str,
    transcript_path: str,
    n_speakers: int,
    speaker_map_path: str | None,
    out_dir: str,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_path = str(out_dir / "meeting.wav")
    convert_to_wav(recording_path, wav_path)

    speaker_timeline = build_speaker_timeline(wav_path, n_speakers)

    transcript_lines = parse_transcript(transcript_path)
    labeled_lines = assign_speakers(transcript_lines, speaker_timeline)

    speaker_map = load_speaker_map(speaker_map_path)
    final_lines = apply_speaker_map(labeled_lines, speaker_map)

    export_json(final_lines, str(out_dir / "transcript.json"))
    export_markdown(final_lines, str(out_dir / "transcript.md"))
    export_docx(final_lines, str(out_dir / "transcript.docx"))
    export_vtt(final_lines, str(out_dir / "transcript.vtt"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", help="Path to meeting recording (mp4/wav/m4a)")
    parser.add_argument("transcript", help="Path to Teams transcript (vtt/txt/json/docx)")
    parser.add_argument("--speakers", type=int, required=True, help="Known number of speakers")
    parser.add_argument("--speaker-map", default=None, help="Optional speaker name mapping JSON")
    parser.add_argument("--out", default="output", help="Output directory")
    args = parser.parse_args()

    run_pipeline(
        recording_path=args.recording,
        transcript_path=args.transcript,
        n_speakers=args.speakers,
        speaker_map_path=args.speaker_map,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
