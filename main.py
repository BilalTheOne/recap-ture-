"""Teams meeting speaker attribution pipeline.

Usage:
    python main.py run <recording> <transcript> --speakers N [--speaker-map FILE]
        [--voiceprints-dir DIR] [--identify-threshold T] --out OUTPUT_DIR

    python main.py enroll <recording> --speakers N --speaker-map FILE
        --voiceprints-dir DIR
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

from biometrics.enroll import enroll_from_meeting
from biometrics.identify import DEFAULT_THRESHOLD, identify_clusters


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
    voiceprints_dir: str | None = None,
    identify_threshold: float = DEFAULT_THRESHOLD,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_path = str(out_dir / "meeting.wav")
    convert_to_wav(recording_path, wav_path)

    speaker_timeline = build_speaker_timeline(wav_path, n_speakers)

    if voiceprints_dir:
        speaker_timeline = identify_clusters(
            wav_path, speaker_timeline, voiceprints_dir, identify_threshold
        )
        speaker_timeline = merge_consecutive_segments(speaker_timeline)

    transcript_lines = parse_transcript(transcript_path)
    labeled_lines = assign_speakers(transcript_lines, speaker_timeline)

    speaker_map = load_speaker_map(speaker_map_path)
    final_lines = apply_speaker_map(labeled_lines, speaker_map)

    export_json(final_lines, str(out_dir / "transcript.json"))
    export_markdown(final_lines, str(out_dir / "transcript.md"))
    export_docx(final_lines, str(out_dir / "transcript.docx"))
    export_vtt(final_lines, str(out_dir / "transcript.vtt"))


def enroll_pipeline(
    recording_path: str,
    n_speakers: int,
    speaker_map_path: str,
    voiceprints_dir: str,
) -> None:
    import tempfile

    speaker_map = load_speaker_map(speaker_map_path)
    if not speaker_map:
        raise ValueError("--speaker-map is required for enrollment (need names to enroll)")

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = str(Path(tmp_dir) / "meeting.wav")
        convert_to_wav(recording_path, wav_path)

        speaker_timeline = build_speaker_timeline(wav_path, n_speakers)
        enrolled = enroll_from_meeting(wav_path, speaker_timeline, speaker_map, voiceprints_dir)

    for name, count in enrolled.items():
        print(f"Enrolled {name}: {count} segments")
    if not enrolled:
        print("No segments enrolled — check that --speaker-map labels match detected speakers.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full speaker attribution pipeline")
    run_parser.add_argument("recording", help="Path to meeting recording (mp4/wav/m4a)")
    run_parser.add_argument("transcript", help="Path to Teams transcript (vtt/txt/json/docx)")
    run_parser.add_argument("--speakers", type=int, required=True, help="Known number of speakers")
    run_parser.add_argument("--speaker-map", default=None, help="Optional speaker name mapping JSON")
    run_parser.add_argument("--out", default="output", help="Output directory")
    run_parser.add_argument(
        "--voiceprints-dir",
        default=None,
        help="Optional directory of enrolled voiceprints to identify speakers by voice biometrics",
    )
    run_parser.add_argument(
        "--identify-threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Cosine similarity threshold required to accept a biometric match",
    )

    enroll_parser = subparsers.add_parser(
        "enroll", help="Enroll speaker voiceprints from a meeting with a known speaker map"
    )
    enroll_parser.add_argument("recording", help="Path to meeting recording (mp4/wav/m4a)")
    enroll_parser.add_argument("--speakers", type=int, required=True, help="Known number of speakers")
    enroll_parser.add_argument(
        "--speaker-map", required=True, help="Speaker label to name mapping JSON (required to enroll)"
    )
    enroll_parser.add_argument(
        "--voiceprints-dir", required=True, help="Directory to store enrolled voiceprints"
    )

    args = parser.parse_args()

    if args.command == "run":
        run_pipeline(
            recording_path=args.recording,
            transcript_path=args.transcript,
            n_speakers=args.speakers,
            speaker_map_path=args.speaker_map,
            out_dir=args.out,
            voiceprints_dir=args.voiceprints_dir,
            identify_threshold=args.identify_threshold,
        )
    elif args.command == "enroll":
        enroll_pipeline(
            recording_path=args.recording,
            n_speakers=args.speakers,
            speaker_map_path=args.speaker_map,
            voiceprints_dir=args.voiceprints_dir,
        )


if __name__ == "__main__":
    main()
