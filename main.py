"""Teams meeting speaker attribution pipeline.

Usage:
    python main.py run <recording> [<transcript>] [--speakers N] [--speaker-map FILE]
        [--voiceprints-dir DIR] [--identify-threshold T] [--interactive-enroll]
        [--whisper-model SIZE] [--distance-threshold T] --out OUTPUT_DIR

    python main.py enroll <recording> --speakers N --speaker-map FILE
        --voiceprints-dir DIR

If <transcript> is omitted, one is generated automatically from the audio
using Whisper. If --speakers is omitted, the speaker count is estimated via
distance-threshold clustering instead of a known fixed count.

With --voiceprints-dir and --interactive-enroll, any speaker cluster that
doesn't match an existing voiceprint is presented for naming on the spot;
naming them enrolls their voice so future recordings recognize them
automatically without needing a --speaker-map.
"""

import argparse
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
from transcript.speaker_map import apply_speaker_map, load_speaker_map
from transcript.transcribe import transcribe_audio

from biometrics.enroll import enroll_from_meeting
from biometrics.identify import DEFAULT_THRESHOLD, identify_clusters
from biometrics.store import add_embeddings


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


def run_pipeline(
    recording_path: str,
    transcript_path: str | None,
    n_speakers: int | None,
    speaker_map_path: str | None,
    out_dir: str,
    voiceprints_dir: str | None = None,
    identify_threshold: float = DEFAULT_THRESHOLD,
    whisper_model: str = "base",
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    interactive_enroll: bool = False,
) -> None:
    if n_speakers is None:
        print(
            "WARNING: --speakers not provided. Estimating speaker count via "
            "distance-threshold clustering, which has been observed to produce "
            "wildly wrong speaker counts. Prefer passing --speakers explicitly."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_path = str(out_dir / "meeting.wav")
    convert_to_wav(recording_path, wav_path)

    speaker_timeline = build_speaker_timeline(wav_path, n_speakers, distance_threshold)

    if voiceprints_dir:
        speaker_timeline, cluster_info = identify_clusters(
            wav_path, speaker_timeline, voiceprints_dir, identify_threshold
        )

        if interactive_enroll:
            renames = {}
            for cluster, info in cluster_info.items():
                if info["name"] is not None:
                    continue
                print(
                    f"New speaker detected ({cluster}, {len(info['embeddings'])} "
                    f"segments, best match score {info['score']:.2f})"
                )
                name = input(
                    "Enter a name to enroll this speaker, or press Enter to skip: "
                ).strip()
                if name:
                    add_embeddings(voiceprints_dir, name, info["embeddings"])
                    renames[cluster] = name

            if renames:
                speaker_timeline = [
                    {**s, "speaker": renames.get(s["speaker"], s["speaker"])}
                    for s in speaker_timeline
                ]

        speaker_timeline = merge_consecutive_segments(speaker_timeline)

    if transcript_path:
        transcript_lines = parse_transcript(transcript_path)
    else:
        transcript_lines = transcribe_audio(wav_path, whisper_model)

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
    run_parser.add_argument(
        "transcript",
        nargs="?",
        default=None,
        help="Path to existing transcript (vtt/txt/json/docx). If omitted, one is "
        "generated automatically from the audio using Whisper.",
    )
    run_parser.add_argument(
        "--speakers",
        type=int,
        default=None,
        help="Known number of speakers (recommended — gives stable, accurate "
        "clustering). If omitted, speaker count is estimated via "
        "distance-threshold clustering: EXPERIMENTAL and prone to wildly "
        "wrong speaker counts, validated to be unreliable in testing.",
    )
    run_parser.add_argument(
        "--distance-threshold",
        type=float,
        default=DEFAULT_DISTANCE_THRESHOLD,
        help="Cosine distance threshold used to estimate speaker count when "
        "--speakers is omitted (lower = more, smaller clusters). No single "
        "value has been found to work reliably across recordings.",
    )
    run_parser.add_argument(
        "--whisper-model",
        default="base",
        help="faster-whisper model size to use when auto-generating a transcript "
        "(tiny/base/small/medium/large-v3)",
    )
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
    run_parser.add_argument(
        "--interactive-enroll",
        action="store_true",
        help="When used with --voiceprints-dir, prompt to name and enroll any "
        "speaker cluster that doesn't match an existing voiceprint, so they're "
        "recognized automatically in future recordings",
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
            whisper_model=args.whisper_model,
            distance_threshold=args.distance_threshold,
            interactive_enroll=args.interactive_enroll,
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
