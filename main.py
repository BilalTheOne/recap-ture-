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

For a browser-based alternative to this CLI, see webapp/app.py.
"""

import argparse
import tempfile
from pathlib import Path

import pipeline
from transcript.speaker_map import load_speaker_map

from biometrics.enroll import enroll_from_meeting
from biometrics.identify import DEFAULT_THRESHOLD
from biometrics.store import enroll_segments


def run_pipeline(
    recording_path: str,
    transcript_path: str | None,
    n_speakers: int | None,
    speaker_map_path: str | None,
    out_dir: str,
    voiceprints_dir: str | None = None,
    identify_threshold: float = DEFAULT_THRESHOLD,
    whisper_model: str = "base",
    distance_threshold: float = pipeline.DEFAULT_DISTANCE_THRESHOLD,
    interactive_enroll: bool = False,
) -> None:
    if n_speakers is None:
        print(
            "WARNING: --speakers not provided. Estimating speaker count via "
            "distance-threshold clustering, which has been observed to produce "
            "wildly wrong speaker counts. Prefer passing --speakers explicitly."
        )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    wav_path = str(out_path / "meeting.wav")
    pipeline.convert_to_wav(recording_path, wav_path)

    speaker_timeline = pipeline.build_speaker_timeline(wav_path, n_speakers, distance_threshold)

    if voiceprints_dir:
        speaker_timeline, cluster_info = pipeline.identify_speakers(
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
                    enroll_segments(voiceprints_dir, name, wav_path, info["embeddings"], info["segments"])
                    renames[cluster] = name

            if renames:
                speaker_timeline = pipeline.apply_cluster_renames(speaker_timeline, renames)
        else:
            from diarization.clustering import merge_consecutive_segments

            speaker_timeline = merge_consecutive_segments(speaker_timeline)

    transcript_lines = pipeline.get_transcript_lines(wav_path, transcript_path, whisper_model)

    speaker_map = load_speaker_map(speaker_map_path)
    pipeline.finalize_and_export(transcript_lines, speaker_timeline, speaker_map, out_dir)


def enroll_pipeline(
    recording_path: str,
    n_speakers: int,
    speaker_map_path: str,
    voiceprints_dir: str,
) -> None:
    speaker_map = load_speaker_map(speaker_map_path)
    if not speaker_map:
        raise ValueError("--speaker-map is required for enrollment (need names to enroll)")

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = str(Path(tmp_dir) / "meeting.wav")
        pipeline.convert_to_wav(recording_path, wav_path)

        speaker_timeline = pipeline.build_speaker_timeline(wav_path, n_speakers)
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
        default=pipeline.DEFAULT_DISTANCE_THRESHOLD,
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
