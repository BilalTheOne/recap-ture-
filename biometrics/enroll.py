"""Enroll speakers by extracting voiceprints from a previously diarized meeting.

Enrollment relies on a meeting that has a speaker-label-to-name mapping (the
same `speaker_map.json` used for output renaming): clean, non-overlapping
segments belonging to a mapped label are used as known samples of that
person's voice. "Speaker_multiple" segments and unmapped labels are skipped,
since enrollment requires a known identity.
"""

from diarization.embeddings import extract_embeddings

from biometrics.store import add_embeddings

_MIN_SEGMENT_DURATION = 1.5  # seconds; very short segments make weak voiceprints


def enroll_from_meeting(
    wav_path: str,
    speaker_timeline: list[dict],
    speaker_map: dict[str, str],
    voiceprints_dir: str,
) -> dict[str, int]:
    """Extract embeddings for each mapped speaker and store them as voiceprints.

    Returns {name: number_of_segments_enrolled}.
    """
    enrollable = [
        segment
        for segment in speaker_timeline
        if segment["speaker"] != "Speaker_multiple"
        and segment["speaker"] in speaker_map
        and segment["end"] - segment["start"] >= _MIN_SEGMENT_DURATION
    ]

    embeddings = extract_embeddings(wav_path, enrollable)

    by_name: dict[str, list[list[float]]] = {}
    for segment, embedding in zip(enrollable, embeddings):
        name = speaker_map[segment["speaker"]]
        by_name.setdefault(name, []).append(embedding["embedding"])

    for name, vectors in by_name.items():
        add_embeddings(voiceprints_dir, name, vectors)

    return {name: len(vectors) for name, vectors in by_name.items()}
