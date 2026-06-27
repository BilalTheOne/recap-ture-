"""Automatic overlapping-speech detection using pyannote.audio.

Marks which speech segments contain more than one simultaneous speaker, so
they can be excluded from per-speaker clustering and labeled
"Speaker_multiple" instead.
"""

from pyannote.audio import Pipeline

_OVERLAP_MODEL = "pyannote/overlapped-speech-detection"


def detect_overlaps(wav_path: str) -> list[dict]:
    """Return overlapping speech regions in `wav_path` as {"start", "end"}."""
    pipeline = Pipeline.from_pretrained(_OVERLAP_MODEL)
    annotation = pipeline(wav_path)

    return [
        {"start": segment.start, "end": segment.end}
        for segment, _, _ in annotation.itertracks(yield_label=True)
    ]


def split_segments_by_overlap(
    segments: list[dict], overlaps: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Split VAD segments into (clean, overlapping) based on overlap regions.

    A segment is considered overlapping if its midpoint falls inside any
    detected overlap region.
    """
    clean, overlapping = [], []
    for segment in segments:
        midpoint = (segment["start"] + segment["end"]) / 2
        is_overlap = any(o["start"] <= midpoint <= o["end"] for o in overlaps)
        (overlapping if is_overlap else clean).append(segment)

    return clean, overlapping
