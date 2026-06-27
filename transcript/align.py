"""Attach speaker labels to transcript lines using timestamp overlap."""


def assign_speakers(transcript_lines: list[dict], speaker_timeline: list[dict]) -> list[dict]:
    """Return transcript lines with a "speaker" field attached.

    For each transcript line, find the speaker timeline segment whose range
    covers the line's midpoint. Lines with no matching segment get
    "speaker": None.
    """
    results = []
    for line in transcript_lines:
        midpoint = (line["start"] + line["end"]) / 2
        speaker = next(
            (
                seg["speaker"]
                for seg in speaker_timeline
                if seg["start"] <= midpoint <= seg["end"]
            ),
            None,
        )
        results.append({**line, "speaker": speaker})

    return results
