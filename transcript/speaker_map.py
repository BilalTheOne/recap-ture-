"""Apply an optional speaker-label-to-real-name mapping.

"Speaker_multiple" is never renamed, since it does not refer to one person.
"""

import json


def load_speaker_map(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def apply_speaker_map(lines: list[dict], speaker_map: dict[str, str]) -> list[dict]:
    return [
        {
            **line,
            "speaker": speaker_map.get(line["speaker"], line["speaker"])
            if line["speaker"] != "Speaker_multiple"
            else line["speaker"],
        }
        for line in lines
    ]
