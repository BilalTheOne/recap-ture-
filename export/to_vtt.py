def _format_vtt_timestamp(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"


def export_vtt(lines: list[dict], path: str) -> None:
    blocks = [
        f"{_format_vtt_timestamp(line['start'])} --> {_format_vtt_timestamp(line['end'])}\n"
        f"<v {line['speaker']}>{line['text']}"
        for line in lines
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n" + "\n\n".join(blocks) + "\n")
