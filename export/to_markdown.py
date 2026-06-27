def _format_timestamp(seconds: float) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def export_markdown(lines: list[dict], path: str) -> None:
    blocks = [
        f"[{_format_timestamp(line['start'])}] {line['speaker']}:\n{line['text']}"
        for line in lines
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(blocks) + "\n")
