import json


def export_json(lines: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lines, f, ensure_ascii=False, indent=2)
