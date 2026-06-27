"""Local file-based storage for enrolled speaker voiceprints.

Each enrolled person is stored as one JSON file under the voiceprints
directory, containing the raw embeddings collected across enrollment
sessions (not just a single averaged vector), so later identification can
match against the full sample set rather than one centroid alone.
"""

import json
import re
from pathlib import Path

import numpy as np


def _safe_filename(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_").lower()
    return f"{slug}.json"


def load_voiceprint(voiceprints_dir: str, name: str) -> dict:
    path = Path(voiceprints_dir) / _safe_filename(name)
    if not path.exists():
        return {"name": name, "embeddings": []}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_voiceprint(voiceprints_dir: str, name: str, embeddings: list[list[float]]) -> None:
    out_dir = Path(voiceprints_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / _safe_filename(name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"name": name, "embeddings": embeddings}, f)


def add_embeddings(voiceprints_dir: str, name: str, new_embeddings: list[list[float]]) -> None:
    """Append newly extracted embeddings to a person's stored voiceprint."""
    existing = load_voiceprint(voiceprints_dir, name)
    existing["embeddings"].extend(new_embeddings)
    save_voiceprint(voiceprints_dir, name, existing["embeddings"])


def load_all_centroids(voiceprints_dir: str) -> dict[str, np.ndarray]:
    """Return {name: centroid_embedding} for every enrolled person."""
    centroids = {}
    voiceprints_path = Path(voiceprints_dir)
    if not voiceprints_path.exists():
        return centroids

    for path in voiceprints_path.glob("*.json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not data["embeddings"]:
            continue
        centroids[data["name"]] = np.array(data["embeddings"]).mean(axis=0)

    return centroids
