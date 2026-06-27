"""Local file-based storage for enrolled speaker voiceprints.

Each enrolled person is stored as one JSON file under the voiceprints
directory, containing the raw embeddings collected across enrollment
sessions (not just a single averaged vector), so later identification can
match against the full sample set rather than one centroid alone.

Alongside each embedding, the short audio clip it was extracted from is
saved under `<voiceprints_dir>/audio/<name>/`, so enrolled voices can be
played back later. Voiceprints enrolled before this feature existed have no
audio on file — `audio_samples` entries for those are `None`.
"""

import json
import re
import subprocess
import uuid
from pathlib import Path

import numpy as np


def _safe_filename(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_").lower()
    return slug


def load_voiceprint(voiceprints_dir: str, name: str) -> dict:
    path = Path(voiceprints_dir) / f"{_safe_filename(name)}.json"
    if not path.exists():
        return {"name": name, "embeddings": [], "audio_samples": []}

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Backfill audio_samples for voiceprints saved before audio was kept.
    audio_samples = data.get("audio_samples", [])
    while len(audio_samples) < len(data["embeddings"]):
        audio_samples.append(None)
    data["audio_samples"] = audio_samples

    return data


def save_voiceprint(
    voiceprints_dir: str,
    name: str,
    embeddings: list[list[float]],
    audio_samples: list[str | None],
) -> None:
    out_dir = Path(voiceprints_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_safe_filename(name)}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"name": name, "embeddings": embeddings, "audio_samples": audio_samples}, f
        )


def save_audio_clip(voiceprints_dir: str, name: str, source_wav: str, start: float, end: float) -> str:
    """Slice [start, end] out of `source_wav` and store it for this person.

    Returns the path relative to `voiceprints_dir`, to be stored in
    `audio_samples`.
    """
    relative_path = f"audio/{_safe_filename(name)}/{uuid.uuid4().hex[:8]}.wav"
    dest_path = Path(voiceprints_dir) / relative_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", source_wav,
            "-ss", str(start), "-to", str(end),
            "-c", "copy", str(dest_path),
        ],
        check=True,
        capture_output=True,
    )
    return relative_path


def add_embeddings(
    voiceprints_dir: str,
    name: str,
    new_embeddings: list[list[float]],
    audio_clips: list[str | None] | None = None,
) -> None:
    """Append newly extracted embeddings (and optional audio clip paths) for a person."""
    existing = load_voiceprint(voiceprints_dir, name)
    existing["embeddings"].extend(new_embeddings)
    existing["audio_samples"].extend(audio_clips or [None] * len(new_embeddings))
    save_voiceprint(voiceprints_dir, name, existing["embeddings"], existing["audio_samples"])


def enroll_segments(
    voiceprints_dir: str,
    name: str,
    wav_path: str,
    embeddings: list[list[float]],
    segments: list[dict],
) -> None:
    """Enroll embeddings together with playable audio clips sliced from `wav_path`.

    `segments` must be aligned by index with `embeddings`, each a
    {"start", "end"} dict in seconds.
    """
    audio_clips = [
        save_audio_clip(voiceprints_dir, name, wav_path, segment["start"], segment["end"])
        for segment in segments
    ]
    add_embeddings(voiceprints_dir, name, embeddings, audio_clips)


def load_all_voiceprints(voiceprints_dir: str) -> dict[str, list[list[float]]]:
    """Return {name: raw_embeddings} for every enrolled person."""
    voiceprints = {}
    voiceprints_path = Path(voiceprints_dir)
    if not voiceprints_path.exists():
        return voiceprints

    for path in voiceprints_path.glob("*.json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data["embeddings"]:
            voiceprints[data["name"]] = data["embeddings"]

    return voiceprints


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
