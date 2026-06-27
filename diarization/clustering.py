"""Cluster speaker embeddings into speaker labels.

The number of speakers is a known input per meeting (not estimated), so
clustering uses a fixed `n_clusters` rather than a distance threshold.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def cluster_speakers(
    segments: list[dict], embeddings: list[dict], n_speakers: int
) -> list[dict]:
    """Assign a "Speaker_N" label to each segment based on its embedding.

    `segments` and `embeddings` must be aligned by index (same order as
    produced by `embeddings.extract_embeddings`).
    """
    vectors = np.array([e["embedding"] for e in embeddings])

    clustering = AgglomerativeClustering(n_clusters=n_speakers)
    labels = clustering.fit_predict(vectors)

    return [
        {
            "start": segment["start"],
            "end": segment["end"],
            "speaker": f"Speaker_{label + 1}",
        }
        for segment, label in zip(segments, labels)
    ]


def merge_consecutive_segments(segments: list[dict]) -> list[dict]:
    """Merge consecutive segments that belong to the same speaker."""
    if not segments:
        return []

    merged = [dict(segments[0])]
    for segment in segments[1:]:
        last = merged[-1]
        if segment["speaker"] == last["speaker"]:
            last["end"] = segment["end"]
        else:
            merged.append(dict(segment))

    return merged
