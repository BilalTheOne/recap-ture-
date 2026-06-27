"""Cluster speaker embeddings into speaker labels.

The number of speakers is normally provided as a known input per meeting
(not estimated), since that removes the need for distance-threshold
clustering — a major source of instability. When it isn't provided, a
cosine distance threshold is used to estimate the speaker count instead;
this is inherently less reliable and may need tuning per recording.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering

DEFAULT_DISTANCE_THRESHOLD = 0.6


def cluster_speakers(
    segments: list[dict],
    embeddings: list[dict],
    n_speakers: int | None = None,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> list[dict]:
    """Assign a "Speaker_N" label to each segment based on its embedding.

    `segments` and `embeddings` must be aligned by index (same order as
    produced by `embeddings.extract_embeddings`).

    If `n_speakers` is given, clustering uses that fixed count. Otherwise,
    the speaker count is estimated by cutting the hierarchical clustering
    tree at `distance_threshold` (cosine distance) — an approximation that
    can split or merge speakers incorrectly if the threshold doesn't suit
    the recording.
    """
    vectors = np.array([e["embedding"] for e in embeddings])

    if len(vectors) < 2:
        # AgglomerativeClustering requires at least 2 samples; with 0 or 1
        # clean segments there's nothing to cluster, so just assign them
        # directly to Speaker_1.
        return [
            {"start": segment["start"], "end": segment["end"], "speaker": "Speaker_1"}
            for segment in segments
        ]

    if n_speakers is not None:
        n_speakers = min(n_speakers, len(vectors))
        clustering = AgglomerativeClustering(n_clusters=n_speakers)
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        )
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
