"""Identify enrolled speakers by matching embeddings against stored
voiceprints using cosine similarity.
"""

import numpy as np

from diarization.embeddings import extract_embeddings

from biometrics.store import load_all_centroids

DEFAULT_THRESHOLD = 0.65


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def match_embedding(
    embedding: list[float],
    centroids: dict[str, np.ndarray],
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[str | None, float]:
    """Return (best_matching_name_or_None, similarity_score).

    None is returned when the best match falls below `threshold`, meaning
    the voice doesn't confidently belong to anyone enrolled.
    """
    if not centroids:
        return None, 0.0

    vector = np.array(embedding)
    scored = [(name, _cosine_similarity(vector, centroid)) for name, centroid in centroids.items()]
    best_name, best_score = max(scored, key=lambda item: item[1])

    if best_score < threshold:
        return None, best_score
    return best_name, best_score


def identify_clusters(
    wav_path: str,
    speaker_timeline: list[dict],
    voiceprints_dir: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[list[dict], dict[str, dict]]:
    """Match each cluster (e.g. "Speaker_1") against enrolled voiceprints.

    Each cluster is matched as a whole, using the average embedding of all
    its segments, rather than segment-by-segment — a per-meeting cluster
    gives a cleaner voiceprint than any single short segment.
    "Speaker_multiple" segments are left untouched.

    Returns (relabeled_timeline, cluster_info), where cluster_info maps each
    original cluster label to {"name": matched_name_or_None, "score": float,
    "embeddings": [[...], ...], "segments": [{"start", "end"}, ...]}. The raw
    embeddings and their source segment timings are included so unmatched
    clusters (new, unenrolled speakers) can be enrolled — with playable audio
    clips — on the spot without re-extracting audio.
    """
    centroids = load_all_centroids(voiceprints_dir)

    identifiable = [s for s in speaker_timeline if s["speaker"] != "Speaker_multiple"]
    embeddings = extract_embeddings(wav_path, identifiable)

    by_cluster: dict[str, list[list[float]]] = {}
    segments_by_cluster: dict[str, list[dict]] = {}
    for segment, embedding in zip(identifiable, embeddings):
        by_cluster.setdefault(segment["speaker"], []).append(embedding["embedding"])
        segments_by_cluster.setdefault(segment["speaker"], []).append(
            {"start": segment["start"], "end": segment["end"]}
        )

    cluster_info = {}
    cluster_to_name = {}
    for cluster, vectors in by_cluster.items():
        centroid = np.array(vectors).mean(axis=0)
        name, score = match_embedding(centroid.tolist(), centroids, threshold)
        cluster_info[cluster] = {
            "name": name,
            "score": score,
            "embeddings": vectors,
            "segments": segments_by_cluster[cluster],
        }
        if name:
            cluster_to_name[cluster] = name

    relabeled = [
        {**segment, "speaker": cluster_to_name.get(segment["speaker"], segment["speaker"])}
        for segment in speaker_timeline
    ]
    return relabeled, cluster_info
