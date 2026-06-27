"""Speaker embedding extraction using SpeechBrain's ECAPA-TDNN."""

import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"

_classifier = None


def _get_classifier() -> EncoderClassifier:
    global _classifier
    if _classifier is None:
        _classifier = EncoderClassifier.from_hparams(source=_MODEL_SOURCE)
    return _classifier


def extract_embeddings(wav_path: str, segments: list[dict]) -> list[dict]:
    """Return one embedding per segment: [{"segment": i, "embedding": [...]}, ...]."""
    classifier = _get_classifier()
    waveform, sample_rate = torchaudio.load(wav_path)

    results = []
    for i, segment in enumerate(segments):
        start_sample = int(segment["start"] * sample_rate)
        end_sample = int(segment["end"] * sample_rate)
        chunk = waveform[:, start_sample:end_sample]

        embedding = classifier.encode_batch(chunk).squeeze().tolist()
        results.append({"segment": i, "embedding": embedding})

    return results
