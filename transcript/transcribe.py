"""Generate a timestamped transcript from audio using faster-whisper.

For recordings that don't come with a pre-existing Teams transcript (e.g.
non-Teams audio files), this produces the same {"start", "end", "text"}
shape that `transcript.parser.parse_transcript` returns, so the rest of the
pipeline doesn't need to know whether the transcript was provided or
generated.
"""

from faster_whisper import WhisperModel

_models: dict[str, WhisperModel] = {}


def _get_model(model_size: str) -> WhisperModel:
    if model_size not in _models:
        _models[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _models[model_size]


def transcribe_audio(wav_path: str, model_size: str = "base") -> list[dict]:
    """Return transcript lines as [{"start", "end", "text"}, ...] (seconds)."""
    model = _get_model(model_size)
    segments, _info = model.transcribe(wav_path)

    return [
        {"start": segment.start, "end": segment.end, "text": segment.text.strip()}
        for segment in segments
    ]
