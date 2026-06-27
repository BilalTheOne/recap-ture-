"""Voice activity detection using Silero VAD."""

import torch


def detect_speech_segments(wav_path: str, sample_rate: int = 16000) -> list[dict]:
    """Return speech regions in `wav_path` as a list of {"start", "end"} (seconds)."""
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
    )
    get_speech_timestamps, _, read_audio, *_ = utils

    audio = read_audio(wav_path, sampling_rate=sample_rate)
    timestamps = get_speech_timestamps(audio, model, sampling_rate=sample_rate)

    return [
        {"start": t["start"] / sample_rate, "end": t["end"] / sample_rate}
        for t in timestamps
    ]
