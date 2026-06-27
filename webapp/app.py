"""Web dashboard for the speaker attribution pipeline.

Upload a recording, optionally provide a known speaker count and an
existing transcript, and let voice biometrics identify known speakers
automatically. Any speaker that doesn't match an enrolled voiceprint is
presented for naming in the browser; naming them enrolls their voice for
future recordings.

Run with:
    uvicorn webapp.app:app --host 0.0.0.0 --port 8000
"""

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

import pipeline
from diarization.clustering import merge_consecutive_segments

from biometrics.store import add_embeddings

VOICEPRINTS_DIR = "voiceprints"
JOBS_DIR = Path("output/web")

app = FastAPI(title="Speaker Attribution Dashboard")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@dataclass
class Job:
    id: str
    out_dir: Path
    status: str = "processing"  # processing | awaiting_names | finalizing | done | error
    cluster_info: dict = field(default_factory=dict)
    speaker_timeline: list = field(default_factory=list)
    renames: dict = field(default_factory=dict)
    final_lines: list = field(default_factory=list)
    error: str | None = None
    resume_event: threading.Event = field(default_factory=threading.Event)


jobs: dict[str, Job] = {}
jobs_lock = threading.Lock()


def _run_job(
    job: Job,
    recording_path: str,
    transcript_path: str | None,
    n_speakers: int | None,
    use_voiceprints: bool,
    whisper_model: str,
) -> None:
    try:
        wav_path = str(job.out_dir / "meeting.wav")
        pipeline.convert_to_wav(recording_path, wav_path)

        speaker_timeline = pipeline.build_speaker_timeline(wav_path, n_speakers)

        if use_voiceprints:
            speaker_timeline, cluster_info = pipeline.identify_speakers(
                wav_path, speaker_timeline, VOICEPRINTS_DIR
            )
            unmatched = {c: info for c, info in cluster_info.items() if info["name"] is None}

            if unmatched:
                job.cluster_info = unmatched
                job.speaker_timeline = speaker_timeline
                job.status = "awaiting_names"
                job.resume_event.wait()
                if job.renames:
                    speaker_timeline = pipeline.apply_cluster_renames(speaker_timeline, job.renames)
            else:
                speaker_timeline = merge_consecutive_segments(speaker_timeline)

        job.status = "finalizing"
        transcript_lines = pipeline.get_transcript_lines(wav_path, transcript_path, whisper_model)
        job.final_lines = pipeline.finalize_and_export(
            transcript_lines, speaker_timeline, None, str(job.out_dir)
        )
        job.status = "done"
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
        job.error = str(exc)
        job.status = "error"


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "upload.html", {})


@app.post("/jobs")
def create_job(
    recording: UploadFile = File(...),
    transcript: UploadFile | None = File(None),
    speakers: str = Form(""),
    use_voiceprints: bool = Form(False),
    whisper_model: str = Form("base"),
):
    job_id = uuid.uuid4().hex[:12]
    out_dir = JOBS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    recording_path = str(out_dir / recording.filename)
    with open(recording_path, "wb") as f:
        f.write(recording.file.read())

    transcript_path = None
    if transcript is not None and transcript.filename:
        transcript_path = str(out_dir / transcript.filename)
        with open(transcript_path, "wb") as f:
            f.write(transcript.file.read())

    n_speakers = int(speakers) if speakers.strip() else None

    job = Job(id=job_id, out_dir=out_dir)
    with jobs_lock:
        jobs[job_id] = job

    thread = threading.Thread(
        target=_run_job,
        args=(job, recording_path, transcript_path, n_speakers, use_voiceprints, whisper_model),
        daemon=True,
    )
    thread.start()

    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)


@app.get("/jobs/{job_id}")
def job_status(request: Request, job_id: str):
    job = jobs[job_id]

    if job.status == "awaiting_names":
        clusters = [
            {"label": label, "segments": len(info["embeddings"]), "score": info["score"]}
            for label, info in job.cluster_info.items()
        ]
        return templates.TemplateResponse(
            request, "naming.html", {"job_id": job_id, "clusters": clusters}
        )

    if job.status == "done":
        return templates.TemplateResponse(
            request, "result.html", {"job_id": job_id, "lines": job.final_lines}
        )

    if job.status == "error":
        return templates.TemplateResponse(
            request, "status.html", {"job_id": job_id, "status": job.status, "error": job.error}
        )

    return templates.TemplateResponse(
        request, "status.html", {"job_id": job_id, "status": job.status, "error": None}
    )


@app.post("/jobs/{job_id}/names")
async def submit_names(request: Request, job_id: str):
    job = jobs[job_id]
    form = await request.form()

    renames = {}
    for cluster, info in job.cluster_info.items():
        name = str(form.get(f"name__{cluster}", "")).strip()
        if name:
            add_embeddings(VOICEPRINTS_DIR, name, info["embeddings"])
            renames[cluster] = name

    job.renames = renames
    job.status = "finalizing"
    job.resume_event.set()

    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)


@app.get("/jobs/{job_id}/download/{fmt}")
def download(job_id: str, fmt: str):
    job = jobs[job_id]
    path = job.out_dir / f"transcript.{fmt}"
    return FileResponse(path, filename=path.name)
