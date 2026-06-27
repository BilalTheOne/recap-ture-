# Teams Meeting Speaker Attribution Project

## Objective

Create an automated pipeline that:

1. Takes a Microsoft Teams meeting recording.
2. Takes the corresponding Teams transcript.
3. Detects and separates speakers from the audio.
4. Maps transcript segments to detected speakers.
5. Produces a final transcript with speaker labels.

The system does **not** need to identify real names. It only needs consistent labels such as:

```text
Speaker_1: Hello everyone.
Speaker_2: Good morning.
Speaker_1: Let's start the meeting.
```

---

# Requirements

## Inputs

### Audio Recording

Supported formats:

* MP4
* WAV
* M4A

### Teams Transcript

Supported formats:

* VTT
* TXT
* JSON
* DOCX (converted to text)

Transcript must contain timestamps.

### Speaker Count

The number of speakers present in the meeting must be provided per meeting
(known input, not estimated).

### Speaker Name Mapping

An optional per-meeting mapping from speaker label to real name can be
provided, e.g.:

```json
{
  "Speaker_1": "Ahmed",
  "Speaker_2": "Sarah"
}
```

If provided, output uses the real names in place of `Speaker_1`/`Speaker_2`.
`Speaker_multiple` is never renamed, since it does not refer to one person.
If no mapping is provided, generic labels are used.

---

# High-Level Architecture

```text
Audio Recording
       |
       v
  Voice Activity
    Detection
       |
       v
 Speech Segments
       |
       v
 Overlap Detection
       |
       +-----------------------+
       |                       |
       v                       v
Overlapping Segments     Clean Segments
       |                       |
       v                       v
Tag "Speaker_multiple"   Speaker Embeddings
       |                       |
       |                       v
       |                  Clustering
       |                  (n_clusters = known speaker count)
       |                       |
       +-----------+-----------+
                   |
                   v
            Speaker Timeline
                   |
                   +-------------------+
                                       |
                                       v
                                Teams Transcript
                                       |
                                       v
                                Timestamp Mapping
                                       |
                                       v
                                Final Transcript
```

Notes:

* Audio captured from a single shared/room microphone can contain multiple
  local speakers under one Teams participant identity, so Teams'
  participant-level speaker attribution is not trusted as a source of truth.
  Diarization is performed directly on the audio instead.
* The number of speakers in a meeting is provided as a known input
  (not estimated), which removes the need for distance-threshold clustering.
* Overlapping speech (multiple people talking at once) is detected
  automatically, not marked manually. Overlapping segments are not assigned
  to an individual speaker; they are labeled `Speaker_multiple`.

---

# Technology Stack

## Audio Processing

### Silero VAD

Purpose:

* Detect speech regions
* Ignore silence
* Split audio into speech chunks

Advantages:

* Fast
* CPU-friendly
* Open source

---

## Overlap Detection

### pyannote.audio (Overlapped Speech Detection)

Purpose:

* Detect time regions where more than one speaker is talking at once
* Runs automatically — no manual marking of overlap regions required

Advantages:

* Pretrained, maintained model (`pyannote/overlapped-speech-detection`)
* Frame-level detection works alongside Silero VAD output

License note:

* `pyannote.audio` is GPL-licensed. Acceptable here since this project is
  for internal use and is not distributed/sold.

Output Example:

```json
[
  {"start": 12.3, "end": 15.0, "overlap": true}
]
```

---

## Speaker Embeddings

### SpeechBrain ECAPA-TDNN

Purpose:

* Convert voice segments into embeddings
* Create speaker fingerprints

Advantages:

* Accurate
* Fast
* Works well on meetings

Output Example:

```python
[0.123, -0.456, 0.998, ...]
```

---

## Clustering

### Agglomerative Clustering

Purpose:

Group embeddings belonging to the same speaker.

The number of speakers (`n_clusters`) is provided as a known input per
meeting rather than estimated via a distance threshold, which removes a
major source of instability.

Segments flagged by overlap detection are excluded from clustering — they
are labeled `Speaker_multiple` directly and never assigned to a single
speaker.

Example:

```text
Segment 1 -> Speaker_1
Segment 2 -> Speaker_2
Segment 3 -> Speaker_1
Segment 4 -> Speaker_3
Segment 5 -> Speaker_multiple   (overlap detected, skipped clustering)
```

---

# Processing Steps

## Step 1

Load meeting recording.

Output:

```text
meeting.mp4
```

---

## Step 2

Convert recording to WAV.

Example:

```bash
ffmpeg -i meeting.mp4 meeting.wav
```

---

## Step 3

Run Silero VAD.

Output:

```json
[
  {"start": 1.2, "end": 4.8},
  {"start": 5.1, "end": 8.0}
]
```

---

## Step 4

Run overlap detection on speech segments.

Output:

```json
[
  {"start": 1.2, "end": 4.8, "overlap": false},
  {"start": 5.1, "end": 8.0, "overlap": true}
]
```

Segments with `overlap: true` are set aside and labeled `Speaker_multiple`
directly — they skip embedding extraction and clustering.

---

## Step 5

Extract speaker embeddings for each non-overlapping segment.

Output:

```json
[
  {
    "segment": 1,
    "embedding": [...]
  }
]
```

---

## Step 6

Cluster embeddings using the known number of speakers (`n_clusters`).

Output:

```json
[
  {
    "start": 1.2,
    "end": 4.8,
    "speaker": "Speaker_1"
  },
  {
    "start": 5.1,
    "end": 8.0,
    "speaker": "Speaker_multiple"
  }
]
```

---

## Step 7

Merge consecutive segments belonging to the same speaker.

Example:

Before:

```text
Speaker_1
Speaker_1
Speaker_1
```

After:

```text
Speaker_1
```

---

## Step 8

Parse Teams transcript.

Extract:

```json
[
  {
    "start": 1.5,
    "end": 3.2,
    "text": "Hello everyone"
  }
]
```

---

## Step 9

Assign speakers using timestamps.

Algorithm:

1. Find transcript timestamp.
2. Find speaker segment covering that timestamp.
3. Attach speaker label (`Speaker_multiple` if the segment was flagged as
   overlapping).

Example:

```json
{
  "speaker": "Speaker_1",
  "text": "Hello everyone"
}
```

---

## Step 10

Apply speaker name mapping, if provided.

Algorithm:

1. For each segment, look up the speaker label in the mapping.
2. If found, replace the label with the real name.
3. If not found (or no mapping provided), keep the generic label.
4. Never rename `Speaker_multiple`.

Example:

```json
{
  "speaker": "Ahmed",
  "text": "Hello everyone"
}
```

---

# Output Format

## JSON

```json
[
  {
    "speaker": "Ahmed",
    "start": 1.5,
    "end": 3.2,
    "text": "Hello everyone"
  }
]
```

---

## Markdown

```text
[00:00:01] Ahmed:
Hello everyone.

[00:00:05] Sarah:
Good morning.

[00:00:08] Ahmed:
Let's begin.
```

---

## DOCX

Formatted as a Word document, one paragraph per line:

```text
[00:00:01] Ahmed:
Hello everyone.

[00:00:05] Sarah:
Good morning.

[00:00:08] Ahmed:
Let's begin.
```

Speaker label is bolded; timestamp and text are plain. Generated with
`python-docx`.

---

## VTT

Standard WebVTT using the `<v Speaker>` voice tag, so it stays a valid,
re-importable subtitle/transcript file:

```vtt
WEBVTT

00:00:01.500 --> 00:00:03.200
<v Ahmed>Hello everyone.

00:00:05.000 --> 00:00:08.000
<v Sarah>Good morning.
```

---

# Performance Goals

## Accuracy

Target:

* Speaker separation > 85%
* Stable labels across meeting

## Speed

Target:

* 1 hour meeting processed in under 10 minutes
* CPU-only operation

---

# Future Improvements

## Web Dashboard

Features:

* Upload recording
* Upload transcript
* View final transcript
* Export results

---

## Database Storage

Store:

* Meetings
* Segments
* Speakers
* Generated transcripts

---

# Project Structure

```text
project/
│
├── input/
│   ├── recordings/
│   ├── transcripts/
│   └── speaker_maps/
│
├── output/
│   ├── json/
│   ├── markdown/
│   ├── docx/
│   └── vtt/
│
├── diarization/
│   ├── vad.py
│   ├── overlap.py
│   ├── embeddings.py
│   ├── clustering.py
│
├── transcript/
│   ├── parser.py
│   ├── align.py
│   ├── speaker_map.py
│
├── export/
│   ├── to_json.py
│   ├── to_markdown.py
│   ├── to_docx.py
│   ├── to_vtt.py
│
├── main.py
│
└── requirements.txt
```

---

# Success Criteria

The system is considered successful when:

* A Teams recording is uploaded.
* The Teams transcript is uploaded.
* Speakers are automatically separated.
* Transcript lines are attributed to speaker IDs.
* Results are generated without manual intervention.
