"""Audio transcription: Beat tracking and note detection using librosa and Basic Pitch."""
from __future__ import annotations
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

BASIC_PITCH_AVAILABLE = False
try:
    from basic_pitch.inference import predict as bp_predict
    BASIC_PITCH_AVAILABLE = True
except ImportError:
    pass


@dataclass
class BeatMap:
    """Variable tempo map derived from beat tracking."""
    beat_times: list[float]  # beat positions in seconds
    tempo_changes: list[tuple[float, float]]  # (time_sec, bpm) pairs
    median_bpm: float


def detect_beats(audio: np.ndarray, sr: int) -> BeatMap:
    """Detect beats and build a variable tempo map using librosa.

    Returns a BeatMap with beat positions and tempo changes.
    """
    import librosa

    # Beat tracking
    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    if len(beat_times) < 2:
        bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
        return BeatMap(
            beat_times=beat_times,
            tempo_changes=[(0.0, bpm)],
            median_bpm=bpm,
        )

    # Compute local BPM from inter-beat intervals
    intervals = np.diff(beat_times)
    local_bpms = 60.0 / intervals

    # Smooth local BPMs with a median filter to reduce noise
    kernel_size = min(7, len(local_bpms))
    if kernel_size % 2 == 0:
        kernel_size = max(1, kernel_size - 1)
    from scipy.ndimage import median_filter
    smoothed_bpms = median_filter(local_bpms, size=kernel_size)

    # Build tempo change list: only emit a change when BPM shifts significantly
    tempo_changes: list[tuple[float, float]] = []
    current_bpm = float(smoothed_bpms[0])
    tempo_changes.append((0.0, current_bpm))

    for i in range(1, len(smoothed_bpms)):
        new_bpm = float(smoothed_bpms[i])
        # Only register a change if BPM shifts by more than 2%
        if abs(new_bpm - current_bpm) / current_bpm > 0.02:
            tempo_changes.append((beat_times[i], new_bpm))
            current_bpm = new_bpm

    median_bpm = float(np.median(local_bpms))

    return BeatMap(
        beat_times=beat_times,
        tempo_changes=tempo_changes,
        median_bpm=median_bpm,
    )


@dataclass
class TranscribedNote:
    """A note detected by Basic Pitch."""
    start_time: float  # seconds
    end_time: float  # seconds
    pitch: int  # MIDI pitch
    amplitude: float  # 0.0-1.0


def transcribe_notes(audio_path: str, min_confidence: float = 0.4) -> list[TranscribedNote]:
    """Transcribe audio to notes using Basic Pitch.

    Args:
        audio_path: Path to audio file (WAV, MP3, etc.)
        min_confidence: Minimum amplitude threshold for notes.

    Returns:
        List of transcribed notes, filtered by confidence.
    """
    if not BASIC_PITCH_AVAILABLE:
        raise RuntimeError(
            "Basic Pitch is not installed. Install with: pip install basic-pitch onnxruntime"
        )

    logger.info("Running Basic Pitch transcription on %s", audio_path)
    model_output, midi_data, note_events = bp_predict(audio_path)

    notes = []
    for event in note_events:
        start_time, end_time, pitch, amplitude = event[0], event[1], event[2], event[3]
        amp = float(amplitude)
        if amp >= min_confidence:
            notes.append(TranscribedNote(
                start_time=float(start_time),
                end_time=float(end_time),
                pitch=int(pitch),
                amplitude=amp,
            ))

    logger.info("Basic Pitch found %d notes (after filtering %d total)", len(notes), len(note_events))
    return notes


def is_basic_pitch_available() -> bool:
    """Check if Basic Pitch transcription is available."""
    return BASIC_PITCH_AVAILABLE
