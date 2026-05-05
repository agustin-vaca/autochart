"""End-to-end pipeline: audio file → Clone Hero chart folder."""
from __future__ import annotations
import logging
import os
import shutil
import tempfile
import wave
import struct
import numpy as np

from chart_generator.audio import analyze_audio_array, detect_bpm, detect_onsets
from chart_generator.mapper import MidiNote, map_notes_to_frets
from chart_generator.chart import (
    ChartSong,
    SyncTrack,
    ChartTrack,
    SectionEvent,
    build_chart_string,
)
from chart_generator.difficulty import derive_difficulty
from chart_generator.output import SongMetadata, generate_song_ini, assemble_output_folder
from chart_generator.preview import generate_preview_html
from chart_generator.separator import separate_guitar

logger = logging.getLogger(__name__)


def _load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load audio from WAV (stdlib) or MP3/OGG (pydub fallback)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        with wave.open(path, "r") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
        if sampwidth == 2:
            dtype = np.int16
        elif sampwidth == 4:
            dtype = np.int32
        else:
            dtype = np.int16
        audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        # Normalize to [-1, 1]
        max_val = float(2 ** (sampwidth * 8 - 1))
        audio = audio / max_val
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)
        return audio, sr
    else:
        # Use ffmpeg subprocess for MP3, OGG, etc.
        import subprocess
        import tempfile
        tmp_wav = tempfile.mktemp(suffix=".wav")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", "22050", "-sample_fmt", "s16", tmp_wav],
                capture_output=True, check=True,
            )
            return _load_audio(tmp_wav)
        finally:
            if os.path.exists(tmp_wav):
                os.unlink(tmp_wav)


def _onsets_to_midi_notes(onset_times: list[float], duration_sec: float) -> list[MidiNote]:
    """Convert onset times to synthetic MidiNotes for chart mapping.

    When we don't have real pitch detection (Basic Pitch), we generate
    notes from onsets with varying pseudo-pitches based on timing patterns.
    """
    if not onset_times:
        return []

    notes: list[MidiNote] = []
    for i, t in enumerate(onset_times):
        # Assign pseudo-pitches based on onset patterns to create variety
        # Use spacing between onsets to infer pitch contour
        if i == 0:
            pitch = 60
        else:
            dt = onset_times[i] - onset_times[i - 1]
            # Shorter intervals → higher pitch, longer → lower
            if dt < 0.15:
                pitch = min(72, notes[-1].pitch + 2)
            elif dt < 0.3:
                pitch = notes[-1].pitch + 1
            elif dt < 0.5:
                pitch = notes[-1].pitch
            else:
                pitch = max(48, notes[-1].pitch - 2)

        pitch = max(40, min(80, pitch))

        # Note duration: until next onset or fixed short duration
        if i + 1 < len(onset_times):
            end = min(t + 0.1, onset_times[i + 1] - 0.01)
        else:
            end = t + 0.1

        notes.append(MidiNote(
            start_time=t,
            end_time=end,
            pitch=pitch,
            confidence=0.8,
        ))

    return notes


def generate_chart_from_audio(
    audio_path: str,
    metadata: SongMetadata,
    output_dir: str,
    album_art_path: str | None = None,
    use_separation: bool = True,
    separation_model: str | None = None,
) -> str:
    """Full pipeline: audio file → Clone Hero chart folder.

    Args:
        audio_path: Path to the input audio file.
        metadata: Song metadata for chart generation.
        output_dir: Directory for the output chart folder.
        album_art_path: Optional path to album art image.
        use_separation: If True, attempt Demucs source separation for
            better guitar note detection. Falls back gracefully.
        separation_model: Demucs model name (default: 'htdemucs').
            Use 'htdemucs_6src' for explicit guitar stem.

    Returns the path to the generated output folder.
    """
    # Load full mix audio (WAV via stdlib, MP3/OGG via ffmpeg fallback)
    audio, sr = _load_audio(audio_path)
    duration_sec = len(audio) / sr

    # BPM detection always uses the full mix (drums help)
    bpm, bpm_confidence = detect_bpm(audio, sr)

    # Attempt source separation for better onset detection
    guitar_audio = None
    guitar_sr = sr
    work_dir = None

    if use_separation:
        try:
            work_dir = tempfile.mkdtemp(prefix="autochart_stems_")
            guitar_path = separate_guitar(audio_path, work_dir, separation_model)
            if guitar_path is not None:
                guitar_audio, guitar_sr = _load_audio(guitar_path)
                logger.info("Using separated guitar stem for onset detection.")
        except Exception as e:
            logger.warning("Source separation error: %s. Using full mix.", e)

    # Onset detection: use guitar stem if available, otherwise full mix
    onset_source = guitar_audio if guitar_audio is not None else audio
    onset_sr = guitar_sr if guitar_audio is not None else sr
    onset_times = detect_onsets(onset_source, onset_sr)

    # Clean up temp separation files
    if work_dir is not None:
        try:
            shutil.rmtree(work_dir)
        except OSError:
            pass

    # Convert onsets to MIDI-like notes
    midi_notes = _onsets_to_midi_notes(onset_times, duration_sec)

    # Map to 5-fret notes
    fret_notes = map_notes_to_frets(midi_notes, bpm=bpm, resolution=192)

    # Build chart structures
    song = ChartSong(
        name=metadata.name,
        artist=metadata.artist,
        genre=metadata.genre,
    )

    sync = SyncTrack()
    sync.add_time_signature(0, 4)
    sync.add_bpm(0, bpm)

    # Build Expert track
    expert = ChartTrack("ExpertSingle")
    for fn in fret_notes:
        expert.add_note(fn.tick, fn.fret, fn.duration_ticks)

    # Derive other difficulties
    hard = derive_difficulty(expert, "HardSingle")
    medium = derive_difficulty(expert, "MediumSingle")
    easy = derive_difficulty(expert, "EasySingle")

    tracks = [easy, medium, hard, expert]

    # Build events (minimal sections)
    events: list[SectionEvent] = []
    if duration_sec > 5:
        events.append(SectionEvent(tick=0, name="Intro"))
        mid_tick = int(duration_sec / 2 * bpm / 60 * 192)
        events.append(SectionEvent(tick=mid_tick, name="Middle"))

    # Generate chart string
    chart_content = build_chart_string(song, sync, events, tracks)

    # Generate song.ini
    song_length_ms = int(duration_sec * 1000)
    ini_content = generate_song_ini(metadata, song_length_ms)

    # Assemble output folder
    result_path = assemble_output_folder(
        output_dir=output_dir,
        chart_content=chart_content,
        ini_content=ini_content,
        audio_source_path=audio_path,
        album_art_path=album_art_path,
    )

    # Generate preview HTML
    audio_filename = "song" + os.path.splitext(audio_path)[1]
    preview_path = os.path.join(result_path, "preview.html")
    generate_preview_html(
        song, sync, events, tracks,
        audio_path=audio_filename,
        output_path=preview_path,
    )

    return result_path
