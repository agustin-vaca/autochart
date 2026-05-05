"""End-to-end pipeline: audio file → Clone Hero chart folder."""
from __future__ import annotations
import logging
import os
import shutil
import tempfile
import wave
import struct
import numpy as np

from chart_generator.audio import detect_bpm, detect_onsets
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
from chart_generator.transcription import (
    detect_beats,
    transcribe_notes,
    is_basic_pitch_available,
    TranscribedNote,
    BeatMap,
)

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


def _time_to_tick_with_tempo_map(
    time_sec: float,
    tempo_changes: list[tuple[float, float]],
    resolution: int = 192,
) -> int:
    """Convert time in seconds to tick using a variable tempo map.

    Args:
        time_sec: Time position in seconds.
        tempo_changes: List of (time_sec, bpm) pairs, sorted by time.
        resolution: Ticks per beat.
    """
    tick = 0.0
    prev_time = 0.0
    prev_bpm = tempo_changes[0][1] if tempo_changes else 120.0

    for change_time, change_bpm in tempo_changes:
        if change_time >= time_sec:
            break
        # Accumulate ticks from prev_time to change_time at prev_bpm
        dt = change_time - prev_time
        tick += dt * (prev_bpm / 60.0) * resolution
        prev_time = change_time
        prev_bpm = change_bpm

    # Accumulate remaining time
    dt = time_sec - prev_time
    tick += dt * (prev_bpm / 60.0) * resolution
    return int(round(tick))


def _transcribed_to_midi_notes(notes: list[TranscribedNote]) -> list[MidiNote]:
    """Convert Basic Pitch transcribed notes to MidiNote format."""
    return [
        MidiNote(
            start_time=n.start_time,
            end_time=n.end_time,
            pitch=n.pitch,
            confidence=n.amplitude,
        )
        for n in notes
    ]


def generate_chart_from_audio(
    audio_path: str,
    metadata: SongMetadata,
    output_dir: str,
    album_art_path: str | None = None,
    use_separation: bool = True,
    separation_model: str | None = None,
) -> str:
    """Full pipeline: audio file → Clone Hero chart folder.

    Uses the best available transcription method:
    1. Demucs separation + Basic Pitch (if available) — real MIDI transcription
    2. Demucs separation + onset detection (fallback)

    Beat tracking provides a variable tempo map for better timing accuracy.

    Args:
        audio_path: Path to the input audio file.
        metadata: Song metadata for chart generation.
        output_dir: Directory for the output chart folder.
        album_art_path: Optional path to album art image.
        use_separation: If True, attempt Demucs source separation.
        separation_model: Demucs model name (default: 'htdemucs').

    Returns the path to the generated output folder.
    """
    # Load full mix audio
    audio, sr = _load_audio(audio_path)
    duration_sec = len(audio) / sr

    # Beat tracking for variable BPM (uses full mix — drums help)
    beat_map = detect_beats(audio, sr)
    logger.info("Beat tracking: median BPM=%.1f, %d tempo changes",
                beat_map.median_bpm, len(beat_map.tempo_changes))

    # Source separation
    transcription_path = None
    guitar_audio = None
    guitar_sr = sr
    work_dir = None

    if use_separation:
        try:
            work_dir = tempfile.mkdtemp(prefix="autochart_stems_")
            guitar_stem_path = separate_guitar(audio_path, work_dir, separation_model)
            if guitar_stem_path is not None:
                transcription_path = guitar_stem_path
                guitar_audio, guitar_sr = _load_audio(guitar_stem_path)
                logger.info("Using separated guitar stem for transcription.")
        except Exception as e:
            logger.warning("Source separation error: %s. Using full mix.", e)

    # Note detection: prefer Basic Pitch, fall back to onset detection
    use_basic_pitch = is_basic_pitch_available()

    if use_basic_pitch:
        # Basic Pitch transcription on separated stem (or full mix)
        bp_path = transcription_path or audio_path
        try:
            transcribed = transcribe_notes(bp_path, min_confidence=0.4)
            midi_notes = _transcribed_to_midi_notes(transcribed)
            logger.info("Basic Pitch: %d notes detected.", len(midi_notes))
        except Exception as e:
            logger.warning("Basic Pitch failed: %s. Falling back to onset detection.", e)
            use_basic_pitch = False

    if not use_basic_pitch:
        # Fallback: onset detection
        onset_source = guitar_audio if guitar_audio is not None else audio
        onset_sr = guitar_sr if guitar_audio is not None else sr
        onset_times = detect_onsets(onset_source, onset_sr)
        midi_notes = _onsets_to_midi_notes(onset_times, duration_sec)
        logger.info("Onset detection: %d notes.", len(midi_notes))

    # Clean up temp separation files
    if work_dir is not None:
        try:
            shutil.rmtree(work_dir)
        except OSError:
            pass

    # Build tempo map for the SyncTrack
    resolution = 192
    sync = SyncTrack()
    sync.add_time_signature(0, 4)

    # Add all tempo changes from beat tracking
    for change_time, change_bpm in beat_map.tempo_changes:
        tick = _time_to_tick_with_tempo_map(change_time, beat_map.tempo_changes, resolution)
        sync.add_bpm(tick, change_bpm)

    # Map notes to frets using variable tempo map
    fret_notes = _map_notes_with_tempo(midi_notes, beat_map.tempo_changes, resolution)

    # Build chart structures
    song = ChartSong(
        name=metadata.name,
        artist=metadata.artist,
        genre=metadata.genre,
    )

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
        mid_tick = _time_to_tick_with_tempo_map(
            duration_sec / 2, beat_map.tempo_changes, resolution)
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


def _map_notes_with_tempo(
    notes: list[MidiNote],
    tempo_changes: list[tuple[float, float]],
    resolution: int = 192,
) -> list:
    """Map MIDI notes to fret notes using a variable tempo map.

    This uses the tempo map for time→tick conversion instead of a fixed BPM,
    then applies the standard fret mapping heuristics.
    """
    from chart_generator.mapper import FretNote, quantize_to_grid

    if not notes:
        return []

    # Collect unique pitches for fret mapping
    pitches = sorted(set(n.pitch for n in notes))
    if len(pitches) == 1:
        pitch_to_base = {pitches[0]: 2}
    else:
        pitch_to_base = {}
        for i, p in enumerate(pitches):
            pitch_to_base[p] = int(round(i * 4 / (len(pitches) - 1)))

    result = []
    prev_fret = None

    for note in sorted(notes, key=lambda n: n.start_time):
        target_fret = pitch_to_base[note.pitch]

        # Limit jumps from previous note
        if prev_fret is not None:
            diff = target_fret - prev_fret
            if diff > 2:
                target_fret = prev_fret + 2
            elif diff < -2:
                target_fret = prev_fret - 2

        target_fret = max(0, min(4, target_fret))

        tick = _time_to_tick_with_tempo_map(note.start_time, tempo_changes, resolution)
        tick = max(0, tick)
        tick = quantize_to_grid(tick, resolution)

        dur_tick = _time_to_tick_with_tempo_map(
            note.start_time + note.duration, tempo_changes, resolution) - tick
        dur_tick = max(0, dur_tick)

        result.append(FretNote(tick=tick, fret=target_fret, duration_ticks=dur_tick))
        prev_fret = target_fret

    return result
