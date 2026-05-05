"""MIDI note to 5-fret Clone Hero mapping with gameplay heuristics."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class MidiNote:
    start_time: float  # seconds
    end_time: float  # seconds
    pitch: int  # MIDI pitch (0-127)
    confidence: float  # 0.0-1.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class FretNote:
    """A note mapped to a Clone Hero fret with tick position."""
    tick: int
    fret: int  # 0-4 (green-orange)
    duration_ticks: int


def quantize_to_grid(tick: int, resolution: int = 192, subdivision: int = 4) -> int:
    """Snap a tick to the nearest grid position.
    subdivision=4 means 16th notes (resolution/4 = 48 tick grid).
    """
    grid_size = resolution // subdivision
    if grid_size <= 0:
        return tick
    return round(tick / grid_size) * grid_size


def _time_to_tick(time_sec: float, bpm: float, resolution: int) -> int:
    """Convert a time in seconds to a tick position."""
    beats = time_sec * bpm / 60.0
    return int(round(beats * resolution))


def map_notes_to_frets(
    notes: list[MidiNote],
    bpm: float,
    resolution: int = 192,
    offset_sec: float = 0.0,
) -> list[FretNote]:
    """Map MIDI notes to 5-fret Clone Hero notes with gameplay heuristics.

    Heuristics applied:
    - Pitch contour preservation (ascending/descending pitch → ascending/descending frets)
    - Repeated pitch consistency (same pitch → same fret)
    - Limited fret jumps (max 2 fret change between adjacent notes)
    - All output frets in 0-4 range
    """
    if not notes:
        return []

    # Collect unique pitches and establish pitch-to-relative-position mapping
    pitches = sorted(set(n.pitch for n in notes))
    if len(pitches) == 1:
        # All same pitch, map to fret 2 (middle)
        pitch_to_base = {pitches[0]: 2}
    else:
        # Map pitch range to 0-4 fret range, preserving order
        pitch_to_base = {}
        for i, p in enumerate(pitches):
            pitch_to_base[p] = int(round(i * 4 / (len(pitches) - 1)))

    result: list[FretNote] = []
    prev_fret: Optional[int] = None

    for note in sorted(notes, key=lambda n: n.start_time):
        target_fret = pitch_to_base[note.pitch]

        # Limit jumps from previous note
        if prev_fret is not None:
            diff = target_fret - prev_fret
            if diff > 2:
                target_fret = prev_fret + 2
            elif diff < -2:
                target_fret = prev_fret - 2

        # Clamp to valid range
        target_fret = max(0, min(4, target_fret))

        tick = _time_to_tick(note.start_time - offset_sec, bpm, resolution)
        tick = max(0, tick)
        tick = quantize_to_grid(tick, resolution)

        dur_ticks = _time_to_tick(note.duration, bpm, resolution)
        dur_ticks = max(0, dur_ticks)

        result.append(FretNote(tick=tick, fret=target_fret, duration_ticks=dur_ticks))
        prev_fret = target_fret

    return result
