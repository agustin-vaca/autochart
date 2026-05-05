"""Tests for MIDI-to-5-fret note mapping."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chart_generator.mapper import (
    MidiNote,
    map_notes_to_frets,
    quantize_to_grid,
)


class TestQuantizeToGrid:
    def test_snap_to_nearest_tick(self):
        # Resolution 192, at 120 BPM: 1 beat = 192 ticks
        # Grid at 48 tick intervals (16th notes)
        result = quantize_to_grid(50, resolution=192, subdivision=4)
        assert result == 48

    def test_exact_on_grid(self):
        result = quantize_to_grid(192, resolution=192, subdivision=4)
        assert result == 192

    def test_snap_up(self):
        result = quantize_to_grid(180, resolution=192, subdivision=4)
        assert result == 192

    def test_zero_stays_zero(self):
        result = quantize_to_grid(0, resolution=192, subdivision=4)
        assert result == 0


class TestMidiNote:
    def test_create(self):
        n = MidiNote(start_time=1.0, end_time=1.5, pitch=60, confidence=0.9)
        assert n.start_time == 1.0
        assert n.duration == 0.5


class TestMapNotesToFrets:
    def test_single_note(self):
        notes = [MidiNote(start_time=0.5, end_time=0.6, pitch=60, confidence=0.9)]
        result = map_notes_to_frets(notes, bpm=120.0, resolution=192)
        assert len(result) == 1
        assert 0 <= result[0].fret <= 4

    def test_ascending_pitch_maps_ascending_frets(self):
        notes = [
            MidiNote(start_time=0.0, end_time=0.1, pitch=50, confidence=0.9),
            MidiNote(start_time=0.5, end_time=0.6, pitch=55, confidence=0.9),
            MidiNote(start_time=1.0, end_time=1.1, pitch=60, confidence=0.9),
            MidiNote(start_time=1.5, end_time=1.6, pitch=65, confidence=0.9),
            MidiNote(start_time=2.0, end_time=2.1, pitch=70, confidence=0.9),
        ]
        result = map_notes_to_frets(notes, bpm=120.0, resolution=192)
        frets = [r.fret for r in result]
        # Should generally preserve ascending contour
        assert frets == sorted(frets)

    def test_descending_pitch_maps_descending_frets(self):
        notes = [
            MidiNote(start_time=0.0, end_time=0.1, pitch=70, confidence=0.9),
            MidiNote(start_time=0.5, end_time=0.6, pitch=65, confidence=0.9),
            MidiNote(start_time=1.0, end_time=1.1, pitch=60, confidence=0.9),
            MidiNote(start_time=1.5, end_time=1.6, pitch=55, confidence=0.9),
            MidiNote(start_time=2.0, end_time=2.1, pitch=50, confidence=0.9),
        ]
        result = map_notes_to_frets(notes, bpm=120.0, resolution=192)
        frets = [r.fret for r in result]
        assert frets == sorted(frets, reverse=True)

    def test_repeated_pitch_maps_same_fret(self):
        notes = [
            MidiNote(start_time=0.0, end_time=0.1, pitch=60, confidence=0.9),
            MidiNote(start_time=0.5, end_time=0.6, pitch=60, confidence=0.9),
            MidiNote(start_time=1.0, end_time=1.1, pitch=60, confidence=0.9),
        ]
        result = map_notes_to_frets(notes, bpm=120.0, resolution=192)
        frets = [r.fret for r in result]
        assert frets[0] == frets[1] == frets[2]

    def test_frets_in_valid_range(self):
        notes = [
            MidiNote(start_time=i * 0.25, end_time=i * 0.25 + 0.1, pitch=40 + i * 3, confidence=0.9)
            for i in range(20)
        ]
        result = map_notes_to_frets(notes, bpm=120.0, resolution=192)
        for r in result:
            assert 0 <= r.fret <= 4

    def test_no_large_fret_jumps(self):
        """Adjacent notes shouldn't jump more than 2 frets typically."""
        notes = [
            MidiNote(start_time=i * 0.25, end_time=i * 0.25 + 0.1, pitch=50 + i, confidence=0.9)
            for i in range(10)
        ]
        result = map_notes_to_frets(notes, bpm=120.0, resolution=192)
        frets = [r.fret for r in result]
        for i in range(1, len(frets)):
            assert abs(frets[i] - frets[i - 1]) <= 2

    def test_empty_input(self):
        result = map_notes_to_frets([], bpm=120.0, resolution=192)
        assert result == []

    def test_output_has_ticks(self):
        notes = [MidiNote(start_time=1.0, end_time=1.5, pitch=60, confidence=0.9)]
        result = map_notes_to_frets(notes, bpm=120.0, resolution=192)
        assert result[0].tick >= 0
        assert isinstance(result[0].tick, int)
