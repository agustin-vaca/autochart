"""Tests for difficulty derivation."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chart_generator.chart import ChartTrack, NoteEvent
from chart_generator.difficulty import derive_difficulty


class TestDeriveDifficulty:
    def _make_expert_track(self):
        track = ChartTrack("ExpertSingle")
        # Simulate a typical passage: notes every 48 ticks (16th notes at 192 res)
        for i in range(64):
            tick = i * 48
            fret = i % 5
            track.add_note(tick, fret, 0)
        # Add some chords
        track.add_note(3072, 0, 0)
        track.add_note(3072, 2, 0)
        track.add_note(3120, 1, 0)
        track.add_note(3120, 3, 0)
        return track

    def test_hard_has_fewer_notes(self):
        expert = self._make_expert_track()
        hard = derive_difficulty(expert, "HardSingle")
        assert hard.name == "HardSingle"
        assert len(hard.notes) < len(expert.notes)
        assert len(hard.notes) > 0

    def test_medium_has_fewer_than_hard(self):
        expert = self._make_expert_track()
        hard = derive_difficulty(expert, "HardSingle")
        medium = derive_difficulty(expert, "MediumSingle")
        assert len(medium.notes) < len(hard.notes)

    def test_easy_has_fewest(self):
        expert = self._make_expert_track()
        hard = derive_difficulty(expert, "HardSingle")
        medium = derive_difficulty(expert, "MediumSingle")
        easy = derive_difficulty(expert, "EasySingle")
        assert len(easy.notes) <= len(medium.notes)
        assert len(easy.notes) > 0

    def test_easy_only_single_notes_per_tick(self):
        """Easy should not have chords."""
        expert = self._make_expert_track()
        easy = derive_difficulty(expert, "EasySingle")
        ticks = [n.tick for n in easy.notes]
        # No duplicate ticks (no chords)
        assert len(ticks) == len(set(ticks))

    def test_frets_in_valid_range(self):
        expert = self._make_expert_track()
        for diff in ["HardSingle", "MediumSingle", "EasySingle"]:
            track = derive_difficulty(expert, diff)
            for n in track.notes:
                assert 0 <= n.fret <= 4

    def test_preserves_tick_order(self):
        expert = self._make_expert_track()
        for diff in ["HardSingle", "MediumSingle", "EasySingle"]:
            track = derive_difficulty(expert, diff)
            ticks = [n.tick for n in track.notes]
            assert ticks == sorted(ticks)

    def test_empty_expert(self):
        expert = ChartTrack("ExpertSingle")
        easy = derive_difficulty(expert, "EasySingle")
        assert len(easy.notes) == 0
