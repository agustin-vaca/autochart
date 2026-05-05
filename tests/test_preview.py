"""Tests for the HTML chart preview generator."""
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chart_generator.chart import (
    ChartSong, SyncTrack, ChartTrack, SectionEvent, build_chart_string, parse_chart_string,
)
from chart_generator.preview import generate_preview_html, chart_to_preview_data

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


class TestChartToPreviewData:
    def _make_chart(self):
        song = ChartSong(name="Test", artist="Tester")
        sync = SyncTrack()
        sync.add_bpm(0, 120.0)
        sync.add_time_signature(0, 4)
        track = ChartTrack("ExpertSingle")
        track.add_note(0, 0, 0)
        track.add_note(192, 1, 0)
        track.add_note(384, 2, 96)
        track.add_note(384, 3, 0)  # chord
        events = [SectionEvent(tick=0, name="Intro")]
        return song, sync, events, [track]

    def test_returns_dict(self):
        data = chart_to_preview_data(*self._make_chart())
        assert isinstance(data, dict)

    def test_has_song_info(self):
        data = chart_to_preview_data(*self._make_chart())
        assert data["song"]["name"] == "Test"
        assert data["song"]["artist"] == "Tester"
        assert data["song"]["resolution"] == 192

    def test_has_bpm(self):
        data = chart_to_preview_data(*self._make_chart())
        assert len(data["syncTrack"]["bpmEvents"]) == 1
        assert data["syncTrack"]["bpmEvents"][0]["bpm"] == 120.0

    def test_has_notes(self):
        data = chart_to_preview_data(*self._make_chart())
        assert len(data["tracks"]) >= 1
        expert = [t for t in data["tracks"] if t["name"] == "ExpertSingle"][0]
        assert len(expert["notes"]) == 4

    def test_notes_have_time_seconds(self):
        data = chart_to_preview_data(*self._make_chart())
        expert = [t for t in data["tracks"] if t["name"] == "ExpertSingle"][0]
        for note in expert["notes"]:
            assert "timeSec" in note
            assert isinstance(note["timeSec"], float)
            assert note["timeSec"] >= 0.0

    def test_has_sections(self):
        data = chart_to_preview_data(*self._make_chart())
        assert len(data["events"]) >= 1
        assert data["events"][0]["name"] == "Intro"

    def test_has_duration(self):
        data = chart_to_preview_data(*self._make_chart())
        assert data["durationSec"] > 0


class TestGeneratePreviewHtml:
    def _make_chart(self):
        song = ChartSong(name="Preview Test", artist="Artist")
        sync = SyncTrack()
        sync.add_bpm(0, 120.0)
        sync.add_time_signature(0, 4)
        track = ChartTrack("ExpertSingle")
        for i in range(20):
            track.add_note(i * 192, i % 5, 0)
        events = [SectionEvent(tick=0, name="Intro")]
        return song, sync, events, [track]

    def test_generates_html_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            preview_path = os.path.join(tmpdir, "preview.html")
            generate_preview_html(
                *self._make_chart(),
                audio_path="song.mp3",
                output_path=preview_path,
            )
            assert os.path.isfile(preview_path)

    def test_html_contains_canvas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            preview_path = os.path.join(tmpdir, "preview.html")
            generate_preview_html(
                *self._make_chart(),
                audio_path="song.mp3",
                output_path=preview_path,
            )
            with open(preview_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "<canvas" in content

    def test_html_contains_audio_ref(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            preview_path = os.path.join(tmpdir, "preview.html")
            generate_preview_html(
                *self._make_chart(),
                audio_path="song.mp3",
                output_path=preview_path,
            )
            with open(preview_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "song.mp3" in content

    def test_html_contains_chart_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            preview_path = os.path.join(tmpdir, "preview.html")
            generate_preview_html(
                *self._make_chart(),
                audio_path="song.mp3",
                output_path=preview_path,
            )
            with open(preview_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "Preview Test" in content
            assert "ExpertSingle" in content

    def test_html_contains_playback_controls(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            preview_path = os.path.join(tmpdir, "preview.html")
            generate_preview_html(
                *self._make_chart(),
                audio_path="song.mp3",
                output_path=preview_path,
            )
            with open(preview_path, "r", encoding="utf-8") as f:
                content = f.read()
            # Should have play/pause functionality
            assert "play" in content.lower()
            assert "pause" in content.lower()

    def test_html_contains_difficulty_selector(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            preview_path = os.path.join(tmpdir, "preview.html")
            song, sync, events, _ = self._make_chart()
            # Make multiple difficulty tracks
            tracks = []
            for name in ["EasySingle", "MediumSingle", "HardSingle", "ExpertSingle"]:
                t = ChartTrack(name)
                for i in range(5):
                    t.add_note(i * 192, i % 5, 0)
                tracks.append(t)
            generate_preview_html(
                song, sync, events, tracks,
                audio_path="song.mp3",
                output_path=preview_path,
            )
            with open(preview_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "Easy" in content
            assert "Expert" in content

    def test_from_chart_file(self):
        """Test generating preview from the reference chart file."""
        ref_path = os.path.join(FIXTURES, "reference.chart")
        with open(ref_path, "r", encoding="utf-8-sig") as f:
            content = f.read()
        song, sync, events, tracks = parse_chart_string(content)

        with tempfile.TemporaryDirectory() as tmpdir:
            preview_path = os.path.join(tmpdir, "preview.html")
            generate_preview_html(
                song, sync, events, tracks,
                audio_path="song.opus",
                output_path=preview_path,
            )
            assert os.path.isfile(preview_path)
            with open(preview_path, "r", encoding="utf-8") as f:
                html = f.read()
            assert "Ode To The Mets" in html
            assert len(html) > 1000  # Non-trivial HTML
