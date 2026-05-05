"""Tests for .chart file generation and parsing."""
import os
import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chart_generator.chart import (
    ChartSong,
    SyncTrack,
    NoteEvent,
    SectionEvent,
    ChartTrack,
    build_chart_string,
    parse_chart_string,
    BPMEvent,
    TSEvent,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


class TestChartDataStructures:
    def test_bpm_event_format(self):
        e = BPMEvent(tick=0, bpm=120.0)
        assert e.to_chart_line() == "  0 = B 120000"

    def test_bpm_event_fractional(self):
        e = BPMEvent(tick=768, bpm=126.663)
        assert e.to_chart_line() == "  768 = B 126663"

    def test_ts_event_default(self):
        e = TSEvent(tick=0, numerator=4)
        assert e.to_chart_line() == "  0 = TS 4"

    def test_ts_event_with_denominator(self):
        e = TSEvent(tick=8448, numerator=9, denominator_exp=3)
        assert e.to_chart_line() == "  8448 = TS 9 3"

    def test_note_event_format(self):
        n = NoteEvent(tick=768, fret=0, duration=0)
        assert n.to_chart_line() == "  768 = N 0 0"

    def test_note_event_with_duration(self):
        n = NoteEvent(tick=1000, fret=3, duration=192)
        assert n.to_chart_line() == "  1000 = N 3 192"

    def test_section_event_format(self):
        s = SectionEvent(tick=768, name="Intro1")
        assert s.to_chart_line() == '  768 = E "section Intro1"'

    def test_note_fret_valid_range(self):
        for fret in range(8):  # 0-7 valid
            n = NoteEvent(tick=0, fret=fret, duration=0)
            assert n.fret == fret

    def test_note_fret_invalid(self):
        with pytest.raises(ValueError):
            NoteEvent(tick=0, fret=8, duration=0)

    def test_note_fret_negative(self):
        with pytest.raises(ValueError):
            NoteEvent(tick=0, fret=-1, duration=0)


class TestChartSong:
    def test_create_song_metadata(self):
        song = ChartSong(
            name="Test Song",
            artist="Test Artist",
            resolution=192,
        )
        assert song.name == "Test Song"
        assert song.artist == "Test Artist"
        assert song.resolution == 192

    def test_default_resolution(self):
        song = ChartSong(name="X", artist="Y")
        assert song.resolution == 192


class TestSyncTrack:
    def test_add_bpm(self):
        st = SyncTrack()
        st.add_bpm(0, 120.0)
        st.add_bpm(768, 130.0)
        assert len(st.bpm_events) == 2

    def test_add_ts(self):
        st = SyncTrack()
        st.add_time_signature(0, 4)
        assert len(st.ts_events) == 1

    def test_to_chart_section(self):
        st = SyncTrack()
        st.add_time_signature(0, 4)
        st.add_bpm(0, 120.0)
        text = st.to_chart_section()
        assert "[SyncTrack]" in text
        assert "0 = TS 4" in text
        assert "0 = B 120000" in text


class TestChartTrack:
    def test_add_notes(self):
        track = ChartTrack("ExpertSingle")
        track.add_note(768, 0, 0)
        track.add_note(768, 2, 0)
        assert len(track.notes) == 2

    def test_to_chart_section(self):
        track = ChartTrack("ExpertSingle")
        track.add_note(768, 0, 0)
        text = track.to_chart_section()
        assert "[ExpertSingle]" in text
        assert "768 = N 0 0" in text

    def test_notes_sorted_by_tick(self):
        track = ChartTrack("ExpertSingle")
        track.add_note(960, 1, 0)
        track.add_note(768, 0, 0)
        text = track.to_chart_section()
        lines = text.strip().split("\n")
        note_lines = [l for l in lines if "= N" in l]
        assert "768" in note_lines[0]
        assert "960" in note_lines[1]


class TestBuildChartString:
    def test_minimal_chart(self):
        song = ChartSong(name="Test", artist="Tester")
        sync = SyncTrack()
        sync.add_bpm(0, 120.0)
        sync.add_time_signature(0, 4)
        track = ChartTrack("ExpertSingle")
        track.add_note(0, 0, 0)

        result = build_chart_string(song, sync, [], [track])

        assert "[Song]" in result
        assert '"Test"' in result
        assert "[SyncTrack]" in result
        assert "[Events]" in result
        assert "[ExpertSingle]" in result

    def test_chart_has_all_difficulty_sections(self):
        song = ChartSong(name="T", artist="A")
        sync = SyncTrack()
        sync.add_bpm(0, 120.0)
        sync.add_time_signature(0, 4)

        tracks = [
            ChartTrack("EasySingle"),
            ChartTrack("MediumSingle"),
            ChartTrack("HardSingle"),
            ChartTrack("ExpertSingle"),
        ]
        for t in tracks:
            t.add_note(0, 0, 0)

        result = build_chart_string(song, sync, [], tracks)
        assert "[EasySingle]" in result
        assert "[MediumSingle]" in result
        assert "[HardSingle]" in result
        assert "[ExpertSingle]" in result


class TestParseChartString:
    def test_parse_reference_chart(self):
        path = os.path.join(FIXTURES, "reference.chart")
        with open(path, "r", encoding="utf-8-sig") as f:
            content = f.read()
        song, sync, events, tracks = parse_chart_string(content)
        assert song.name == "Ode To The Mets"
        assert song.artist == "The Strokes"
        assert song.resolution == 192
        assert len(sync.bpm_events) > 0
        assert any(t.name == "ExpertSingle" for t in tracks)

    def test_roundtrip_minimal(self):
        song = ChartSong(name="Roundtrip", artist="Test")
        sync = SyncTrack()
        sync.add_bpm(0, 110.0)
        sync.add_time_signature(0, 4)
        track = ChartTrack("ExpertSingle")
        track.add_note(0, 0, 0)
        track.add_note(192, 1, 0)
        track.add_note(384, 2, 96)

        text = build_chart_string(song, sync, [], [track])
        song2, sync2, events2, tracks2 = parse_chart_string(text)

        assert song2.name == "Roundtrip"
        assert song2.artist == "Test"
        assert len(sync2.bpm_events) == 1
        assert sync2.bpm_events[0].bpm == 110.0
        assert len(tracks2) == 1
        assert len(tracks2[0].notes) == 3
