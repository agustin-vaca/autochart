"""Tests for file output (song.ini, folder assembly)."""
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chart_generator.output import (
    SongMetadata,
    generate_song_ini,
    assemble_output_folder,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


class TestSongMetadata:
    def test_create(self):
        m = SongMetadata(
            name="Test Song",
            artist="Test Artist",
            album="Test Album",
            genre="Rock",
            year="2024",
            charter="AutoChart",
        )
        assert m.name == "Test Song"
        assert m.charter == "AutoChart"


class TestGenerateSongIni:
    def test_basic_output(self):
        m = SongMetadata(
            name="Test Song",
            artist="Test Artist",
            album="Test Album",
            genre="Rock",
            year="2024",
            charter="ChartHero",
        )
        ini = generate_song_ini(m, song_length_ms=180000)
        assert "[song]" in ini
        assert "name = Test Song" in ini
        assert "artist = Test Artist" in ini
        assert "album = Test Album" in ini
        assert "genre = Rock" in ini
        assert "year = 2024" in ini
        assert "charter = ChartHero" in ini
        assert "song_length = 180000" in ini

    def test_has_diff_guitar(self):
        m = SongMetadata(name="X", artist="Y")
        ini = generate_song_ini(m, song_length_ms=60000)
        assert "diff_guitar" in ini

    def test_matches_reference_format(self):
        """song.ini format should match what Clone Hero expects."""
        m = SongMetadata(name="X", artist="Y")
        ini = generate_song_ini(m, song_length_ms=60000)
        lines = ini.strip().split("\n")
        assert lines[0] == "[song]"
        # All data lines should be key = value format
        for line in lines[1:]:
            if line.strip():
                assert " = " in line


class TestAssembleOutputFolder:
    def test_creates_folder_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "Test Artist - Test Song (ChartHero)")
            chart_content = "[Song]\n{\n}\n"
            ini_content = "[song]\nname = Test\n"
            # Create a tiny test audio file
            audio_path = os.path.join(tmpdir, "test.mp3")
            with open(audio_path, "wb") as f:
                f.write(b"\x00" * 100)

            assemble_output_folder(
                output_dir=output_path,
                chart_content=chart_content,
                ini_content=ini_content,
                audio_source_path=audio_path,
                album_art_path=None,
            )

            assert os.path.isdir(output_path)
            assert os.path.isfile(os.path.join(output_path, "notes.chart"))
            assert os.path.isfile(os.path.join(output_path, "song.ini"))

    def test_copies_album_art(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output")
            chart_content = "[Song]\n{\n}\n"
            ini_content = "[song]\nname = Test\n"
            audio_path = os.path.join(tmpdir, "test.mp3")
            with open(audio_path, "wb") as f:
                f.write(b"\x00" * 100)
            art_path = os.path.join(tmpdir, "cover.jpg")
            with open(art_path, "wb") as f:
                f.write(b"\xff\xd8" + b"\x00" * 100)

            assemble_output_folder(
                output_dir=output_path,
                chart_content=chart_content,
                ini_content=ini_content,
                audio_source_path=audio_path,
                album_art_path=art_path,
            )

            assert os.path.isfile(os.path.join(output_path, "album.jpg"))
