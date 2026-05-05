"""Tests for end-to-end chart generation pipeline."""
import os
import sys
import struct
import tempfile
import wave
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chart_generator.pipeline import generate_chart_from_audio
from chart_generator.output import SongMetadata
from chart_generator.chart import parse_chart_string

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _make_test_wav(path: str, duration: float = 10.0, sr: int = 22050):
    """Create a test WAV with a click track + tonal content."""
    n_samples = int(duration * sr)
    t = np.linspace(0, duration, n_samples, dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    beat_interval = 0.5  # 120 BPM
    click_len = int(0.01 * sr)
    time_pos = 0.0
    while time_pos < duration:
        idx = int(time_pos * sr)
        end = min(idx + click_len, n_samples)
        audio[idx:end] += 0.7 * np.sin(2 * np.pi * 1000 * np.arange(end - idx) / sr).astype(np.float32)
        time_pos += beat_interval
    # Write as 16-bit PCM WAV using stdlib wave
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        pcm = (audio * 32767).astype(np.int16)
        wf.writeframes(pcm.tobytes())


class TestEndToEndPipeline:
    def test_generates_valid_chart_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "test.wav")
            _make_test_wav(wav_path, duration=10.0)

            output_dir = os.path.join(tmpdir, "output")
            meta = SongMetadata(name="Test Song", artist="Test Artist")

            result_path = generate_chart_from_audio(
                audio_path=wav_path,
                metadata=meta,
                output_dir=output_dir,
            )

            assert os.path.isdir(result_path)
            assert os.path.isfile(os.path.join(result_path, "notes.chart"))
            assert os.path.isfile(os.path.join(result_path, "song.ini"))

    def test_chart_has_all_difficulties(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "test.wav")
            _make_test_wav(wav_path, duration=10.0)

            output_dir = os.path.join(tmpdir, "output")
            meta = SongMetadata(name="Test", artist="Artist")

            result_path = generate_chart_from_audio(
                audio_path=wav_path,
                metadata=meta,
                output_dir=output_dir,
            )

            chart_path = os.path.join(result_path, "notes.chart")
            with open(chart_path, "r", encoding="utf-8") as f:
                content = f.read()

            song, sync, events, tracks = parse_chart_string(content)
            track_names = [t.name for t in tracks]
            assert "ExpertSingle" in track_names
            assert "HardSingle" in track_names
            assert "MediumSingle" in track_names
            assert "EasySingle" in track_names

    def test_chart_has_notes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "test.wav")
            _make_test_wav(wav_path, duration=10.0)

            output_dir = os.path.join(tmpdir, "output")
            meta = SongMetadata(name="Test", artist="Artist")

            result_path = generate_chart_from_audio(
                audio_path=wav_path,
                metadata=meta,
                output_dir=output_dir,
            )

            chart_path = os.path.join(result_path, "notes.chart")
            with open(chart_path, "r", encoding="utf-8") as f:
                content = f.read()

            _, _, _, tracks = parse_chart_string(content)
            expert = [t for t in tracks if t.name == "ExpertSingle"][0]
            assert len(expert.notes) > 0

    def test_chart_bpm_reasonable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "test.wav")
            _make_test_wav(wav_path, duration=10.0)

            output_dir = os.path.join(tmpdir, "output")
            meta = SongMetadata(name="Test", artist="Artist")

            result_path = generate_chart_from_audio(
                audio_path=wav_path,
                metadata=meta,
                output_dir=output_dir,
            )

            chart_path = os.path.join(result_path, "notes.chart")
            with open(chart_path, "r", encoding="utf-8") as f:
                content = f.read()

            _, sync, _, _ = parse_chart_string(content)
            assert len(sync.bpm_events) >= 1
            # BPM should be in a reasonable range
            assert 30 < sync.bpm_events[0].bpm < 300

    def test_song_ini_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "test.wav")
            _make_test_wav(wav_path, duration=10.0)

            output_dir = os.path.join(tmpdir, "output")
            meta = SongMetadata(name="My Song", artist="My Artist", album="My Album")

            result_path = generate_chart_from_audio(
                audio_path=wav_path,
                metadata=meta,
                output_dir=output_dir,
            )

            ini_path = os.path.join(result_path, "song.ini")
            with open(ini_path, "r", encoding="utf-8") as f:
                ini_content = f.read()

            assert "name = My Song" in ini_content
            assert "artist = My Artist" in ini_content
            assert "album = My Album" in ini_content
