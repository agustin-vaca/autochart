"""Tests for Demucs source separation module."""
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chart_generator.separator import (
    get_guitar_stem_path,
    separate_guitar,
    separate_stems,
    is_available,
)


class TestGetGuitarStemPath:
    def test_prefers_guitar_stem(self):
        stems = {"guitar": "/tmp/guitar.wav", "other": "/tmp/other.wav", "drums": "/tmp/drums.wav"}
        assert get_guitar_stem_path(stems) == "/tmp/guitar.wav"

    def test_falls_back_to_other(self):
        stems = {"vocals": "/tmp/vocals.wav", "other": "/tmp/other.wav", "drums": "/tmp/drums.wav"}
        assert get_guitar_stem_path(stems) == "/tmp/other.wav"

    def test_returns_none_if_no_match(self):
        stems = {"vocals": "/tmp/vocals.wav", "drums": "/tmp/drums.wav"}
        assert get_guitar_stem_path(stems) is None

    def test_empty_dict(self):
        assert get_guitar_stem_path({}) is None


class TestSeparateGuitar:
    @patch("chart_generator.separator.DEMUCS_AVAILABLE", False)
    def test_returns_none_when_demucs_unavailable(self):
        result = separate_guitar("/some/audio.wav", "/tmp/output")
        assert result is None

    @patch("chart_generator.separator.DEMUCS_AVAILABLE", True)
    @patch("chart_generator.separator.separate_stems")
    def test_returns_guitar_path_on_success(self, mock_separate):
        mock_separate.return_value = {
            "vocals": "/tmp/stems/vocals.wav",
            "drums": "/tmp/stems/drums.wav",
            "bass": "/tmp/stems/bass.wav",
            "other": "/tmp/stems/other.wav",
        }
        result = separate_guitar("/input/song.mp3", "/tmp/stems")
        assert result == "/tmp/stems/other.wav"
        mock_separate.assert_called_once_with("/input/song.mp3", "/tmp/stems", None)

    @patch("chart_generator.separator.DEMUCS_AVAILABLE", True)
    @patch("chart_generator.separator.separate_stems")
    def test_returns_none_on_runtime_error(self, mock_separate):
        mock_separate.side_effect = RuntimeError("Demucs crashed")
        result = separate_guitar("/input/song.mp3", "/tmp/stems")
        assert result is None

    @patch("chart_generator.separator.DEMUCS_AVAILABLE", True)
    @patch("chart_generator.separator.separate_stems")
    def test_passes_model_name(self, mock_separate):
        mock_separate.return_value = {"guitar": "/tmp/stems/guitar.wav"}
        result = separate_guitar("/input/song.mp3", "/tmp/stems", "htdemucs_6src")
        mock_separate.assert_called_once_with("/input/song.mp3", "/tmp/stems", "htdemucs_6src")
        assert result == "/tmp/stems/guitar.wav"


class TestSeparateStems:
    @patch("chart_generator.separator.DEMUCS_AVAILABLE", False)
    def test_raises_when_demucs_unavailable(self):
        with pytest.raises(RuntimeError, match="Demucs is not installed"):
            separate_stems("/some/audio.wav", "/tmp/output")


class TestIsAvailable:
    @patch("chart_generator.separator.DEMUCS_AVAILABLE", True)
    def test_returns_true_when_available(self):
        assert is_available() is True

    @patch("chart_generator.separator.DEMUCS_AVAILABLE", False)
    def test_returns_false_when_unavailable(self):
        assert is_available() is False
