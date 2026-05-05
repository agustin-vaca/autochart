"""Tests for the audio analysis pipeline."""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chart_generator.audio import (
    detect_bpm,
    detect_onsets,
    AudioAnalysis,
    analyze_audio_array,
)


def _make_click_track(bpm: float, duration: float = 10.0, sr: int = 22050) -> np.ndarray:
    """Synthesize a simple click track at a given BPM for testing."""
    n_samples = int(duration * sr)
    audio = np.zeros(n_samples, dtype=np.float32)
    beat_interval = 60.0 / bpm
    click_len = int(0.01 * sr)  # 10ms click
    t = 0.0
    while t < duration:
        idx = int(t * sr)
        end = min(idx + click_len, n_samples)
        audio[idx:end] = 0.8 * np.sin(2 * np.pi * 1000 * np.arange(end - idx) / sr)
        t += beat_interval
    return audio


class TestDetectBPM:
    def test_120_bpm_click(self):
        audio = _make_click_track(120.0, duration=15.0)
        bpm, confidence = detect_bpm(audio, sr=22050)
        assert abs(bpm - 120.0) < 5.0, f"Expected ~120 BPM, got {bpm}"
        assert confidence > 0.0

    def test_90_bpm_click(self):
        audio = _make_click_track(90.0, duration=15.0)
        bpm, confidence = detect_bpm(audio, sr=22050)
        assert abs(bpm - 90.0) < 5.0, f"Expected ~90 BPM, got {bpm}"

    def test_returns_positive_bpm(self):
        audio = _make_click_track(140.0, duration=10.0)
        bpm, confidence = detect_bpm(audio, sr=22050)
        assert bpm > 0


class TestDetectOnsets:
    def test_finds_clicks(self):
        audio = _make_click_track(120.0, duration=5.0)
        onsets = detect_onsets(audio, sr=22050)
        # At 120 BPM for 5 seconds, expect ~10 beats
        assert len(onsets) >= 5
        assert len(onsets) <= 15

    def test_onset_times_are_sorted(self):
        audio = _make_click_track(100.0, duration=5.0)
        onsets = detect_onsets(audio, sr=22050)
        assert onsets == sorted(onsets)

    def test_onset_times_are_positive(self):
        audio = _make_click_track(120.0, duration=5.0)
        onsets = detect_onsets(audio, sr=22050)
        for t in onsets:
            assert t >= 0.0


class TestAnalyzeAudioArray:
    def test_returns_analysis(self):
        audio = _make_click_track(120.0, duration=10.0)
        result = analyze_audio_array(audio, sr=22050)
        assert isinstance(result, AudioAnalysis)
        assert result.bpm > 0
        assert len(result.onset_times) > 0
        assert result.duration_sec > 0
