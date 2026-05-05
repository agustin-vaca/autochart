"""Audio analysis: BPM detection, onset detection using scipy/numpy (no librosa/numba)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import signal


@dataclass
class AudioAnalysis:
    bpm: float
    bpm_confidence: float
    onset_times: list[float]  # seconds
    duration_sec: float
    sr: int


def _onset_envelope(audio: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """Compute onset strength envelope using spectral flux."""
    # Short-time Fourier transform
    nperseg = 2048
    _, _, stft = signal.stft(audio, fs=sr, nperseg=nperseg, noverlap=nperseg - hop_length)
    mag = np.abs(stft)

    # Spectral flux: half-wave rectified difference
    flux = np.diff(mag, axis=1)
    flux = np.maximum(0, flux)
    envelope = flux.sum(axis=0)

    return envelope


def detect_bpm(audio: np.ndarray, sr: int = 22050) -> tuple[float, float]:
    """Detect BPM using autocorrelation of onset envelope.

    Returns (bpm, confidence) where confidence is 0-1.
    """
    hop_length = 512
    envelope = _onset_envelope(audio, sr, hop_length)

    if len(envelope) < 2:
        return 120.0, 0.0

    # Normalize
    envelope = envelope - envelope.mean()
    if envelope.std() > 0:
        envelope = envelope / envelope.std()

    # Autocorrelation
    corr = np.correlate(envelope, envelope, mode="full")
    corr = corr[len(corr) // 2:]  # Keep positive lags only

    # Convert lag range to BPM range (30-300 BPM)
    fps = sr / hop_length
    min_lag = int(fps * 60 / 300)  # 300 BPM
    max_lag = int(fps * 60 / 30)   # 30 BPM
    max_lag = min(max_lag, len(corr) - 1)

    if min_lag >= max_lag or max_lag <= 0:
        return 120.0, 0.0

    search = corr[min_lag:max_lag + 1]
    best_idx = np.argmax(search) + min_lag

    bpm = 60.0 * fps / best_idx

    # Octave correction: if BPM is suspiciously low, check if doubling is stronger
    if bpm < 80 and best_idx > 1:
        half_lag = best_idx // 2
        if min_lag <= half_lag < len(corr) and corr[half_lag] > search[best_idx - min_lag] * 0.7:
            bpm = bpm * 2
            best_idx = half_lag

    confidence = float(search[best_idx - min_lag] / corr[0]) if corr[0] > 0 and min_lag <= best_idx <= max_lag else 0.0
    confidence = max(0.0, min(1.0, confidence))

    return float(bpm), confidence


def detect_onsets(audio: np.ndarray, sr: int = 22050) -> list[float]:
    """Detect onset times in seconds using peak picking on onset envelope."""
    hop_length = 512
    envelope = _onset_envelope(audio, sr, hop_length)

    if len(envelope) < 3:
        return []

    # Adaptive threshold: mean + 1 std
    threshold = envelope.mean() + envelope.std() * 0.5

    # Peak picking
    peaks, properties = signal.find_peaks(envelope, height=threshold, distance=3)

    # Convert frame indices to time
    fps = sr / hop_length
    onset_times = [float(p / fps) for p in peaks]

    return onset_times


def analyze_audio_array(audio: np.ndarray, sr: int = 22050) -> AudioAnalysis:
    """Run full audio analysis on a numpy array."""
    # Ensure mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    duration_sec = len(audio) / sr
    bpm, confidence = detect_bpm(audio, sr)
    onsets = detect_onsets(audio, sr)

    return AudioAnalysis(
        bpm=bpm,
        bpm_confidence=confidence,
        onset_times=onsets,
        duration_sec=duration_sec,
        sr=sr,
    )
