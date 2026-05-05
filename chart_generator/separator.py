"""Audio source separation using Demucs for guitar isolation."""
from __future__ import annotations
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEMUCS_AVAILABLE = False
try:
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    DEMUCS_AVAILABLE = True
except ImportError:
    pass


def is_available() -> bool:
    """Check if Demucs source separation is available."""
    return DEMUCS_AVAILABLE


def separate_stems(
    audio_path: str,
    output_dir: str,
    model_name: str | None = None,
) -> dict[str, str]:
    """Separate audio into stems using Demucs.

    Args:
        audio_path: Path to the input audio file.
        output_dir: Directory to write separated stem WAV files.
        model_name: Demucs model name. Defaults to 'htdemucs'.
            Use 'htdemucs_6src' for explicit guitar stem.

    Returns:
        Dict mapping stem name to output WAV path.
        E.g. {"drums": "/tmp/.../drums.wav", "bass": "...", ...}

    Raises:
        RuntimeError: If Demucs is not available or separation fails.
    """
    if not DEMUCS_AVAILABLE:
        raise RuntimeError(
            "Demucs is not installed. Install with: pip install demucs torch torchaudio"
        )

    if model_name is None:
        model_name = "htdemucs"

    os.makedirs(output_dir, exist_ok=True)

    try:
        model = get_model(model_name)
        model.eval()

        # Load audio: use ffmpeg CLI to convert to WAV, then read with scipy
        # This avoids torchaudio.load() which may require torchcodec/shared FFmpeg DLLs
        import subprocess
        import tempfile as _tempfile
        import wave as _wave

        tmp_wav = _tempfile.mktemp(suffix=".wav", dir=output_dir)
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path, "-ar", str(model.samplerate),
                 "-ac", "2", "-sample_fmt", "s16", tmp_wav],
                capture_output=True, check=True,
            )
            with _wave.open(tmp_wav, "r") as wf:
                n_channels = wf.getnchannels()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)

            audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            audio_np = audio_np.reshape(-1, n_channels).T  # shape: (channels, samples)
            waveform = torch.from_numpy(audio_np)
            sr = model.samplerate
        finally:
            if os.path.exists(tmp_wav):
                os.unlink(tmp_wav)

        # Resample to model's sample rate if needed
        if sr != model.samplerate:
            waveform = torchaudio.functional.resample(waveform, sr, model.samplerate)

        # Ensure stereo (model expects 2 channels)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]

        # Add batch dimension
        ref = waveform.mean(0)
        waveform = (waveform - ref.mean()) / ref.std()
        sources = apply_model(model, waveform[None], device="cpu")[0]
        sources = sources * ref.std() + ref.mean()

    except Exception as e:
        raise RuntimeError(f"Demucs separation failed: {e}") from e

    # Save each stem as WAV using scipy
    import scipy.io.wavfile as _wavfile

    stem_paths: dict[str, str] = {}
    for i, stem_name in enumerate(model.sources):
        stem_path = os.path.join(output_dir, f"{stem_name}.wav")
        stem_np = sources[i].cpu().numpy()
        # Convert from (channels, samples) to (samples, channels) for scipy
        if stem_np.ndim == 2:
            stem_np = stem_np.T
        # Clip and convert to int16
        stem_np = np.clip(stem_np, -1.0, 1.0)
        stem_int16 = (stem_np * 32767).astype(np.int16)
        _wavfile.write(stem_path, model.samplerate, stem_int16)
        stem_paths[stem_name] = stem_path

    return stem_paths


def get_guitar_stem_path(stem_paths: dict[str, str]) -> str | None:
    """Find the best guitar stem from separation output.

    Prefers explicit 'guitar' stem (htdemucs_6src), falls back to 'other'
    stem (htdemucs) which typically contains guitar.

    Returns:
        Path to the guitar/other stem WAV, or None if not found.
    """
    if "guitar" in stem_paths:
        return stem_paths["guitar"]
    if "other" in stem_paths:
        return stem_paths["other"]
    return None


def separate_guitar(
    audio_path: str,
    output_dir: str,
    model_name: str | None = None,
) -> str | None:
    """High-level: separate audio and return path to guitar stem.

    Args:
        audio_path: Path to the input audio file.
        output_dir: Directory to write stem files.
        model_name: Demucs model name (default: 'htdemucs').

    Returns:
        Path to guitar stem WAV, or None if separation unavailable/failed.
        On unexpected failure, logs a warning and returns None.
    """
    if not DEMUCS_AVAILABLE:
        logger.info("Demucs not available, skipping source separation.")
        return None

    try:
        stem_paths = separate_stems(audio_path, output_dir, model_name)
    except RuntimeError as e:
        logger.warning("Source separation failed: %s", e)
        return None

    guitar_path = get_guitar_stem_path(stem_paths)
    if guitar_path is None:
        logger.warning(
            "No guitar/other stem found in separation output. "
            "Available stems: %s", list(stem_paths.keys())
        )
    return guitar_path
