# AutoChart 🎸

AI-powered Clone Hero chart generator. Feed it a song, get a playable chart.

## What it does

Takes an audio file (MP3, OGG, WAV) and generates a complete Clone Hero-compatible chart folder with:
- **notes.chart** — all 4 difficulties (Easy/Medium/Hard/Expert)
- **song.ini** — song metadata
- **preview.html** — interactive note highway preview with audio playback
- Audio file copied to output

## Quick Start

```bash
# Install dependencies
pip install numpy scipy

# Generate a chart
python cli.py "song.mp3" --name "Song Name" --artist "Artist" --album "Album"
```

The chart folder is created and a preview opens in your browser. Press **Space** to play.

## Preview Features

- 🎵 Audio playback synced to scrolling note highway
- 🎸 Clone Hero-style colored fret gems (green/red/yellow/blue/orange)
- 📊 Difficulty selector (Easy/Medium/Hard/Expert)
- ⚡ Adjustable scroll speed (1x–4x)
- ⌨️ Keyboard: Space = play/pause, ←/→ = seek ±5s

## CLI Options

```
python cli.py <audio_file> [options]

Required:
  audio               Path to audio file (MP3, OGG, WAV)
  --name NAME         Song name
  --artist ARTIST     Artist name

Optional:
  --album ALBUM       Album name
  --genre GENRE       Genre (default: Rock)
  --year YEAR         Year
  --charter CHARTER   Charter name (default: AutoChart)
  --art ART           Path to album art image
  --output DIR        Output directory (default: current dir)
  --no-preview        Skip opening preview in browser
  --no-separation     Skip Demucs source separation
  --separation-model  Demucs model (default: htdemucs)
```

## How It Works

1. **BPM Detection** — autocorrelation-based tempo estimation with octave correction
2. **Onset Detection** — spectral flux analysis to find note attack points
3. **Fret Mapping** — gameplay heuristics: pitch contour preservation, hand-position consistency, jump limiting
4. **Difficulty Derivation** — rhythmic backbone reduction (Expert → Hard → Medium → Easy)
5. **Chart Generation** — outputs valid `.chart` format readable by Clone Hero

## Dependencies

All free and open-source:
- Python 3.10+
- NumPy (BSD)
- SciPy (BSD)
- FFmpeg (LGPL) — for MP3/OGG input

## Project Structure

```
chart_generator/
├── audio.py       # BPM & onset detection (scipy/numpy)
├── mapper.py      # MIDI→5-fret mapping with gameplay heuristics
├── chart.py       # .chart file data structures, generation & parsing
├── difficulty.py  # Derive Easy/Medium/Hard from Expert
├── output.py      # song.ini generation & folder assembly
├── preview.py     # HTML preview with note highway & audio playback
└── pipeline.py    # End-to-end: audio → Clone Hero folder
tests/             # 74 tests covering all modules
cli.py             # Command-line interface
```

## Roadmap

- [x] Guitar source separation (Demucs) for better note detection
- [ ] Real audio-to-MIDI (Basic Pitch) instead of onset-based mapping
- [ ] Desktop app (Tauri)
- [ ] Optional cloud AI refinement (bring your own API key)
- [ ] .sng file packaging
- [ ] Chart editor/review UI

## License

MIT
