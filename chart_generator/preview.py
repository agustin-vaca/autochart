"""Generate an HTML preview with a scrolling note highway synced to audio playback."""
from __future__ import annotations
import json
import os
from typing import Optional

from chart_generator.chart import ChartSong, SyncTrack, ChartTrack, SectionEvent


def _tick_to_seconds(tick: int, bpm_events: list[dict], resolution: int) -> float:
    """Convert a tick position to seconds using BPM map."""
    if not bpm_events:
        return 0.0

    time_sec = 0.0
    prev_tick = 0
    prev_bpm = bpm_events[0]["bpm"]

    for ev in bpm_events:
        if ev["tick"] > tick:
            break
        dt_ticks = ev["tick"] - prev_tick
        time_sec += dt_ticks / resolution * 60.0 / prev_bpm
        prev_tick = ev["tick"]
        prev_bpm = ev["bpm"]

    dt_ticks = tick - prev_tick
    time_sec += dt_ticks / resolution * 60.0 / prev_bpm
    return time_sec


def chart_to_preview_data(
    song: ChartSong,
    sync: SyncTrack,
    events: list[SectionEvent],
    tracks: list[ChartTrack],
) -> dict:
    """Convert chart data structures into a JSON-serializable dict for the preview."""
    bpm_data = [{"tick": e.tick, "bpm": e.bpm} for e in sync.bpm_events]

    track_data = []
    max_tick = 0
    for track in tracks:
        notes = []
        for n in sorted(track.notes, key=lambda x: (x.tick, x.fret)):
            t_sec = _tick_to_seconds(n.tick, bpm_data, song.resolution)
            dur_sec = _tick_to_seconds(n.tick + n.duration, bpm_data, song.resolution) - t_sec
            notes.append({
                "tick": n.tick,
                "fret": n.fret,
                "duration": n.duration,
                "timeSec": t_sec,
                "durationSec": max(dur_sec, 0.05),
            })
            max_tick = max(max_tick, n.tick + n.duration)
        track_data.append({"name": track.name, "notes": notes})

    event_data = []
    for e in events:
        event_data.append({
            "tick": e.tick,
            "name": e.name,
            "timeSec": _tick_to_seconds(e.tick, bpm_data, song.resolution),
        })

    duration_sec = _tick_to_seconds(max_tick, bpm_data, song.resolution) + 2.0

    return {
        "song": {
            "name": song.name,
            "artist": song.artist,
            "resolution": song.resolution,
        },
        "syncTrack": {
            "bpmEvents": bpm_data,
            "tsEvents": [{"tick": e.tick, "numerator": e.numerator} for e in sync.ts_events],
        },
        "tracks": track_data,
        "events": event_data,
        "durationSec": duration_sec,
    }


_PREVIEW_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AutoChart Preview — {song_name} by {artist}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  background: #0a0a0f;
  color: #e0e0e0;
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  overflow: hidden;
  height: 100vh;
}}
#header {{
  position: fixed; top: 0; left: 0; right: 0; z-index: 10;
  background: linear-gradient(180deg, rgba(10,10,15,0.97) 80%, rgba(10,10,15,0));
  padding: 12px 24px;
  display: flex; align-items: center; gap: 20px;
}}
#header h1 {{
  font-size: 18px; font-weight: 600;
  background: linear-gradient(90deg, #00e676, #00bcd4);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
#header .meta {{ font-size: 13px; color: #888; }}
#controls {{
  display: flex; align-items: center; gap: 12px; margin-left: auto;
}}
button {{
  background: #1a1a2e; border: 1px solid #333; color: #e0e0e0;
  padding: 6px 16px; border-radius: 6px; cursor: pointer; font-size: 13px;
  transition: all 0.15s;
}}
button:hover {{ background: #2a2a4e; border-color: #555; }}
button.active {{ background: #00e676; color: #000; border-color: #00e676; font-weight: 600; }}
select {{
  background: #1a1a2e; border: 1px solid #333; color: #e0e0e0;
  padding: 5px 10px; border-radius: 6px; font-size: 13px;
}}
#time-display {{
  font-family: 'Cascadia Code', 'Consolas', monospace;
  font-size: 14px; color: #aaa; min-width: 100px;
}}
#progress-bar {{
  position: fixed; bottom: 0; left: 0; right: 0; height: 4px;
  background: #1a1a2e; z-index: 10; cursor: pointer;
}}
#progress-fill {{
  height: 100%; background: linear-gradient(90deg, #00e676, #00bcd4);
  width: 0%; transition: width 0.1s linear;
}}
#section-label {{
  position: fixed; top: 60px; left: 50%; transform: translateX(-50%);
  font-size: 14px; color: #666; z-index: 5; font-style: italic;
}}
canvas {{
  display: block; width: 100vw; height: 100vh;
}}
#speed-control {{
  display: flex; align-items: center; gap: 6px;
}}
#speed-control label {{ font-size: 12px; color: #888; }}
</style>
</head>
<body>
<div id="header">
  <h1>♫ AutoChart Preview</h1>
  <span class="meta">{song_name} — {artist}</span>
  <div id="controls">
    <select id="difficulty">
      {difficulty_options}
    </select>
    <div id="speed-control">
      <label>Speed</label>
      <select id="speed">
        <option value="4">4x</option>
        <option value="3">3x</option>
        <option value="2" selected>2x</option>
        <option value="1.5">1.5x</option>
        <option value="1">1x</option>
      </select>
    </div>
    <button id="playBtn" onclick="togglePlay()">▶ Play</button>
    <button onclick="restart()">⟲ Restart</button>
    <span id="time-display">0:00 / 0:00</span>
  </div>
</div>
<div id="section-label"></div>
<canvas id="highway"></canvas>
<div id="progress-bar" onclick="seek(event)">
  <div id="progress-fill"></div>
</div>
<audio id="audio" src="{audio_path}" preload="auto"></audio>

<script>
const CHART_DATA = {chart_json};

const FRET_COLORS = [
  '#00e676', // Green
  '#f44336', // Red
  '#ffeb3b', // Yellow
  '#2196f3', // Blue
  '#ff9800', // Orange
];
const FRET_GLOW = [
  'rgba(0,230,118,0.4)',
  'rgba(244,67,54,0.4)',
  'rgba(255,235,59,0.4)',
  'rgba(33,150,243,0.4)',
  'rgba(255,152,0,0.4)',
];

const canvas = document.getElementById('highway');
const ctx = canvas.getContext('2d');
const audio = document.getElementById('audio');
const playBtn = document.getElementById('playBtn');
const diffSelect = document.getElementById('difficulty');
const speedSelect = document.getElementById('speed');
const timeDisplay = document.getElementById('time-display');
const progressFill = document.getElementById('progress-fill');
const sectionLabel = document.getElementById('section-label');

let playing = false;
let currentTrack = CHART_DATA.tracks.find(t => t.name === 'ExpertSingle') || CHART_DATA.tracks[0];
let highwaySpeed = 2; // seconds visible ahead of strike line

function resize() {{
  canvas.width = window.innerWidth * devicePixelRatio;
  canvas.height = window.innerHeight * devicePixelRatio;
  ctx.scale(devicePixelRatio, devicePixelRatio);
}}
window.addEventListener('resize', resize);
resize();

diffSelect.addEventListener('change', () => {{
  const name = diffSelect.value;
  currentTrack = CHART_DATA.tracks.find(t => t.name === name) || currentTrack;
}});

speedSelect.addEventListener('change', () => {{
  highwaySpeed = parseFloat(speedSelect.value);
}});

function togglePlay() {{
  if (playing) {{
    audio.pause();
    playing = false;
    playBtn.textContent = '▶ Play';
    playBtn.classList.remove('active');
  }} else {{
    audio.play();
    playing = true;
    playBtn.textContent = '⏸ Pause';
    playBtn.classList.add('active');
  }}
}}

function restart() {{
  audio.currentTime = 0;
  if (!playing) togglePlay();
}}

function seek(e) {{
  const rect = e.target.getBoundingClientRect();
  const pct = (e.clientX - rect.left) / rect.width;
  audio.currentTime = pct * audio.duration;
}}

function formatTime(sec) {{
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return m + ':' + s.toString().padStart(2, '0');
}}

function getCurrentSection(time) {{
  let current = '';
  for (const ev of CHART_DATA.events) {{
    if (ev.timeSec <= time) current = ev.name;
    else break;
  }}
  return current;
}}

function draw() {{
  const W = window.innerWidth;
  const H = window.innerHeight;
  const now = audio.currentTime || 0;
  const dur = audio.duration || CHART_DATA.durationSec;

  ctx.clearRect(0, 0, W, H);

  // Highway dimensions
  const laneCount = 5;
  const highwayWidth = Math.min(400, W * 0.5);
  const laneWidth = highwayWidth / laneCount;
  const hx = (W - highwayWidth) / 2; // highway left edge
  const strikeY = H * 0.85; // strike line position
  const topY = 80;
  const visibleHeight = strikeY - topY;

  // Draw highway background
  ctx.fillStyle = 'rgba(20, 20, 35, 0.85)';
  ctx.fillRect(hx - 10, topY, highwayWidth + 20, visibleHeight + 40);

  // Lane dividers
  ctx.strokeStyle = 'rgba(255,255,255,0.07)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= laneCount; i++) {{
    const x = hx + i * laneWidth;
    ctx.beginPath();
    ctx.moveTo(x, topY);
    ctx.lineTo(x, strikeY + 30);
    ctx.stroke();
  }}

  // Beat lines (horizontal)
  if (CHART_DATA.syncTrack.bpmEvents.length > 0) {{
    const bpm = CHART_DATA.syncTrack.bpmEvents[0].bpm;
    const beatInterval = 60.0 / bpm;
    const firstBeat = Math.floor(now / beatInterval) * beatInterval;
    for (let bt = firstBeat; bt < now + highwaySpeed + beatInterval; bt += beatInterval) {{
      const dt = bt - now;
      if (dt < -0.01) continue;
      const y = strikeY - (dt / highwaySpeed) * visibleHeight;
      if (y < topY || y > strikeY + 20) continue;
      const isMeasure = Math.abs(bt / beatInterval % 4) < 0.1;
      ctx.strokeStyle = isMeasure ? 'rgba(255,255,255,0.18)' : 'rgba(255,255,255,0.06)';
      ctx.lineWidth = isMeasure ? 1.5 : 0.5;
      ctx.beginPath();
      ctx.moveTo(hx, y);
      ctx.lineTo(hx + highwayWidth, y);
      ctx.stroke();
    }}
  }}

  // Strike line
  const grad = ctx.createLinearGradient(hx, strikeY, hx + highwayWidth, strikeY);
  FRET_COLORS.forEach((c, i) => {{
    grad.addColorStop(i / (laneCount - 1), c);
  }});
  ctx.strokeStyle = grad;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(hx, strikeY);
  ctx.lineTo(hx + highwayWidth, strikeY);
  ctx.stroke();

  // Strike line fret circles
  for (let i = 0; i < laneCount; i++) {{
    const cx = hx + (i + 0.5) * laneWidth;
    ctx.beginPath();
    ctx.arc(cx, strikeY, laneWidth * 0.3, 0, Math.PI * 2);
    ctx.strokeStyle = FRET_COLORS[i];
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.3;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }}

  // Draw notes
  const noteRadius = Math.min(laneWidth * 0.35, 18);
  for (const note of currentTrack.notes) {{
    const dt = note.timeSec - now;
    if (dt < -0.3 || dt > highwaySpeed + 0.5) continue;

    const y = strikeY - (dt / highwaySpeed) * visibleHeight;
    if (y < topY - 20 || y > strikeY + 50) continue;

    const cx = hx + (note.fret + 0.5) * laneWidth;
    const fret = Math.min(note.fret, 4);

    // Sustain tail
    if (note.durationSec > 0.06) {{
      const endDt = note.timeSec + note.durationSec - now;
      const endY = strikeY - (endDt / highwaySpeed) * visibleHeight;
      const clampedEndY = Math.max(topY, endY);
      const clampedY = Math.min(strikeY + 20, y);
      if (clampedEndY < clampedY) {{
        ctx.fillStyle = FRET_GLOW[fret];
        ctx.fillRect(cx - 4, clampedEndY, 8, clampedY - clampedEndY);
      }}
    }}

    // Note glow
    ctx.shadowColor = FRET_COLORS[fret];
    ctx.shadowBlur = dt < 0.05 && dt > -0.05 ? 25 : 10;

    // Note gem
    ctx.beginPath();
    ctx.arc(cx, y, noteRadius, 0, Math.PI * 2);
    ctx.fillStyle = FRET_COLORS[fret];
    ctx.fill();

    // Inner highlight
    ctx.beginPath();
    ctx.arc(cx - noteRadius * 0.2, y - noteRadius * 0.2, noteRadius * 0.35, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255,255,255,0.35)';
    ctx.fill();

    // Hit flash
    if (dt < 0 && dt > -0.15) {{
      ctx.beginPath();
      ctx.arc(cx, strikeY, noteRadius * 1.5 * (1 + Math.abs(dt) * 5), 0, Math.PI * 2);
      ctx.fillStyle = FRET_GLOW[fret];
      ctx.globalAlpha = 1 - Math.abs(dt) / 0.15;
      ctx.fill();
      ctx.globalAlpha = 1;
    }}

    ctx.shadowBlur = 0;
  }}

  // UI updates
  timeDisplay.textContent = formatTime(now) + ' / ' + formatTime(dur);
  progressFill.style.width = (now / dur * 100) + '%';
  sectionLabel.textContent = getCurrentSection(now);

  requestAnimationFrame(draw);
}}

// Keyboard controls
document.addEventListener('keydown', (e) => {{
  if (e.code === 'Space') {{ e.preventDefault(); togglePlay(); }}
  if (e.code === 'ArrowLeft') {{ audio.currentTime = Math.max(0, audio.currentTime - 5); }}
  if (e.code === 'ArrowRight') {{ audio.currentTime = Math.min(audio.duration, audio.currentTime + 5); }}
}});

requestAnimationFrame(draw);
</script>
</body>
</html>"""


def generate_preview_html(
    song: ChartSong,
    sync: SyncTrack,
    events: list[SectionEvent],
    tracks: list[ChartTrack],
    audio_path: str,
    output_path: str,
) -> str:
    """Generate a self-contained HTML preview file with note highway and audio playback.

    Args:
        song: Chart song metadata
        sync: Sync track with BPM/TS events
        events: Section events
        tracks: Note tracks (all difficulties)
        audio_path: Relative path to audio file (from the HTML file's location)
        output_path: Where to write the HTML file

    Returns:
        Path to the generated HTML file.
    """
    data = chart_to_preview_data(song, sync, events, tracks)

    # Build difficulty options
    diff_names = {"EasySingle": "Easy", "MediumSingle": "Medium", "HardSingle": "Hard", "ExpertSingle": "Expert"}
    diff_options = []
    for t in data["tracks"]:
        label = diff_names.get(t["name"], t["name"])
        selected = "selected" if t["name"] == "ExpertSingle" else ""
        diff_options.append(f'<option value="{t["name"]}" {selected}>{label} ({len(t["notes"])} notes)</option>')

    html = _PREVIEW_HTML_TEMPLATE.format(
        song_name=song.name,
        artist=song.artist,
        chart_json=json.dumps(data),
        audio_path=audio_path,
        difficulty_options="\n      ".join(diff_options),
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path
