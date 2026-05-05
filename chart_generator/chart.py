"""Clone Hero .chart file data structures, generation, and parsing."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import re


@dataclass
class BPMEvent:
    tick: int
    bpm: float

    def to_chart_line(self) -> str:
        return f"  {self.tick} = B {int(round(self.bpm * 1000))}"


@dataclass
class TSEvent:
    tick: int
    numerator: int
    denominator_exp: Optional[int] = None

    def to_chart_line(self) -> str:
        if self.denominator_exp is not None:
            return f"  {self.tick} = TS {self.numerator} {self.denominator_exp}"
        return f"  {self.tick} = TS {self.numerator}"


@dataclass
class NoteEvent:
    tick: int
    fret: int
    duration: int

    def __post_init__(self):
        if self.fret < 0 or self.fret > 7:
            raise ValueError(f"Fret must be 0-7, got {self.fret}")

    def to_chart_line(self) -> str:
        return f"  {self.tick} = N {self.fret} {self.duration}"


@dataclass
class SectionEvent:
    tick: int
    name: str

    def to_chart_line(self) -> str:
        return f'  {self.tick} = E "section {self.name}"'


@dataclass
class ChartSong:
    name: str
    artist: str
    resolution: int = 192
    offset: int = 0
    player2: str = "bass"
    difficulty: int = 0
    preview_start: int = 0
    preview_end: int = 0
    genre: str = "rock"
    media_type: str = "cd"
    music_stream: str = "song.ogg"

    def to_chart_section(self) -> str:
        lines = [
            "[Song]",
            "{",
            f'  Name = "{self.name}"',
            f'  Artist = "{self.artist}"',
            f"  Offset = {self.offset}",
            f"  Resolution = {self.resolution}",
            f"  Player2 = {self.player2}",
            f"  Difficulty = {self.difficulty}",
            f"  PreviewStart = {self.preview_start}",
            f"  PreviewEnd = {self.preview_end}",
            f'  Genre = "{self.genre}"',
            f'  MediaType = "{self.media_type}"',
            f'  MusicStream = "{self.music_stream}"',
            "}",
        ]
        return "\n".join(lines)


class SyncTrack:
    def __init__(self):
        self.bpm_events: list[BPMEvent] = []
        self.ts_events: list[TSEvent] = []

    def add_bpm(self, tick: int, bpm: float):
        self.bpm_events.append(BPMEvent(tick=tick, bpm=bpm))

    def add_time_signature(self, tick: int, numerator: int, denominator_exp: Optional[int] = None):
        self.ts_events.append(TSEvent(tick=tick, numerator=numerator, denominator_exp=denominator_exp))

    def to_chart_section(self) -> str:
        events = []
        for ts in self.ts_events:
            events.append((ts.tick, 0, ts.to_chart_line()))
        for bpm in self.bpm_events:
            events.append((bpm.tick, 1, bpm.to_chart_line()))
        events.sort(key=lambda x: (x[0], x[1]))
        lines = ["[SyncTrack]", "{"]
        lines.extend(e[2] for e in events)
        lines.append("}")
        return "\n".join(lines)


class ChartTrack:
    def __init__(self, name: str):
        self.name = name
        self.notes: list[NoteEvent] = []

    def add_note(self, tick: int, fret: int, duration: int):
        self.notes.append(NoteEvent(tick=tick, fret=fret, duration=duration))

    def to_chart_section(self) -> str:
        sorted_notes = sorted(self.notes, key=lambda n: (n.tick, n.fret))
        lines = [f"[{self.name}]", "{"]
        lines.extend(n.to_chart_line() for n in sorted_notes)
        lines.append("}")
        return "\n".join(lines)


def build_chart_string(
    song: ChartSong,
    sync_track: SyncTrack,
    events: list[SectionEvent],
    tracks: list[ChartTrack],
) -> str:
    parts = [song.to_chart_section(), sync_track.to_chart_section()]

    # Events section
    event_lines = ["[Events]", "{"]
    for ev in sorted(events, key=lambda e: e.tick):
        event_lines.append(ev.to_chart_line())
    event_lines.append("}")
    parts.append("\n".join(event_lines))

    # Note tracks
    for track in tracks:
        parts.append(track.to_chart_section())

    return "\n".join(parts)


def parse_chart_string(content: str) -> tuple[ChartSong, SyncTrack, list[SectionEvent], list[ChartTrack]]:
    sections: dict[str, list[str]] = {}
    current_section = None
    current_lines: list[str] = []

    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if current_section is not None:
                sections[current_section] = current_lines
            current_section = stripped[1:-1]
            current_lines = []
        elif stripped not in ("{", "}"):
            current_lines.append(stripped)

    if current_section is not None:
        sections[current_section] = current_lines

    # Parse Song section
    song_data = {}
    for line in sections.get("Song", []):
        m = re.match(r'(\w+)\s*=\s*"?([^"]*)"?', line)
        if m:
            song_data[m.group(1)] = m.group(2).strip()

    song = ChartSong(
        name=song_data.get("Name", "Unknown"),
        artist=song_data.get("Artist", "Unknown"),
        resolution=int(song_data.get("Resolution", "192")),
        offset=int(song_data.get("Offset", "0")),
    )

    # Parse SyncTrack
    sync = SyncTrack()
    for line in sections.get("SyncTrack", []):
        m = re.match(r"(\d+)\s*=\s*B\s+(\d+)", line)
        if m:
            sync.add_bpm(int(m.group(1)), int(m.group(2)) / 1000.0)
            continue
        m = re.match(r"(\d+)\s*=\s*TS\s+(\d+)(?:\s+(\d+))?", line)
        if m:
            denom = int(m.group(3)) if m.group(3) else None
            sync.add_time_signature(int(m.group(1)), int(m.group(2)), denom)

    # Parse Events
    events: list[SectionEvent] = []
    for line in sections.get("Events", []):
        m = re.match(r'(\d+)\s*=\s*E\s+"section\s+(.+)"', line)
        if m:
            events.append(SectionEvent(tick=int(m.group(1)), name=m.group(2)))

    # Parse note tracks
    tracks: list[ChartTrack] = []
    track_names = ["EasySingle", "MediumSingle", "HardSingle", "ExpertSingle"]
    for tn in track_names:
        if tn in sections:
            track = ChartTrack(tn)
            for line in sections[tn]:
                m = re.match(r"(\d+)\s*=\s*N\s+(\d+)\s+(\d+)", line)
                if m:
                    track.add_note(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            tracks.append(track)

    return song, sync, events, tracks
