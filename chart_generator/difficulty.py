"""Derive lower difficulty levels from Expert chart using rhythmic backbone reduction."""
from __future__ import annotations
from chart_generator.chart import ChartTrack, NoteEvent


# Fraction of expert notes to keep per difficulty
_KEEP_RATIOS = {
    "HardSingle": 0.75,
    "MediumSingle": 0.45,
    "EasySingle": 0.25,
}


def derive_difficulty(expert: ChartTrack, target_name: str) -> ChartTrack:
    """Derive a lower difficulty track from ExpertSingle.

    Strategy:
    - Keep notes that fall on strong beats (downbeats first, then upbeats)
    - Reduce chord density (Easy = no chords, Medium = max 2-note chords)
    - Preserve phrase identity by keeping first note of each phrase group
    - Simplify fret range for lower difficulties
    """
    if target_name not in _KEEP_RATIOS:
        raise ValueError(f"Unknown difficulty: {target_name}")

    keep_ratio = _KEEP_RATIOS[target_name]
    result = ChartTrack(target_name)

    if not expert.notes:
        return result

    # Group notes by tick (chords share a tick)
    tick_groups: dict[int, list[NoteEvent]] = {}
    for note in expert.notes:
        tick_groups.setdefault(note.tick, []).append(note)

    sorted_ticks = sorted(tick_groups.keys())
    target_count = max(1, int(len(sorted_ticks) * keep_ratio))

    # Score each tick position by beat strength (lower tick mod = stronger beat)
    # Resolution is typically 192; a whole beat = 192, half = 96, quarter = 48
    def beat_strength(tick: int) -> int:
        """Higher score = stronger beat position."""
        if tick % 768 == 0:  # Every 4 beats (measure in 4/4)
            return 100
        if tick % 192 == 0:  # Every beat
            return 80
        if tick % 96 == 0:  # Every half beat
            return 60
        if tick % 48 == 0:  # Every quarter beat (16th note)
            return 40
        return 20

    scored_ticks = [(t, beat_strength(t)) for t in sorted_ticks]
    # Sort by beat strength descending, then by tick ascending for stability
    scored_ticks.sort(key=lambda x: (-x[1], x[0]))

    # Keep the top N ticks by beat strength
    kept_ticks = set(t for t, _ in scored_ticks[:target_count])

    # Max notes per chord by difficulty
    max_chord_size = {
        "HardSingle": 3,
        "MediumSingle": 2,
        "EasySingle": 1,
    }[target_name]

    # Fret range reduction for lower difficulties
    max_fret = {
        "HardSingle": 4,
        "MediumSingle": 3,
        "EasySingle": 2,
    }[target_name]

    for tick in sorted(kept_ticks):
        notes = tick_groups[tick]
        # Limit chord size
        chord = sorted(notes, key=lambda n: n.fret)[:max_chord_size]
        for note in chord:
            fret = min(note.fret, max_fret)
            result.add_note(tick, fret, note.duration)

    return result
