#!/usr/bin/env python3
"""ChartHero CLI — Generate Clone Hero charts from audio files."""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chart_generator.pipeline import generate_chart_from_audio
from chart_generator.output import SongMetadata


def main():
    parser = argparse.ArgumentParser(
        description="AutoChart: AI-powered Clone Hero chart generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python cli.py song.mp3 --name "My Song" --artist "My Band"
  python cli.py song.ogg --name "Test" --artist "Me" --album "Album" --art cover.jpg
  python cli.py song.wav --name "X" --artist "Y" --output ./charts/
""",
    )
    parser.add_argument("audio", help="Path to audio file (MP3, OGG, WAV)")
    parser.add_argument("--name", required=True, help="Song name")
    parser.add_argument("--artist", required=True, help="Artist name")
    parser.add_argument("--album", default="", help="Album name")
    parser.add_argument("--genre", default="Rock", help="Genre (default: Rock)")
    parser.add_argument("--year", default="", help="Year")
    parser.add_argument("--charter", default="AutoChart", help="Charter name (default: AutoChart)")
    parser.add_argument("--art", default=None, help="Path to album art image")
    parser.add_argument("--output", default=".", help="Output directory (default: current dir)")
    parser.add_argument("--no-preview", action="store_true", help="Skip opening preview in browser")

    args = parser.parse_args()

    if not os.path.isfile(args.audio):
        print(f"Error: Audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    if args.art and not os.path.isfile(args.art):
        print(f"Error: Album art not found: {args.art}", file=sys.stderr)
        sys.exit(1)

    meta = SongMetadata(
        name=args.name,
        artist=args.artist,
        album=args.album,
        genre=args.genre,
        year=args.year,
        charter=args.charter,
    )

    folder_name = f"{args.artist} - {args.name} ({args.charter})"
    output_dir = os.path.join(args.output, folder_name)

    print(f"🎸 AutoChart — Generating Clone Hero chart")
    print(f"   Audio: {args.audio}")
    print(f"   Song:  {args.name} by {args.artist}")
    print()

    start = time.time()

    print("📊 Analyzing audio...")
    result_path = generate_chart_from_audio(
        audio_path=args.audio,
        metadata=meta,
        output_dir=output_dir,
        album_art_path=args.art,
    )

    elapsed = time.time() - start

    print(f"✅ Chart generated in {elapsed:.1f}s")
    print(f"📁 Output: {result_path}")
    print()
    print("Files created:")
    for f in sorted(os.listdir(result_path)):
        size = os.path.getsize(os.path.join(result_path, f))
        print(f"   {f} ({size:,} bytes)")

    # Open preview in browser
    preview_path = os.path.join(result_path, "preview.html")
    if os.path.isfile(preview_path) and not args.no_preview:
        import webbrowser
        preview_url = "file:///" + os.path.abspath(preview_path).replace("\\", "/")
        print(f"\n🌐 Opening preview in browser...")
        webbrowser.open(preview_url)


if __name__ == "__main__":
    main()
