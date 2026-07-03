import argparse
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from scraper import scrape_all
from srt_parser import parse_srt_directory
from aligner import align_transcripts_with_srt
from scene_segmenter import add_scenes
import character_extractor


DIALOGUE_COLUMNS = [
    "episode_id", "season", "episode_num", "episode_title",
    "scene_id", "line_num", "speaker", "line",
    "char_mentions", "mention_count",
    "timestamp_start", "timestamp_end", "srt_match_ratio",
]


def run(data_dir: Path, skip_scrape: bool, gap_seconds: float, seasons: list[int] | None = None) -> pd.DataFrame:
    srt_dir = data_dir / "raw" / "srt"
    raw_csv = data_dir / "suits_dialogue_raw.csv"
    out_dir = data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    if skip_scrape:
        if not raw_csv.exists():
            sys.exit(f"[ERROR] {raw_csv} not found. Run without --skip-scrape first.")
        print(f"Loading {raw_csv}")
        transcript_df = pd.read_csv(raw_csv)
        print(f"  {len(transcript_df):,} lines loaded")
    else:
        transcript_df = scrape_all(data_dir, seasons=seasons)

    srt_files = list(srt_dir.rglob("*.srt")) if srt_dir.exists() else []
    if srt_files:
        _header(f"STEP 2: Parsing {len(srt_files)} SRT files")
        srt_df = parse_srt_directory(srt_dir)

        _header("STEP 3: Aligning dialogue ↔ SRT timestamps")
        merged_df = align_transcripts_with_srt(transcript_df, srt_df)
        step_offset = 1
    else:
        print(
            "\n[INFO] No SRT files — skipping timestamp alignment.\n"
            "       Drop .srt files in data/raw/srt/ and re-run to add timestamps.\n"
        )
        merged_df = transcript_df.copy()
        merged_df["timestamp_start"] = None
        merged_df["timestamp_end"] = None
        merged_df["srt_match_ratio"] = None
        step_offset = 0

    dialogue_df = add_scenes(merged_df, gap_seconds=gap_seconds)

    scene_chars, cooc, ep_interactions = character_extractor.run(dialogue_df, out_dir)

    available = [c for c in DIALOGUE_COLUMNS if c in dialogue_df.columns]
    dialogue_df = dialogue_df[available].reset_index(drop=True)
    dialogue_path = out_dir / "suits_dialogue.csv"
    dialogue_df.to_csv(dialogue_path, index=False)

    print(f"  suits_dialogue.csv          {len(dialogue_df):>8,} lines")
    print(f"  suits_scene_characters.csv  {len(scene_chars):>8,} (scene, char) pairs")
    print(f"  suits_cooccurrence.csv      {len(cooc):>8,} scene edges")
    print(f"  suits_episode_interactions  {len(ep_interactions):>8,} episode edges")
    print()
    print(f"  Seasons:  {sorted(dialogue_df['season'].unique())}")
    print(f"  Episodes: {dialogue_df['episode_id'].nunique()}")
    print(f"  Scenes:   {dialogue_df['scene_id'].nunique():,}")

    lines_with_mentions = (dialogue_df["mention_count"] > 0).sum()
    print(f"  Lines with character mentions: {lines_with_mentions:,} ({100*lines_with_mentions/len(dialogue_df):.1f}%)")

    if dialogue_df["timestamp_start"].notna().any():
        ts_pct = 100 * dialogue_df["timestamp_start"].notna().mean()
        print(f"  Lines with timestamps: {ts_pct:.1f}%")

    if not cooc.empty:
        top_pairs = (
            cooc.groupby(["char_a", "char_b"])["edge_weight"]
            .sum().sort_values(ascending=False).head(8)
        )
        for (a, b), w in top_pairs.items():
            print(f"    {a:<22} ↔ {b:<22} {w:>7.1f}")

    return dialogue_df


def _header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Build the Suits dialogue + graph dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--skip-scrape", action="store_true",
                        help="Load existing suits_dialogue_raw.csv instead of scraping")
    parser.add_argument("--gap", type=float, default=30.0,
                        help="Silence gap in seconds that starts a new scene (default: 30)")
    parser.add_argument("--seasons", type=int, nargs="+", default=None,
                        help="Scrape only these seasons, e.g. --seasons 1 2")
    args = parser.parse_args()
    run(Path(args.data_dir), skip_scrape=args.skip_scrape, gap_seconds=args.gap, seasons=args.seasons)


if __name__ == "__main__":
    main()
