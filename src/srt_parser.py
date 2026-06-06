"""
Parses OpenSubtitles-format SRT files into a structured DataFrame.

Place SRT files under data/raw/srt/ — see PLACE_SRT_FILES_HERE.txt.
Filenames must contain the season/episode code, e.g.:
    S01E01.srt  |  suits_s01e01.srt  |  Suits.S01E01.HDTV.srt

Run directly:
    python src/srt_parser.py --srt-dir data/raw/srt --out data/srt_parsed.csv
"""

import re
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Matches "HH:MM:SS,mmm --> HH:MM:SS,mmm"
_TS_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s+-->\s+"
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)

# HTML / formatting tags common in SRT files
_TAG_RE = re.compile(r"<[^>]+>|\{[^}]+\}")

# Musical notes and sound effect markers — not dialogue
_EFFECT_RE = re.compile(r"^[\[{(♪♫]|^\s*#")


def _ts_to_seconds(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_srt_file(path: Path, season: int, episode_num: int) -> list[dict]:
    episode_id = f"S{season:02d}E{episode_num:02d}"

    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as e:
        print(f"  Cannot read {path}: {e}")
        return []

    blocks = re.split(r"\n{2,}", text.strip())
    rows = []

    for block in blocks:
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        if len(lines) < 2:
            continue

        # Find the timestamp line (not necessarily line index 1 — some SRTs skip the counter)
        ts_line_idx = None
        for i, l in enumerate(lines):
            if _TS_RE.match(l):
                ts_line_idx = i
                break
        if ts_line_idx is None:
            continue

        m = _TS_RE.match(lines[ts_line_idx])
        ts_start = _ts_to_seconds(m.group(1), m.group(2), m.group(3), m.group(4))
        ts_end   = _ts_to_seconds(m.group(5), m.group(6), m.group(7), m.group(8))

        dialogue_lines = lines[ts_line_idx + 1:]
        if not dialogue_lines:
            continue

        # Clean and join multi-line subtitles
        cleaned = []
        for dl in dialogue_lines:
            dl = _TAG_RE.sub("", dl).strip()
            dl = dl.replace("​", "")  # zero-width space
            if dl and not _EFFECT_RE.match(dl):
                cleaned.append(dl)

        dialogue = " ".join(cleaned).strip()
        if len(dialogue) < 3:
            continue

        rows.append({
            "episode_id": episode_id,
            "season": season,
            "episode_num": episode_num,
            "line": dialogue,
            "timestamp_start": ts_start,
            "timestamp_end": ts_end,
        })

    return rows


_SEASON_EP_RE = re.compile(r"[Ss](\d{1,2})[Ee](\d{1,2})")


def parse_srt_directory(srt_dir: Path) -> pd.DataFrame:
    srt_files = sorted(srt_dir.rglob("*.srt"))

    if not srt_files:
        print(f"No .srt files found under {srt_dir}")
        return pd.DataFrame()

    all_rows: list[dict] = []
    skipped = []

    for f in tqdm(srt_files, desc="Parsing SRT files"):
        m = _SEASON_EP_RE.search(f.name)
        if not m:
            skipped.append(f.name)
            continue
        season, ep = int(m.group(1)), int(m.group(2))
        rows = parse_srt_file(f, season, ep)
        all_rows.extend(rows)

    if skipped:
        print(f"Skipped (no S##E## in filename): {', '.join(skipped)}")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values(["episode_id", "timestamp_start"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--srt-dir", default="data/raw/srt")
    parser.add_argument("--out", default="data/srt_parsed.csv")
    args = parser.parse_args()

    df = parse_srt_directory(Path(args.srt_dir))
    if not df.empty:
        df.to_csv(args.out, index=False)
        print(f"Saved {len(df):,} subtitle blocks → {args.out}")
        print(f"Episodes covered: {df['episode_id'].nunique()}")
