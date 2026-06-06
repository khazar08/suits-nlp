"""
Aligns speaker-attributed transcript lines (from scraper)
with timestamps (from SRT parser) using fuzzy text matching.

Strategy:
  For each episode, we build a normalized version of both text sources,
  then use difflib.SequenceMatcher to find the best SRT block for each
  transcript line. Lines that don't match above min_ratio get None timestamps.
"""

import re
import difflib
from typing import Optional

import pandas as pd
from tqdm import tqdm


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _find_best_srt_match(
    query: str,
    candidates: list[str],
    candidate_indices: list[int],
    min_ratio: float,
    search_window: Optional[tuple[int, int]] = None,
) -> Optional[int]:
    """
    Return the index into `candidates` with the best match to `query`,
    or None if the best ratio is below min_ratio.

    search_window: (lo, hi) restricts search to a contiguous range of candidates.
    This speeds up alignment dramatically for long episodes by exploiting the
    fact that transcript order ≈ subtitle order.
    """
    lo, hi = search_window if search_window else (0, len(candidates))
    lo = max(0, lo)
    hi = min(len(candidates), hi)

    best_ratio = 0.0
    best_idx = None

    for i in range(lo, hi):
        ratio = difflib.SequenceMatcher(None, query, candidates[i], autojunk=False).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i

    return best_idx if best_ratio >= min_ratio else None


def align_transcripts_with_srt(
    transcript_df: pd.DataFrame,
    srt_df: pd.DataFrame,
    min_ratio: float = 0.55,
    window_half: int = 30,
) -> pd.DataFrame:
    """
    Merge speaker labels from transcript_df with timestamps from srt_df.

    transcript_df columns: episode_id, speaker, line, ...
    srt_df columns:        episode_id, line, timestamp_start, timestamp_end

    Returns a DataFrame with all transcript_df columns plus
    timestamp_start and timestamp_end (None where no SRT match found).
    """
    result_rows: list[dict] = []
    srt_episodes = set(srt_df["episode_id"].unique()) if not srt_df.empty else set()

    for ep_id, t_group in tqdm(
        transcript_df.groupby("episode_id"),
        desc="Aligning episodes",
        total=transcript_df["episode_id"].nunique(),
    ):
        t_group = t_group.reset_index(drop=True)

        if ep_id not in srt_episodes:
            for _, row in t_group.iterrows():
                result_rows.append({
                    **row.to_dict(),
                    "timestamp_start": None,
                    "timestamp_end": None,
                    "srt_match_ratio": None,
                })
            continue

        s_group = srt_df[srt_df["episode_id"] == ep_id].reset_index(drop=True)
        srt_norms = [_normalize(l) for l in s_group["line"]]

        last_matched_srt = 0  # use as center of search window

        for t_idx, t_row in t_group.iterrows():
            t_norm = _normalize(t_row["line"])
            if len(t_norm) < 5:
                result_rows.append({
                    **t_row.to_dict(),
                    "timestamp_start": None,
                    "timestamp_end": None,
                    "srt_match_ratio": None,
                })
                continue

            lo = max(0, last_matched_srt - window_half)
            hi = min(len(srt_norms), last_matched_srt + window_half)

            best_idx = _find_best_srt_match(
                t_norm, srt_norms, list(range(len(srt_norms))),
                min_ratio, (lo, hi)
            )

            # Expand window if no match found in restricted range
            if best_idx is None:
                best_idx = _find_best_srt_match(
                    t_norm, srt_norms, list(range(len(srt_norms))),
                    min_ratio
                )

            if best_idx is not None:
                srt_row = s_group.iloc[best_idx]
                ratio = difflib.SequenceMatcher(
                    None, t_norm, srt_norms[best_idx], autojunk=False
                ).ratio()
                result_rows.append({
                    **t_row.to_dict(),
                    "timestamp_start": srt_row["timestamp_start"],
                    "timestamp_end": srt_row["timestamp_end"],
                    "srt_match_ratio": round(ratio, 3),
                })
                last_matched_srt = best_idx
            else:
                result_rows.append({
                    **t_row.to_dict(),
                    "timestamp_start": None,
                    "timestamp_end": None,
                    "srt_match_ratio": None,
                })

    df = pd.DataFrame(result_rows)

    if not df.empty:
        matched = df["timestamp_start"].notna().sum()
        total = len(df)
        print(f"Timestamp alignment: {matched:,}/{total:,} lines matched ({100*matched/total:.1f}%)")

    return df
