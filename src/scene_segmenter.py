"""
Assigns scene IDs to dialogue rows.

Two strategies, selected automatically:
  1. Time-gap segmentation  — uses SRT timestamps (preferred)
  2. Character-change segmentation — fallback when timestamps are absent

Scene IDs are formatted as:  S01E01_SC001
"""

import pandas as pd


def segment_by_time_gap(df: pd.DataFrame, gap_seconds: float = 30.0) -> pd.DataFrame:
    """
    New scene whenever there is a silence gap > gap_seconds between subtitle blocks.

    Requires: timestamp_start, timestamp_end columns (float seconds).
    """
    df = df.copy()
    df = df.sort_values(["episode_id", "timestamp_start"]).reset_index(drop=True)

    scene_col = []

    for ep_id, group in df.groupby("episode_id", sort=False):
        group = group.reset_index(drop=True)
        scene = 1
        ep_scenes = [scene]

        for i in range(1, len(group)):
            gap = group.at[i, "timestamp_start"] - group.at[i - 1, "timestamp_end"]
            if gap > gap_seconds:
                scene += 1
            ep_scenes.append(scene)

        scene_col.extend(ep_scenes)

    df["scene_num"] = scene_col
    df["scene_id"] = (
        df["episode_id"] + "_SC" + df["scene_num"].astype(str).str.zfill(3)
    )
    df = df.drop(columns=["scene_num"])
    return df


def segment_by_character_change(df: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """
    Fallback: new scene when the set of speakers in a sliding window
    changes completely relative to the previous window.

    window: number of lines to consider for the active character set.
    """
    df = df.copy().reset_index(drop=True)
    scene_col: list[int] = []

    for ep_id, group in df.groupby("episode_id", sort=False):
        group = group.reset_index(drop=True)
        n = len(group)
        scene = 1
        ep_scenes = [scene]

        prev_chars: set[str] = set(group.loc[: window - 1, "speaker"].unique()) - {"UNKNOWN"}

        for i in range(1, n):
            lo = max(0, i - window)
            curr_chars: set[str] = set(group.loc[lo:i, "speaker"].unique()) - {"UNKNOWN"}

            if prev_chars and curr_chars and prev_chars.isdisjoint(curr_chars):
                scene += 1
                prev_chars = curr_chars
            elif curr_chars:
                prev_chars = curr_chars

            ep_scenes.append(scene)

        scene_col.extend(ep_scenes)

    df["scene_num"] = scene_col
    df["scene_id"] = (
        df["episode_id"] + "_SC" + df["scene_num"].astype(str).str.zfill(3)
    )
    df = df.drop(columns=["scene_num"])
    return df


def add_scenes(df: pd.DataFrame, gap_seconds: float = 30.0) -> pd.DataFrame:
    """Auto-selects strategy based on whether timestamps are present."""
    has_timestamps = (
        "timestamp_start" in df.columns
        and df["timestamp_start"].notna().mean() > 0.3
    )

    if has_timestamps:
        print(f"Using time-gap scene segmentation (gap = {gap_seconds}s)")
        return segment_by_time_gap(df, gap_seconds=gap_seconds)
    else:
        print("Using character-change scene segmentation (no timestamps)")
        return segment_by_character_change(df)
