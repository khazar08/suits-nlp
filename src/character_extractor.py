"""
Builds scene-level character co-occurrence data from dialogue lines.

Since Springfield transcripts have no speaker labels, we use character NAME
MENTIONS in dialogue as a proxy for scene presence. This is the standard
approach for TV-show network analysis and is sufficient to build the
interaction graph specified in the SPEC.

Output:
  suits_scene_characters.csv  — one row per (scene, character) pair
  suits_cooccurrence.csv      — one row per (scene, char_a, char_b) pair
"""

import re
from itertools import combinations

import pandas as pd

from scraper import CHARACTER_MAP, MENTION_RE, SUITS_CHARACTERS

# Window (in dialogue lines) to look around a line when assigning scene presence.
# A character mentioned in a 5-line window is considered "present" in that scene.
MENTION_WINDOW = 3


def extract_scene_characters(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each scene, collect the set of characters mentioned in its dialogue.
    Returns a long-format DataFrame with columns:
      scene_id, episode_id, season, episode_num, character, mention_count
    """
    if "scene_id" not in df.columns:
        raise ValueError("DataFrame must have a scene_id column — run scene_segmenter first.")

    rows = []
    for (scene_id, ep_id), group in df.groupby(["scene_id", "episode_id"], sort=False):
        season = group["season"].iloc[0]
        ep_num = group["episode_num"].iloc[0]

        # Collect all character mentions across all lines in this scene
        mention_counts: dict[str, int] = {}
        for mentions_str in group["char_mentions"].dropna():
            if not mentions_str:
                continue
            for char in mentions_str.split("|"):
                char = char.strip()
                if char:
                    mention_counts[char] = mention_counts.get(char, 0) + 1

        for char, count in mention_counts.items():
            rows.append({
                "scene_id": scene_id,
                "episode_id": ep_id,
                "season": season,
                "episode_num": ep_num,
                "character": char,
                "mention_count": count,
            })

    return pd.DataFrame(rows)


def build_cooccurrence(scene_chars_df: pd.DataFrame, min_mentions: int = 1) -> pd.DataFrame:
    """
    For every scene, create edges between all pairs of characters present.
    Edge weight = geometric mean of the two characters' mention counts.

    min_mentions: minimum times a character must be mentioned to count as present.

    Returns DataFrame with columns:
      scene_id, episode_id, season, episode_num,
      char_a, char_b, weight_a, weight_b, edge_weight
    """
    rows = []

    for (scene_id, ep_id), group in scene_chars_df.groupby(["scene_id", "episode_id"], sort=False):
        season = group["season"].iloc[0]
        ep_num = group["episode_num"].iloc[0]

        present = group[group["mention_count"] >= min_mentions].set_index("character")["mention_count"]
        chars = list(present.index)

        if len(chars) < 2:
            continue

        for a, b in combinations(sorted(chars), 2):
            # Geometric mean of mention counts as edge weight
            weight = (present[a] * present[b]) ** 0.5
            rows.append({
                "scene_id": scene_id,
                "episode_id": ep_id,
                "season": season,
                "episode_num": ep_num,
                "char_a": a,
                "char_b": b,
                "mentions_a": present[a],
                "mentions_b": present[b],
                "edge_weight": round(weight, 3),
            })

    return pd.DataFrame(rows)


def episode_interaction_summary(cooc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate co-occurrence data to episode level.
    Returns a DataFrame with total interaction weight per (episode, char_a, char_b).
    Used for building the temporal graph G₁, G₂, … Gₙ.
    """
    agg = (
        cooc_df.groupby(["episode_id", "season", "episode_num", "char_a", "char_b"])
        .agg(
            scene_count=("scene_id", "count"),
            total_weight=("edge_weight", "sum"),
        )
        .reset_index()
    )
    return agg.sort_values(["episode_id", "total_weight"], ascending=[True, False])


def run(dialogue_df: pd.DataFrame, out_dir) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full extraction pipeline. Returns (scene_chars, cooccurrence, episode_interactions).
    Saves CSVs to out_dir.
    """
    import pathlib
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting scene-level character presence...")
    scene_chars = extract_scene_characters(dialogue_df)
    scene_chars.to_csv(out_dir / "suits_scene_characters.csv", index=False)
    print(f"  {len(scene_chars):,} (scene, character) pairs → suits_scene_characters.csv")

    print("Building co-occurrence edges...")
    cooc = build_cooccurrence(scene_chars)
    cooc.to_csv(out_dir / "suits_cooccurrence.csv", index=False)
    print(f"  {len(cooc):,} scene-level edges → suits_cooccurrence.csv")

    print("Aggregating to episode-level interactions...")
    ep_interactions = episode_interaction_summary(cooc)
    ep_interactions.to_csv(out_dir / "suits_episode_interactions.csv", index=False)
    print(f"  {len(ep_interactions):,} episode-level edges → suits_episode_interactions.csv")

    return scene_chars, cooc, ep_interactions
