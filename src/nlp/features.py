"""
NLP Feature Extraction Orchestrator — Step 2 of the Suits pipeline.

Reads:   data/processed/suits_dialogue.csv
Writes:  data/processed/suits_features.csv

New columns added per dialogue line:
  ── Sentiment (VADER) ──────────────────────────────────
  vader_compound, vader_pos, vader_neg, vader_neu
  sentiment_label           positive | negative | neutral

  ── Emotion ────────────────────────────────────────────
  emotion_label             anger | fear | joy | sadness |
                            disgust | surprise | trust | neutral
  emotion_score             confidence (0–1)
  emotion_mode              keyword | transformer

  ── Linguistic Complexity ──────────────────────────────
  word_count, char_count
  unique_word_ratio         type-token ratio (lexical diversity)
  flesch_ease               Flesch Reading Ease
  gunning_fog               Gunning Fog grade level
  avg_word_length
  is_question, is_exclamation

  ── Power Language ─────────────────────────────────────
  power_category            commanding | defensive | manipulative |
                            assertive | neutral
  power_score               relative dominance (0–1)
  commanding_score, defensive_score,
  manipulative_score, assertive_score

  ── Entities & Legal Language ──────────────────────────
  entities_people           pipe-separated person names
  entities_orgs             pipe-separated org names
  legal_term_count
  legal_terms               pipe-separated legal terms found
  has_legal_language        bool

Usage:
    python src/nlp/features.py
    python src/nlp/features.py --use-transformers   # slower, more accurate emotion
    python src/nlp/features.py --batch-size 128
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from nlp import sentiment as _sent
from nlp import emotion as _emo
from nlp import complexity as _cpx
from nlp import power_language as _pow
from nlp import entities as _ent


CHUNK = 512   # lines processed per tqdm tick


def extract_features(
    df: pd.DataFrame,
    use_transformers: bool = False,
    batch_size: int = 64,
) -> pd.DataFrame:
    texts = df["line"].fillna("").tolist()
    n = len(texts)

    print(f"Processing {n:,} dialogue lines...")

    # ── Sentiment (VADER — vectorized, instant) ───────────────────────────────
    print("  [1/4] Sentiment (VADER)...")
    sent_rows = _sent.score_batch(texts)
    sent_df = pd.DataFrame(sent_rows, index=df.index)

    # ── Complexity ─────────────────────────────────────────────────────────────
    print("  [2/4] Linguistic complexity (textstat)...")
    cpx_rows = []
    for i in tqdm(range(0, n, CHUNK), desc="    complexity", leave=False):
        cpx_rows.extend(_cpx.score_batch(texts[i : i + CHUNK]))
    cpx_df = pd.DataFrame(cpx_rows, index=df.index)

    # ── Power language ─────────────────────────────────────────────────────────
    print("  [3/4] Power language (rule-based)...")
    pow_rows = []
    for i in tqdm(range(0, n, CHUNK), desc="    power", leave=False):
        pow_rows.extend(_pow.score_batch(texts[i : i + CHUNK]))
    pow_df = pd.DataFrame(pow_rows, index=df.index)

    # ── Entities & legal terms (spaCy) ────────────────────────────────────────
    print("  [4/4] Named entities + legal terms (spaCy)...")
    ent_rows = []
    for i in tqdm(range(0, n, CHUNK), desc="    entities", leave=False):
        batch = texts[i : i + CHUNK]
        ent_rows.extend(_ent.score_batch(batch))
    ent_df = pd.DataFrame(ent_rows, index=df.index)

    # ── Emotion (after entities, may trigger model download) ──────────────────
    print(f"  [+] Emotion ({'transformer' if use_transformers else 'keyword'} mode)...")
    emo_rows = []
    for i in tqdm(range(0, n, batch_size), desc="    emotion", leave=False):
        batch = texts[i : i + batch_size]
        emo_rows.extend(_emo.score_batch(batch, use_transformers=use_transformers, batch_size=batch_size))
    emo_df = pd.DataFrame(emo_rows, index=df.index)

    # ── Merge all ─────────────────────────────────────────────────────────────
    result = pd.concat(
        [df, sent_df, emo_df, cpx_df, pow_df, ent_df],
        axis=1,
    )

    return result


def run(
    data_dir: Path,
    use_transformers: bool = False,
    batch_size: int = 64,
) -> pd.DataFrame:
    in_path  = data_dir / "processed" / "suits_dialogue.csv"
    out_path = data_dir / "processed" / "suits_features.csv"

    if not in_path.exists():
        sys.exit(
            f"[ERROR] {in_path} not found.\n"
            "Run the main pipeline first: python src/pipeline.py"
        )

    print(f"Loading {in_path}...")
    df = pd.read_csv(in_path)
    print(f"  {len(df):,} lines across {df['episode_id'].nunique()} episodes\n")

    features_df = extract_features(df, use_transformers=use_transformers, batch_size=batch_size)
    features_df.to_csv(out_path, index=False)

    # ── Print summary stats ───────────────────────────────────────────────────
    print(f"\nSaved {len(features_df):,} rows → {out_path}")
    print("\n── Sentiment distribution ──────────────────────────")
    print(features_df["sentiment_label"].value_counts().to_string())
    print("\n── Emotion distribution ────────────────────────────")
    print(features_df["emotion_label"].value_counts().to_string())
    print("\n── Power language distribution ─────────────────────")
    print(features_df["power_category"].value_counts().to_string())
    print(f"\n── Legal language ──────────────────────────────────")
    legal = features_df["has_legal_language"].sum()
    print(f"Lines with legal terms: {legal:,} ({100*legal/len(features_df):.1f}%)")
    top_terms = (
        features_df["legal_terms"]
        .dropna()
        .str.split("|")
        .explode()
        .loc[lambda s: s != ""]
        .value_counts()
        .head(10)
    )
    print("Top legal terms:")
    print(top_terms.to_string())

    return features_df


def main():
    parser = argparse.ArgumentParser(description="Suits NLP Feature Extraction")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--use-transformers", action="store_true",
        help="Use j-hartmann/emotion-english-distilroberta-base for emotion (slower, more accurate)"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    run(Path(args.data_dir), use_transformers=args.use_transformers, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
