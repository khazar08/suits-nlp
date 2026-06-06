"""
Prediction Pipeline — Step 5 Orchestrator

Reads from data/processed/:
  suits_influence.csv, suits_centrality.csv, suits_features.csv,
  suits_dominance.csv, suits_power_trajectory.csv

Writes to data/processed/:
  suits_predict_features.csv   — full feature matrix with targets
  suits_eval_results.csv       — per-model evaluation metrics
  suits_feature_importance.csv — top features from RF + XGBoost

Saves trained models to data/models/

Usage:
    python src/predict/pipeline.py                    # train all 3 models
    python src/predict/pipeline.py --model rf         # only random forest
    python src/predict/pipeline.py --predict S05E08   # predict next after S05E08
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report,
    top_k_accuracy_score, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from predict.features import (
    build_feature_matrix, build_lstm_matrix, get_X_y, MAIN_CHARS, FEATURE_COLS
)
from predict.models import RandomForestPredictor, XGBoostPredictor, LSTMPredictor

MIN_EPISODES = 10   # minimum for any meaningful training


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(processed_dir: Path) -> dict[str, pd.DataFrame]:
    required = {
        "influence":   "suits_influence.csv",
        "centrality":  "suits_centrality.csv",
        "nlp":         "suits_features.csv",
        "dominance":   "suits_dominance.csv",
    }
    data = {}
    for key, fname in required.items():
        path = processed_dir / fname
        if not path.exists():
            sys.exit(f"[ERROR] {path} not found. Run earlier pipeline steps first.")
        data[key] = pd.read_csv(path)
    return data


# ── Temporal train / test split ───────────────────────────────────────────────

def temporal_split(
    feature_df: pd.DataFrame,
    train_seasons: list[int] | None = None,
    test_season: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split strictly by time — no leaking future data into training.

    Default: train S01-S07, test S08-S09.
    With only 1 season of data, uses 80/20 episode split.
    """
    seasons = sorted(feature_df["season"].unique())
    n_seasons = len(seasons)

    if n_seasons == 1:
        ep_ids = sorted(feature_df["episode_idx"].unique())
        split_idx = int(0.8 * len(ep_ids))
        train_eps = ep_ids[:split_idx]
        train = feature_df[feature_df["episode_idx"].isin(train_eps)]
        test  = feature_df[~feature_df["episode_idx"].isin(train_eps)]
        return train, test

    if train_seasons is None:
        cutoff = seasons[int(0.75 * n_seasons)]
        train = feature_df[feature_df["season"] <= cutoff]
        test  = feature_df[feature_df["season"] > cutoff]
    else:
        train = feature_df[feature_df["season"].isin(train_seasons)]
        test_s = [test_season] if test_season else [s for s in seasons if s not in train_seasons]
        test  = feature_df[feature_df["season"].isin(test_s)]

    return train, test


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    feature_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """
    Per-episode top-1 and top-3 accuracy.
    For each test episode, rank all characters by predicted probability,
    check if the actual dominant character is in top-1 / top-3.
    """
    dom = test_df[["episode_id", "next_dominant"]].dropna().drop_duplicates()
    test_episodes = dom["episode_id"].tolist()

    top1_hits, top3_hits = 0, 0
    predictions = []

    for ep_id in test_episodes:
        true_dom = dom[dom["episode_id"] == ep_id]["next_dominant"].iloc[0]
        ranked = model.predict_proba_dominance(feature_df, ep_id)
        if not ranked:
            continue

        chars_ranked = list(ranked.keys())
        predicted = chars_ranked[0]
        top3 = chars_ranked[:3]

        predictions.append({
            "episode_id":   ep_id,
            "true_dominant": true_dom,
            "predicted":    predicted,
            "top3":         " | ".join(top3),
            "correct_top1": predicted == true_dom,
            "correct_top3": true_dom in top3,
            **{f"prob_{c.split()[0]}": ranked.get(c, 0) for c in MAIN_CHARS},
        })

        top1_hits += int(predicted == true_dom)
        top3_hits += int(true_dom in top3)

    n = len(test_episodes)
    result = {
        "model":        model.name,
        "n_episodes":   n,
        "top1_accuracy": round(top1_hits / n, 4) if n else 0,
        "top3_accuracy": round(top3_hits / n, 4) if n else 0,
    }
    return result, pd.DataFrame(predictions)


# ── Inference: predict next episode ──────────────────────────────────────────

def predict_next(model, feature_df: pd.DataFrame, episode_id: str) -> None:
    """Pretty-print prediction for the episode after episode_id."""
    ranked = model.predict_proba_dominance(feature_df, episode_id)
    if not ranked:
        print(f"  No prediction available for episode after {episode_id}")
        return

    print(f"\n{'═'*52}")
    print(f"  PREDICTION: dominant character after {episode_id}")
    print(f"  Model: {model.name}")
    print(f"{'═'*52}")
    for i, (char, prob) in enumerate(list(ranked.items())[:5], 1):
        bar = "█" * int(prob * 40)
        marker = " ◄ PREDICTED" if i == 1 else ""
        print(f"  {i}. {char:<25} {prob:.3f}  {bar}{marker}")


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run(
    data_dir: Path,
    models_to_train: list[str] = ("rf", "xgb", "lstm"),
    predict_after: str | None = None,
) -> dict:
    processed = data_dir / "processed"
    models_dir = data_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────────────────
    _h("Loading data")
    data = load_data(processed)
    n_eps = data["influence"]["episode_id"].nunique()
    print(f"  Episodes: {n_eps}  |  Characters in influence: {data['influence']['character'].nunique()}")

    if n_eps < MIN_EPISODES:
        print(f"\n  ⚠  Only {n_eps} episode(s) of data.")
        print(f"     Models need ≥{MIN_EPISODES} episodes to train meaningfully.")
        print(f"     Run the full scrape (python src/pipeline.py) first.")
        print(f"\n  Running feature engineering demo on available data...")

    # ── Feature engineering ───────────────────────────────────────────────────
    _h("Building feature matrix")
    feature_df = build_feature_matrix(
        data["influence"], data["centrality"],
        data["nlp"], data["dominance"],
    )
    feature_df.to_csv(processed / "suits_predict_features.csv", index=False)
    print(f"  Shape: {feature_df.shape}")
    print(f"  Characters: {sorted(feature_df['character'].unique())}")

    X, y = get_X_y(feature_df)
    print(f"  Training-ready rows (lag satisfied): {len(X)}")
    print(f"  Target balance — dominant: {y.sum()}, non-dominant: {(y==0).sum()}")

    if n_eps < MIN_EPISODES or len(X) < 10:
        print("\n  Skipping model training — not enough episodes.")
        print("  Feature matrix saved. Re-run after full scrape.")
        return {"feature_df": feature_df}

    # ── Train / test split ────────────────────────────────────────────────────
    _h("Temporal train / test split")
    train_df, test_df = temporal_split(feature_df)
    X_train, y_train = get_X_y(train_df)
    X_test,  y_test  = get_X_y(test_df)
    print(f"  Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

    results, predictors = [], {}

    # ── Random Forest ─────────────────────────────────────────────────────────
    if "rf" in models_to_train:
        _h("Training Random Forest")
        rf = RandomForestPredictor(n_estimators=300)
        rf.fit(X_train, y_train)
        rf.save(models_dir / "rf_model.pkl")
        predictors["rf"] = rf

        metrics, preds_df = evaluate_model(rf, feature_df, test_df)
        results.append(metrics)
        print(f"  Top-1 accuracy: {metrics['top1_accuracy']:.3f}")
        print(f"  Top-3 accuracy: {metrics['top3_accuracy']:.3f}")

        fi = rf.feature_importance().head(15)
        print("\n  Top 15 features:")
        for feat, imp in fi.items():
            print(f"    {feat:<40} {imp:.4f}")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    if "xgb" in models_to_train:
        _h("Training XGBoost")
        try:
            xgb = XGBoostPredictor()
            xgb.fit(X_train, y_train)
            xgb.save(models_dir / "xgb_model")
            predictors["xgb"] = xgb

            metrics, _ = evaluate_model(xgb, feature_df, test_df)
            results.append(metrics)
            print(f"  Top-1 accuracy: {metrics['top1_accuracy']:.3f}")
            print(f"  Top-3 accuracy: {metrics['top3_accuracy']:.3f}")
        except ImportError:
            print("  XGBoost not installed — skipping.")

    # ── LSTM ──────────────────────────────────────────────────────────────────
    if "lstm" in models_to_train:
        _h("Training LSTM")
        X_wide, y_wide = build_lstm_matrix(feature_df)
        train_eps = train_df["episode_id"].unique()
        X_wide_train = X_wide[X_wide.index.isin(train_eps)]
        y_wide_train = y_wide[y_wide.index.isin(train_eps)]

        lstm = LSTMPredictor(input_dim=X_wide.shape[1])
        lstm._feat_cols = list(X_wide.columns)
        lstm.fit(X_wide_train, y_wide_train)
        lstm.save(models_dir / "lstm_model.pt")
        predictors["lstm"] = lstm

        # Evaluate LSTM episode-by-episode
        dom = test_df[["episode_id","next_dominant"]].dropna().drop_duplicates()
        top1, top3 = 0, 0
        for _, row in dom.iterrows():
            ranked = lstm.predict_proba_dominance(X_wide, row["episode_id"])
            chars = list(ranked.keys())
            top1 += int(chars[0] == row["next_dominant"])
            top3 += int(row["next_dominant"] in chars[:3])
        n = len(dom)
        metrics = {
            "model": "lstm",
            "n_episodes": n,
            "top1_accuracy": round(top1/n, 4) if n else 0,
            "top3_accuracy": round(top3/n, 4) if n else 0,
        }
        results.append(metrics)
        print(f"  Top-1 accuracy: {metrics['top1_accuracy']:.3f}")
        print(f"  Top-3 accuracy: {metrics['top3_accuracy']:.3f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    _h("Evaluation Summary")
    if results:
        eval_df = pd.DataFrame(results)
        eval_df.to_csv(processed / "suits_eval_results.csv", index=False)
        print(eval_df.to_string(index=False))

    # ── Inference ─────────────────────────────────────────────────────────────
    if predict_after and predictors:
        _h(f"Inference — predicting after {predict_after}")
        best_model = predictors.get("xgb") or predictors.get("rf") or list(predictors.values())[0]
        predict_next(best_model, feature_df, predict_after)

    return {"feature_df": feature_df, "predictors": predictors,
            "results": results if results else []}


def _h(title: str) -> None:
    print(f"\n{'═'*58}\n  {title}\n{'═'*58}")


def main():
    parser = argparse.ArgumentParser(description="Suits Dominance Predictor")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model",    default="rf,xgb,lstm",
                        help="Comma-separated: rf, xgb, lstm")
    parser.add_argument("--predict",  default=None,
                        help="Predict dominant character after this episode (e.g. S05E08)")
    args = parser.parse_args()

    run(
        data_dir=Path(args.data_dir),
        models_to_train=args.model.split(","),
        predict_after=args.predict,
    )


if __name__ == "__main__":
    main()
