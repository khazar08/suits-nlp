"""
Prediction Models — Step 5.2

Three models with a unified interface:

  RandomForestPredictor   — interpretable baseline, handles small datasets
  XGBoostPredictor        — gradient boosting, usually best on tabular data
  LSTMPredictor           — sequence model capturing temporal patterns

All expose:
  .fit(X_train, y_train)
  .predict_proba_dominance(feature_df, episode_id) → {char: prob}
  .save(path) / .load(path)
"""

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from predict.features import MAIN_CHARS, FEATURE_COLS


# ── Shared prediction helper ──────────────────────────────────────────────────

def _rank_characters(
    clf,
    feature_df: pd.DataFrame,
    episode_id: str,
    feature_cols: list[str],
) -> dict[str, float]:
    """
    Run binary classifier over all characters for a given episode.
    Returns {character: P(is_dominant)} normalized to sum to 1.
    """
    ep_rows = feature_df[feature_df["episode_id"] == episode_id].copy()
    if ep_rows.empty:
        return {}

    available = [c for c in feature_cols if c in ep_rows.columns]
    probs: dict[str, float] = {}

    for _, row in ep_rows.iterrows():
        char = row["character"]
        x = row[available].fillna(0).values.reshape(1, -1)
        prob = clf.predict_proba(x)[0][1]   # P(class=1, i.e. is_dominant)
        probs[char] = prob

    total = sum(probs.values())
    if total > 1e-9:
        probs = {k: round(v / total, 4) for k, v in probs.items()}

    return dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))


# ── Random Forest ─────────────────────────────────────────────────────────────

class RandomForestPredictor:
    name = "random_forest"

    def __init__(self, n_estimators: int = 300, **kwargs):
        from sklearn.ensemble import RandomForestClassifier
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            **kwargs,
        )
        self._feature_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestPredictor":
        self._feature_cols = list(X.columns)
        self.clf.fit(X, y)
        return self

    def predict_proba_dominance(
        self, feature_df: pd.DataFrame, episode_id: str
    ) -> dict[str, float]:
        return _rank_characters(self.clf, feature_df, episode_id, self._feature_cols)

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self.clf.feature_importances_, index=self._feature_cols
        ).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"clf": self.clf, "cols": self._feature_cols}, f)

    @classmethod
    def load(cls, path: Path) -> "RandomForestPredictor":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        m = cls.__new__(cls)
        m.clf, m._feature_cols = obj["clf"], obj["cols"]
        return m


# ── XGBoost ───────────────────────────────────────────────────────────────────

class XGBoostPredictor:
    name = "xgboost"

    def __init__(self, **kwargs):
        try:
            import xgboost as xgb
            self._xgb = xgb
        except Exception as e:
            raise ImportError(
                f"XGBoost unavailable ({e}). "
                "On Mac: brew install libomp, then pip install xgboost"
            ) from e

        defaults = dict(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        defaults.update(kwargs)
        self.clf = xgb.XGBClassifier(**defaults)
        self._feature_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostPredictor":
        self._feature_cols = list(X.columns)
        # Compute scale_pos_weight for class imbalance
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        self.clf.set_params(scale_pos_weight=n_neg / max(n_pos, 1))
        self.clf.fit(X, y)
        return self

    def predict_proba_dominance(
        self, feature_df: pd.DataFrame, episode_id: str
    ) -> dict[str, float]:
        return _rank_characters(self.clf, feature_df, episode_id, self._feature_cols)

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self.clf.feature_importances_, index=self._feature_cols
        ).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.clf.save_model(str(path.with_suffix(".json")))
        (path.parent / f"{path.stem}_cols.json").write_text(
            json.dumps(self._feature_cols)
        )

    @classmethod
    def load(cls, path: Path) -> "XGBoostPredictor":
        import xgboost as xgb
        path = Path(path)
        m = cls.__new__(cls)
        m._xgb = xgb
        m.clf = xgb.XGBClassifier()
        m.clf.load_model(str(path.with_suffix(".json")))
        m._feature_cols = json.loads(
            (path.parent / f"{path.stem}_cols.json").read_text()
        )
        return m


# ── LSTM ──────────────────────────────────────────────────────────────────────

class LSTMPredictor:
    """
    Sequence model: given T=5 consecutive episodes, predict next dominant.
    Trained in wide format: each episode is a flat vector of all main chars' features.
    """
    name = "lstm"

    def __init__(
        self,
        input_dim: int = 0,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.3,
        seq_len: int = 5,
        n_classes: int = len(MAIN_CHARS),
        lr: float = 1e-3,
        max_epochs: int = 100,
        patience: int = 10,
    ):
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers
        self.dropout    = dropout
        self.seq_len    = seq_len
        self.n_classes  = n_classes
        self.lr         = lr
        self.max_epochs = max_epochs
        self.patience   = patience
        self._net       = None
        self._feat_cols: list[str] = []

    def _build_net(self, input_dim: int):
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self, inp, hid, layers, drop, n_cls):
                super().__init__()
                self.lstm = nn.LSTM(inp, hid, layers, batch_first=True,
                                    dropout=drop if layers > 1 else 0)
                self.drop = nn.Dropout(drop)
                self.fc   = nn.Linear(hid, n_cls)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(self.drop(out[:, -1, :]))  # last time step

        return _Net(input_dim, self.hidden_dim, self.n_layers,
                    self.dropout, self.n_classes)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LSTMPredictor":
        """
        X: wide episode matrix (n_episodes × features)
        y: integer index into MAIN_CHARS (n_episodes,)
        """
        import torch, torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        self._feat_cols = list(X.columns)
        self.input_dim  = X.shape[1]

        # Filter valid targets
        mask = y >= 0
        X_arr = X[mask].values.astype(np.float32)
        y_arr = y[mask].values.astype(np.int64)

        # Build sliding window sequences
        seqs, labels = [], []
        for i in range(len(X_arr) - self.seq_len):
            seqs.append(X_arr[i : i + self.seq_len])
            labels.append(y_arr[i + self.seq_len])

        if len(seqs) < 10:
            print(f"  [LSTM] Only {len(seqs)} sequences — need ≥10 to train meaningfully.")
            self._net = self._build_net(self.input_dim)
            return self

        X_t = torch.tensor(np.array(seqs))
        y_t = torch.tensor(np.array(labels))

        # Train / val split (80/20)
        split = int(0.8 * len(X_t))
        ds_train = TensorDataset(X_t[:split], y_t[:split])
        ds_val   = TensorDataset(X_t[split:], y_t[split:])
        dl_train = DataLoader(ds_train, batch_size=16, shuffle=True)
        dl_val   = DataLoader(ds_val,   batch_size=16)

        device = (
            "mps"  if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        net = self._build_net(self.input_dim).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()

        best_val, no_improve = float("inf"), 0
        best_state = None

        for epoch in range(self.max_epochs):
            net.train()
            for xb, yb in dl_train:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss_fn(net(xb), yb).backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()

            net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in dl_val:
                    val_loss += loss_fn(net(xb.to(device)), yb.to(device)).item()
            val_loss /= max(len(dl_val), 1)

            if val_loss < best_val - 1e-4:
                best_val, no_improve = val_loss, 0
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

            if (epoch + 1) % 10 == 0:
                print(f"    epoch {epoch+1:3d}  val_loss={val_loss:.4f}")

        if best_state:
            net.load_state_dict(best_state)
        self._net = net.cpu()
        print(f"  [LSTM] Training done — best val_loss={best_val:.4f}")
        return self

    def predict_proba_dominance(
        self, X_wide: pd.DataFrame, episode_id: str
    ) -> dict[str, float]:
        """
        X_wide: full wide episode matrix (index = episode_ids).
        Returns {char: prob} for the episode AFTER episode_id.
        """
        import torch, torch.nn.functional as F

        if self._net is None:
            return {}

        ep_ids = list(X_wide.index)
        if episode_id not in ep_ids:
            return {}

        idx = ep_ids.index(episode_id)
        start = max(0, idx - self.seq_len + 1)
        seq = X_wide.iloc[start : idx + 1][self._feat_cols].values.astype(np.float32)

        # Pad if needed
        if len(seq) < self.seq_len:
            pad = np.zeros((self.seq_len - len(seq), seq.shape[1]), dtype=np.float32)
            seq = np.vstack([pad, seq])

        x = torch.tensor(seq[np.newaxis])   # (1, seq_len, features)
        self._net.eval()
        with torch.no_grad():
            logits = self._net(x)
            probs  = F.softmax(logits, dim=-1).squeeze().numpy()

        return {MAIN_CHARS[i]: round(float(probs[i]), 4) for i in range(len(MAIN_CHARS))}

    def save(self, path: Path) -> None:
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {k: v for k, v in self.__dict__.items() if k != "_net"}
        torch.save({"state": self._net.state_dict() if self._net else None,
                    "meta": meta}, path)

    @classmethod
    def load(cls, path: Path) -> "LSTMPredictor":
        import torch
        obj = torch.load(path, map_location="cpu")
        m = cls(**{k: v for k, v in obj["meta"].items() if k != "_net"})
        if obj["state"] and m.input_dim > 0:
            m._net = m._build_net(m.input_dim)
            m._net.load_state_dict(obj["state"])
        return m
