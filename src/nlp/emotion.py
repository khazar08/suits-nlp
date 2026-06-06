"""
Emotion classification for Suits dialogue lines.

Two modes — selected automatically based on what's installed:

  FAST (default):
    Keyword-based heuristic. Instant, no GPU. Returns one of:
    anger | fear | joy | sadness | disgust | surprise | trust | neutral

  TRANSFORMER (--use-transformers flag in features.py):
    Uses j-hartmann/emotion-english-distilroberta-base (HuggingFace).
    Classifies into: anger | disgust | fear | joy | neutral | sadness | surprise
    Requires: pip install transformers torch
    Runs in batches on CPU or MPS (Apple Silicon).
"""

import re

# ── Keyword banks for fast mode ───────────────────────────────────────────────

_KEYWORDS: dict[str, list[str]] = {
    "anger": [
        "angry", "furious", "rage", "outraged", "livid", "irate",
        "damn", "hell", "bastard", "pissed", "infuriated", "enraged",
        "how dare", "unacceptable", "ridiculous", "absurd", "disgusting",
        "you bastard", "son of a bitch", "screw you",
    ],
    "fear": [
        "afraid", "scared", "terrified", "frightened", "worried", "nervous",
        "anxious", "panic", "dread", "terror", "threat", "threaten",
        "what if they", "what happens if", "I'm not sure we", "risk",
    ],
    "joy": [
        "love", "happy", "great", "wonderful", "excited", "brilliant",
        "congratulations", "perfect", "amazing", "fantastic", "celebrate",
        "proud", "thrilled", "delighted", "ecstatic", "won", "victory",
    ],
    "sadness": [
        "sorry", "sad", "unfortunate", "regret", "miss", "grief", "hurt",
        "disappointed", "heartbroken", "lonely", "miserable", "cry",
        "tears", "mourn", "lost", "failure", "failed", "let you down",
    ],
    "disgust": [
        "disgusting", "pathetic", "pitiful", "shameful", "embarrassing",
        "revolting", "repulsive", "vile", "contempt", "beneath",
        "low", "dirty", "corrupt", "sleazy",
    ],
    "surprise": [
        "really", "seriously", "no way", "impossible", "unbelievable",
        "what", "how", "I didn't expect", "didn't see that",
        "shocked", "stunned", "astonished", "blow my mind",
    ],
    "trust": [
        "trust", "believe", "honest", "promise", "guarantee", "faith",
        "loyal", "reliable", "depend on", "count on", "confident in",
        "I've got your back", "you can count on me",
    ],
}

# Pre-compile as single pattern per emotion
_COMPILED_KW: dict[str, re.Pattern] = {
    emotion: re.compile(
        "|".join(re.escape(kw) for kw in sorted(kws, key=len, reverse=True)),
        re.IGNORECASE,
    )
    for emotion, kws in _KEYWORDS.items()
}


def _score_fast(text: str) -> dict:
    hits = {em: len(pat.findall(text)) for em, pat in _COMPILED_KW.items()}
    total = sum(hits.values())

    if total == 0:
        return {"emotion_label": "neutral", "emotion_score": 0.0, "emotion_mode": "keyword"}

    dominant = max(hits, key=hits.get)  # type: ignore[arg-type]
    return {
        "emotion_label": dominant,
        "emotion_score": round(hits[dominant] / total, 4),
        "emotion_mode": "keyword",
    }


def score_batch_fast(texts: list[str]) -> list[dict]:
    return [_score_fast(t) for t in texts]


# ── Transformer mode ──────────────────────────────────────────────────────────

_pipe = None  # lazy-loaded HuggingFace pipeline
_TRANSFORMER_MODEL = "j-hartmann/emotion-english-distilroberta-base"


def _load_transformer():
    global _pipe
    if _pipe is not None:
        return _pipe

    from transformers import pipeline
    import torch

    device = (
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    print(f"  Loading emotion model on {device}: {_TRANSFORMER_MODEL}")
    _pipe = pipeline(
        "text-classification",
        model=_TRANSFORMER_MODEL,
        device=device,
        truncation=True,
        max_length=128,
    )
    return _pipe


def score_batch_transformer(texts: list[str], batch_size: int = 64) -> list[dict]:
    pipe = _load_transformer()
    results: list[dict] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Truncate each text to avoid tokenizer warnings on very long lines
        batch = [t[:512] for t in batch]
        preds = pipe(batch)
        for pred in preds:
            results.append({
                "emotion_label": pred["label"].lower(),
                "emotion_score": round(pred["score"], 4),
                "emotion_mode": "transformer",
            })

    return results


def score_batch(texts: list[str], use_transformers: bool = False, batch_size: int = 64) -> list[dict]:
    if use_transformers:
        try:
            return score_batch_transformer(texts, batch_size=batch_size)
        except Exception as e:
            print(f"  Transformer emotion failed ({e}), falling back to keyword mode.")
    return score_batch_fast(texts)
