"""
VADER sentiment scoring — fast, no GPU needed, tuned for conversational text.
Returns compound score + positive/negative/neutral breakdown per line.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_sia: SentimentIntensityAnalyzer | None = None


def _get_sia() -> SentimentIntensityAnalyzer:
    global _sia
    if _sia is None:
        _sia = SentimentIntensityAnalyzer()
    return _sia


def score(text: str) -> dict:
    s = _get_sia().polarity_scores(text)
    c = s["compound"]
    return {
        "vader_compound": round(c, 4),
        "vader_pos":      round(s["pos"], 4),
        "vader_neg":      round(s["neg"], 4),
        "vader_neu":      round(s["neu"], 4),
        "sentiment_label": (
            "positive" if c >= 0.05 else
            "negative" if c <= -0.05 else
            "neutral"
        ),
    }


def score_batch(texts: list[str]) -> list[dict]:
    sia = _get_sia()
    results = []
    for text in texts:
        s = sia.polarity_scores(text)
        c = s["compound"]
        results.append({
            "vader_compound": round(c, 4),
            "vader_pos":      round(s["pos"], 4),
            "vader_neg":      round(s["neg"], 4),
            "vader_neu":      round(s["neu"], 4),
            "sentiment_label": (
                "positive" if c >= 0.05 else
                "negative" if c <= -0.05 else
                "neutral"
            ),
        })
    return results
