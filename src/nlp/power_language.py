"""
Legal Power Language Detector for Suits dialogue.

Classifies each line into one of five categories using regex pattern matching:
  commanding    — issuing orders, asserting authority
  defensive     — denying, deflecting, justifying
  manipulative  — conditional threats, leveraging relationships
  assertive     — confident declarations, guarantees, promises
  neutral       — none of the above

Also returns per-category raw scores (pattern hit counts) for fine-grained analysis.
"""

import re
from typing import NamedTuple


class PowerScores(NamedTuple):
    commanding:   float
    defensive:    float
    manipulative: float
    assertive:    float


# ── Pattern banks ─────────────────────────────────────────────────────────────

_COMMANDING = [
    r"\byou (will|must|shall|are going to)\b",
    r"\b(do it|get it done|make it happen|handle it)\b",
    r"\bI (need|want|require|expect) you to\b",
    r"\b(that['']s (an order|final|non-negotiable|settled))\b",
    r"\b(listen to me|you['']re fired|you['']re done)\b",
    r"\b(I['']m telling you|I['']m ordering)\b",
    r"\b(close the deal|make the call|sign (it|the papers?))\b",
    r"\b(get out|leave now|you['']re dismissed)\b",
    r"\bI decide\b",
    r"\b(not (up for|open to) discussion)\b",
]

_DEFENSIVE = [
    r"\bI (didn['']t|never|wouldn['']t|haven['']t|couldn['']t)\b",
    r"\bthat['']s not (true|right|what|how|fair|my fault)\b",
    r"\b(it wasn['']t me|I had nothing to do with)\b",
    r"\b(let me explain|I can explain|you don['']t understand)\b",
    r"\byou['']re wrong\b",
    r"\b(I was just|I only|I merely)\b",
    r"\bI didn['']t (know|mean|intend|realize)\b",
    r"\b(that['']s not what I said|I never said that)\b",
    r"\b(I was following orders|I had no choice)\b",
    r"\bdon['']t blame me\b",
    r"\b(I['']m not (guilty|responsible|at fault))\b",
]

_MANIPULATIVE = [
    r"\bif you (don['']t|won['']t|can['']t|refuse)\b",
    r"\bunless you\b",
    r"\byou owe (me|us)\b",
    r"\bafter everything (I['']ve|we['']ve) done\b",
    r"\b(I could ruin|I could destroy|I can make your life)\b",
    r"\byou know what (happens|will happen) if\b",
    r"\b(I have something on|I know what you did)\b",
    r"\b(you need me|without me you['']re|you can['']t survive without)\b",
    r"\b(think carefully|think about what you['']re doing)\b",
    r"\b(it would be a shame if|imagine if)\b",
    r"\byou['']ll regret\b",
    r"\bdon['']t make me\b",
]

_ASSERTIVE = [
    r"\bI (guarantee|promise|assure|swear)\b",
    r"\b(trust me|believe me)\b",
    r"\bI['']m (absolutely|completely|totally) (certain|sure|confident)\b",
    r"\bmark my words\b",
    r"\bI['']ll (make sure|personally|stake my reputation)\b",
    r"\bI know (for a fact|exactly|precisely)\b",
    r"\b(there is no doubt|without a doubt|no question)\b",
    r"\bI['']m the best\b",
    r"\b(I win|we will win|this is a win)\b",
    r"\bcount on (me|it|that)\b",
    r"\bI stand by\b",
]

# Pre-compile all patterns
_COMPILED = {
    "commanding":   [re.compile(p, re.IGNORECASE) for p in _COMMANDING],
    "defensive":    [re.compile(p, re.IGNORECASE) for p in _DEFENSIVE],
    "manipulative": [re.compile(p, re.IGNORECASE) for p in _MANIPULATIVE],
    "assertive":    [re.compile(p, re.IGNORECASE) for p in _ASSERTIVE],
}

_CATEGORY_ORDER = ["commanding", "defensive", "manipulative", "assertive"]


def score(text: str) -> dict:
    hits: dict[str, int] = {}
    for cat, patterns in _COMPILED.items():
        hits[cat] = sum(1 for p in patterns if p.search(text))

    total = sum(hits.values())

    if total == 0:
        return {
            "power_category":    "neutral",
            "power_score":       0.0,
            "commanding_score":  0.0,
            "defensive_score":   0.0,
            "manipulative_score": 0.0,
            "assertive_score":   0.0,
        }

    dominant = max(hits, key=hits.get)  # type: ignore[arg-type]
    dom_score = hits[dominant] / total  # relative dominance

    return {
        "power_category":     dominant,
        "power_score":        round(dom_score, 4),
        "commanding_score":   hits["commanding"],
        "defensive_score":    hits["defensive"],
        "manipulative_score": hits["manipulative"],
        "assertive_score":    hits["assertive"],
    }


def score_batch(texts: list[str]) -> list[dict]:
    return [score(t) for t in texts]
