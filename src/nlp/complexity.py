"""
Linguistic complexity metrics per dialogue line.
No external NLP dependencies — pure Python + regex.

Metrics:
  word_count         — total words
  char_count         — total non-space characters
  unique_word_ratio  — type-token ratio (lexical diversity)
  flesch_ease        — Flesch Reading Ease (higher = simpler)
  gunning_fog        — Gunning Fog grade level
  avg_word_length    — mean characters per word
  is_question        — ends with ?
  is_exclamation     — ends with !
"""

import re

_SENT_SPLIT = re.compile(r"[.!?]+")
_VOWEL_RUN = re.compile(r"[aeiouy]+", re.IGNORECASE)
_TRAILING_E = re.compile(r"e$", re.IGNORECASE)


def _syllables(word: str) -> int:
    word = re.sub(r"[^a-zA-Z]", "", word)
    if not word:
        return 0
    count = len(_VOWEL_RUN.findall(word))
    # Silent trailing -e reduces count only if more than 1 syllable
    if _TRAILING_E.search(word) and count > 1:
        count -= 1
    return max(1, count)


def _n_sentences(text: str) -> int:
    parts = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    return max(1, len(parts))


def score(text: str) -> dict:
    text = text.strip()
    if not text:
        return _empty()

    words = text.split()
    n_words = len(words)
    if n_words == 0:
        return _empty()

    n_sents    = _n_sentences(text)
    syllables  = [_syllables(w) for w in words]
    n_sylls    = sum(syllables)
    char_count = sum(len(w) for w in words)

    # Flesch Reading Ease
    asl = n_words / n_sents          # avg sentence length
    asw = n_sylls / n_words          # avg syllables/word
    flesch = 206.835 - 1.015 * asl - 84.6 * asw

    # Gunning Fog
    complex_count = sum(1 for s in syllables if s >= 3)
    fog = 0.4 * (asl + 100 * complex_count / n_words)

    unique_ratio = len({w.lower() for w in words}) / n_words

    return {
        "word_count":        n_words,
        "char_count":        char_count,
        "unique_word_ratio": round(unique_ratio, 4),
        "flesch_ease":       round(flesch, 2),
        "gunning_fog":       round(fog, 2),
        "avg_word_length":   round(char_count / n_words, 3),
        "is_question":       text.rstrip().endswith("?"),
        "is_exclamation":    text.rstrip().endswith("!"),
    }


def score_batch(texts: list[str]) -> list[dict]:
    return [score(t) for t in texts]


def _empty() -> dict:
    return {
        "word_count": 0, "char_count": 0,
        "unique_word_ratio": 0.0, "flesch_ease": 0.0,
        "gunning_fog": 0.0, "avg_word_length": 0.0,
        "is_question": False, "is_exclamation": False,
    }
