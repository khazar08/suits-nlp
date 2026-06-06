"""
Topic Evolution Model — Step 5 (BERTopic / LDA)

Discovers latent narrative arcs in Suits dialogue and tracks their intensity
episode-by-episode, giving you "Season 1 = fraud + mentorship, Season 4 = power
struggle" as a data artefact rather than a guess.

Fast mode  (default): sklearn LDA — no extra deps, runs in seconds.
Full mode  (--use-bertopic): BERTopic with sentence-transformers embeddings
           — richer topics, needs:  pip install bertopic sentence-transformers
"""

import re
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# ── English stopwords (no NLTK download needed) ──────────────────────────────
_STOPWORDS = set("""
a about above after again against all also am an and any are aren't as at
be because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further get got had hadn't has hasn't have haven't having he he'd
he'll he's her here here's hers herself him himself his how how's i i'd
i'll i'm i've if in into is isn't it it's its itself just know let's
like look me more most mustn't my myself no nor not now of off on once only
or other ought our ours ourselves out over own right said same see shall
she she'd she'll she's should shouldn't so some such than that that's the
their theirs them themselves then there there's these they they'd they'll
they're they've think this those through to too under until up very was
wasn't we we'd we'll we're we've were weren't what what's when when's
where where's which while who who's whom why why's will with won't would
wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())

# Suits-specific noise words to suppress from topics
_SUITS_NOISE = {
    "harvey", "mike", "louis", "donna", "jessica", "rachel",
    "specter", "litt", "ross", "pearson", "zane", "paulsen",
    "yeah", "okay", "hey", "right", "know", "just", "want",
    "got", "going", "come", "let", "say", "mean", "think",
    "look", "tell", "said", "get", "good", "way", "thing",
}

_ALL_STOP = _STOPWORDS | _SUITS_NOISE

# ── Text preprocessing ────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"'\w*", "", text)         # remove contractions
    return " ".join(w for w in text.split() if w not in _ALL_STOP and len(w) > 2)


def aggregate_episode_text(dialogue_df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate all cleaned dialogue lines per episode into one document."""
    agg = (
        dialogue_df.groupby(["episode_id", "season", "episode_num"])["line"]
        .apply(lambda lines: " ".join(lines.dropna().astype(str)))
        .reset_index()
        .rename(columns={"line": "raw_text"})
    )
    agg["text"] = agg["raw_text"].apply(_clean)
    agg = agg.sort_values(["season", "episode_num"]).reset_index(drop=True)
    return agg


# ── LDA Topic Model ───────────────────────────────────────────────────────────

class LDATopicModel:
    def __init__(self, n_topics: int = 10, n_words: int = 15, seed: int = 42):
        self.n_topics = n_topics
        self.n_words  = n_words
        self.seed     = seed
        self._vect:  Optional[CountVectorizer] = None
        self._model: Optional[LatentDirichletAllocation] = None
        self.vocab:  Optional[np.ndarray] = None

    def fit(self, texts: list[str]) -> "LDATopicModel":
        self._vect = CountVectorizer(
            max_features=8_000,
            min_df=2,
            max_df=0.90,
            ngram_range=(1, 2),
            stop_words=list(_ALL_STOP),
        )
        dtm = self._vect.fit_transform(texts)
        self.vocab = np.array(self._vect.get_feature_names_out())

        self._model = LatentDirichletAllocation(
            n_components=self.n_topics,
            max_iter=30,
            learning_method="online",
            random_state=self.seed,
            n_jobs=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(dtm)

        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        """Returns (n_docs × n_topics) topic-distribution matrix."""
        dtm = self._vect.transform(texts)
        return self._model.transform(dtm)

    def top_words(self) -> pd.DataFrame:
        """Top N words per topic as a long-format DataFrame."""
        rows = []
        for t_idx, comp in enumerate(self._model.components_):
            top = np.argsort(comp)[::-1][: self.n_words]
            for rank, w_idx in enumerate(top):
                rows.append({
                    "topic_id":  t_idx,
                    "rank":      rank + 1,
                    "keyword":   self.vocab[w_idx],
                    "weight":    round(comp[w_idx] / comp.sum(), 6),
                })
        return pd.DataFrame(rows)

    def topic_label(self, topic_id: int, n: int = 3) -> str:
        comp = self._model.components_[topic_id]
        top = np.argsort(comp)[::-1][:n]
        words = ", ".join(self.vocab[top])
        return _TOPIC_HINTS.get(frozenset(self.vocab[top[:2]]), f"Topic {topic_id}: {words}")


# ── Heuristic topic labels (matched by top-2 keyword pair) ───────────────────
# Will rarely fire on real data; mainly documents intent.
_TOPIC_HINTS: dict[frozenset, str] = {
    frozenset({"firm", "partner"}): "Firm Politics",
    frozenset({"case", "client"}): "Client Cases",
    frozenset({"deal", "merger"}): "Corporate Deals",
    frozenset({"trial", "court"}): "Litigation",
    frozenset({"harvard", "law"}): "Legal Career",
    frozenset({"money", "pay"}): "Financial Pressure",
    frozenset({"trust", "lie"}): "Betrayal & Trust",
    frozenset({"love", "relationship"}): "Personal Relationships",
    frozenset({"power", "control"}): "Power Struggles",
    frozenset({"secret", "truth"}): "Secrets & Revelations",
}


# ── BERTopic (optional) ───────────────────────────────────────────────────────

class BERTopicModel:
    """
    Wrapper around BERTopic for richer topic discovery.
    Requires: pip install bertopic sentence-transformers
    """
    def __init__(self, n_topics: int = 10):
        try:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "BERTopic dependencies missing.\n"
                "pip install bertopic sentence-transformers"
            )
        self.n_topics = n_topics
        self._model = BERTopic(nr_topics=n_topics, verbose=False)
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self._topics: Optional[list] = None

    def fit(self, texts: list[str]) -> "BERTopicModel":
        embeddings = self._embedder.encode(texts, show_progress_bar=True)
        self._topics, _ = self._model.fit_transform(texts, embeddings)
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        topics, probs = self._model.transform(texts)
        return np.array(probs) if probs is not None else np.zeros((len(texts), self.n_topics))

    def top_words(self) -> pd.DataFrame:
        rows = []
        for t_id in set(self._topics):
            if t_id == -1:
                continue
            words = self._model.get_topic(t_id)
            for rank, (word, weight) in enumerate(words[:15]):
                rows.append({"topic_id": t_id, "rank": rank+1,
                             "keyword": word, "weight": round(weight, 6)})
        return pd.DataFrame(rows)

    def topic_label(self, topic_id: int, n: int = 3) -> str:
        words = self._model.get_topic(topic_id)
        if not words:
            return f"Topic {topic_id}"
        return ", ".join(w for w, _ in words[:n])
