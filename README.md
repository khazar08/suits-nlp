# Suits NLP

A research project that reverse-engineers the power dynamics of the TV show *Suits* using web scraping, NLP, graph machine learning, and predictive modeling. Tracks character influence episode-by-episode, predicts who dominates next, and surfaces narrative arc shifts across all nine seasons.

---

## What it does

**Collects** every line of dialogue from all 134 episodes via Springfield! Springfield! transcripts, optionally aligned to SRT subtitle timestamps.

**Extracts** per-line NLP features: VADER sentiment, Flesch/Gunning Fog readability, power language category (commanding / defensive / manipulative / assertive), emotion label (7 classes), named entities, and legal term density.

**Builds** a weighted character co-occurrence graph per episode — nodes are characters, edges are shared scenes, weights are interaction intensity. Computes PageRank, betweenness, eigenvector, and degree centrality.

**Scores** each character's influence per episode as a weighted combination of PageRank, mention volume share, and sentiment control. Tracks this as a power trajectory across the full series.

**Predicts** who dominates the next episode using three models: Random Forest, XGBoost, and a PyTorch LSTM trained on temporal sequences of influence features.

**Discovers** latent narrative themes using LDA topic modeling and tracks their intensity shift across seasons — e.g. "Season 1 = mentorship + fraud" vs "Season 6 = power struggle + litigation".

**Visualizes** everything in a 5-tab Streamlit dashboard with interactive Plotly charts, a live network graph, and a real-time dominance predictor.

---

## Project structure

```
suits-nlp/
├── src/
│   ├── pipeline.py              # Step 1 orchestrator: scrape → scene → graph data
│   ├── scraper.py               # Scrapes Springfield! Springfield! transcripts
│   ├── srt_parser.py            # Parses SRT subtitle files for timestamps
│   ├── aligner.py               # Fuzzy-matches transcript lines to SRT timestamps
│   ├── scene_segmenter.py       # Segments dialogue into scenes
│   ├── character_extractor.py   # Builds co-occurrence edges from scenes
│   ├── nlp/
│   │   ├── sentiment.py         # VADER sentiment scoring
│   │   ├── complexity.py        # Flesch, Gunning Fog, TTR (pure Python)
│   │   ├── power_language.py    # Regex classifier: commanding/defensive/manipulative/assertive
│   │   ├── emotion.py           # Keyword-based + optional HuggingFace emotion model
│   │   ├── entities.py          # spaCy NER + 60-term legal lexicon
│   │   └── features.py          # Step 2 orchestrator: runs all NLP modules
│   ├── graph/
│   │   ├── builder.py           # NetworkX graph construction per episode
│   │   ├── metrics.py           # Centrality, influence score, power trajectory
│   │   └── pipeline.py          # Step 3+4 orchestrator: graphs → influence → dominance
│   ├── predict/
│   │   ├── features.py          # Lag features, momentum, dominance streak
│   │   ├── models.py            # RandomForest, XGBoost, PyTorch LSTM
│   │   └── pipeline.py          # Step 5 orchestrator: train → evaluate → predict
│   ├── topics/
│   │   ├── model.py             # LDA (sklearn) + optional BERTopic wrapper
│   │   └── pipeline.py          # Step 6 orchestrator: fit → episode distributions → rolling avg
│   └── dashboard/
│       └── app.py               # Step 7: 5-tab Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Setup

**Python 3.10+** required.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

On Mac, if XGBoost fails to import:

```bash
brew install libomp
pip install xgboost
```

Optional — for richer topic modeling:

```bash
pip install bertopic sentence-transformers
```

---

## Running the full pipeline

Each step writes CSVs to `data/processed/`. Run them in order.

```bash
# Step 1 — Scrape transcripts and build scene/character data (~3.5 min for all 134 episodes)
python3 src/pipeline.py

# Step 2 — NLP feature extraction (~5 min)
python3 src/nlp/features.py

# Step 3+4 — Build graphs, compute centrality and influence scores
python3 src/graph/pipeline.py

# Step 5 — Train prediction models (RF + XGBoost + LSTM)
python3 src/predict/pipeline.py

# Step 6 — Topic modeling
python3 src/topics/pipeline.py

# Step 7 — Launch dashboard
streamlit run src/dashboard/app.py
```

### Optional flags

```bash
python3 src/nlp/features.py --use-transformers     # HuggingFace emotion model instead of keyword-based
python3 src/topics/pipeline.py --n-topics 12       # change number of LDA topics
python3 src/topics/pipeline.py --use-bertopic      # BERTopic instead of LDA
python3 src/predict/pipeline.py --model rf         # train only Random Forest
python3 src/predict/pipeline.py --predict S05E08   # print ranked prediction after a specific episode
```

### SRT subtitles (optional but recommended)

Download SRT files from OpenSubtitles and place them in `data/raw/srt/`. Filenames must contain the episode code, e.g. `Suits.S01E01.srt`. The pipeline auto-detects them and uses timestamp-based scene segmentation (>30s gap = new scene) instead of the character-change heuristic fallback.

---

## Dashboard tabs

| Tab | What it shows |
|-----|---------------|
| **Character Power** | Influence trajectory per character over all episodes, season separators, avg influence and dominance count bar charts |
| **Network Graph** | Interactive co-occurrence graph for any episode — node size = influence, edge width = interaction weight, adjustable minimum weight filter |
| **Emotion Tracker** | Episode-level sentiment trend, stacked area emotion breakdown (anger/joy/fear/sadness/surprise/trust), power language mix, legal language density |
| **Story Evolution** | LDA topic intensity trend lines over all episodes, dominant narrative arc color strip, expandable keyword table per topic |
| **Dominance Predictor** | Select any episode, choose RF or LSTM, get ranked probability chart of who dominates next |

---

## How influence is scored

```
Influence = 0.4 × PageRank_norm
           + 0.3 × Mention_Volume_Share
           + 0.3 × Sentiment_Control_norm
```

All three components are min-max normalized per episode before combining. The result is ranked 1–N per episode and smoothed with a 3-episode rolling average for trajectory charts.

---

## Models

| Model | Architecture | Notes |
|-------|-------------|-------|
| Random Forest | 300 trees, class-balanced, max_depth=8 | Interpretable; outputs feature importances |
| XGBoost | 400 estimators, scale_pos_weight for imbalance | Usually best on tabular data |
| LSTM | 2-layer PyTorch, hidden=64, seq_len=5, dropout=0.3 | Captures temporal momentum; auto-detects MPS/CUDA/CPU |

Training uses a **temporal split** (train on earlier seasons, test on later ones) to prevent future data leakage. Evaluation reports top-1 and top-3 per-episode accuracy.

---

## Data notes

- Springfield! Springfield! transcripts contain no speaker labels — character presence is inferred from name mentions within lines.
- Scene segmentation without SRT timestamps is approximate; short episodes or episodes with few character transitions may appear as a single scene.
- The `data/` directory is excluded from version control. All files in it are generated by the pipeline.
