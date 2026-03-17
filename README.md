# Intelligent Idea Analysis Engine (IIAE)

**Live Demo:** [intelengine.streamlit.app](https://intelengine.streamlit.app/)

A semantic duplicate detection system for idea management at scale. Rather than relying on keyword matching, the IIAE analyzes the *meaning* of submitted ideas using deep learning — catching paraphrased duplicates that traditional search would miss entirely.

---

## Overview

As idea platforms scale to hundreds of thousands of entries, two problems emerge: databases lose value when filled with duplicates, and manual review of every submission becomes infeasible. The IIAE automates duplicate validation before ideas enter the database by comparing semantic meaning, not just words.

The system validates new submissions against a dataset of 1,600 existing ideas and visualizes the semantic landscape in 2D.

---

## How It Works

### Baseline Model — TF-IDF + K-Means
A traditional NLP approach using word-frequency analysis. Fast and interpretable, but treats "car" and "automobile" as completely different concepts.

- `TfidfVectorizer` with `max_features=5000`, `ngram_range=(1,2)`
- `KMeans` with `n_clusters=8`
- Silhouette Score: **0.19** (weak cluster separation — expected for keyword-based methods)

### Core Model — Sentence-BERT (SBERT)
Uses the `all-MiniLM-L6-v2` transformer model to generate 384-dimensional sentence embeddings, placing semantically similar ideas close together in vector space regardless of vocabulary differences.

**Duplicate detection via cosine similarity:**

| Similarity | Decision |
|---|---|
| > 0.80 | ❌ Reject as duplicate |
| 0.60 – 0.80 | ⚠️ Warning: semantically similar |
| < 0.60 | ✅ Accept as unique |

**Example:**
| Idea Pair | TF-IDF | SBERT |
|---|---|---|
| *"A next-gen gamified coding tutor..."* vs *"Developing a gamified coding tutor..."* | 0.67 | 0.98 |
| *"Machine learning for crop yield"* vs *"machine learning for crop yield"* | 1.00 | 1.00 |

SBERT correctly identifies paraphrased duplicates that TF-IDF misses.

### Dimensionality Reduction & Visualization
SBERT operates in 384-dimensional space. Two reduction methods are used:

- **PCA** — millisecond-speed reduction for real-time visualization in the Streamlit app (~12% variance captured)
- **t-SNE** — slower but produces clearer category separation; used for static report visualizations (`perplexity=30`, `n_iter=1000`)

---

## Dataset

100,000 ideas were synthetically generated using Python's `Faker` library with a structured `category_map`. Each idea follows the template: **Action + Technology + Benefit**. A sample of 1,600 ideas (200 per category) is used in the app.

| Category | Count |
|---|---|
| AI & Robotics | 200 |
| Health | 200 |
| Ocean | 200 |
| Cybersecurity | 200 |
| Education | 200 |
| Food & Biotech | 200 |
| Energy & Environment | 200 |
| Space | 200 |

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/intelligent-idea-analysis-engine.git
cd intelligent-idea-analysis-engine

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
# Launch the Streamlit web application
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### Other Scripts

```bash
# Generate static t-SNE visualizations (PNG + HTML)
python visualization.py

# Run the baseline TF-IDF + K-Means model
python dm_baseline.py

# Regenerate the synthetic dataset
python idea_generator.py

# Regenerate BERT embeddings (takes ~2 minutes with batch processing)
python Embedding_engine.py
```

---

## Project Structure

```
project/
│
├── app.py                  # Main Streamlit application
├── visualization.py        # t-SNE visualization generation
├── dm_baseline.py          # Baseline TF-IDF + K-Means model
├── idea_generator.py       # Synthetic data generation
├── Embedding_engine.py     # BERT embedding generation
├── idea_sample.csv         # Dataset (1,600 ideas)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Performance

- **Query speed:** ~0.2 seconds per validation
- **Embedding generation:** ~2 minutes for 1,600 ideas (batch_size=128, vs ~83 min sequentially)
- **Data generation:** ~seconds for 100,000 ideas (8 parallel processes via `multiprocessing`)

---

## Known Limitations

- Similarity thresholds (0.60 / 0.80) were manually tuned — not optimized on a labeled dataset
- Memory usage scales linearly with dataset size (FAISS indexing recommended for millions of ideas)
- General-purpose SBERT may underperform on highly specialized domain terminology without fine-tuning
- Ideas sharing a theme but differing significantly in approach may occasionally be flagged as duplicates

---

## Tech Stack

- [Streamlit](https://streamlit.io/) — web interface
- [Sentence-Transformers](https://www.sbert.net/) — SBERT embeddings (`all-MiniLM-L6-v2`)
- [scikit-learn](https://scikit-learn.org/) — TF-IDF, K-Means, PCA, t-SNE
- [Faker](https://faker.readthedocs.io/) — synthetic data generation

---

*Built by Christian Garmann Schjelderup*
