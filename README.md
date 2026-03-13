# Intelligent Idea Analysis Engine (IIAE)
**Exam Project – Christian Garmann Schjelderup**

Live Demo: [https://intelengine.streamlit.app/](https://intelengine.streamlit.app/)

---

## Project Overview

This project addresses a core challenge in idea management at scale: how do you determine whether an incoming idea is genuinely new, or simply a rephrasing of one already in the database? Standard keyword-based database searches will only match exact words, missing conceptually identical ideas expressed with different vocabulary.

The **Intelligent Idea Analysis Engine (IIAE)** solves this by analyzing the *semantic meaning* of ideas using a deep learning model, rather than relying on surface-level word matching. The system validates new submissions against a dataset of 1,600 existing ideas and visualizes the semantic landscape in 2D.

---

## Background

### Idea Management and the Scalability Problem
The project is built around a fictional but realistic scenario: an open platform where users submit ideas and pay to access the database. As such platforms scale to hundreds of thousands of entries, two problems emerge. First, the database loses value if filled with duplicates. Second, manual review of every submission becomes impossible. The IIAE automates duplicate validation before ideas enter the database (Sections 2.1, 4).

### Natural Language Processing (NLP)
NLP was applied at two levels:
- **Baseline level:** TF-IDF for word-frequency analysis
- **Advanced level:** Sentence-BERT (SBERT) for contextual semantic understanding (Section 2.2)

### Data Mining (DM) vs. Computational Intelligence (CI)
TF-IDF treats "car" and "automobile" as entirely different – it compares words, not meaning. To overcome this limitation, the CI model uses SBERT with the `all-MiniLM-L6-v2` model, generating 384-dimensional embeddings that place semantically similar concepts close together in vector space (Sections 2.3–2.4).

### Dimensionality Reduction and Visualisation
SBERT operates in 384-dimensional space. Two methods were used to reduce this to 2D:
- **PCA (Principal Component Analysis):** Fast (milliseconds), used for real-time Streamlit visualization. Captured ~12% of variance.
- **t-SNE (t-Distributed Stochastic Neighbour Embedding):** Slower but produces clearer category separation, used for static report visualizations (Section 2.5).

---

## Data Warehouse

### Data Generation
100,000 ideas were generated synthetically using a Python script with the `Faker` library and a `category_map`. Each idea follows the template: **Action + Technology + Benefit**. Python's `multiprocessing` with 8 parallel processes reduced generation time from minutes to seconds (Section 5.2).

### From SQL Server to CSV
Data was initially stored in a SQL Server database with columns for `OriginalText`, `Category`, and `EmbeddingVarbinary`. After encountering driver compatibility and deployment issues (see Section 10.7), 200 ideas per category were exported to CSV – 1,600 ideas in total.

### Final Dataset Summary

| Property | Value |
|---|---|
| Total ideas generated | 100,000 |
| Ideas in final dataset | 1,600 |
| Number of categories | 8 |
| Language | English |
| Format | CSV |

| Category | Count | Percentage |
|---|---|---|
| AI & Robotics | 200 | 12.5% |
| Health | 200 | 12.5% |
| Ocean | 200 | 12.5% |
| Cybersecurity | 200 | 12.5% |
| Education | 200 | 12.5% |
| Food & Biotech | 200 | 12.5% |
| Energy & Environment | 200 | 12.5% |
| Space | 200 | 12.5% |

---

## Data Pre-processing

Because the data was template-generated, no cleaning (typo correction, URL removal, etc.) was necessary. Preprocessing consisted of two stages:

1. **Parallel idea generation** using `multiprocessing` (8 processes)
2. **BERT encoding** with batch processing (`batch_size=128`), reducing embedding time from ~83 minutes to ~2 minutes

Embeddings were serialized using Python's `pickle` for storage in SQL Server's `VARBINARY` column. Each embedding occupies 1,536 bytes (384 dimensions × 4 bytes per float32) (Section 7).

---

## Algorithms

### Algorithm 1: TF-IDF + K-Means (Baseline DM Model)
- `TfidfVectorizer` with `max_features=5000`, `ngram_range=(1, 2)`, `stop_words='english'`
- `KMeans` with `n_clusters=8`, `n_init=10`
- **Silhouette Score: 0.1893** – indicating weak cluster separation (Section 9.1)

### Algorithm 2: Sentence-BERT (CI Model)
- Model: `all-MiniLM-L6-v2` (BERT-based Sentence Transformer)
- Generates 384-dimensional embeddings per idea
- Duplicate detection via cosine similarity:

```
similarity = (A · B) / (||A|| × ||B||)
```

Thresholds (Section 8.1):

| Threshold | Decision |
|---|---|
| > 0.80 | Reject as duplicate |
| 0.60 – 0.80 | Warning: semantically similar |
| < 0.60 | Accept as unique |

### Algorithm 3: Dimensionality Reduction (PCA & t-SNE)
- PCA used in the live app for real-time 2D positioning
- t-SNE used for static visualizations with `perplexity=30`, `n_iter=1000` (Section 9.3)

---

## Performance Comparison

| Idea Pair | TF-IDF | BERT | Notes |
|---|---|---|---|
| "A next-gen gamified coding tutor..." vs "Developing a gamified coding tutor..." | 0.67 | 0.98 | Different wording, same meaning – BERT wins |
| "Machine learning for crop yield" vs "machine learning for crop yield" | 1.00 | 1.00 | Identical text – both succeed |

TF-IDF can detect keyword overlap but fails to recognize paraphrasing. SBERT handles this correctly (Section 8.2).

---

## Running the Project

### 1. Streamlit Web Application (Primary Interface)
```bash
streamlit run app.py
```
Features: real-time idea validation, semantic map with PCA, top-5 most similar ideas, interactive hover.

### 2. Visualization Generation
```bash
python visualization.py
```
Output: static PNG and interactive HTML using t-SNE.

### 3. Baseline Model
```bash
python dm_baseline.py
```
Output: TF-IDF + K-Means clustering, Elbow Method plot, Silhouette Score plot.

---

## File Structure

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
└── README.md               # Documentation
```

---

## Key Findings

**What worked:**
- SBERT successfully detects semantic duplicates regardless of wording
- Real-time validation at ~0.2 seconds per query
- t-SNE produced clear category clustering without labeled training data

**Limitations:**
- Similarity thresholds (0.60 and 0.80) were manually tuned, not optimized on a labeled dataset
- Memory usage scales linearly with database size (FAISS indexing recommended for millions of ideas)
- General-purpose SBERT may miss highly specialized terminology without fine-tuning
- Ideas sharing a theme but differing in approach may be incorrectly flagged as duplicates

(Section 10.5)

---

## Technical Challenges Encountered

| Challenge | Impact | Solution |
|---|---|---|
| Escape sequences in SQL Server paths | Blocked connection | Raw strings (`r'...'`) |
| Driver compatibility | Upload failures | ODBC Driver 17 |
| Binary type conflicts | Data corruption | Explicit `VARBINARY` + `pickle` |
| SQLAlchemy 2.0+ syntax | Query failures | `text()` wrapper |
| Sequential embedding processing | 80+ minute runtime | Batch processing (`batch_size=128`) |
| SQL Server deployment on Streamlit Cloud | Deployment issues | Migrated to CSV |

(Section 10.7)

---

## Conclusion

All six project objectives were successfully achieved (Section 11.1):
1. Complete data pipeline generating, processing, and storing 100,000 ideas
2. Text preprocessing via template-based generation and batch BERT encoding
3. Baseline DM model (TF-IDF + K-Means, Silhouette Score: 0.19)
4. CI model (SBERT) with successful semantic duplicate detection
5. Real-time validation feature (0.2s per query)
6. Interactive visualizations using PCA and t-SNE

The core conclusion is that **Computational Intelligence is essential for semantic tasks that traditional Data Mining cannot solve**. Keyword matching will miss the majority of paraphrased duplicates; semantic understanding via transformer-based models is required (Section 10.6).

---

## References

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *arXiv preprint arXiv:1908.10084*. https://arxiv.org/abs/1908.10084

Vaswani, A., et al. (2017). Attention Is All You Need. *arXiv preprint arXiv:1706.03762*. https://arxiv.org/abs/1706.03762

van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. *Journal of Machine Learning Research, 9*, 2579–2605. https://www.jmlr.org/papers/v9/vandermaaten08a.html

Sentence Transformers Documentation. https://www.sbert.net/

Hugging Face Model Card: all-MiniLM-L6-v2. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

Scikit-learn Documentation. https://scikit-learn.org/

---

*Project by Christian Garmann Schjelderup*
