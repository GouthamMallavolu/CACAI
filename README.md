# Context-Aware Information Retrieval for Conversational AI Assistants

## Project Overview
This repository implements a context-aware conversational Information Retrieval (IR) system that supports:
- BM25 (lexical ranked retrieval)
- Dense retrieval (sentence embeddings + FAISS)
- Hybrid retrieval (BM25 + Dense score fusion with an alpha trade-off)
- Conversational query reformulation (topic entity memory + pronoun resolution)
- RAG-style answer generation (evidence-grounded, slower)
- Offline evaluation using P@10, MAP, and nDCG@10
- Streamlit frontend for interactive demos

The system is designed for conversational queries where follow-up turns may omit key context (e.g., “it/its/they”), causing traditional keyword retrieval to return irrelevant results.

## Repository Structure
    Project_Mallavolu_Goutham/
    |
    |-- code/
    |   |-- app.py                # Streamlit frontend
    |   |-- ir_system.py          # Main conversational IR controller (reformulate + retrieve + RAG)
    |   |-- bm25.py               # BM25 implementation
    |   |-- dense.py              # Dense retrieval (SentenceTransformer + FAISS, cached embeddings)
    |   |-- hybrid.py             # Score fusion (alpha*BM25 + (1-alpha)*Dense)
    |   |-- reformulation.py      # Topic entity extraction + pronoun replacement
    |   |-- rag.py                # Optional RAG answer generation (local model)
    |   |-- evaluation.py         # Metrics: P@10, MAP, nDCG@10
    |   |-- main.py               # Offline evaluation entry-point (may be named main_eval.py in some setups)
    |   |-- requirements.txt      # Python dependencies
    |   |-- README.md 
    |
    |-- data/
    |   |-- corpus.csv            # Document collection (>= 400 docs recommended)
    |   |-- queries.csv           # Evaluation queries
    |   |-- qrels_filtered.csv    # Relevance judgments aligned to corpus doc_ids
    |
    |-- outputs/
    |   |-- embeddings_*.npy      # Cached dense embeddings (auto-generated)
    |   |-- metrics_summary.csv   # Evaluation results (auto-generated)
    |
    |-- walkthrough/
    |   |-- walkthrough.pdf       # Annotated screenshots walkthrough (max 6 pages)
    |
    |-- IEEE_Paper/                     

## System Architecture (High-Level)
User Query
→ (optional) Conversational Reformulation (topic entity + pronoun resolution)
→ Retrieval Layer:
   - BM25 (lexical scoring)
   - Dense retrieval (FAISS cosine similarity on embeddings)
   - Hybrid fusion: FinalScore = α·BM25 + (1−α)·Dense
→ Top-K Ranked Documents (doc_id + score + preview)
→ (optional) RAG Answer Generation from top evidence passages
→ Streamlit frontend output

## Core Components (What each does)

### BM25 (bm25.py)
Lexical ranked retrieval using TF/IDF-style weighting with document-length normalization. Strong when queries contain exact keywords present in relevant documents.

### Dense Retrieval (dense.py)
Encodes documents and queries with a SentenceTransformer model (e.g., all-MiniLM-L6-v2), normalizes vectors, and retrieves nearest neighbors using FAISS (IndexFlatIP). Helps when wording differs (semantic similarity). Dense embeddings are cached to outputs/ to speed up subsequent runs.

### Hybrid Fusion (hybrid.py)
Combines BM25 and Dense scores:
FinalScore(d) = α·BM25(d) + (1−α)·Dense(d)
- α = 1.0 → BM25 only
- α = 0.0 → Dense only
- typical starting point: α ≈ 0.6

### Conversational Reformulation (reformulation.py)
Tracks a topic entity from definitional queries (e.g., “what is X”) and rewrites follow-up queries by replacing pronouns (“it”, “its”, “they”) with that entity.

### Optional RAG (rag.py)
Generates a short answer grounded in retrieved passages. This is slower than retrieval and should be enabled only for demonstrations. The UI includes a toggle for RAG.

### Evaluation (evaluation.py + main.py)
Offline evaluation computes P@10, MAP, and nDCG@10 for BM25, Dense, and Hybrid and writes outputs/metrics_summary.csv.

## Installation

### 1) Create and activate an environment (recommended)
Conda:

    conda create -n conversational_ir python=3.10
    conda activate conversational_ir

venv:

    python -m venv venv
    source venv/bin/activate      # macOS/Linux
    venv\Scripts\activate       # Windows

### 2) Install dependencies

    pip install -r code/requirements.txt

Notes:
- NumPy is pinned in requirements.txt for compatibility with common scientific Python wheels.
- If FAISS installation fails on your platform:

    pip install faiss-cpu

## Dataset Setup
Place these files in data/:

    data/
    ├── corpus.csv
    ├── queries.csv
    └── qrels_filtered.csv

Expected columns:

corpus.csv:

    doc_id,text

queries.csv:

    query_id,query

qrels_filtered.csv:

    query_id,doc_id,relevance

Important: qrels_filtered.csv doc_id values must exist in corpus.csv doc_id. If they do not match, evaluation will be invalid.

## Running the System

### Streamlit frontend (interactive demo)

    cd code
    streamlit run app.py

Recommended demo sequence (with reformulation ON):
- What is global warming?
- What are its effects?
- How does it affect agriculture?

Compare methods by switching BM25 vs Dense vs Hybrid (alpha slider). Enable RAG only once for a short demo due to latency.

### Offline evaluation (metrics)

    cd code
    python main.py
    
Outputs:
- outputs/metrics_summary.csv
- outputs/embeddings_*.npy (dense cache)

## Reproducibility Steps
To reproduce results consistently:

1. Use Python 3.10.
2. Install dependencies using code/requirements.txt.
3. Use the same dataset files in data/.
4. If you want to rebuild the dense index from scratch, delete cached embeddings:

       rm outputs/embeddings_*.npy

5. Run evaluation (python main.py or python main_eval.py).
6. Compare outputs/metrics_summary.csv against reported values in the memo/paper.

## Troubleshooting
- First run is slower: models download and embeddings are built and cached.
- RAG is slower: it runs text generation; keep it OFF for faster retrieval-only testing.
- If results show doc_ids not found in corpus, ensure doc_id types match (avoid “.0” float formatting) and rebuild embeddings cache.
