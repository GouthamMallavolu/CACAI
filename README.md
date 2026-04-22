# Context-Aware Information Retrieval for Conversational AI Assistants

## Project Overview

This project implements a **context-aware conversational Information Retrieval (IR) system** that combines:

- **BM25 (lexical retrieval)**
- **Dense Retrieval (Sentence Embeddings + FAISS)**
- **Hybrid Retrieval (BM25 + Dense fusion)**
- **Conversational Query Reformulation**
- **Optional RAG-style Answer Generation**
- **Evaluation using P@10, MAP, nDCG@10**
- **Interactive Streamlit Frontend**

The system is designed to support follow-up conversational queries and demonstrate both traditional and modern retrieval techniques.

# Project Structure
```
Project_Mallavolu_Goutham/
│
├── code/
│   ├── app.py                  # Streamlit frontend
│   ├── ir_system.py            # Main conversational IR engine
│   ├── bm25.py                 # BM25 lexical retrieval
│   ├── dense.py                # Dense retrieval (SentenceTransformers + FAISS)
│   ├── hybrid.py               # Hybrid score fusion logic
│   ├── reformulation.py        # Conversational query reformulation
│   ├── rag.py                  # Optional RAG answer generation
│   ├── evaluation.py           # Evaluation metrics (P@10, MAP, nDCG@10)
│   ├── main_eval.py            # Offline evaluation script
│   └── requirements.txt        # Dependencies
│
├── data/
│   ├── corpus.csv              # Document collection
│   ├── queries.csv             # Evaluation queries
│   └── qrels_filtered.csv      # Relevance judgments
│
├── outputs/
│   ├── embeddings_*.npy        # Cached dense embeddings
│   └── metrics_summary.csv     # Evaluation results
│
└── README.md                   # Project documentation
```

# System Components

## BM25 Retrieval
Implements traditional lexical retrieval using:
- Term Frequency (TF)
- Inverse Document Frequency (IDF)
- Document length normalization

BM25 performs strongly for exact keyword-based queries.

## Dense Retrieval
Uses:
- SentenceTransformer model: `all-MiniLM-L6-v2`
- FAISS `IndexFlatIP`
- L2-normalized embeddings (inner product ≈ cosine similarity)

Dense retrieval captures semantic similarity even when wording differs.

## Hybrid Retrieval

Hybrid ranking combines BM25 and Dense scores:
- FinalScore = α · BM25 + (1 − α) · Dense

Where:
- α = 1 → Pure BM25
- α = 0 → Pure Dense
- Default α ≈ 0.6

Hybrid retrieval improves robustness across query types.

## Conversational Query Reformulation

Handles follow-up queries by:
- Extracting topic entity from definitional queries
- Replacing pronouns like “it”, “its”, “they” with the stored entity

Example:
- Q1: What is global warming?
- Q2: What are its effects?
- Reformulated: What are global warming effects?

## RAG Answer Generation

If enabled, the system:
- Uses top retrieved passages as evidence
- Generates a short grounded answer using `google/flan-t5-small`
- Includes fallback logic to prevent hallucinations

RAG improves interaction quality but increases latency.

# Evaluation Metrics

The system evaluates retrieval performance using:

- **Precision@10 (P@10)** – Relevant documents in the top 10
- **Mean Average Precision (MAP)** – Ranking quality across all relevant documents
- **nDCG@10** – Ranking quality emphasizing higher-ranked documents

Results are saved in: `outputs/metrics_summary.csv`

# System Architecture
                     ┌───────────────────────┐
                     │      User Query       │
                     └────────────┬──────────┘
                                  │
                                  ▼
              ┌──────────────────────────────────┐
              │ Conversational Query Reformulator│
              │ (Pronoun Resolution + Context)   │
              └────────────┬─────────────────────┘
                           │
                           ▼
              ┌──────────────────────────────────┐
              │        Retrieval Layer           │
              │                                  │
              │  ┌────────────┐   ┌───────────┐  │
              │  │   BM25     │   │  Dense    │  │
              │  │ (Lexical)  │   │ (FAISS)   │  │
              │  └──────┬─────┘   └─────┬─────┘  │
              │         │               │        │
              │         └─Hybrid Fusion-|        |
              └────────────┬─────────────────────┘
                           │
                           ▼
              ┌──────────────────────────────────┐
              │      Top-K Ranked Documents      │
              └────────────┬─────────────────────┘
                           │
                           ▼
                     ┌───────────────────────────┐
                     │  RAG Answer Generation    │
                     │ (Grounded in Evidence)    │
                     └────────────┬──────────────┘
                                  │
                                  ▼
                     ┌──────────────────────┐
                     │  Streamlit Frontend  │
                     │  Results + Answer    │
                     └──────────────────────┘

## Installation

### 1. Clone the repository

    git clone <your-repo-url>
    cd Project_Mallavolu_Goutham

### 2. Create and activate an environment (recommended)

Using conda:

    conda create -n conversational_ir python=3.10
    conda activate conversational_ir

Using venv:

    python -m venv venv
    source venv/bin/activate      # macOS/Linux
    venv\\Scripts\\activate         # Windows

### 3. Install dependencies

    pip install -r code/requirements.txt

If FAISS is not available in your environment:

    pip install faiss-cpu

If PyTorch is missing:

    pip install torch torchvision torchaudio

## Dataset setup

Place the following files inside the `data/` directory:

    data/
    ├── corpus.csv
    ├── queries.csv
    └── qrels_filtered.csv

Expected formats:

corpus.csv

    doc_id,text

queries.csv

    query_id,query

qrels_filtered.csv

    query_id,doc_id,relevance

## Run the system

### Streamlit frontend

    cd code
    streamlit run app.py

### Offline evaluation

    cd code
    python main_eval.py

This generates:

    outputs/metrics_summary.csv

## Reproducibility

To reproduce results consistently:

1. Use Python 3.10.
2. Install dependencies using `code/requirements.txt`.
3. Use the same dataset files (`corpus.csv`, `queries.csv`, `qrels_filtered.csv`).
4. If you want to rebuild the dense index from scratch, delete the cached embeddings:

       rm outputs/embeddings_*.npy

5. Run evaluation:

       cd code
       python main_eval.py

6. Compare the newly generated `outputs/metrics_summary.csv` with the reported results.

## Notes

- The first run may be slower because sentence-transformer models and embeddings are downloaded/built and then cached.
- Enabling RAG in the UI increases latency because it runs text generation; keep it off for faster retrieval-only testing.
"""
