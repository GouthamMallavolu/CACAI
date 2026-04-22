import streamlit as st
import pandas as pd
from pathlib import Path

from ir_system import ConversationalIR

st.set_page_config(page_title="Conversational IR", layout="wide")
st.title("Context-Aware Information Retrieval for Conversational AI Assistants")

DATA_DIR = Path("../data")

@st.cache_resource
def load_system():
    return ConversationalIR()

@st.cache_data
def load_corpus_preview():
    df = pd.read_csv(DATA_DIR / "corpus.csv")
    df["doc_id"] = df["doc_id"].astype(str)
    return dict(zip(df["doc_id"], df["text"]))

ir = load_system()
corpus_preview = load_corpus_preview()

col1, col2, col3 = st.columns(3)
with col1:
    method = st.selectbox("Method", ["Hybrid", "BM25", "Dense"])
with col2:
    alpha = st.slider("Hybrid alpha (BM25 weight)", 0.0, 1.0, 0.6, 0.05)
with col3:
    use_rag = st.checkbox("Enable RAG answer (slower)", value=False)

use_reform = st.checkbox("Enable conversational reformulation", value=True)

query = st.text_input("Enter a query (try a follow-up like: 'What are its effects?')")

if st.button("Search") and query.strip():
    rq, ranked, answer = ir.search(
        query=query,
        method=method,
        alpha=alpha,
        use_reformulation=use_reform,
        use_rag=use_rag,
        top_k=10
    )

    st.subheader("Reformulated Query")
    st.code(rq, language="text")

    if use_rag:
        st.subheader("RAG Answer")
        st.write(answer if answer else "")

    st.subheader("Top 10 Results")
    for i, (doc_id, score) in enumerate(ranked, start=1):
        preview = corpus_preview.get(str(doc_id), "")[:260]
        st.markdown(f"**{i}. Doc {doc_id}** — score: `{score:.4f}`")
        st.write(preview + ("..." if len(preview) > 0 else ""))
        st.divider()