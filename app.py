from pathlib import Path
import sys
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.rag import rag_answer

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖📚",
    layout="wide",
)

st.title("AI Research Assistant")
st.caption("Local RAG Assistant using Chroma, Langchain, and MLX-LM.")

st.markdown(
    """
Ask a question about the research papers indexed in your local vector database.
"""
)

query = st.text_area(
    "Research Question",
    placeholder="What are the optimization methods used in sparse subspace clustering?",
    height=120,
)

col1, col2 = st.columns([1, 4])

with col1:
    ask_button = st.button("Ask", type="primary")
with col2:
    clear_button = st.button("Clear")

if clear_button:
    st.rerun()
if ask_button:
    if not query.strip():
        st.warning("Please enter a research question.")
    else:
        with st.spinner("Retrieving relevant papers and generating answer..."):
            answer = rag_answer(query)
        st.subheader("Answer")
        st.write(answer)
