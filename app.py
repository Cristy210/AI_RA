"""Streamlit frontend for the AI Research Assistant.

This application provides a chat-based interface for querying a local
retrieval-augmented generation (RAG) pipeline built with Chroma,
LangChain, and an MLX-powered language model.
"""

from pathlib import Path
import sys
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.rag import rag_answer_stream, load_retriever
from llm.mlx_model import MLXModel

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖📚",
    layout="wide",
)

@st.cache_resource
def get_retriever():
    """Load and cache the Chroma retriever.

    The retriever is created only once per Streamlit session and reused
    across subsequent reruns to avoid repeatedly loading the vector
    database from disk.

    Retrieves the top `k` most relevant document chunks for
    each user query before passing them to the language model for
    answer generation.

    Returns:
        BaseRetriever: Configured LangChain retriever.
    """
    return load_retriever(k=40)

@st.cache_resource
def get_llm():
    """Load and cache the MLX language model.

    The model is initialized only once per Streamlit session, avoiding
    repeated model loading during user interactions.

    Returns:
        MLXModel: Initialized MLX language model.
    """
    return MLXModel(max_tokens=700)

retriever = get_retriever()
llm = get_llm()

st.title("AI Research Assistant")
# st.caption("Local RAG Assistant using Chroma, Langchain, and MLX-LM.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask a question about your indexed research papers...")

if query:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        response_stream = rag_answer_stream(
            query,
            retriever=retriever,
            llm=llm,
        )
        answer = st.write_stream(response_stream)
    
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )