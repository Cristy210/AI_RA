"""Streamlit frontend for the AI Research Assistant."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from llm.mlx_model import MLXModel
from src.database_manager import (
    get_database_status,
    list_research_databases,
)
from src.ingest_arxiv import build_research_database
from src.rag import rag_answer_stream, load_retriever


st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖📚",
    layout="wide",
)


@st.cache_resource
def get_retriever(database_name: str):
    """Load and cache the retriever for one research database."""
    return load_retriever(
        database_name=database_name,
    )


@st.cache_resource
def get_llm() -> MLXModel:
    """Load and cache the MLX language model."""
    return MLXModel(max_tokens=700)


def reset_chat() -> None:
    """Clear the current chat history."""

    st.session_state.messages = []



st.title("AI Research Assistant")

databases = list_research_databases()

with st.sidebar:
    st.header("Research Database")

    database_options = {
        database["display_name"]: database["name"] for database in databases
    }

    selection_options = [
        "Create a new database",
        *database_options.keys(),
    ]

    selected_option = st.selectbox(
        "Choose a database",
        options=selection_options,
    )

    if selected_option == "Create a new database":
        with st.form("create_database_form"):
            new_database_name = st.text_input(
                "Database name",
                placeholder="Medical Image Segmentation",
            )

            research_topic = st.text_input(
                "arXiv research topic",
                placeholder="medical image segmentation",
            )

            max_results = st.number_input(
                "Maximum number of papers",
                min_value=1,
                max_value=100,
                value=10,
                step=1,
            )

            submitted = st.form_submit_button("Build Database")
        if submitted:
            if not new_database_name.strip():
                st.error("Enter a database name.")
            elif not research_topic.strip():
                st.error("Enter a research topic.")
            else:
                with st.spinner("Downloading papers and building the database..."):
                    result = build_research_database(
                        query=research_topic,
                        database_name=new_database_name,
                        max_results=int(max_results),
                    )
                if result["status"] == "already_exists":
                    st.warning(result["message"])
                elif result["status"] == "no_results":
                    st.warning(result["message"])

                else:
                    get_retriever.clear()
                    reset_chat()

                    st.session_state.active_database = result["database_name"]

                    st.success(
                        f"Created '{result['display_name']}' with"
                        f"{result['chunks_indexed']} chunks."
                    )

                    st.rerun()
    else:
        selected_database = database_options[selected_option]

        previous_database = st.session_state.get("active_database")

        if previous_database != selected_database:
            st.session_state.active_database = selected_database
            reset_chat()

        status = get_database_status(selected_database)

        st.success(f"Using: {status['display_name']}")
        st.caption(
            f"Topic: {status['research_topic']}\n\n"
            f"Papers: {status['papers_processed']}\n\n"
            f"Chunks: {status['document_chunks']}"
        )

active_database = st.session_state.get("active_database")

if active_database is None:
    st.info("Select an existing research database or create a new one to begin.")
    st.stop()

retriever = get_retriever(active_database)
llm = get_llm()

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
            query=query,
            database_name=active_database,
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
