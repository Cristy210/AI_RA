"""Core retrievel-augmented generation (RAG) pipeline.

This module provides utilities for retrieving relevant document chunks
from a Chroma Vector database, constructing prompts from the retrieved
context, and generating responses using the chosen MLX-LM Model.
"""

from pathlib import Path
import sys

from llm.mlx_model import MLXModel
from langchain_core.documents import Document
from src.database_manager import load_vectorstore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

DB_DIR = PROJECT_ROOT / "vectorstore" / "research_papers"


def load_retriever(
    database_name: str,
    k: int = 25,
):
    """Create a retriever for a selected research database.

    Args:
        database_name: Name of the Chroma collection to query.
        k: Number of relevant document chunks to retrieve.

    Returns:
        Configured Langchain retriever.

    Raises:
        ValueError: If the selected database is empty.
    """
    vectorstore = load_vectorstore(database_name)
    document_count = vectorstore._collection.count()

    if document_count == 0:
        raise ValueError(f"Research database '{database_name}' is empty.")

    return vectorstore.as_retriever(search_kwargs={"k": k})


def format_context(docs: list[Document]) -> str:
    """Format retrieved documents into a prompt-ready context string.

    Each retrieved document is labeled with its source number, title,
    source identifier, and page number before being concatenated into a
    single context block for the language model.

    Args:
        docs: Retrieved document chunks.

    Returns:
        str: Formatted context containing all retrieved document chunks.
    """
    context_blocks = []

    for i, doc in enumerate(docs, start=1):
        title = doc.metadata.get("title")
        source = doc.metadata.get("entry_id")
        page = doc.metadata.get("page")

        block = f"""
[Source {i}]
Title: {title}
Source: {source}
Page: {page}

{doc.page_content}
"""
        context_blocks.append(block)
    return "\n\n".join(context_blocks)


def build_prompt(query: str, context: str) -> str:
    """Construct the prompt for the language model.

    Combines the user's research question with the retrieved document
    context and task-specific instructions to guide the language model
    toward generating an evidence-based response.

    Args:
        query (str): User's research question.
        context (str): Formatted retrieved document context.

    Returns:
        str: Complete prompt supplied to the language model.
    """
    return f"""

Use the retrieved excerpts from research papers to answer the question;

Guidelines:
- Base the answer on the provided excerpts.
- If the excerpts are insufficient, say so clearly.
- When possible, mention the relevant paper/source numbers.
- Keep the answer clear, technical, and concise. 
- Format mathematical notaitons using Markdown LaTeX.
- use inline math with `$...$`.
- use display equations with `$$...$$`.
- For example, write `$\\ell_1$`, `$C^T$`, `$|C| + |C^T|$`, and `$\\operatorname{{diag}}(C)=0$`.

Question:
{query}

Retrieved excerpts:
{context}

Response:
"""


def rag_answer(
    query: str,
    database_name: str,
    retriever=None,
    llm=None,
) -> str:
    """Generate a complete answer from the selected research database.

    Args:
        query (str): User's research question.
        database_name: Research database to query.
        retriever: Preloaded document retriever. If ``None``, a default
        retriever is created.
        llm: Preloaded MLX language model. If ``None``, a default model
        is initialized.

    Returns:
        str: Generated answer based on the retrieved document context.
    """
    query = query.strip()

    if not query:
        raise ValueError("Query must not be empty")

    if retriever is None:
        retriever = load_retriever(database_name=database_name)

    if llm is None:
        llm = MLXModel(max_tokens=700)

    docs = retriever.invoke(query)
    context = format_context(docs)
    prompt = build_prompt(query, context)

    return llm.generate_response(prompt)


def rag_answer_stream(
    query: str,
    database_name: str,
    retriever=None,
    llm=None,
):
    """Generate a streaming answer from a selected database.

    Args:
        query (str): User's research question.
        database_name: Research database to query.
        retriever (BaseRetriever, optional): Preloaded document retriever.
            If ``None``, a default retriever is created.
        llm (MLXModel, optional): Preloaded language model. If ``None``,
            a default model is initialized.

    Yields:
        str: Consecutive segments of the generated response.
    """
    query = query.strip()

    if not query:
        raise ValueError("Query must not be empty")

    if retriever is None:
        retriever = load_retriever(
            database_name=database_name,
        )

    if llm is None:
        llm = MLXModel(max_tokens=700)

    docs = retriever.invoke(query)
    context = format_context(docs)
    prompt = build_prompt(query, context)

    yield from llm.stream_response(prompt)


if __name__ == "__main__":
    query = "What is the main objective in sparse subspace clustering methods?"

    answer = rag_answer(query=query, database_name="subspace_clustering")

    print("\n" + "=" * 80)
    print("Final Answer")
    print("=" * 80)
    print(answer)
