"""Core retrievel-augmented generation (RAG) pipeline.

This module provides utilities for retrieving relevant document chunks
from a Chroma Vector database, constructing prompts from the retrieved 
context, and generating responses using the chosen MLX-LM Model. 
"""

from pathlib import Path
import sys

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from llm.mlx_model import MLXModel

DB_DIR = PROJECT_ROOT / "vectorstore" / "research_papers"

def load_retriever(k:int = 5):
    """Create a Chroma document retriever.

    Initializes the embedding model, loads the persistent Chroma vector
    database, and returns a LangChain retriever configured to return the
    top-k most relevant document chunks for semantic search.

    Args:
        k (int, optional): Number of document chunks to retrieve for each
            query. Defaults to 5.

    Returns:
        BaseRetriever: Configured LangChain retriever backed by Chroma.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name = "BAAI/bge-small-en-v1.5",
        encode_kwargs = {"normalize_embeddings": True},
    )

    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    print("Using DB: ", DB_DIR)
    print("Number of doc chunks:", db._collection.count())

    return db.as_retriever(search_kwargs={"k": k})

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

def build_prompt(query:str, context:str) -> str:
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

Question:
{query}

Retrieved excerpts:
{context}

Response:
"""

def rag_answer(query:str) -> str:
    """Generate a response using the complete RAG pipeline.

    Retrieves the most relevant document chunks, formats them into a
    prompt, invokes the MLX language model, and returns the generated
    answer.

    Args:
        query (str): User's research question.

    Returns:
        str: Generated answer based on the retrieved document context.
    """
    retriever = load_retriever(k=10)
    docs = retriever.invoke(query)

    context = format_context(docs)
    prompt = build_prompt(query, context)
    llm = MLXModel()
    answer = llm.generate_response(prompt)

    return answer

def rag_answer_stream(query:str, retriever=None, llm=None):
    """Generate a streaming response using the RAG pipeline.

    Uses the supplied retriever and language model if provided;
    otherwise, initializes default instances. The response is yielded
    incrementally to support real-time streaming in the Streamlit
    interface.

    Args:
        query (str): User's research question.
        retriever (BaseRetriever, optional): Preloaded document retriever.
            If ``None``, a default retriever is created.
        llm (MLXModel, optional): Preloaded language model. If ``None``,
            a default model is initialized.

    Yields:
        str: Consecutive segments of the generated response.
    """
    if retriever is None:
        retriever = load_retriever(k=5)
    
    if llm is None:
        llm = MLXModel(max_tokens=700)
    
    docs = retriever.invoke(query)
    context = format_context(docs)
    prompt = build_prompt(query, context)

    yield from llm.stream_response(prompt)

if __name__ == "__main__":
    query = "What is the main objective in sparse subspace clustering methods?"

    answer = rag_answer(query)

    print("\n" + "="*80)
    print("Final Answer")
    print("="*80)
    print(answer)