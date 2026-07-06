from pathlib import Path
import sys

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from llm.mlx_model import MLXModel

DB_DIR = PROJECT_ROOT / "vectorstore" / "research_papers"

def load_retriever(k:int = 5):
    embeddings = HuggingFaceEmbeddings(
        model_name = "BAAI/bge-small-en-v1.5",
        encode_kwargs = {"normalize_embeddings": True},
    )

    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    print("Using DB: ", DB_DIR)
    print("Number of docs:", db._collection.count())

    return db.as_retriever(search_kwargs={"k": k})

def format_context(docs) -> str:
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
    retriever = load_retriever(k=10)
    docs = retriever.invoke(query)

    context = format_context(docs)
    prompt = build_prompt(query, context)
    llm = MLXModel()
    answer = llm.generate_response(prompt)

    return answer

if __name__ == "__main__":
    query = "What is the main objective in sparse subspace clustering methods?"

    answer = rag_answer(query)

    print("\n" + "="*80)
    print("Final Answer")
    print("="*80)
    print(answer)