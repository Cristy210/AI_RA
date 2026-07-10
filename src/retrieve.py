from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_DIR = PROJECT_ROOT / "vectorstore" / "research_papers"

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True},
)

db = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings,
)
print("Using DB:", DB_DIR)
print("Number of docs:", db._collection.count())

retriever = db.as_retriever(search_kwargs={"k": 5})

query = "What are the optimization methods used in sparse subspace clustering?"
results = retriever.invoke(query)

for i, doc in enumerate(results, start=1):
    print("=" * 80)
    print(f"Result {i}")
    print("Source:", doc.metadata.get("entry_id"))
    print("Title:", doc.metadata.get("title"))
    print(doc.page_content[:800])
