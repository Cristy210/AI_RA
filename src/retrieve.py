"""Manually test retrieval from a selected research database."""

from src.rag import load_retriever


def main() -> None:
    """Retrieve relevant chunks from a sample database."""

    database_name = "sparse_subspace_clustering"

    retriever = load_retriever(
        database_name=database_name,
        k=5,
    )

    query = (
        "What are the optimization methods used in "
        "sparse subspace clustering?"
    )

    results = retriever.invoke(query)

    for index, doc in enumerate(results, start=1):
        print("=" * 80)
        print(f"Result {index}")
        print("Source:", doc.metadata.get("entry_id"))
        print("Title:", doc.metadata.get("title"))
        print(doc.page_content[:800])


if __name__ == "__main__":
    main()