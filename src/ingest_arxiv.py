"""Create domain-specific research databases from arXiv papers."""

import logging
import requests

import arxiv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)
from docling_core.types.doc import TableItem
from langchain_core.documents import Document
from transformers import AutoTokenizer

from src.database_manager import (
    DB_DIR,
    EMBEDDING_MODEL,
    PAPERS_DIR,
    database_exists,
    normalize_database_name,
    register_database,
)
from src.utils.downloader import download_pdf

logger = logging.getLogger(__name__)

PAPERS_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)


def search_arxiv_papers(
    query: str,
    max_candidates: int,
) -> list[arxiv.Result]:
    """
    Search arXiv and return candidate papers without downloading them.
    """
    client = arxiv.Client()

    search = arxiv.Search(
        query=query,
        max_results=max_candidates,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    return list(client.results(search))


def score_paper_relevance(
    paper: arxiv.Result,
    query: str,
) -> float:
    """
    Score a paper using query-term overlap with its title and abstract.
    """
    query_terms = set(query.lower().split())

    if not query_terms:
        return 0.0

    title_terms = set(paper.title.lower().split())
    abstract_terms = set(paper.summary.lower().split())

    title_score = len(query_terms & title_terms) / len(query_terms)
    abstract_score = len(query_terms & abstract_terms) / len(query_terms)

    return 0.7 * title_score + 0.3 * abstract_score


def filter_relevant_papers(
    papers: list[arxiv.Result],
    query: str,
    max_results: int,
) -> list[arxiv.Result]:
    """
    Rank candidate papers and return the most relevant ones.
    """

    ranked_papers = sorted(
        papers,
        key=lambda paper: score_paper_relevance(paper, query),
        reverse=True,
    )

    return ranked_papers[:max_results]


def download_arxiv_papers(
    papers: list[arxiv.Result],
    database_name: str,
) -> list[dict]:
    """Search arXiv and download papers into a database-specific directory.

    Args:
        papers: list of papers from arXiv database
        database_name: Name of the research database associated with the papers.
    Returns:
        Metadata for the retrieved papers.
    """

    collection_name = normalize_database_name(database_name)
    database_paper_dir = PAPERS_DIR / collection_name
    database_paper_dir.mkdir(parents=True, exist_ok=True)

    downloaded_papers = []

    for paper in papers:
        paper_id = paper.entry_id.split("/")[-1]
        pdf_path = database_paper_dir / f"{paper_id}.pdf"

        if not pdf_path.exists():
            logger.info("Downloading paper: %s", paper.title)

            try:
                download_pdf(paper.pdf_url, pdf_path)
            except requests.RequestException as error:
                logger.warning(
                    "Skipping paper '%s'. PDF download failed from %s: %s",
                    paper.title,
                    paper.pdf_url,
                    error,
                )
                continue
        else:
            logger.info("Using cached paper: %s", paper.title)

        downloaded_papers.append(
            {
                "paper_id": paper_id,
                "pdf_path": pdf_path,
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "published": str(paper.published.date()),
                "summary": paper.summary,
                "pdf_url": paper.pdf_url,
                "entry_id": paper.entry_id,
            }
        )

    return downloaded_papers

def load_pdfs_as_chunks(papers: list[dict]) -> list[Document]:
    """
    Parse research PDFs and create chunks containing text or tables.
    """

    from docling.datamodel.accelerator_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    )
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption


    pipeline_options = PdfPipelineOptions(
        do_table_structure=True,
        accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice.CPU,
            num_threads=4,
        ),
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            ),
        }
    )
    
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL),
        max_tokens=450,
    )
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=False,
        repeat_table_header=True,
    )
    documents = []
    for paper in papers:
        parsed = converter.convert(str(paper["pdf_path"])).document

        for chunk_index, chunk in enumerate(
            chunker.chunk(dl_doc=parsed)
        ):
            items = chunk.meta.doc_items
            pages = sorted({
                provenance.page_no
                for item in items
                for provenance in item.prov
            })

            table_refs = sorted({
                item.self_ref
                for item in items
                if isinstance(item, TableItem)
            })

            content = chunker.contextualize(chunk=chunk)

            if not content.strip():
                continue

            metadata = {
                "paper_id": paper["paper_id"],
                "title": paper["title"],
                "authors": ",".join(paper["authors"]),
                "published": paper["published"],
                "pdf_url": paper["pdf_url"],
                "entry_id": paper["entry_id"],
                "source": str(paper["pdf_path"]),
                "content_type": "table" if table_refs else "text",
                "chunk_index": chunk_index,
                "pages":",".join(map(str, pages)),
                "table_ids":",".join(
                    f"{paper['paper_id']}:{ref}"
                    for ref in table_refs
                ),
            }

            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                )
            )
    return documents



def build_research_database(
    query: str,
    database_name: str,
    max_results: int = 10,
) -> dict:
    """Build a named Chroma collection from relevant arXiv papers.

    Args:
        query: Research topic used to search arXiv
        database_name: User-facing name for the research database.
        max_results: Maximum number of papers to process.

    Returns:
        Summary of the database building operation.

    Raises:
        ValueError: If the topic, database name, or result count is invalid.
    """

    query = query.strip()
    collection_name = normalize_database_name(database_name)

    if not query:
        raise ValueError("Research topic must not be empty")

    if max_results < 1:
        raise ValueError(f"max_results must be greater than zero. Got {max_results}")

    if database_exists(collection_name):
        return {
            "status": "already_exists",
            "database_name": collection_name,
            "message": (
                f"Database '{collection_name}' already exists. "
                "Select it from the existing database instead."
            ),
        }

    candidate_papers = search_arxiv_papers(
        query=query,
        max_candidates=max_results * 5,
    )

    relevant_papers = filter_relevant_papers(
        papers=candidate_papers,
        query=query,
        max_results=max_results,
    )

    papers = download_arxiv_papers(
        papers=relevant_papers,
        database_name=collection_name,
    )

    if not papers:
        return {
            "status": "no_results",
            "database_name": collection_name,
            "message": f"No arXiv papers were found for '{query}'.",
        }

    chunks = load_pdfs_as_chunks(papers)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
    )

    vectorstore.add_documents(chunks)

    register_database(
        database_name=database_name,
        research_topic=query,
        papers_processed=len(papers),
        chunks_indexed=len(chunks),
    )

    return {
        "status": "created",
        "database_name": collection_name,
        "display_name": database_name.strip(),
        "research_topic": query,
        "papers_processed": len(papers),
        "table_chunks_indexed": sum(
            chunk.metadata["content_type"] == "table"
            for chunk in chunks
        ),
        "chunks_indexed": len(chunks),
    }


def main() -> None:
    """Build a sample research database from the command line."""

    result = build_research_database(
        query="sparse subspace clustering",
        database_name="Sparse Subspace Clustering",
        max_results=30,
    )

    print(result)


if __name__ == "__main__":
    main()
