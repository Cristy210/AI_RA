from pathlib import Path
import arxiv

from utils.downloader import download_pdf
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

QUERY = "sparse subspace clustering"
MAX_RESULTS = 10

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PAPER_DIR = PROJECT_ROOT / "data" / "papers"
DB_DIR = PROJECT_ROOT / "vectorstore" / "research_papers"

PAPER_DIR.mkdir(parents=True, exist_ok=True)

def download_arxiv_papers(query:str, max_results:int):
    client = arxiv.Client()

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    downloaded_papers = []

    for paper in client.results(search):
        paper_id = paper.entry_id.split("/")[-1]
        pdf_path = PAPER_DIR / f"{paper_id}.pdf"

        if not pdf_path.exists():
            print(f"Downloading: {paper.title}")
            download_pdf(paper.pdf_url, pdf_path)
        else:
            print(f"Already exists: {paper.title}")
        
        downloaded_papers.append(
            {
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
            
# papers = download_arxiv_papers(QUERY, MAX_RESULTS)
# print(papers)

def load_pdfs_as_docs(papers):
    docs = []

    for paper in papers:
        loader = PyPDFLoader(str(paper["pdf_path"]))
        pages  = loader.load()

        for page in pages:
            page.metadata.update(
                {
                    "title": paper["title"],
                    "authors": ",".join(paper["authors"]),
                    "published": paper["published"],
                    "summary": paper["summary"],
                    "pdf_url": paper["pdf_url"],
                    "entry_id": paper["entry_id"],
                }
            )
        docs.extend(pages)
    
    return docs

def main():
    papers = download_arxiv_papers(QUERY, MAX_RESULTS)
    docs = load_pdfs_as_docs(papers)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name = "BAAI/bge-small-en-v1.5",
        encode_kwargs = {"normalize_embeddings": True},
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )

    print("\nDone.")
    print(f"Downloaded/loaded papers: {len(papers)}")
    print(f"Loaded PDF pages: {len(docs)}")
    print(f"Indexed Chunks: {len(chunks)}")
    print(f"Saved Vector DB to: {DB_DIR}")

if __name__ == "__main__":
    main()