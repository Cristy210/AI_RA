"""Create domain-specific research databases from arXiv papers."""

from pathlib import Path
import logging

import arxiv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.database_manager import(
    DB_DIR,
    EMBEDDING_MODEL,
    database_exists,
    normalize_database_name,
    register_database,
)
from src.utils.downloader import download_pdf

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = PROJECT_ROOT / "data" / "papers"

PAPER_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

def build_research_database(
    query:str,
    database_name: str,
    max_results: int = 10,
) -> dict:
    """Build a named research database from arXiv papers."""

    query = query.strip()
    collection_name = normalize_database_name(database_name)

    if not query:
        raise ValueError("Research topic must not be empty")
    if max_results < 1:
        raise ValueError(f"max_results must be greater than zero. Got {max_results}")
    if database_exists(collection_name):
        return{
            "status": "already_exists",
            "database_name": collection_name,
            "message": (
                f"Database '{collection_name} already exists. "
                "Select it from the existing database instead."
            ),
        }
    
    paper_dir = PAPE