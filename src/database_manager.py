"""Utilities for managing research databases."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = PROJECT_ROOT / "vectorstore" / "research_papers"
REGISTRY_PATH = PROJECT_ROOT / "vectorstore" / "databases.json"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

def normalize_database_name(name:str) -> str:
    """Convert a database name into a safe Chroma collection name."""

    normalized = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        name.strip.lower(),
    ).strip("_")

    if not normalized:
        raise ValueError("Database name must not be empty")
    
    return normalized

def load_database_registry() -> dict:
    """Load the research database registry."""

    if not REGISTRY_PATH.exists():
        return {}
    
    with REGISTRY_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)
    
def save_database_registry(registry: dict) -> None:
    """Persist the research database registry."""

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REGISTRY_PATH.open("w", encoding="utf-8") as file:
        json.dump(registry, file, indent=2)

def register_database(
        database_name: str,
        research_topic: str,
        papers_processed: int,
        chunks_indexed: int,
) -> None:
    """Add or update a database in the registry."""

    normalized_name = normalize_database_name(database_name)
    registry = load_database_registry()
    registry[normalized_name] = {
        "name": normalized_name,
        "display_name": database_name.strip(),
        "research_topic": research_topic.strip(),
        "papers_processed": papers_processed,
        "chunks_indexed": chunks_indexed,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    
    save_database_registry(registry)

def list_research_databases() -> list[dict]:
    """Return registered research databases."""

    registry = load_database_registry()

    return sorted(
        registry.values(),
        key=lambda item: item["display_name"].lower(),
    )

def database_exists(database_name: str) -> bool:
    """Check whether a database is registered."""

    normalized_name = normalize_database_name(database_name)
    registry = load_database_registry()

    return normalized_name in registry

def load_vectorstore(database_name: str) -> Chroma:
    """Load an existing Chroma collection."""

    collection_name = normalize_database_name(database_name)

    embeddings = HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    return Chroma(
        collection_name=collection_name,
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
    )

def get_database_status(database_name: str) -> dict:
    """Return metadata and chunk count for a database."""
    
    collection_name = normalize_database_name(database_name)
    registry = load_database_registry()

    if collection_name not in registry:
        raise ValueError(
            f"Research Database '{collection_name}' does not exist."
        )
    
    vectorstore = load_vectorstore(collection_name)
    metadata = registry[collection_name]

    return{
        **metadata,
        "document_chunks": vectorstore._collection.count(),
        "ready": vectorstore._collection.count() > 0,
    }