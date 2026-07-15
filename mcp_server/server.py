"""MCP server for the AI Research Assistant."""

from mcp.server.fastmcp import FastMCP

from llm.mlx_model import MLXModel
from src.database_manager import (
    get_database_status,
    list_research_databases,
)
from src.ingest_arxiv import build_research_database
from src.rag import rag_answer

mcp = FastMCP("AI Research Assistant")

llm = MLXModel(max_tokens=700)


@mcp.tool()
def list_databases() -> list[dict]:
    """List the available research databases."""

    return list_research_databases()


@mcp.tool()
def build_database(
    research_topic: str,
    database_name: str,
    max_results: int = 10,
) -> dict:
    """Build a research database from relevant arXiv papers.

    Args:
        research_topic: Topic used to search arXiv.
        database_name: Name assigned to the research database.
        max_results: Maximum number of papers to index.

    Returns:
        Summary of the completed database-building operation.
    """

    return build_research_database(
        query=research_topic, database_name=database_name, max_results=max_results
    )


@mcp.tool()
def database_status(database_name: str) -> dict:
    """Return status information for a research database."""

    return get_database_status(database_name)


@mcp.tool()
def ask_research_question(
    query: str,
    database_name: str,
) -> str:
    """Answer a research using a selected research database.

    Args:
        query: Research question to answer.
        database_name: Research database containing relevant papers
    Returns:
        Evidence-based response generated from retrieved paper excerpts.
    """

    return rag_answer(
        query=query,
        database_name=database_name,
        llm=llm,
    )


if __name__ == "__main__":
    mcp.run()
