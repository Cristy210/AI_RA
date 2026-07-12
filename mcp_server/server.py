from mcp.server.fastmcp import FastMCP

from llm.mlx_model import MLXModel
from src.rag import load_retriever, rag_answer

mcp = FastMCP("AI Research Assistant")

retriever = load_retriever(k=10)
llm = MLXModel(max_tokens=700)

@mcp.tool()
def ask_research_question(query:str) -> str:
    """Answer a research question using locally indexed research papers
    
    Args:
        query: Research question to answer.
    Returns:
        Evidence-based response generated from retrieved paper excerpts.
    """

    return rag_answer(
        query=query,
        retriever=retriever,
        llm=llm,
    )

if __name__ == "__main__":
    mcp.run()