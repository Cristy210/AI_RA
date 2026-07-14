"""Utilities for downloading research paper PDFs."""

from pathlib import Path
import requests


def download_pdf(pdf_url: str, save_path: Path) -> Path:
    """Download a PDF  to the requested local path.
    
    Args:
        pdf_url: URL for the pdf file.
        save_path: Local path where the PDF should be saved.
    
    Returns:
        Path to the downloaded PDF.
    
    Raises:
        requests.HTTPError: If the HTTP request is unsuccessful.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(
        pdf_url,
        timeout=30,
        headers={
            "User-Agent": "AI-Research-Assistant/2.0",
        },
    )
    response.raise_for_status()

    save_path.write_bytes(response.content)

    return save_path