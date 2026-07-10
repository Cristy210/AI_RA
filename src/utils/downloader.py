from pathlib import Path
import requests


def download_pdf(pdf_url: str, save_path: Path):
    response = requests.get(pdf_url, timeout=30)
    response.raise_for_status()

    with open(save_path, "wb") as f:
        f.write(response.content)
