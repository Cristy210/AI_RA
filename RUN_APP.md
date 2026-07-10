# Running the AI-Research Assistant

This guide walks through the complete process of running the AI research assistant locally, from forking the repository to asking your first research question. 

> **Note**
> This project is designed for **Apple Silicon Macs** and uses **MLX-LM** for local LLM inference. 

# 1. Fork the repository 

Navigate to the repository.

```text
https://github.com/Cristy210/AI_RA
```

Click **Fork** in the upper-right corner of GitHub to create your own copy. 

# 2. Clone the repository

Open a terminal and clone your fork. 

```bash
git clone https://github.com/<your-github-username>/AI_RA.git
```

Navigate into the project repository. 

```bash
cd AI_RA
```

# 3. Create the Conda Environment

The repository already contains an `environment.yml` file with all required dependencies. 

Create the Environment: 

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate AIRA
```

# 4. Build the Document Vector Database

Before running the application, the research papers must be downloaded and indexed. 

Run:

```bash
python src/ingest_arxiv.py
```

The script performs the following operations:

1. Searches arXiv for research papers. 
2. Downloads PDF files. 
3. Parses each PDF.
4. Splits documents into text chunks. 
5. Generates embeddings using 

```
BAAI/bge-small-en-v1.5
```

6. Stores the embeddings inside a persistent Chroma vector database. 

With the current configuration, the scripts downloads upto 30 papers matching 

```text
sparse subspace clustering
```

After indexing finishes successfully, you should see output similar to

```text
Done.
Downloaded/loaded papers: 30
Loaded PDF pages: 394
Indexed Chunks: 1545
```

> **Note**
>
> This indexing step only needs to be performed once for a given document collection. 

# 5. Launch the Application 

From the repository root, run

```bash
streamlit run app.py
```

Streamlit will start a local server. 

Open the URL in your browser. 

# 6. Wait for the language model to load

On startup, the application loads the local MLX language model

```text
mlx-community/Mistral-7B-Instruct-v0.3-4bit
```

The first launch may take several minutes because the model must be downloaded. 
Subsequent launches are significantly faster because the model is cached locally. 

# 7. Ask your first question

Once the application loads, the Streamlit interface will display a chat input.

Press **Enter**.

---

# Running the Application Again

After the vector database has been built once, you only need to

```bash
cd AI_RA

conda activate AIRA

streamlit run app.py
```

The existing vector database stored in

```
vectorstore/research_papers/
```

will automatically be reused.

You only need to rerun

```bash
python src/ingest_arxiv.py
```

if you want to download new papers or rebuild the vector database.

---

# Customizing the Research Topic

The arXiv search query is defined near the top of

```text
src/ingest_arxiv.py
```

For example,

```python
QUERY = "sparse subspace clustering"
MAX_RESULTS = 30
```

Modify these values to build a knowledge base for another research area.

Example:

```python
QUERY = "retrieval augmented generation"
MAX_RESULTS = 50
```

After modifying the query, rebuild the vector database:

```bash
python src/ingest_arxiv.py
```

Then launch the application again.