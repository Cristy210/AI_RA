# Running the Streamlit Application

This guide walks through running the AI Research Assistant locally—from cloning the repository to asking your first research question.

> **Note**
>
> This project is designed for **Apple Silicon Macs** and uses **MLX-LM** for fully local large language model inference.

---

# 1. Fork the Repository

Navigate to the repository:

```text
https://github.com/Cristy210/AI_RA
```

Click **Fork** in the upper-right corner of GitHub to create your own copy.

---

# 2. Clone Your Fork

```bash
git clone https://github.com/<your-github-username>/AI_RA.git

cd AI_RA
```

---

# 3. Create the Conda Environment

The repository includes an `environment.yml` file containing all required dependencies.

Create the environment:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate AIRA
```

---

# 4. Launch the Streamlit Application

From the project root, start Streamlit:

```bash
streamlit run app.py
```

Streamlit will start a local web server.

Open the URL displayed in your terminal (typically `http://localhost:8501`).

---

# 5. Wait for the Language Model

When the application starts for the first time, it loads the local MLX language model:

```text
mlx-community/Mistral-7B-Instruct-v0.3-4bit
```

The initial launch may take a few minutes while the model is downloaded and cached.

Subsequent launches are significantly faster.

---

# 6. Create Your First Research Database

When the application opens, the sidebar displays two options:

- Create a new database
- Select an existing database

Choose **Create a new database**.

Provide:

| Field | Example |
|-------|---------|
| Database Name | Subspace_Clustering |
| arXiv Research Topic | subspace clustering |
| Maximum Papers | 10 |

Click **Build Database**.

The application will automatically:

1. Search arXiv
2. Download matching papers
3. Parse PDFs
4. Split documents into text chunks
5. Generate embeddings using

```text
BAAI/bge-small-en-v1.5
```

6. Store the embeddings in a persistent Chroma vector database
7. Register the database for future use

Depending on the number of papers selected, this process may take several minutes.

---

# 7. Ask Your First Question

Once indexing completes, the selected database becomes active.

Use the chat input at the bottom of the page to ask a research question, for example:

```text
What optimization methods are used in subspace clustering algorithms?
```

The application will:

1. Retrieve the most relevant document chunks.
2. Construct a grounded prompt.
3. Generate an evidence-based response using the local MLX model.

---

# Working with Multiple Databases

The application supports multiple research databases.

Use the sidebar to switch between databases without rebuilding embeddings.

Each database stores:

- Research topic
- Number of indexed papers
- Number of document chunks

Selecting a different database automatically loads its corresponding Chroma collection.

---

# Reusing Existing Databases

Previously created databases are stored locally and automatically detected the next time the application starts.

Simply launch Streamlit again:

```bash
conda activate AIRA

streamlit run app.py
```

Then select the desired database from the sidebar.

No re-indexing is required unless you want to create a new research database.

---

# Creating a Different Knowledge Base

To build a database for another research area:

1. Select **Create a new database**
2. Enter a new database name
3. Enter a different arXiv search topic
4. Click **Build Database**

For example:

| Database Name | Research Topic |
|--------------|----------------|
| julia_ml | Julia adaptation in machine learning |
| LLMs | large language models |
| graph_clustering | graph clustering |

Each topic is indexed independently and can be selected from the sidebar whenever needed.