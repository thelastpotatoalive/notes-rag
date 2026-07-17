# 📚 Notes RAG — Ask Questions About Your Own PDFs

A fully **local** Retrieval-Augmented Generation (RAG) system that answers questions about your own PDF notes — lecture slides, textbooks, anything. No API keys, no cloud, no cost. Everything runs on your machine and your documents never leave it.

Built to understand how modern AI assistants (customer support bots, documentation search, research tools) actually work under the hood.

---

## What It Does

Point it at a folder of PDFs, and it lets you ask questions in plain English. Instead of guessing from general knowledge, it retrieves the most relevant passages from *your* notes and uses a local language model to answer — with sources cited.

```
Your question: What is the threshold voltage of a MOSFET?

==================================================
ANSWER:
The threshold voltage (Vth) is the minimum gate-to-source
voltage needed to create a conducting channel between the
source and drain...

SOURCES: lecture3_transistors.pdf
==================================================
```

---

## How It Works

The system runs in two phases:

**Setup phase** (`ingest.py`) — done once:
1. Reads text from every PDF in the `notes/` folder
2. Splits each document into overlapping chunks
3. Embeds each chunk into a vector using a neural network
4. Stores the vectors in a local database

**Query phase** (`ask.py`) — every time you ask:
1. Embeds your question the same way
2. Finds the most similar chunks by meaning (not keywords)
3. Feeds those chunks to a local LLM as context
4. Returns a grounded answer with sources

---

## Tech Stack

| Component | Tool | Role |
|---|---|---|
| PDF reading | `pypdf` | Extracts text from PDFs |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | Turns text into vectors |
| Vector store | `chromadb` | Stores & searches vectors locally |
| LLM | `Ollama` (llama3.2) | Generates answers on-device |

Everything runs locally — the embedding model and LLM both live on disk after a one-time download.

---

## Setup

**1. Install [Ollama](https://ollama.com/download)** and pull a model:
```bash
ollama pull llama3.2
```

**2. Install Python dependencies:**
```bash
git clone https://github.com/thelastpotatoalive/notes-rag.git
cd notes-rag
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**3. Add your PDFs** to a `notes/` folder in the project directory.

---

## Usage

**Process your notes** (run once, and again whenever you add or change PDFs):
```bash
python ingest.py
```

**Ask questions:**
```bash
python ask.py
```

Type your questions at the prompt. Type `quit` to exit.

---

## Project Structure

```
notes-rag/
├── rag.py            # core engine: embedding, retrieval, generation
├── ingest.py         # reads PDFs, chunks, embeds, stores
├── ask.py            # interactive Q&A loop
├── requirements.txt  # dependencies
└── notes/            # your PDFs (git-ignored, stays private)
```

---

## Design Decisions

**Why chunk with overlap?** Documents are split into ~500-word chunks with a 50-word overlap. The overlap ensures that ideas falling on a chunk boundary still appear intact in at least one chunk, which improves retrieval accuracy.

**Why local?** Running the embedding model and LLM on-device means your notes stay completely private, there's no per-query cost, and it works offline. The trade-off is that a local model is less powerful than a large cloud API — but for answering questions from a fixed set of notes, it's more than sufficient.

**Why reset on ingest?** Each ingestion clears the vector store and rebuilds it from the current `notes/` folder, so removing a PDF actually removes its content from the system.

---

## What I Learned

- How retrieval-augmented generation works end to end
- How embeddings turn language into searchable vectors, and why semantic search beats keyword matching
- How vector databases store and query high-dimensional data
- Prompt engineering — constraining an LLM to answer only from provided context
- Running LLMs locally with Ollama and GPU acceleration

---

