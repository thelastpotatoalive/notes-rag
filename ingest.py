"""
ingest.py — Loads PDFs into the vector store.
Run this once before asking questions (and again when you add new notes).
Usage: python ingest.py
"""

import os
from pypdf import PdfReader
from rag import embed, get_collection


def read_pdf(path):
    """Extracts all text from a single PDF file, page by page."""
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:                 # some pages (e.g. images) return nothing
            text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits a long text into smaller overlapping chunks.
    - chunk_size: roughly how many words per chunk
    - overlap: how many words each chunk shares with the previous one
    Overlap prevents losing meaning at the boundaries between chunks.
    """
    words = text.split()          # split into individual words
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])   # join this slice back into text
        chunks.append(chunk)
        start += chunk_size - overlap        # move forward, but step back by overlap

    return chunks

def ingest(notes_folder="notes"):
    """
    Main function: reads every PDF in the notes folder, chunks each one,
    embeds the chunks, and stores them in the vector store.
    """
    collection = get_collection()

    # Find all PDF files in the notes folder
    pdf_files = [f for f in os.listdir(notes_folder) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDFs found in '{notes_folder}'. Add some and try again.")
        return

    print(f"Found {len(pdf_files)} PDF(s). Processing...")

    chunk_id = 0
    for pdf in pdf_files:
        path = os.path.join(notes_folder, pdf)
        print(f"  Reading {pdf}...")

        text = read_pdf(path)
        chunks = chunk_text(text)
        print(f"    Split into {len(chunks)} chunks")

        # Embed all chunks from this PDF
        vectors = embed(chunks)

        # Store them in ChromaDB
        collection.add(
            ids=[f"chunk_{chunk_id + i}" for i in range(len(chunks))],
            embeddings=vectors,
            documents=chunks,
            metadatas=[{"source": pdf} for _ in chunks],
        )
        chunk_id += len(chunks)

    print(f"\nDone. Stored {chunk_id} chunks total.")
    print("You can now run: python ask.py")


if __name__ == "__main__":
    ingest()