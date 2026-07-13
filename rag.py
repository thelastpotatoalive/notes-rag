"""
rag.py - Core RAG engine
Shared functions for embedding, storing, retrieving and answering.
Used by both ingest.py and ask.py
"""

import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# --- Settings---
EMBED_MODEL = "all-MiniLM-L6-v2" # the embedding model 
LLM_MODEL = "llama3.2"           # the local model Ollama runs
DB_FOLDER = "vector_store"       # where chromadb saves data on disk
COLLECTION = "notes"             # name of our chunk collection

# Load the embedding model once, when this file is first imported.
# This avoids reloading the model every time we embed something.
print("Loading emebedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

def embed(texts):
    """
    Turns text into vectors (embeddings).
    Accepts a single string or a list of strings.
    Returns a list of vectors.
    """
    # If someone passes one string, wrap it in a list for consistency
    if isinstance(texts, str):
        texts = [texts]

    # The model does the actual work - convert text to number lists
    vectors = embedder.encode(texts).tolist()
    return vectors

# Connect to ChromaDB - it saves to disk in DB_FOLDER
# PersistentClient means the data survives after the program closes.
client = chromadb.PersistentClient(path=DB_FOLDER)
                                   
def get_collection():
    """
    Returns our collection of note chunks, creating it if it doesn't exist yet.
    A 'collection' is like a table - it holds all our embedded chunks.
    """
    return client.get_or_create_collection(name=COLLECTION)

def reset_collection():
    """Deletes the entire collection so we can start fresh."""
    try:
        client.delete_collection(name=COLLECTION)
    except Exception:
        pass  # collection didn't exist yet — that's fine
    return client.get_or_create_collection(name=COLLECTION)

def retrieve(question, n_results=4):
    """
    Finds the most relevant chunks from our notes for a given question
    1. Embeds the question into a vector
    2. Asks ChromaDB for the closest matching chunks 
    Returns a list of chunk texts and their sources.
    """
    collection = get_collection()
    
    # Turn the question into a vector (same method used on the notes)
    question_vector = embed(question)

    # Ask ChromaDB for the n closest chunks
    results = collection.query(
        query_embeddings=question_vector,
        n_results=n_results,
    )

    # Unpack the results into a cleaner format
    chunks = results["documents"][0]    # the chunk texts
    metadatas = results["metadatas"][0] # info about each chunk (e.g.source file)

    return chunks, metadatas 

def answer(question, n_results=4):
    """
    The full RAG pipeline:
    1. Retrieve relevant chunks for the question
    2. Build a prompt that includes those chunks as context
    3. Ask the local LLM to answer using only that context
    Returns the answer text and the sources used.
    """

    # Step 1 - get relevant chunks
    chunks, metadatas = retrieve(question, n_results)

    # Step 2 - combine the chunks into one context block
    context = "\n\n".join(chunks)

    # Build the prompt - this is where we instruct the LLM
    prompt = f"""Your are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I couldn't find that in your notes."

Context:
{context}

Question: {question}

Answer:"""

    # Step 3 - send the prompt to the local LLM via Ollama
    response = ollama.chat(
        model=LLM_MODEL,
        messages = [{"role": "user", "content": prompt}],
    )

    answer_text = response["message"]["content"]

    # Collect the unique source files for citation
    sources = list(set(m["source"] for m  in metadatas))

    return answer_text, sources