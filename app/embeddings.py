# embeddings.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle

model = SentenceTransformer("all-MiniLM-L6-v2")

INDEX_FILE = "faiss_index.index"
DOCS_FILE = "documents.pkl"

def load_all_documents():
    all_texts = []
    for file_name in os.listdir("data"):
        if file_name.endswith(".txt"):
            with open(f"data/{file_name}", "r", encoding="utf-8") as f:
                text = f.read()
            docs = text.split("\n\n")
            all_texts.extend([doc.strip() for doc in docs if doc.strip()])
    return all_texts

def create_or_load_index():
    # Load all documents from data folder
    documents = load_all_documents()

    # Encode and create FAISS index
    embeddings = model.encode(documents)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # Save index and documents
    faiss.write_index(index, INDEX_FILE)
    with open(DOCS_FILE, "wb") as f:
        pickle.dump(documents, f)

    return index, documents

def search(index, documents, query, top_k=2):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [documents[i] for i in indices[0]]
