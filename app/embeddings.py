# embeddings.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle
import logging
import hashlib
from PyPDF2 import PdfReader
from docx import Document

logger = logging.getLogger(__name__)

model = SentenceTransformer("all-MiniLM-L6-v2")

INDEX_FILE = "faiss_index.index"
DOCS_FILE = "documents.pkl"
DATA_DIR = "data"

# ----------------- TEXT EXTRACTION -----------------
def extract_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == ".pdf":
        text = ""
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        logger.warning(f"Unsupported file type: {ext}")
        return ""  # ignore unsupported files

# ----------------- LOAD DOCUMENTS -----------------
def load_all_documents():
    all_texts = []
    for file_name in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file_name)
        text = extract_text_from_file(file_path)
        if text.strip():  # ignore empty content
            all_texts.append(text.strip())
    return all_texts

# ----------------- HASH UTILITY -----------------
def file_hash(file_path: str) -> str:
    """Return SHA256 hash of the file content."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

# ----------------- INDEX CREATION & LOADING -----------------
def create_or_load_index():
    current_docs = load_all_documents()

    # CASE 1: No index exists
    if not os.path.exists(INDEX_FILE) or not os.path.exists(DOCS_FILE):
        if not current_docs:
            return None, []

        embeddings = model.encode(current_docs)
        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))
        logger.info(f"FAISS index size now: {index.ntotal}")

        faiss.write_index(index, INDEX_FILE)
        with open(DOCS_FILE, "wb") as f:
            pickle.dump(current_docs, f)
        return index, current_docs

    # CASE 2 & 3: Index exists
    index = faiss.read_index(INDEX_FILE)
    try:
        with open(DOCS_FILE, "rb") as f:
            stored_docs = pickle.load(f)
        if index.ntotal != len(stored_docs):
            logger.error("Index and metadata mismatch!")
            raise ValueError("FAISS index corrupted or out of sync.")
    except Exception:
        logger.error("Failed to load documents.pkl. Rebuilding index from scratch...")
        os.remove(INDEX_FILE)
        os.remove(DOCS_FILE)
        return create_or_load_index()

    # Detect new documents (avoid duplicates using hash)
    existing_hashes = {hashlib.sha256(doc.encode("utf-8")).hexdigest() for doc in stored_docs}
    new_docs = [doc for doc in current_docs if hashlib.sha256(doc.encode("utf-8")).hexdigest() not in existing_hashes]

    if not new_docs:
        return index, stored_docs  # no new docs

    # Incrementally add new embeddings
    new_embeddings = model.encode(new_docs)
    index.add(np.array(new_embeddings))
    logger.info(f"FAISS index size now: {index.ntotal}")

    stored_docs.extend(new_docs)

    # Save updated index + metadata
    faiss.write_index(index, INDEX_FILE)
    with open(DOCS_FILE, "wb") as f:
        pickle.dump(stored_docs, f)

    return index, stored_docs

# ----------------- SEARCH -----------------
def search(index, documents, query, top_k=2):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    logger.info(f"Top-k indices retrieved: {indices[0]}")
    logger.info(f"Distances: {distances[0]}")
    return [documents[i] for i in indices[0]]