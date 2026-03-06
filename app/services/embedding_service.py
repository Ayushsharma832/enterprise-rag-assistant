# embedding_service.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os, pickle, hashlib, logging
from PyPDF2 import PdfReader
from docx import Document
from app.core.config import DATA_DIR, INDEX_FILE, DOCS_FILE

logger = logging.getLogger(__name__)
model = SentenceTransformer("all-MiniLM-L6-v2")


# ----------------- FILE TYPES -----------------
def supported_file_types():
    return [".txt", ".pdf", ".docx"]


# ----------------- TEXT EXTRACTION -----------------
def extract_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join([page.extract_text() or "" for page in reader.pages])

    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    else:
        logger.warning(f"Unsupported file type: {ext}")
        return ""


# ----------------- CHUNKING -----------------
def split_into_chunks(text, chunk_size=200, overlap=50):
    """
    Splits text into overlapping chunks.
    chunk_size = number of words
    overlap = overlapping words between chunks
    """

    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ----------------- LOAD DOCUMENTS -----------------
def load_all_documents():
    all_chunks = []

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    for file_name in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file_name)

        text = extract_text_from_file(file_path)

        if not text.strip():
            continue

        chunks = split_into_chunks(text)

        all_chunks.extend(chunks)

    return all_chunks


# ----------------- HASH UTILITY -----------------
def file_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ----------------- INDEX CREATION & LOADING -----------------
def create_or_load_index():

    current_chunks = load_all_documents()

    # First time index creation
    if not os.path.exists(INDEX_FILE) or not os.path.exists(DOCS_FILE):

        if not current_chunks:
            return None, []

        embeddings = model.encode(current_chunks)

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)

        index.add(np.array(embeddings))

        faiss.write_index(index, INDEX_FILE)

        with open(DOCS_FILE, "wb") as f:
            pickle.dump(current_chunks, f)

        return index, current_chunks


    # Load existing index
    index = faiss.read_index(INDEX_FILE)

    try:
        with open(DOCS_FILE, "rb") as f:
            stored_chunks = pickle.load(f)

        if index.ntotal != len(stored_chunks):
            raise ValueError("FAISS index corrupted or out of sync.")

    except Exception:

        os.remove(INDEX_FILE)
        os.remove(DOCS_FILE)

        return create_or_load_index()


    # Detect new chunks
    existing_hashes = {file_hash(doc) for doc in stored_chunks}

    new_chunks = [
        chunk for chunk in current_chunks
        if file_hash(chunk) not in existing_hashes
    ]

    if new_chunks:

        logger.info(f"Adding {len(new_chunks)} new chunks to index")

        new_embeddings = model.encode(new_chunks)

        index.add(np.array(new_embeddings))

        stored_chunks.extend(new_chunks)

        faiss.write_index(index, INDEX_FILE)

        with open(DOCS_FILE, "wb") as f:
            pickle.dump(stored_chunks, f)

    return index, stored_chunks


# ----------------- SEARCH -----------------
def search(index, documents, query, top_k=4):

    query_embedding = model.encode([query])

    distances, indices = index.search(
        np.array(query_embedding),
        top_k
    )

    return [documents[i] for i in indices[0]]