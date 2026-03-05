from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os, pickle, hashlib, logging
from PyPDF2 import PdfReader
from docx import Document
from app.core.config import DATA_DIR, INDEX_FILE, DOCS_FILE

logger = logging.getLogger(__name__)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Supported file types
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
        return "\n".join([page.extract_text() for page in reader.pages])
    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        logger.warning(f"Unsupported file type: {ext}")
        return ""

# ----------------- LOAD DOCUMENTS -----------------
def load_all_documents():
    all_texts = []
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    for file_name in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file_name)
        text = extract_text_from_file(file_path)
        if text.strip():
            all_texts.append(text.strip())
    return all_texts

# ----------------- HASH UTILITY -----------------
def file_hash(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

# ----------------- INDEX CREATION & LOADING -----------------
def create_or_load_index():
    current_docs = load_all_documents()

    if not os.path.exists(INDEX_FILE) or not os.path.exists(DOCS_FILE):
        if not current_docs:
            return None, []
        embeddings = model.encode(current_docs)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))
        faiss.write_index(index, INDEX_FILE)
        with open(DOCS_FILE, "wb") as f:
            pickle.dump(current_docs, f)
        return index, current_docs

    index = faiss.read_index(INDEX_FILE)
    try:
        with open(DOCS_FILE, "rb") as f:
            stored_docs = pickle.load(f)
        if index.ntotal != len(stored_docs):
            raise ValueError("FAISS index corrupted or out of sync.")
    except Exception:
        os.remove(INDEX_FILE)
        os.remove(DOCS_FILE)
        return create_or_load_index()

    existing_hashes = {hashlib.sha256(doc.encode("utf-8")).hexdigest() for doc in stored_docs}
    new_docs = [doc for doc in current_docs if hashlib.sha256(doc.encode("utf-8")).hexdigest() not in existing_hashes]
    if new_docs:
        new_embeddings = model.encode(new_docs)
        index.add(np.array(new_embeddings))
        stored_docs.extend(new_docs)
        faiss.write_index(index, INDEX_FILE)
        with open(DOCS_FILE, "wb") as f:
            pickle.dump(stored_docs, f)

    return index, stored_docs

# ----------------- SEARCH -----------------
def search(index, documents, query, top_k=2):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [documents[i] for i in indices[0]]