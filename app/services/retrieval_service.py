from rank_bm25 import BM25Okapi
from app.services.embedding_service import search
import logging

logger = logging.getLogger(__name__)

bm25 = None
documents = []

def initialize_bm25(stored_docs):
    """
    Initialize BM25 index from stored documents
    """
    global bm25, documents

    documents = stored_docs

    tokenized_docs = [
        doc.lower().split(" ")
        for doc in stored_docs
    ]

    bm25 = BM25Okapi(tokenized_docs)

    logger.info("BM25 index initialized.")


def retrieve_top_chunks(index, stored_docs, query, top_k=4):
    """
    Hybrid retrieval using FAISS + BM25
    """

    if not index or not stored_docs:
        return []

    global bm25

    # ---------------- FAISS semantic search ----------------
    faiss_results = search(index, stored_docs, query, top_k)

    # Convert FAISS docs to indices
    faiss_indices = [
        stored_docs.index(doc)
        for doc in faiss_results
        if doc in stored_docs
    ]

    # ---------------- BM25 keyword search ----------------
    tokenized_query = query.lower().split(" ")

    bm25_scores = bm25.get_scores(tokenized_query)

    top_bm25 = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:top_k]

    # ---------------- Merge results ----------------
    combined_indices = list(set(faiss_indices + top_bm25))

    retrieved_docs = [
        stored_docs[i]
        for i in combined_indices
    ]

    return retrieved_docs