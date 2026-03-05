from app.services.embedding_service import search

def retrieve_top_chunks(index, stored_docs, query, top_k=2):
    if not index or not stored_docs:
        return []
    return search(index, stored_docs, query, top_k)